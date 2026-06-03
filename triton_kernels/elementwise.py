from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _freqs_as_real(freqs_cis: torch.Tensor) -> torch.Tensor:
    if freqs_cis.is_complex():
        return torch.view_as_real(freqs_cis).flatten(-2)
    return freqs_cis


def _read_i32_from_plan_torch(plan_tensor: torch.Tensor, field: int) -> torch.Tensor:
    if plan_tensor.dtype == torch.uint8:
        return plan_tensor[:, field * 4 : field * 4 + 4].contiguous().view(torch.int32).squeeze(-1)
    if plan_tensor.dtype == torch.int32:
        return plan_tensor.view(torch.int32).reshape(plan_tensor.shape[0], -1)[:, field]
    raise TypeError(f"Unsupported plan dtype: {plan_tensor.dtype}")


def _plan_tensor_as_i32(plan_tensor: torch.Tensor) -> torch.Tensor:
    if plan_tensor.dtype == torch.int32:
        return plan_tensor.view(torch.int32).reshape(plan_tensor.shape[0], -1)
    if plan_tensor.dtype == torch.uint8:
        return plan_tensor.view(torch.int32).reshape(plan_tensor.shape[0], -1)
    raise TypeError(f"Unsupported plan dtype: {plan_tensor.dtype}")


def fused_rope_inplace_torch(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    cos = freqs_cis.real.index_select(0, positions.to(torch.long)).repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.index_select(0, positions.to(torch.long)).repeat_interleave(2, dim=-1)
    if inverse:
        sin = -sin

    def _apply(x: torch.Tensor) -> torch.Tensor:
        local_cos = cos
        local_sin = sin
        while local_cos.ndim < x.ndim:
            local_cos = local_cos.unsqueeze(1)
            local_sin = local_sin.unsqueeze(1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return ((x.float() * local_cos) + (rotated.float() * local_sin)).to(x.dtype)

    q.copy_(_apply(q))
    if k is not None:
        k.copy_(_apply(k))


@triton.jit
def _fused_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    positions_ptr,
    batch_size: tl.constexpr,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    rope_dim: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    freqs_stride_pos: tl.constexpr,
    freqs_stride_d: tl.constexpr,
    has_k: tl.constexpr,
    inverse: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    pair_offsets = pid_d * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    mask = pair_offsets < (rope_dim // 2)

    pos = tl.load(positions_ptr + pid_b).to(tl.int64)
    freq_real = tl.load(
        freqs_ptr + pos * freqs_stride_pos + pair_offsets * 2 * freqs_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    freq_imag = tl.load(
        freqs_ptr + pos * freqs_stride_pos + (pair_offsets * 2 + 1) * freqs_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if inverse:
        freq_imag = -freq_imag

    if pid_h < q_heads:
        q_base = pid_b * q_stride_b + pid_h * q_stride_h
        q_even_offsets = q_base + pair_offsets * 2 * q_stride_d
        q_odd_offsets = q_base + (pair_offsets * 2 + 1) * q_stride_d
        q_even = tl.load(q_ptr + q_even_offsets, mask=mask, other=0.0).to(tl.float32)
        q_odd = tl.load(q_ptr + q_odd_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(q_ptr + q_even_offsets, q_even * freq_real - q_odd * freq_imag, mask=mask)
        tl.store(q_ptr + q_odd_offsets, q_even * freq_imag + q_odd * freq_real, mask=mask)

    if has_k and pid_h < k_heads:
        k_base = pid_b * k_stride_b + pid_h * k_stride_h
        k_even_offsets = k_base + pair_offsets * 2 * k_stride_d
        k_odd_offsets = k_base + (pair_offsets * 2 + 1) * k_stride_d
        k_even = tl.load(k_ptr + k_even_offsets, mask=mask, other=0.0).to(tl.float32)
        k_odd = tl.load(k_ptr + k_odd_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(k_ptr + k_even_offsets, k_even * freq_real - k_odd * freq_imag, mask=mask)
        tl.store(k_ptr + k_odd_offsets, k_even * freq_imag + k_odd * freq_real, mask=mask)


def fused_rope_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    if q.device.type != "cuda":
        fused_rope_inplace_torch(q, k, freqs_cis, positions, inverse)
        return
    if q.ndim not in (2, 3) or (k is not None and k.ndim not in (2, 3)):
        fused_rope_inplace_torch(q, k, freqs_cis, positions, inverse)
        return
    if q.shape[0] != positions.shape[0] or q.shape[-1] % 2 != 0:
        fused_rope_inplace_torch(q, k, freqs_cis, positions, inverse)
        return
    if k is not None and (k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]):
        fused_rope_inplace_torch(q, k, freqs_cis, positions, inverse)
        return

    freqs_real = _freqs_as_real(freqs_cis)
    if not freqs_real.is_cuda:
        freqs_real = freqs_real.to(device=q.device)
    positions = positions.to(device=q.device)

    batch_size = q.shape[0]
    q_heads = q.shape[1] if q.ndim == 3 else 1
    k_heads = k.shape[1] if k is not None and k.ndim == 3 else (1 if k is not None else 0)
    max_heads = max(q_heads, k_heads)
    q_stride_b = q.stride(0)
    q_stride_h = q.stride(1) if q.ndim == 3 else 0
    q_stride_d = q.stride(-1)
    if k is None:
        k_arg = q
        k_stride_b = k_stride_h = k_stride_d = 0
    else:
        k_arg = k
        k_stride_b = k.stride(0)
        k_stride_h = k.stride(1) if k.ndim == 3 else 0
        k_stride_d = k.stride(-1)

    block_pairs = min(128, triton.next_power_of_2(max(1, q.shape[-1] // 2)))
    grid = (batch_size, max_heads, triton.cdiv(q.shape[-1] // 2, block_pairs))
    _fused_rope_kernel[grid](
        q,
        k_arg,
        freqs_real,
        positions,
        batch_size,
        q_heads,
        k_heads,
        q.shape[-1],
        q_stride_b,
        q_stride_h,
        q_stride_d,
        k_stride_b,
        k_stride_h,
        k_stride_d,
        freqs_real.stride(0),
        freqs_real.stride(1),
        k is not None,
        inverse,
        BLOCK_PAIRS=block_pairs,
    )


def compressor_positions_from_plan_torch(
    plan_tensor: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    seq_lens = _read_i32_from_plan_torch(plan_tensor, 0)
    return (seq_lens.to(torch.int32) - int(compress_ratio)).clamp(min=0)


@triton.jit
def _compressor_positions_from_plan_kernel(
    plan_ptr,
    positions_ptr,
    rows: tl.constexpr,
    plan_stride_row: tl.constexpr,
    compress_ratio: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    seq_len = _load_plan_i32(plan_ptr, offsets, plan_stride_row, 0, mask)
    positions = tl.maximum(seq_len - compress_ratio, 0)
    tl.store(positions_ptr + offsets, positions, mask=mask)


def compressor_positions_from_plan(
    plan_tensor: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    if plan_tensor.device.type != "cuda":
        return compressor_positions_from_plan_torch(plan_tensor, compress_ratio)
    if plan_tensor.dtype == torch.int32:
        plan_i32 = _plan_tensor_as_i32(plan_tensor)
        return (plan_i32[:, 0] - int(compress_ratio)).clamp(min=0).to(torch.int32)
    if plan_tensor.dtype != torch.uint8:
        return compressor_positions_from_plan_torch(plan_tensor, compress_ratio)
    rows = plan_tensor.shape[0]
    positions = torch.empty((rows,), dtype=torch.int32, device=plan_tensor.device)
    if rows == 0:
        return positions
    block = 256
    _compressor_positions_from_plan_kernel[(triton.cdiv(rows, block),)](
        plan_tensor,
        positions,
        rows,
        plan_tensor.stride(0),
        int(compress_ratio),
        BLOCK=block,
    )
    return positions


def compressor_decode_mask_positions_torch(
    kv_compressed: torch.Tensor,
    plan_tensor: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    seq_lens = _read_i32_from_plan_torch(plan_tensor, 0)
    is_boundary = (seq_lens % int(compress_ratio) == 0).view(-1, *([1] * (kv_compressed.ndim - 1)))
    kv_compressed.copy_(torch.where(is_boundary, kv_compressed, torch.zeros_like(kv_compressed)))
    return (seq_lens.to(torch.int32) - int(compress_ratio)).clamp(min=0)


def compressor_prefill_metadata_torch(
    plan_tensor: torch.Tensor,
    out_loc: torch.Tensor,
    compress_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_lens = _read_i32_from_plan_torch(plan_tensor, 0)
    ragged_ids = _read_i32_from_plan_torch(plan_tensor, 1) & 0xFFFF
    positions = (seq_lens.to(torch.int32) - int(compress_ratio)).clamp(min=0)
    return positions, out_loc[ragged_ids.to(torch.long)]


@triton.jit
def _load_plan_i32(plan_ptr, row, stride_row: tl.constexpr, field: tl.constexpr, mask, other: tl.constexpr = 0):
    base = row * stride_row + field * 4
    b0 = tl.load(plan_ptr + base + 0, mask=mask, other=other).to(tl.int32)
    b1 = tl.load(plan_ptr + base + 1, mask=mask, other=other).to(tl.int32)
    b2 = tl.load(plan_ptr + base + 2, mask=mask, other=other).to(tl.int32)
    b3 = tl.load(plan_ptr + base + 3, mask=mask, other=other).to(tl.int32)
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


@triton.jit
def _compressor_decode_mask_positions_kernel(
    kv_ptr,
    plan_ptr,
    positions_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    kv_stride_row: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    plan_stride_row: tl.constexpr,
    compress_ratio: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    block_d = tl.program_id(1)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < dim
    seq_len = _load_plan_i32(plan_ptr, row, plan_stride_row, 0, row < rows)
    boundary = (seq_len % compress_ratio) == 0
    vals = tl.load(kv_ptr + row * kv_stride_row + offsets_d * kv_stride_dim, mask=mask_d, other=0.0)
    vals = tl.where(boundary, vals, 0.0)
    tl.store(kv_ptr + row * kv_stride_row + offsets_d * kv_stride_dim, vals, mask=mask_d)
    pos = tl.maximum(seq_len - compress_ratio, 0)
    tl.store(positions_ptr + row, pos, mask=block_d == 0)


def compressor_decode_mask_positions(
    kv_compressed: torch.Tensor,
    plan_tensor: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    if kv_compressed.device.type != "cuda":
        return compressor_decode_mask_positions_torch(kv_compressed, plan_tensor, compress_ratio)
    if plan_tensor.dtype == torch.int32:
        seq_lens = _plan_tensor_as_i32(plan_tensor)[:, 0].to(torch.int32)
        is_boundary = (seq_lens % int(compress_ratio) == 0).view(-1, *([1] * (kv_compressed.ndim - 1)))
        kv_compressed.copy_(torch.where(is_boundary, kv_compressed, torch.zeros_like(kv_compressed)))
        return (seq_lens - int(compress_ratio)).clamp(min=0)
    if plan_tensor.dtype != torch.uint8:
        return compressor_decode_mask_positions_torch(kv_compressed, plan_tensor, compress_ratio)
    rows = kv_compressed.shape[0]
    dim = kv_compressed.numel() // max(rows, 1)
    if rows == 0:
        return torch.empty((0,), dtype=torch.int32, device=kv_compressed.device)
    kv_view = kv_compressed.reshape(rows, dim)
    positions = torch.empty((rows,), dtype=torch.int32, device=kv_compressed.device)
    block_d = min(1024, triton.next_power_of_2(max(1, dim)))
    grid = (rows, triton.cdiv(dim, block_d))
    _compressor_decode_mask_positions_kernel[grid](
        kv_view,
        plan_tensor,
        positions,
        rows,
        dim,
        kv_view.stride(0),
        kv_view.stride(1),
        plan_tensor.stride(0),
        int(compress_ratio),
        BLOCK_D=block_d,
    )
    return positions


@triton.jit
def _compressor_prefill_metadata_kernel(
    plan_ptr,
    out_loc_ptr,
    positions_ptr,
    selected_out_loc_ptr,
    rows: tl.constexpr,
    plan_stride_row: tl.constexpr,
    compress_ratio: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    seq_len = _load_plan_i32(plan_ptr, offsets, plan_stride_row, 0, mask)
    ragged_id = _load_plan_i32(plan_ptr, offsets, plan_stride_row, 1, mask) & 0xFFFF
    positions = tl.maximum(seq_len - compress_ratio, 0)
    selected = tl.load(out_loc_ptr + ragged_id, mask=mask, other=0)
    tl.store(positions_ptr + offsets, positions, mask=mask)
    tl.store(selected_out_loc_ptr + offsets, selected, mask=mask)


def compressor_prefill_metadata(
    plan_tensor: torch.Tensor,
    out_loc: torch.Tensor,
    compress_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if plan_tensor.device.type != "cuda":
        return compressor_prefill_metadata_torch(plan_tensor, out_loc, compress_ratio)
    if plan_tensor.dtype == torch.int32:
        plan_i32 = _plan_tensor_as_i32(plan_tensor)
        positions = (plan_i32[:, 0] - int(compress_ratio)).clamp(min=0).to(torch.int32)
        ragged_ids = (plan_i32[:, 1] & 0xFFFF).to(torch.long)
        return positions, out_loc[ragged_ids]
    if plan_tensor.dtype != torch.uint8:
        return compressor_prefill_metadata_torch(plan_tensor, out_loc, compress_ratio)
    rows = plan_tensor.shape[0]
    positions = torch.empty((rows,), dtype=torch.int32, device=plan_tensor.device)
    selected_out_loc = torch.empty((rows,), dtype=out_loc.dtype, device=out_loc.device)
    if rows == 0:
        return positions, selected_out_loc
    block = 256
    _compressor_prefill_metadata_kernel[(triton.cdiv(rows, block),)](
        plan_tensor,
        out_loc,
        positions,
        selected_out_loc,
        rows,
        plan_tensor.stride(0),
        int(compress_ratio),
        BLOCK=block,
    )
    return positions, selected_out_loc
