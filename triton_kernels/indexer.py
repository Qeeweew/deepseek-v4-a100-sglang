from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _hadamard_inplace_or_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    from sglang.jit_kernel.hadamard import _jit_hadamard_module

    module = _jit_hadamard_module(x.dtype)
    x2 = x.reshape(-1, x.shape[-1])
    out2 = out.reshape(-1, out.shape[-1])
    module.hadamard_transform(x2, out2, x.shape[-1] ** -0.5)
    return out


def _freqs_real(freqs_cis: torch.Tensor) -> torch.Tensor:
    if freqs_cis.is_complex():
        return torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    return freqs_cis.contiguous()


@triton.jit
def _apply_rotary_tail_kernel(
    x_ptr,
    freqs_ptr,
    positions_ptr,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_freq_pos,
    stride_freq_dim,
    num_freqs: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    pair_offsets = pid_d * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    mask = pair_offsets < (rope_dim // 2)
    base_dim = head_dim - rope_dim
    pos = tl.load(positions_ptr + pid_b).to(tl.int64)
    pos = tl.minimum(tl.maximum(pos, 0), num_freqs - 1)
    base = pid_b * stride_x_batch + pid_h * stride_x_head

    offs_even = base + (base_dim + pair_offsets * 2) * stride_x_dim
    offs_odd = base + (base_dim + pair_offsets * 2 + 1) * stride_x_dim

    x_even = tl.load(x_ptr + offs_even, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + offs_odd, mask=mask, other=0.0).to(tl.float32)

    freq_even = tl.load(
        freqs_ptr + pos * stride_freq_pos + pair_offsets * 2 * stride_freq_dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    freq_odd = tl.load(
        freqs_ptr + pos * stride_freq_pos + (pair_offsets * 2 + 1) * stride_freq_dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out_even = x_even * freq_even - x_odd * freq_odd
    out_odd = x_even * freq_odd + x_odd * freq_even
    tl.store(x_ptr + offs_even, out_even, mask=mask)
    tl.store(x_ptr + offs_odd, out_odd, mask=mask)


def apply_rotary_tail_inplace(
    x: torch.Tensor,
    freqs_cis_or_real: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    freqs = (
        freqs_cis_or_real
        if not freqs_cis_or_real.is_complex()
        else _freqs_real(freqs_cis_or_real)
    )
    rope_dim = freqs.shape[1]
    if rope_dim == 0:
        return x
    if rope_dim % 2 != 0 or rope_dim > x.shape[-1]:
        raise ValueError(f"invalid rotary dim {rope_dim} for q dim {x.shape[-1]}")
    if freqs.shape[0] <= 0:
        raise ValueError("freqs must contain at least one position")
    block_pairs = min(64, triton.next_power_of_2(max(1, rope_dim // 2)))
    grid = (x.shape[0], x.shape[1], triton.cdiv(rope_dim // 2, block_pairs))
    _apply_rotary_tail_kernel[grid](
        x,
        freqs,
        positions,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        freqs.stride(0),
        freqs.stride(1),
        freqs.shape[0],
        head_dim=x.shape[-1],
        rope_dim=rope_dim,
        BLOCK_PAIRS=block_pairs,
    )
    return x


def bf16_paged_mqa_logits_torch(
    q: torch.Tensor,
    kvcache_bf16: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata, clean_logits
    if seq_lens.ndim == 2:
        seq_lens = seq_lens.squeeze(-1)
    if weight.ndim == 3:
        weight = weight.squeeze(-1)

    batch_size, _, num_heads, head_dim = q.shape
    block_size = kvcache_bf16.shape[1]
    assert head_dim == 128
    assert block_size == 64
    assert kvcache_bf16.dtype == torch.bfloat16

    num_pages = (max_seq_len + block_size - 1) // block_size
    pages = page_table[:, :num_pages].to(torch.long)
    values = kvcache_bf16.squeeze(2)[pages].contiguous()
    values = values.view(batch_size, num_pages * block_size, head_dim).float()
    qv = q[:, 0].float()
    scores = torch.einsum("bld,bhd->blh", values, qv)
    scores = F.relu(scores)
    logits = (scores * weight[:, None, :].float()).sum(dim=-1)
    logits = logits[:, :max_seq_len].contiguous()

    pos = torch.arange(max_seq_len, device=logits.device).unsqueeze(0)
    logits = logits.masked_fill(pos >= seq_lens.to(torch.long).unsqueeze(1), float("-inf"))
    return logits


@triton.jit
def _bf16_paged_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    out_ptr,
    max_seq_len: tl.constexpr,
    k_pages: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_block: tl.constexpr,
    k_stride_d: tl.constexpr,
    weight_stride_b: tl.constexpr,
    weight_stride_h: tl.constexpr,
    page_table_stride_b: tl.constexpr,
    page_table_stride_p: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_l: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
    NUM_BLOCK_ITERS: tl.constexpr,
):
    b = tl.program_id(0)
    worker = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads

    for block_iter in range(0, NUM_BLOCK_ITERS):
        block_l = worker + block_iter * NUM_WORKERS
        tile_start = block_l * BLOCK_L
        if tile_start < max_seq_len:
            if tile_start < seq_len:
                offs_l = tile_start + tl.arange(0, BLOCK_L)
                valid_l = (offs_l < max_seq_len) & (offs_l < seq_len)

                page_offsets = offs_l // 64
                slot_offsets = offs_l - page_offsets * 64
                page_ids = tl.load(
                    page_table_ptr
                    + b * page_table_stride_b
                    + page_offsets * page_table_stride_p,
                    mask=valid_l,
                    other=0,
                ).to(tl.int64)
                valid_page = valid_l & (page_ids >= 0) & (page_ids < k_pages)
                safe_page_ids = tl.minimum(
                    tl.maximum(page_ids, 0), tl.maximum(k_pages - 1, 0)
                )

                acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)
                for d_start in range(0, head_dim, BLOCK_D):
                    offs_d = d_start + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < head_dim
                    q_vals = tl.load(
                        q_ptr
                        + b * q_stride_b
                        + offs_h[None, :] * q_stride_h
                        + offs_d[:, None] * q_stride_d,
                        mask=mask_h[None, :] & mask_d[:, None],
                        other=0.0,
                    )
                    k_vals = tl.load(
                        k_ptr
                        + safe_page_ids[:, None] * k_stride_page
                        + slot_offsets[:, None] * k_stride_block
                        + offs_d[None, :] * k_stride_d,
                        mask=valid_page[:, None] & mask_d[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(k_vals, q_vals, input_precision="tf32")

                acc = tl.where(acc > 0.0, acc, 0.0)
                weights = tl.load(
                    weight_ptr + b * weight_stride_b + offs_h * weight_stride_h,
                    mask=mask_h,
                    other=0.0,
                ).to(tl.float32)
                logits = tl.sum(acc * weights[None, :], axis=1)
                logits = tl.where(valid_l, logits, float("-inf"))
                tl.store(
                    out_ptr + b * out_stride_b + offs_l * out_stride_l,
                    logits,
                    mask=offs_l < max_seq_len,
                )
            else:
                offs_l = tile_start + tl.arange(0, BLOCK_L)
                tl.store(
                    out_ptr + b * out_stride_b + offs_l * out_stride_l,
                    tl.full((BLOCK_L,), float("-inf"), dtype=tl.float32),
                    mask=offs_l < max_seq_len,
                )
def bf16_paged_mqa_logits(
    q: torch.Tensor,
    kvcache_bf16: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    if q.device.type != "cuda":
        return bf16_paged_mqa_logits_torch(
            q, kvcache_bf16, weight, seq_lens, page_table, deep_gemm_metadata, max_seq_len, clean_logits
        )
    if seq_lens.ndim == 2:
        seq_lens = seq_lens.squeeze(-1)
    if weight.ndim == 3:
        weight = weight.squeeze(-1)

    batch_size, _, num_heads, head_dim = q.shape
    if (
        head_dim != 128
        or kvcache_bf16.shape[1] != 64
        or kvcache_bf16.dtype != torch.bfloat16
    ):
        return bf16_paged_mqa_logits_torch(
            q, kvcache_bf16, weight, seq_lens, page_table, deep_gemm_metadata, max_seq_len, clean_logits
        )

    block_l = 256
    block_d = 64
    block_h = triton.next_power_of_2(num_heads)
    num_seq_blocks = triton.cdiv(max_seq_len, block_l)
    num_workers = min(128, triton.next_power_of_2(max(1, num_seq_blocks)))
    num_block_iters = triton.cdiv(num_seq_blocks, num_workers)

    out = torch.empty((batch_size, max_seq_len), dtype=torch.float32, device=q.device)
    seq_lens_arg = seq_lens.to(torch.int32) if seq_lens.dtype not in (torch.int32, torch.int64) else seq_lens
    grid = (batch_size, num_workers)
    _bf16_paged_mqa_logits_kernel[grid](
        q,
        kvcache_bf16,
        weight,
        seq_lens_arg,
        page_table,
        out,
        int(max_seq_len),
        kvcache_bf16.shape[0],
        int(num_heads),
        int(head_dim),
        q.stride(0),
        q.stride(2),
        q.stride(3),
        kvcache_bf16.stride(0),
        kvcache_bf16.stride(1),
        kvcache_bf16.stride(3),
        weight.stride(0),
        weight.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
        NUM_WORKERS=num_workers,
        NUM_BLOCK_ITERS=num_block_iters,
        num_warps=4,
        num_stages=3,
    )
    return out


def bf16_indexer_q_torch(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor | None = None,
    weights_out: torch.Tensor | None = None,
    scratch_q: torch.Tensor | None = None,
    allow_inplace_input: bool = False,
    freqs_real: torch.Tensor | None = None,
):
    from sglang.jit_kernel.hadamard import hadamard_transform

    if allow_inplace_input and q_input.dtype == torch.bfloat16 and q_input.is_contiguous():
        q = q_input
    elif scratch_q is not None:
        scratch_q.copy_(q_input)
        q = scratch_q
    else:
        q = q_input.to(torch.bfloat16).contiguous().clone()
    apply_rotary_tail_inplace(q, freqs_real if freqs_real is not None else freqs_cis, positions)
    if q_out is None:
        q = hadamard_transform(q, scale=q.shape[-1] ** -0.5)
    else:
        q = _hadamard_inplace_or_out(q, q_out)
    weights = (weight.float() * float(weight_scale)).unsqueeze(-1)
    if weights_out is not None:
        weights_out.copy_(weights)
        weights = weights_out
    return q, weights


@triton.jit
def _scale_weights_kernel(
    weight_ptr,
    out_ptr,
    numel,
    weight_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, w * weight_scale, mask=mask)


def bf16_indexer_q(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor | None = None,
    weights_out: torch.Tensor | None = None,
    scratch_q: torch.Tensor | None = None,
    allow_inplace_input: bool = False,
    freqs_real: torch.Tensor | None = None,
):
    if q_input.device.type != "cuda":
        return bf16_indexer_q_torch(
            q_input,
            weight,
            weight_scale,
            freqs_cis,
            positions,
            q_out=q_out,
            weights_out=weights_out,
            scratch_q=scratch_q,
            allow_inplace_input=allow_inplace_input,
            freqs_real=freqs_real,
        )

    from sglang.jit_kernel.hadamard import hadamard_transform

    if allow_inplace_input and q_input.dtype == torch.bfloat16 and q_input.is_contiguous():
        q = q_input
    elif scratch_q is not None:
        scratch_q.copy_(q_input)
        q = scratch_q
    else:
        q = q_input.to(torch.bfloat16).contiguous().clone()
    apply_rotary_tail_inplace(q, freqs_real if freqs_real is not None else freqs_cis, positions)
    if q_out is None:
        q = hadamard_transform(q, scale=q.shape[-1] ** -0.5)
    else:
        q = _hadamard_inplace_or_out(q, q_out)

    weight_flat = weight.contiguous().view(-1)
    weights = (
        weights_out
        if weights_out is not None
        else torch.empty((*weight.shape, 1), device=weight.device, dtype=torch.float32)
    )
    block = 256
    _scale_weights_kernel[(triton.cdiv(weight_flat.numel(), block),)](
        weight_flat,
        weights.view(-1),
        weight_flat.numel(),
        float(weight_scale),
        BLOCK=block,
    )
    return q, weights
