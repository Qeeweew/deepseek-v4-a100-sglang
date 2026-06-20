from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


def scatter_bf16_rows_torch(dst: torch.Tensor, loc: torch.Tensor, src: torch.Tensor) -> None:
    head_dim = src.shape[-1]
    flat_dst = dst.view(-1, head_dim)
    flat_src = src.view(-1, head_dim)
    flat_dst[loc.to(torch.long)] = flat_src.to(flat_dst.dtype)


@triton.jit
def _scatter_bf16_rows_kernel(
    dst_ptr,
    loc_ptr,
    src_ptr,
    rows: tl.constexpr,
    dst_total_rows: tl.constexpr,
    head_dim: tl.constexpr,
    dst_stride_row: tl.constexpr,
    dst_stride_dim: tl.constexpr,
    src_stride_row: tl.constexpr,
    src_stride_dim: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    block_d = tl.program_id(1)
    offs_r = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_r = offs_r < rows
    mask_d = offsets_d < head_dim
    dst_row_idx = tl.load(loc_ptr + offs_r, mask=mask_r, other=0).to(tl.int64)
    valid_dst = mask_r & (dst_row_idx >= 0) & (dst_row_idx < dst_total_rows)
    safe_dst_row_idx = tl.minimum(
        tl.maximum(dst_row_idx, 0),
        tl.maximum(dst_total_rows - 1, 0),
    )
    vals = tl.load(
        src_ptr + offs_r[:, None] * src_stride_row + offsets_d[None, :] * src_stride_dim,
        mask=mask_r[:, None] & mask_d[None, :],
        other=0.0,
    )
    tl.store(
        dst_ptr + safe_dst_row_idx[:, None] * dst_stride_row + offsets_d[None, :] * dst_stride_dim,
        vals,
        mask=valid_dst[:, None] & mask_d[None, :],
    )


def scatter_bf16_rows(dst: torch.Tensor, loc: torch.Tensor, src: torch.Tensor) -> None:
    if dst.device.type != "cuda" or src.device.type != "cuda":
        scatter_bf16_rows_torch(dst, loc, src)
        return
    head_dim = src.shape[-1]
    rows = src.numel() // head_dim
    if rows == 0:
        return
    flat_dst = dst.view(-1, head_dim)
    flat_src = src.view(-1, head_dim)
    block_d = min(1024, triton.next_power_of_2(max(1, head_dim)))
    block_r = 4
    grid = (triton.cdiv(rows, block_r), triton.cdiv(head_dim, block_d))
    _scatter_bf16_rows_kernel[grid](
        flat_dst,
        loc,
        flat_src,
        rows,
        flat_dst.shape[0],
        head_dim,
        flat_dst.stride(0),
        flat_dst.stride(1),
        flat_src.stride(0),
        flat_src.stride(1),
        BLOCK_R=block_r,
        BLOCK_D=block_d,
    )


def scatter_int8_indexer_rows_torch(dst: torch.Tensor, loc: torch.Tensor, src: torch.Tensor) -> None:
    head_dim = src.shape[-1]
    page_size = 64
    bytes_per_page = page_size * (head_dim + 4)
    if dst.dtype != torch.uint8 or dst.shape[-1] != bytes_per_page:
        raise ValueError(f"expected uint8 indexer cache with page bytes {bytes_per_page}")
    flat_src = src.view(-1, head_dim).float()
    loc_l = loc.to(torch.long)
    valid = (loc_l >= 0) & (loc_l < dst.shape[0] * page_size)
    if not valid.any():
        return
    loc_l = loc_l[valid]
    flat_src = flat_src[valid]
    page = loc_l // page_size
    slot = loc_l % page_size
    scale = flat_src.abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    q = torch.round(flat_src / scale[:, None]).clamp(-127, 127).to(torch.int8)
    for i in range(q.shape[0]):
        row = dst[page[i]]
        row[slot[i] * head_dim : (slot[i] + 1) * head_dim] = q[i].view(torch.uint8)
        scale_off = page_size * head_dim + slot[i] * 4
        row[scale_off : scale_off + 4] = scale[i : i + 1].view(torch.uint8)


@triton.jit
def _scatter_int8_indexer_rows_kernel(
    dst_ptr,
    dst_f32_ptr,
    loc_ptr,
    src_ptr,
    rows: tl.constexpr,
    num_pages: tl.constexpr,
    head_dim: tl.constexpr,
    bytes_per_page: tl.constexpr,
    floats_per_page: tl.constexpr,
    page_size: tl.constexpr,
    src_stride_row: tl.constexpr,
    src_stride_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    loc = tl.load(loc_ptr + row).to(tl.int64)
    valid = (loc >= 0) & (loc < num_pages * page_size)
    safe_loc = tl.minimum(tl.maximum(loc, 0), tl.maximum(num_pages * page_size - 1, 0))
    page = safe_loc // page_size
    slot = safe_loc - page * page_size
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    vals = tl.load(
        src_ptr + row * src_stride_row + offs_d * src_stride_dim,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    abs_max = tl.max(tl.abs(vals), axis=0)
    scale = tl.maximum(abs_max, 1.0e-8) / 127.0
    q = tl.extra.libdevice.nearbyint(vals / scale)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    value_offset = page * bytes_per_page + slot * head_dim + offs_d
    tl.store(dst_ptr + value_offset, q, mask=valid & mask_d)
    scale_offset = page * bytes_per_page + page_size * head_dim + slot * 4
    scale_offset_f32 = page * floats_per_page + (page_size * head_dim) // 4 + slot
    tl.store(dst_f32_ptr + scale_offset_f32, scale, mask=valid)


def scatter_int8_indexer_rows(dst: torch.Tensor, loc: torch.Tensor, src: torch.Tensor) -> None:
    if dst.device.type != "cuda" or src.device.type != "cuda":
        scatter_int8_indexer_rows_torch(dst, loc, src)
        return
    head_dim = src.shape[-1]
    page_size = 64
    bytes_per_page = page_size * (head_dim + 4)
    if dst.dtype != torch.uint8 or dst.shape[-1] != bytes_per_page:
        raise ValueError(f"expected uint8 indexer cache with page bytes {bytes_per_page}")
    rows = src.numel() // head_dim
    if rows == 0:
        return
    flat_src = src.view(-1, head_dim)
    dst_f32 = dst.view(torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    _scatter_int8_indexer_rows_kernel[(rows,)](
        dst,
        dst_f32,
        loc,
        flat_src,
        rows,
        dst.shape[0],
        head_dim,
        bytes_per_page,
        bytes_per_page // 4,
        page_size,
        flat_src.stride(0),
        flat_src.stride(1),
        BLOCK_D=block_d,
    )


def trim_and_pad_rows_torch(
    page_indices: torch.Tensor,
    lengths: torch.Tensor,
    q_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if page_indices.ndim == 3:
        page_indices = page_indices.squeeze(1)
    if page_indices.shape[0] >= q_tokens:
        return page_indices[:q_tokens], lengths[:q_tokens].to(torch.int32)
    pad_rows = q_tokens - page_indices.shape[0]
    page_indices = torch.nn.functional.pad(page_indices, (0, 0, 0, pad_rows), value=-1)
    lengths = torch.nn.functional.pad(lengths.to(torch.int32), (0, pad_rows), value=1)
    return page_indices, lengths


@triton.jit
def _trim_and_pad_rows_kernel(
    src_idx_ptr,
    src_len_ptr,
    dst_idx_ptr,
    dst_len_ptr,
    src_rows: tl.constexpr,
    q_tokens: tl.constexpr,
    topk: tl.constexpr,
    src_idx_stride_row: tl.constexpr,
    src_idx_stride_col: tl.constexpr,
    dst_idx_stride_row: tl.constexpr,
    dst_idx_stride_col: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offs_c = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < topk
    in_bounds = row < src_rows
    vals = tl.load(
        src_idx_ptr + row * src_idx_stride_row + offs_c * src_idx_stride_col,
        mask=in_bounds & mask_c,
        other=-1,
    )
    tl.store(
        dst_idx_ptr + row * dst_idx_stride_row + offs_c * dst_idx_stride_col,
        vals,
        mask=(row < q_tokens) & mask_c,
    )
    if col_block == 0:
        length = tl.load(src_len_ptr + row, mask=in_bounds, other=1).to(tl.int32)
        tl.store(dst_len_ptr + row, length, mask=row < q_tokens)


def trim_and_pad_rows(
    page_indices: torch.Tensor,
    lengths: torch.Tensor,
    q_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if page_indices.device.type != "cuda":
        return trim_and_pad_rows_torch(page_indices, lengths, q_tokens)
    if page_indices.ndim == 3:
        page_indices = page_indices.squeeze(1)
    src_rows, topk = page_indices.shape
    if src_rows >= q_tokens:
        return page_indices[:q_tokens], lengths[:q_tokens].to(torch.int32)
    out_idx = torch.empty((q_tokens, topk), device=page_indices.device, dtype=page_indices.dtype)
    out_len = torch.empty((q_tokens,), device=lengths.device, dtype=torch.int32)
    out_idx.fill_(-1)
    out_len.fill_(1)
    if src_rows == 0:
        return out_idx, out_len
    block_c = min(256, triton.next_power_of_2(max(1, topk)))
    grid = (q_tokens, triton.cdiv(topk, block_c))
    _trim_and_pad_rows_kernel[grid](
        page_indices,
        lengths.to(torch.int32) if lengths.dtype != torch.int32 else lengths,
        out_idx,
        out_len,
        src_rows,
        q_tokens,
        topk,
        page_indices.stride(0),
        page_indices.stride(1),
        out_idx.stride(0),
        out_idx.stride(1),
        BLOCK_C=block_c,
    )
    return out_idx, out_len


def gather_bf16_kv_torch(
    buffer: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    total_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_tokens = indices.shape[0]
    head_dim = buffer.shape[-1]
    flat = buffer.view(-1, head_dim)
    gathered = torch.zeros((q_tokens, total_topk, head_dim), device=buffer.device, dtype=buffer.dtype)
    invalid = torch.ones((q_tokens, total_topk), device=indices.device, dtype=torch.bool)
    available_topk = min(total_topk, indices.shape[1])
    if q_tokens == 0 or available_topk == 0:
        return gathered, invalid
    if lengths.shape[0] < q_tokens:
        padded_lengths = torch.zeros((q_tokens,), device=lengths.device, dtype=torch.int64)
        padded_lengths[: lengths.shape[0]] = lengths.to(torch.int64)
        lengths = padded_lengths
    else:
        lengths = lengths[:q_tokens].to(torch.int64)
    lengths = lengths.view(q_tokens, 1)
    pos = torch.arange(available_topk, device=indices.device, dtype=torch.int64).view(1, available_topk)
    flat_rows = flat.shape[0]
    idx = indices[:, :available_topk]
    valid = (pos < lengths) & (idx >= 0) & (idx < flat_rows)
    rows = idx.to(torch.int64).clamp(min=0, max=max(0, flat_rows - 1))
    partial = flat[rows]
    gathered[:, :available_topk] = torch.where(valid.unsqueeze(-1), partial, torch.zeros_like(partial))
    invalid[:, :available_topk] = ~valid
    return gathered, invalid


@triton.jit
def _gather_bf16_kv_kernel(
    buffer_ptr,
    indices_ptr,
    lengths_ptr,
    out_ptr,
    invalid_ptr,
    idx_rows: tl.constexpr,
    idx_topk: tl.constexpr,
    lengths_rows: tl.constexpr,
    buffer_rows: tl.constexpr,
    out_rows: tl.constexpr,
    out_topk_capacity: tl.constexpr,
    total_topk: tl.constexpr,
    head_dim: tl.constexpr,
    idx_stride_q: tl.constexpr,
    idx_stride_k: tl.constexpr,
    out_stride_q: tl.constexpr,
    out_stride_k: tl.constexpr,
    out_stride_d: tl.constexpr,
    invalid_stride_q: tl.constexpr,
    invalid_stride_k: tl.constexpr,
    buffer_stride_row: tl.constexpr,
    buffer_stride_d: tl.constexpr,
    out_topk_offset: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q = tl.program_id(0)
    k_block = tl.program_id(1)
    block_d = tl.program_id(2)
    q_u64 = q.to(tl.uint64)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim
    out_k = offs_k + out_topk_offset
    idx_valid = (q < idx_rows) & (offs_k < idx_topk)
    out_valid = (q < out_rows) & (out_k < out_topk_capacity) & (offs_k < total_topk)
    q_len_valid = q < lengths_rows
    length = tl.load(lengths_ptr + q, mask=q_len_valid, other=0)
    row_index = tl.load(
        indices_ptr + q * idx_stride_q + offs_k * idx_stride_k,
        mask=idx_valid & (offs_k < total_topk),
        other=-1,
    )
    valid = (
        idx_valid
        & q_len_valid
        & (offs_k < length)
        & (row_index >= 0)
        & (row_index < buffer_rows)
        & (offs_k < total_topk)
    )
    safe_row = tl.minimum(tl.maximum(row_index, 0), tl.maximum(buffer_rows - 1, 0))
    vals = tl.load(
        buffer_ptr + safe_row[:, None] * buffer_stride_row + offsets_d[None, :] * buffer_stride_d,
        mask=valid[:, None] & mask_d[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr
        + q_u64 * out_stride_q
        + out_k[:, None] * out_stride_k
        + offsets_d[None, :] * out_stride_d,
        vals,
        mask=out_valid[:, None] & mask_d[None, :],
    )
    tl.store(
        invalid_ptr + q_u64 * invalid_stride_q + out_k * invalid_stride_k,
        ~valid,
        mask=(block_d == 0) & out_valid,
    )


def _launch_gather(
    buffer: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    invalid_mask: torch.Tensor,
    total_topk: int,
    out_topk_offset: int = 0,
) -> None:
    q_tokens = out.shape[0]
    idx_rows = indices.shape[0]
    idx_topk = indices.shape[1]
    lengths_rows = lengths.shape[0]
    out_topk_capacity = out.shape[1]
    head_dim = buffer.shape[-1]
    if q_tokens == 0 or total_topk == 0:
        return
    if out_topk_offset < 0 or out_topk_offset >= out_topk_capacity:
        return
    writable_topk = out_topk_capacity - out_topk_offset
    effective_topk = min(total_topk, idx_topk, writable_topk)
    if effective_topk <= 0:
        return
    flat = buffer.view(-1, head_dim)
    buffer_rows = flat.shape[0]
    block_d = min(1024, triton.next_power_of_2(max(1, head_dim)))
    # Larger K tiles reduce launch overhead on narrower head dimensions.
    block_k = 16 if head_dim <= 256 else 8
    grid = (q_tokens, triton.cdiv(effective_topk, block_k), triton.cdiv(head_dim, block_d))
    _gather_bf16_kv_kernel[grid](
        flat,
        indices,
        lengths.to(torch.int32) if lengths.dtype not in (torch.int32, torch.int64) else lengths,
        out,
        invalid_mask,
        idx_rows,
        idx_topk,
        lengths_rows,
        buffer_rows,
        q_tokens,
        out_topk_capacity,
        effective_topk,
        head_dim,
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        invalid_mask.stride(0),
        invalid_mask.stride(1),
        flat.stride(0),
        flat.stride(1),
        int(out_topk_offset),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
    )


def gather_bf16_kv(
    buffer: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    total_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if buffer.device.type != "cuda":
        return gather_bf16_kv_torch(buffer, indices, lengths, total_topk)
    q_tokens = indices.shape[0]
    head_dim = buffer.shape[-1]
    out = torch.empty((q_tokens, total_topk, head_dim), dtype=buffer.dtype, device=buffer.device)
    invalid_mask = torch.empty((q_tokens, total_topk), dtype=torch.bool, device=buffer.device)
    if indices.shape[1] < total_topk:
        out[:, indices.shape[1] : total_topk].zero_()
        invalid_mask[:, indices.shape[1] : total_topk].fill_(True)
    indices = indices[:, :total_topk]
    _launch_gather(buffer, indices, lengths, out, invalid_mask, total_topk)
    return out, invalid_mask


def gather_bf16_kv_into(
    buffer: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    total_topk: int,
    out: torch.Tensor,
    invalid_mask: torch.Tensor,
    out_topk_offset: int = 0,
) -> None:
    if out_topk_offset < 0:
        return
    out_topk_end = min(out.shape[1], out_topk_offset + total_topk)
    writable_topk = max(0, out_topk_end - out_topk_offset)
    if buffer.device.type != "cuda":
        gathered, invalid = gather_bf16_kv_torch(buffer, indices, lengths, total_topk)
        rows = min(out.shape[0], gathered.shape[0])
        out[:, out_topk_offset:out_topk_end].zero_()
        invalid_mask[:, out_topk_offset:out_topk_end].fill_(True)
        out[:rows, out_topk_offset:out_topk_end].copy_(gathered[:rows, :writable_topk])
        invalid_mask[:rows, out_topk_offset:out_topk_end].copy_(invalid[:rows, :writable_topk])
        return
    if writable_topk <= 0:
        return
    available_topk = min(indices.shape[1], total_topk, writable_topk)
    if available_topk < writable_topk:
        tail_start = out_topk_offset + available_topk
        out[:, tail_start:out_topk_end].zero_()
        invalid_mask[:, tail_start:out_topk_end].fill_(True)
    _launch_gather(
        buffer,
        indices[:, :total_topk],
        lengths,
        out,
        invalid_mask,
        total_topk,
        out_topk_offset,
    )
