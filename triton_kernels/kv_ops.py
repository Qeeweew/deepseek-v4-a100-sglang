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
    dst_rows = tl.load(loc_ptr + offs_r, mask=mask_r, other=0).to(tl.int64)
    vals = tl.load(
        src_ptr + offs_r[:, None] * src_stride_row + offsets_d[None, :] * src_stride_dim,
        mask=mask_r[:, None] & mask_d[None, :],
        other=0.0,
    )
    tl.store(
        dst_ptr + dst_rows[:, None] * dst_stride_row + offsets_d[None, :] * dst_stride_dim,
        vals,
        mask=mask_r[:, None] & mask_d[None, :],
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
        head_dim,
        flat_dst.stride(0),
        flat_dst.stride(1),
        flat_src.stride(0),
        flat_src.stride(1),
        BLOCK_R=block_r,
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
    lengths = lengths.to(torch.int64).view(q_tokens, 1)
    pos = torch.arange(total_topk, device=indices.device, dtype=torch.int64).view(1, total_topk)
    valid = (pos < lengths) & (indices[:, :total_topk] >= 0)
    rows = indices[:, :total_topk].to(torch.int64).clamp(min=0)
    gathered = flat[rows]
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    return gathered, ~valid


@triton.jit
def _gather_bf16_kv_kernel(
    buffer_ptr,
    indices_ptr,
    lengths_ptr,
    out_ptr,
    invalid_ptr,
    src_rows: tl.constexpr,
    q_tokens: tl.constexpr,
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
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim
    q_valid = q < src_rows
    length = tl.load(lengths_ptr + q, mask=q_valid, other=0).to(tl.int64)
    row_index = tl.load(
        indices_ptr + q * idx_stride_q + offs_k * idx_stride_k,
        mask=q_valid & (offs_k < total_topk),
        other=-1,
    ).to(tl.int64)
    valid = q_valid & (offs_k < length) & (row_index >= 0) & (offs_k < total_topk)
    safe_row = tl.maximum(row_index, 0)
    vals = tl.load(
        buffer_ptr + safe_row[:, None] * buffer_stride_row + offsets_d[None, :] * buffer_stride_d,
        mask=valid[:, None] & mask_d[None, :],
        other=0.0,
    )
    out_k = offs_k + out_topk_offset
    tl.store(
        out_ptr + q * out_stride_q + out_k[:, None] * out_stride_k + offsets_d[None, :] * out_stride_d,
        vals,
        mask=(offs_k[:, None] < (out_topk_offset + total_topk)) & mask_d[None, :],
    )
    tl.store(
        invalid_ptr + q * invalid_stride_q + out_k * invalid_stride_k,
        ~valid,
        mask=(block_d == 0) & (offs_k < total_topk),
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
    src_rows = indices.shape[0]
    head_dim = buffer.shape[-1]
    if q_tokens == 0 or total_topk == 0:
        return
    flat = buffer.view(-1, head_dim)
    block_d = min(1024, triton.next_power_of_2(max(1, head_dim)))
    block_k = 8
    grid = (q_tokens, triton.cdiv(total_topk, block_k), triton.cdiv(head_dim, block_d))
    _gather_bf16_kv_kernel[grid](
        flat,
        indices,
        lengths.to(torch.int32) if lengths.dtype not in (torch.int32, torch.int64) else lengths,
        out,
        invalid_mask,
        src_rows,
        q_tokens,
        total_topk,
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
    indices = indices[:, :total_topk]
    q_tokens = indices.shape[0]
    head_dim = buffer.shape[-1]
    out = torch.empty((q_tokens, total_topk, head_dim), dtype=buffer.dtype, device=buffer.device)
    invalid_mask = torch.empty((q_tokens, total_topk), dtype=torch.bool, device=buffer.device)
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
    if buffer.device.type != "cuda":
        gathered, invalid = gather_bf16_kv_torch(buffer, indices, lengths, total_topk)
        rows = min(out.shape[0], gathered.shape[0])
        out[:, out_topk_offset : out_topk_offset + total_topk].zero_()
        invalid_mask[:, out_topk_offset : out_topk_offset + total_topk].fill_(True)
        out[:rows, out_topk_offset : out_topk_offset + total_topk].copy_(gathered[:rows])
        invalid_mask[:rows, out_topk_offset : out_topk_offset + total_topk].copy_(invalid[:rows])
        return
    _launch_gather(
        buffer,
        indices[:, :total_topk],
        lengths,
        out,
        invalid_mask,
        total_topk,
        out_topk_offset,
    )
