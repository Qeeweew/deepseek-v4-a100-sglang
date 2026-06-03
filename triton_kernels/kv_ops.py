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
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    block_d = tl.program_id(1)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets_d < head_dim
    dst_row = tl.load(loc_ptr + row).to(tl.int64)
    vals = tl.load(src_ptr + row * src_stride_row + offsets_d * src_stride_dim, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_row * dst_stride_row + offsets_d * dst_stride_dim, vals, mask=mask)


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
    grid = (rows, triton.cdiv(head_dim, block_d))
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
        BLOCK_D=block_d,
    )


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
    BLOCK_D: tl.constexpr,
):
    q = tl.program_id(0)
    k = tl.program_id(1)
    block_d = tl.program_id(2)
    offsets_d = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim
    length = tl.load(lengths_ptr + q).to(tl.int64)
    row_index = tl.load(indices_ptr + q * idx_stride_q + k * idx_stride_k).to(tl.int64)
    valid = (k < length) & (row_index >= 0)
    safe_row = tl.maximum(row_index, 0)
    vals = tl.load(
        buffer_ptr + safe_row * buffer_stride_row + offsets_d * buffer_stride_d,
        mask=mask_d & valid,
        other=0.0,
    )
    out_k = k + out_topk_offset
    tl.store(
        out_ptr + q * out_stride_q + out_k * out_stride_k + offsets_d * out_stride_d,
        vals,
        mask=mask_d,
    )
    tl.store(
        invalid_ptr + q * invalid_stride_q + out_k * invalid_stride_k,
        ~valid,
        mask=block_d == 0,
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
    q_tokens = indices.shape[0]
    head_dim = buffer.shape[-1]
    if q_tokens == 0 or total_topk == 0:
        return
    flat = buffer.view(-1, head_dim)
    block_d = min(1024, triton.next_power_of_2(max(1, head_dim)))
    grid = (q_tokens, total_topk, triton.cdiv(head_dim, block_d))
    _gather_bf16_kv_kernel[grid](
        flat,
        indices,
        lengths.to(torch.int32) if lengths.dtype not in (torch.int32, torch.int64) else lengths,
        out,
        invalid_mask,
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
        out[:, out_topk_offset : out_topk_offset + total_topk].copy_(gathered)
        invalid_mask[:, out_topk_offset : out_topk_offset + total_topk].copy_(invalid)
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
