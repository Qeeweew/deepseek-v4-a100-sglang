from __future__ import annotations

import torch
import triton
import triton.language as tl


_PATCHED = False


@triton.jit
def _keyed_add(x, y):
    key_mask: tl.constexpr = 0xFFFF0000
    kx = x & key_mask
    ky = y & key_mask
    return tl.where(kx == ky, x + y - kx, y)


@triton.jit
def _bitmatrix_metadata_compute_stage2_p2(
    ColSortedIndx,
    RowSortedIndx,
    NonzeroIndx,
    n_tokens,
    ColPartialSum,
    stride_pm,
    stride_pn,
    ColOffs,
    TOKS_PER_ROW: tl.constexpr,
    BLOCK_PER_TOK: tl.constexpr,
    BLOCK_SIZE_P2: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_P2 <= 32768)
    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    nonzero_indx_size = n_tokens * TOKS_PER_ROW
    pid_m = tl.program_id(0)

    offs_local = tl.arange(0, BLOCK_SIZE_P2)
    offs_global = pid_m * (BLOCK_PER_TOK * TOKS_PER_ROW) + offs_local
    valid = offs_local < (BLOCK_PER_TOK * TOKS_PER_ROW)
    valid = valid & (offs_global < nonzero_indx_size)
    col_indx = tl.load(NonzeroIndx + offs_global, mask=valid, other=-1).to(tl.uint32)

    kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    col_indx = kv_pairs >> 16
    offs_local_sorted = kv_pairs & 0xFFFF
    offs_global = pid_m * (BLOCK_PER_TOK * TOKS_PER_ROW) + offs_local_sorted
    valid = (col_indx != 0xFFFF) & (offs_local_sorted < (BLOCK_PER_TOK * TOKS_PER_ROW))

    x = (kv_pairs & 0xFFFF0000) | 0x00000001
    cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xFFFF

    row_sorted_indx = tl.load(
        ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=valid
    )
    row_sorted_indx += tl.load(ColOffs + col_indx, mask=valid)
    row_sorted_indx += exclusive_run_lengths
    tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=valid)
    tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=valid)


def _cdiv(x, y):
    return (x + y - 1) // y


def _make_bitmatrix_metadata_no_topk_p2(nonzero_indx, bitmatrix):
    from triton_kernels.tensor_details.bitmatrix import BitmatrixMetadata
    from triton_kernels.tensor_details.bitmatrix import _bitmatrix_metadata_compute_stage1
    from triton_kernels.tensor_details.bitmatrix_details.sum_bitmatrix_rows import (
        sum_bitmatrix_rows,
    )

    assert nonzero_indx.ndim == 2
    partial_block_m = 32
    col_sum, col_partial_sum = sum_bitmatrix_rows(
        bitmatrix, partials_block_size=partial_block_m
    )
    device = bitmatrix.device
    n_indx = nonzero_indx.numel()
    n_cols = bitmatrix.shape[1]
    col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
    col_sorted_indx = combined_indx[:n_indx]
    row_sorted_indx = combined_indx[n_indx:]

    memset_block = 1024
    memset_grid = (_cdiv(n_indx * 2, memset_block) + n_cols + 1,)
    _bitmatrix_metadata_compute_stage1[memset_grid](
        combined_indx,
        n_indx * 2,
        -1,
        memset_block,
        col_sum,
        col_offs,
        col_sum.shape[0],
        col_partial_sum,
        col_partial_sum.shape[0],
        col_partial_sum.stride(0),
        col_partial_sum.stride(1),
        BLOCK_M=512,
        BLOCK_N=512,
    )

    toks_per_row = nonzero_indx.shape[-1]
    block_size = partial_block_m * toks_per_row
    block_size_p2 = triton.next_power_of_2(block_size)
    compute_grid = (_cdiv(bitmatrix.shape_max[0], partial_block_m),)
    _bitmatrix_metadata_compute_stage2_p2[compute_grid](
        col_sorted_indx,
        row_sorted_indx,
        nonzero_indx,
        bitmatrix.shape[0],
        col_partial_sum,
        col_partial_sum.stride(0),
        col_partial_sum.stride(1),
        col_offs,
        TOKS_PER_ROW=toks_per_row,
        BLOCK_PER_TOK=partial_block_m,
        BLOCK_SIZE_P2=block_size_p2,
    )
    return BitmatrixMetadata(
        col_sum=col_sum,
        col_sorted_indx=col_sorted_indx,
        row_sorted_indx=row_sorted_indx,
    )


def patch_oai_bitmatrix_metadata() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import triton_kernels.tensor as tensor_mod
    import triton_kernels.tensor_details.bitmatrix as bitmatrix_mod

    tensor_mod.make_bitmatrix_metadata = _make_bitmatrix_metadata_no_topk_p2
    bitmatrix_mod.make_bitmatrix_metadata = _make_bitmatrix_metadata_no_topk_p2
    _PATCHED = True
