from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def bf16_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
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

    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_bf16.shape[1]
    assert head_dim == 128
    assert block_size == 64
    assert kvcache_bf16.dtype == torch.bfloat16

    num_pages = (max_seq_len + block_size - 1) // block_size
    pages = page_table[:, :num_pages].to(torch.long)
    values = kvcache_bf16.squeeze(2)[pages].contiguous()
    values = values.view(batch_size, num_pages * block_size, head_dim).float()
    q = q_fp8[:, 0].float()
    scores = torch.einsum("bld,bhd->blh", values, q)
    scores = F.relu(scores)
    logits = (scores * weight[:, None, :].float()).sum(dim=-1)
    logits = logits[:, :max_seq_len].contiguous()

    pos = torch.arange(max_seq_len, device=logits.device).unsqueeze(0)
    logits = logits.masked_fill(pos >= seq_lens.to(torch.long).unsqueeze(1), 0.0)
    return logits


@triton.jit
def _fp8_e4m3fn_to_f32(bits):
    bits_u = bits.to(tl.uint32)
    sign = (bits_u >> 7) & 0x1
    exp = (bits_u >> 3) & 0xF
    mant = bits_u & 0x7
    mant_f = mant.to(tl.float32)
    exp_f = exp.to(tl.float32)
    subnormal = mant_f * 0.001953125
    normal = (1.0 + mant_f * 0.125) * tl.exp2(exp_f - 7.0)
    val = tl.where(exp == 0, subnormal, normal)
    val = tl.where(sign == 1, -val, val)
    return val


@triton.jit
def _bf16_paged_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    weight_ptr,
    seq_lens_ptr,
    page_table_ptr,
    out_ptr,
    max_seq_len: tl.constexpr,
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
):
    b = tl.program_id(0)
    block_l = tl.program_id(1)
    offs_l = block_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, BLOCK_H)
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
    valid_l = (offs_l < max_seq_len) & (offs_l < seq_len)

    page_offsets = offs_l // 64
    slot_offsets = offs_l - page_offsets * 64
    page_ids = tl.load(
        page_table_ptr + b * page_table_stride_b + page_offsets * page_table_stride_p,
        mask=valid_l,
        other=0,
    ).to(tl.int64)

    acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)
    for d_start in range(0, head_dim, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        k_vals = tl.load(
            k_ptr
            + page_ids[:, None] * k_stride_page
            + slot_offsets[:, None] * k_stride_block
            + offs_d[None, :] * k_stride_d,
            mask=valid_l[:, None] & mask_d[None, :],
            other=0.0,
        )
        q_bits = tl.load(
            q_ptr
            + b * q_stride_b
            + offs_h[None, :] * q_stride_h
            + offs_d[:, None] * q_stride_d,
            mask=(offs_h[None, :] < num_heads) & mask_d[:, None],
            other=0,
        )
        q_vals = _fp8_e4m3fn_to_f32(q_bits)
        acc += tl.dot(k_vals.to(tl.float32), q_vals, input_precision="tf32")

    acc = tl.maximum(acc, 0.0)
    weights = tl.load(
        weight_ptr + b * weight_stride_b + offs_h * weight_stride_h,
        mask=offs_h < num_heads,
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(acc * weights[None, :], axis=1)
    logits = tl.where(valid_l, logits, 0.0)
    tl.store(out_ptr + b * out_stride_b + offs_l * out_stride_l, logits, mask=offs_l < max_seq_len)


def bf16_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_bf16: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    if q_fp8.device.type != "cuda":
        return bf16_paged_mqa_logits_torch(
            q_fp8, kvcache_bf16, weight, seq_lens, page_table, deep_gemm_metadata, max_seq_len, clean_logits
        )
    if seq_lens.ndim == 2:
        seq_lens = seq_lens.squeeze(-1)
    if weight.ndim == 3:
        weight = weight.squeeze(-1)

    batch_size, _, num_heads, head_dim = q_fp8.shape
    if (
        q_fp8.dtype != torch.float8_e4m3fn
        or head_dim != 128
        or kvcache_bf16.shape[1] != 64
        or kvcache_bf16.dtype != torch.bfloat16
    ):
        return bf16_paged_mqa_logits_torch(
            q_fp8, kvcache_bf16, weight, seq_lens, page_table, deep_gemm_metadata, max_seq_len, clean_logits
        )

    out = torch.empty((batch_size, max_seq_len), dtype=torch.float32, device=q_fp8.device)
    block_l = 16
    block_d = 32
    block_h = triton.next_power_of_2(num_heads)
    grid = (batch_size, triton.cdiv(max_seq_len, block_l))
    q_u8 = q_fp8.view(torch.uint8)
    _bf16_paged_mqa_logits_kernel[grid](
        q_u8,
        kvcache_bf16,
        weight,
        seq_lens.to(torch.int32) if seq_lens.dtype not in (torch.int32, torch.int64) else seq_lens,
        page_table,
        out,
        int(max_seq_len),
        int(num_heads),
        int(head_dim),
        q_u8.stride(0),
        q_u8.stride(2),
        q_u8.stride(3),
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
    )
    return out
