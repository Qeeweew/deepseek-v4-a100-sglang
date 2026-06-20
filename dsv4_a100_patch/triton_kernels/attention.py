from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


def _bucket_total_tokens(total_tokens: int) -> int:
    if total_tokens <= 0:
        return 1
    n = 1
    while n < total_tokens:
        n <<= 1
    return n


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "total_topk", "d_qk"],
)
@triton.jit
def _direct_sparse_attention_kernel(
    Q,
    KV,
    Indices,
    Lengths,
    AttnSink,
    Output,
    LSE,
    sm_scale,
    kv_rows,
    total_tokens,
    total_tokens_bucket,
    h_q,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_d,
    stride_idx_t,
    stride_idx_k,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    STORE_LSE: tl.constexpr,
    total_topk: tl.constexpr,
    d_qk: tl.constexpr,
    d_v: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_QK_BLOCKS: tl.constexpr,
    NUM_V_BLOCKS: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")
    POS_INF = float("+inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    stride_idx_t_64 = tl.cast(stride_idx_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64
    idx_base = Indices + pid_t_64 * stride_idx_t_64
    seq_len = tl.load(Lengths + pid_t).to(tl.int32)

    offs_d_0 = tl.arange(0, BLOCK_D)
    offs_d_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_d_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_d_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d_0 = offs_d_0 < d_qk
    mask_d_1 = offs_d_1 < d_qk
    mask_d_2 = offs_d_2 < d_qk
    mask_d_3 = offs_d_3 < d_qk

    q_chunk_0 = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_d_0[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d_0[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    if NUM_QK_BLOCKS > 1:
        q_chunk_1 = tl.load(
            q_base + offs_h[:, None] * stride_q_h + offs_d_1[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d_1[None, :],
            other=0.0,
        ).to(tl.bfloat16)
    if NUM_QK_BLOCKS > 2:
        q_chunk_2 = tl.load(
            q_base + offs_h[:, None] * stride_q_h + offs_d_2[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d_2[None, :],
            other=0.0,
        ).to(tl.bfloat16)
    if NUM_QK_BLOCKS > 3:
        q_chunk_3 = tl.load(
            q_base + offs_h[:, None] * stride_q_h + offs_d_3[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d_3[None, :],
            other=0.0,
        ).to(tl.bfloat16)

    for n_start in range(0, total_topk, BLOCK_N):
        should_compute = n_start < seq_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < total_topk
            row_idx = tl.load(idx_base + offs_n * stride_idx_k, mask=mask_n, other=-1).to(tl.int64)
            valid = mask_n & (offs_n < seq_len) & (row_idx >= 0) & (row_idx < kv_rows)
            safe_row_idx = tl.minimum(tl.maximum(row_idx, 0), tl.maximum(kv_rows - 1, 0))

            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
            k_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_d_0[None, :] * stride_kv_d
            k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_0[None, :], other=0.0).to(tl.bfloat16)
            qk += tl.dot(q_chunk_0, tl.trans(k_chunk))
            if NUM_QK_BLOCKS > 1:
                k_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_d_1[None, :] * stride_kv_d
                k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_1[None, :], other=0.0).to(tl.bfloat16)
                qk += tl.dot(q_chunk_1, tl.trans(k_chunk))
            if NUM_QK_BLOCKS > 2:
                k_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_d_2[None, :] * stride_kv_d
                k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_2[None, :], other=0.0).to(tl.bfloat16)
                qk += tl.dot(q_chunk_2, tl.trans(k_chunk))
            if NUM_QK_BLOCKS > 3:
                k_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_d_3[None, :] * stride_kv_d
                k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_3[None, :], other=0.0).to(tl.bfloat16)
                qk += tl.dot(q_chunk_3, tl.trans(k_chunk))

            qk = qk * sm_scale
            qk = tl.where(valid[None, :], qk, NEG_INF)

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
            p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
            l_new = alpha * l_i + tl.sum(p, axis=1)
            p_bf16 = p.to(tl.bfloat16)

            offs_v = offs_d_0
            v_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
            v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
            acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v)

            if NUM_V_BLOCKS > 1:
                offs_v = offs_d_1
                v_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v)

            if NUM_V_BLOCKS > 2:
                offs_v = offs_d_2
                v_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v)

            if NUM_V_BLOCKS > 3:
                offs_v = offs_d_3
                v_ptrs = KV + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v)

            m_i = m_new
            l_i = l_new

    is_lonely_q = l_i == 0.0

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    is_lonely_q_2d = is_lonely_q[:, None]
    output_scale_2d = output_scale[:, None]
    acc_0 = tl.where(is_lonely_q_2d, 0.0, acc_0 * output_scale_2d)
    if NUM_V_BLOCKS > 1:
        acc_1 = tl.where(is_lonely_q_2d, 0.0, acc_1 * output_scale_2d)
    if NUM_V_BLOCKS > 2:
        acc_2 = tl.where(is_lonely_q_2d, 0.0, acc_2 * output_scale_2d)
    if NUM_V_BLOCKS > 3:
        acc_3 = tl.where(is_lonely_q_2d, 0.0, acc_3 * output_scale_2d)
    if STORE_LSE:
        if HAS_ATTN_SINK:
            lse = m_i + tl.math.log2(denominator) / LOG2E
            lse = tl.where(is_lonely_q, attn_sink_vals, lse)
        else:
            lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
            lse = tl.where(is_lonely_q, POS_INF, lse)
        stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
        tl.store(LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h, lse, mask=mask_h)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64
    offs_h_2d = offs_h[:, None]
    mask_h_2d = mask_h[:, None]

    offs_v_0 = tl.arange(0, BLOCK_D)
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)

    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_0[None, :] * stride_o_d,
        acc_0.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_0[None, :] < d_v),
    )
    if NUM_V_BLOCKS > 1:
        tl.store(
            o_base + offs_h_2d * stride_o_h + offs_v_1[None, :] * stride_o_d,
            acc_1.to(tl.bfloat16),
            mask=mask_h_2d & (offs_v_1[None, :] < d_v),
        )
    if NUM_V_BLOCKS > 2:
        tl.store(
            o_base + offs_h_2d * stride_o_h + offs_v_2[None, :] * stride_o_d,
            acc_2.to(tl.bfloat16),
            mask=mask_h_2d & (offs_v_2[None, :] < d_v),
        )
    if NUM_V_BLOCKS > 3:
        tl.store(
            o_base + offs_h_2d * stride_o_h + offs_v_3[None, :] * stride_o_d,
            acc_3.to(tl.bfloat16),
            mask=mask_h_2d & (offs_v_3[None, :] < d_v),
        )


def direct_sparse_attention(
    q: torch.Tensor,
    buffer: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    store_lse: bool = True,
):
    total_tokens, h_q, d_qk = q.shape
    d_v = buffer.shape[-1]
    total_topk = indices.shape[1]
    if output is None:
        output = torch.empty((total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((total_tokens, h_q), dtype=torch.float32, device=q.device)
    num_qk_blocks = triton.cdiv(d_qk, 128)
    num_v_blocks = triton.cdiv(d_v, 128)
    if num_qk_blocks > 4 or num_v_blocks > 4:
        raise ValueError(f"direct_sparse_attention only supports d_qk/d_v <= 512, got {d_qk}/{d_v}")
    has_attn_sink = attn_sink is not None
    attn_sink_tensor = attn_sink if has_attn_sink else lse[:1]
    grid = lambda meta: (triton.cdiv(h_q, meta["BLOCK_H"]), total_tokens)
    _direct_sparse_attention_kernel[grid](
        q,
        buffer,
        indices,
        lengths.to(torch.int32) if lengths.dtype not in (torch.int32, torch.int64) else lengths,
        attn_sink_tensor,
        output,
        lse,
        sm_scale,
        buffer.view(-1, d_v).shape[0],
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        buffer.view(-1, d_v).stride(0),
        buffer.view(-1, d_v).stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=has_attn_sink,
        STORE_LSE=store_lse,
        total_topk=total_topk,
        d_qk=d_qk,
        d_v=d_v,
        NUM_QK_BLOCKS=num_qk_blocks,
        NUM_V_BLOCKS=num_v_blocks,
    )
    return output, lse

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "total_topk", "d_qk"],
)
@triton.jit
def _direct_dual_sparse_attention_kernel(
    Q,
    KV0,
    KV1,
    Indices0,
    Lengths0,
    Indices1,
    Lengths1,
    AttnSink,
    Output,
    LSE,
    sm_scale,
    kv0_rows,
    kv1_rows,
    total_tokens,
    total_tokens_bucket,
    h_q,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv0_t,
    stride_kv0_d,
    stride_kv1_t,
    stride_kv1_d,
    stride_idx0_t,
    stride_idx0_k,
    stride_idx1_t,
    stride_idx1_k,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    STORE_LSE: tl.constexpr,
    primary_topk: tl.constexpr,
    total_topk: tl.constexpr,
    d_qk: tl.constexpr,
    d_v: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_QK_BLOCKS: tl.constexpr,
    NUM_V_BLOCKS: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")
    POS_INF = float("+inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    stride_idx0_t_64 = tl.cast(stride_idx0_t, tl.int64)
    stride_idx1_t_64 = tl.cast(stride_idx1_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64
    idx0_base = Indices0 + pid_t_64 * stride_idx0_t_64
    idx1_base = Indices1 + pid_t_64 * stride_idx1_t_64
    seq_len0 = tl.load(Lengths0 + pid_t).to(tl.int32)
    seq_len1 = tl.load(Lengths1 + pid_t).to(tl.int32)

    offs_d_0 = tl.arange(0, BLOCK_D)
    offs_d_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_d_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_d_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d_0 = offs_d_0 < d_qk
    mask_d_1 = offs_d_1 < d_qk
    mask_d_2 = offs_d_2 < d_qk
    mask_d_3 = offs_d_3 < d_qk

    q_chunk_0 = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_d_0[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d_0[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    if NUM_QK_BLOCKS > 1:
        q_chunk_1 = tl.load(
            q_base + offs_h[:, None] * stride_q_h + offs_d_1[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d_1[None, :],
            other=0.0,
        ).to(tl.bfloat16)
    if NUM_QK_BLOCKS > 2:
        q_chunk_2 = tl.load(
            q_base + offs_h[:, None] * stride_q_h + offs_d_2[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d_2[None, :],
            other=0.0,
        ).to(tl.bfloat16)
    if NUM_QK_BLOCKS > 3:
        q_chunk_3 = tl.load(
            q_base + offs_h[:, None] * stride_q_h + offs_d_3[None, :] * stride_q_d,
            mask=mask_h[:, None] & mask_d_3[None, :],
            other=0.0,
        ).to(tl.bfloat16)

    for source in range(2):
        if source == 0:
            local_topk = primary_topk
            idx_base = idx0_base
            seq_len = seq_len0
            kv_ptr = KV0
            kv_rows = kv0_rows
            stride_kv_t = stride_kv0_t
            stride_kv_d = stride_kv0_d
            stride_idx_k = stride_idx0_k
        else:
            local_topk = total_topk - primary_topk
            idx_base = idx1_base
            seq_len = seq_len1
            kv_ptr = KV1
            kv_rows = kv1_rows
            stride_kv_t = stride_kv1_t
            stride_kv_d = stride_kv1_d
            stride_idx_k = stride_idx1_k

        for n_start in range(0, local_topk, BLOCK_N):
            should_compute = n_start < seq_len
            if should_compute:
                offs_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < local_topk
                row_idx = tl.load(
                    idx_base + offs_n * stride_idx_k,
                    mask=mask_n,
                    other=-1,
                ).to(tl.int64)
                valid = mask_n & (offs_n < seq_len) & (row_idx >= 0) & (row_idx < kv_rows)
                safe_row_idx = tl.minimum(tl.maximum(row_idx, 0), tl.maximum(kv_rows - 1, 0))

                qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
                k_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_d_0[None, :] * stride_kv_d
                k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_0[None, :], other=0.0).to(tl.bfloat16)
                qk += tl.dot(q_chunk_0, tl.trans(k_chunk))
                if NUM_QK_BLOCKS > 1:
                    k_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_d_1[None, :] * stride_kv_d
                    k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_1[None, :], other=0.0).to(tl.bfloat16)
                    qk += tl.dot(q_chunk_1, tl.trans(k_chunk))
                if NUM_QK_BLOCKS > 2:
                    k_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_d_2[None, :] * stride_kv_d
                    k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_2[None, :], other=0.0).to(tl.bfloat16)
                    qk += tl.dot(q_chunk_2, tl.trans(k_chunk))
                if NUM_QK_BLOCKS > 3:
                    k_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_d_3[None, :] * stride_kv_d
                    k_chunk = tl.load(k_ptrs, mask=valid[:, None] & mask_d_3[None, :], other=0.0).to(tl.bfloat16)
                    qk += tl.dot(q_chunk_3, tl.trans(k_chunk))

                qk = qk * sm_scale
                qk = tl.where(valid[None, :], qk, NEG_INF)

                m_ij = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
                p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
                l_new = alpha * l_i + tl.sum(p, axis=1)
                p_bf16 = p.to(tl.bfloat16)

                offs_v = offs_d_0
                v_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v)

                if NUM_V_BLOCKS > 1:
                    offs_v = offs_d_1
                    v_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                    v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                    acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v)

                if NUM_V_BLOCKS > 2:
                    offs_v = offs_d_2
                    v_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                    v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                    acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v)

                if NUM_V_BLOCKS > 3:
                    offs_v = offs_d_3
                    v_ptrs = kv_ptr + safe_row_idx[:, None] * stride_kv_t + offs_v[None, :] * stride_kv_d
                    v = tl.load(v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0).to(tl.bfloat16)
                    acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v)

                m_i = m_new
                l_i = l_new

    is_lonely_q = l_i == 0.0

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    is_lonely_q_2d = is_lonely_q[:, None]
    output_scale_2d = output_scale[:, None]
    acc_0 = tl.where(is_lonely_q_2d, 0.0, acc_0 * output_scale_2d)
    if NUM_V_BLOCKS > 1:
        acc_1 = tl.where(is_lonely_q_2d, 0.0, acc_1 * output_scale_2d)
    if NUM_V_BLOCKS > 2:
        acc_2 = tl.where(is_lonely_q_2d, 0.0, acc_2 * output_scale_2d)
    if NUM_V_BLOCKS > 3:
        acc_3 = tl.where(is_lonely_q_2d, 0.0, acc_3 * output_scale_2d)
    if STORE_LSE:
        if HAS_ATTN_SINK:
            lse = m_i + tl.math.log2(denominator) / LOG2E
            lse = tl.where(is_lonely_q, attn_sink_vals, lse)
        else:
            lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
            lse = tl.where(is_lonely_q, POS_INF, lse)
        stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
        tl.store(LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h, lse, mask=mask_h)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64
    offs_h_2d = offs_h[:, None]
    mask_h_2d = mask_h[:, None]
    offs_v_0 = tl.arange(0, BLOCK_D)
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_0[None, :] * stride_o_d,
        acc_0.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_0[None, :] < d_v),
    )
    if NUM_V_BLOCKS > 1:
        tl.store(o_base + offs_h_2d * stride_o_h + offs_v_1[None, :] * stride_o_d, acc_1.to(tl.bfloat16), mask=mask_h_2d & (offs_v_1[None, :] < d_v))
    if NUM_V_BLOCKS > 2:
        tl.store(o_base + offs_h_2d * stride_o_h + offs_v_2[None, :] * stride_o_d, acc_2.to(tl.bfloat16), mask=mask_h_2d & (offs_v_2[None, :] < d_v))
    if NUM_V_BLOCKS > 3:
        tl.store(o_base + offs_h_2d * stride_o_h + offs_v_3[None, :] * stride_o_d, acc_3.to(tl.bfloat16), mask=mask_h_2d & (offs_v_3[None, :] < d_v))


def direct_dual_sparse_attention(
    q: torch.Tensor,
    primary_buffer: torch.Tensor,
    primary_indices: torch.Tensor,
    primary_lengths: torch.Tensor,
    extra_buffer: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_lengths: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    store_lse: bool = True,
):
    total_tokens, h_q, d_qk = q.shape
    d_v = primary_buffer.shape[-1]
    primary_topk = primary_indices.shape[1]
    total_topk = primary_topk + extra_indices.shape[1]
    if output is None:
        output = torch.empty((total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((total_tokens, h_q), dtype=torch.float32, device=q.device)
    num_qk_blocks = triton.cdiv(d_qk, 128)
    num_v_blocks = triton.cdiv(d_v, 128)
    if num_qk_blocks > 4 or num_v_blocks > 4:
        raise ValueError(f"direct_dual_sparse_attention only supports d_qk/d_v <= 512, got {d_qk}/{d_v}")
    has_attn_sink = attn_sink is not None
    attn_sink_tensor = attn_sink if has_attn_sink else lse[:1]
    primary_lengths = primary_lengths.to(torch.int32) if primary_lengths.dtype not in (torch.int32, torch.int64) else primary_lengths
    extra_lengths = extra_lengths.to(torch.int32) if extra_lengths.dtype not in (torch.int32, torch.int64) else extra_lengths

    grid = lambda meta: (triton.cdiv(h_q, meta["BLOCK_H"]), total_tokens)
    _direct_dual_sparse_attention_kernel[grid](
        q,
        primary_buffer,
        extra_buffer,
        primary_indices,
        primary_lengths,
        extra_indices,
        extra_lengths,
        attn_sink_tensor,
        output,
        lse,
        sm_scale,
        primary_buffer.view(-1, d_v).shape[0],
        extra_buffer.view(-1, d_v).shape[0],
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        primary_buffer.view(-1, d_v).stride(0),
        primary_buffer.view(-1, d_v).stride(1),
        extra_buffer.view(-1, d_v).stride(0),
        extra_buffer.view(-1, d_v).stride(1),
        primary_indices.stride(0),
        primary_indices.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=has_attn_sink,
        STORE_LSE=store_lse,
        primary_topk=primary_topk,
        total_topk=total_topk,
        d_qk=d_qk,
        d_v=d_v,
        NUM_QK_BLOCKS=num_qk_blocks,
        NUM_V_BLOCKS=num_v_blocks,
    )
    return output, lse
