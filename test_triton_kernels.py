import os

import pytest
import torch

from triton_kernels import (
    bf16_indexer_q,
    bf16_indexer_q_torch,
    bf16_paged_mqa_logits,
    bf16_paged_mqa_logits_torch,
    compressor_decode_mask_positions,
    compressor_decode_mask_positions_torch,
    compressor_prefill_metadata,
    compressor_prefill_metadata_torch,
    compressor_positions_from_plan,
    compressor_positions_from_plan_torch,
    direct_dual_sparse_attention,
    direct_sparse_attention,
    fused_rope_inplace,
    fused_rope_inplace_torch,
    gather_bf16_kv,
    gather_bf16_kv_into,
    gather_bf16_kv_torch,
    scatter_bf16_rows,
    scatter_bf16_rows_torch,
    trim_and_pad_rows,
    trim_and_pad_rows_torch,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench_cuda_graph(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _perf_pair(label, eager_fn, graph_fn=None, eager_warmup=10, eager_iters=30, graph_warmup=10, graph_iters=100):
    eager_ms = _bench(eager_fn, warmup=eager_warmup, iters=eager_iters)
    graph_ms = _bench_cuda_graph(graph_fn or eager_fn, warmup=graph_warmup, iters=graph_iters)
    print(f"perf {label} eager={eager_ms:.3f}ms cuda_graph={graph_ms:.3f}ms")
    return eager_ms, graph_ms


def _pack_plan(seq_lens, ragged_ids=None):
    if ragged_ids is None:
        ragged_ids = torch.arange(seq_lens.numel(), device=seq_lens.device, dtype=torch.int32)
    plan_i32 = torch.zeros((seq_lens.numel(), 4), device=seq_lens.device, dtype=torch.int32)
    plan_i32[:, 0] = seq_lens.to(torch.int32)
    plan_i32[:, 1] = ragged_ids.to(torch.int32)
    return plan_i32.view(torch.uint8)


def test_fused_rope_inplace_accuracy_and_perf():
    torch.manual_seed(1)
    device = "cuda"
    batch, q_heads, k_heads, dim, seqlen = 128, 64, 1, 64, 4096
    q_ref = torch.randn(batch, q_heads, dim, device=device, dtype=torch.bfloat16)
    k_ref = torch.randn(batch, k_heads, dim, device=device, dtype=torch.bfloat16)
    q_tri = q_ref.clone()
    k_tri = k_ref.clone()
    freqs = torch.polar(
        torch.ones((seqlen, dim // 2), device=device),
        torch.randn((seqlen, dim // 2), device=device),
    )
    positions = torch.randint(0, seqlen, (batch,), device=device, dtype=torch.int32)

    fused_rope_inplace_torch(q_ref, k_ref, freqs, positions)
    fused_rope_inplace(q_tri, k_tri, freqs, positions)
    torch.cuda.synchronize()
    torch.testing.assert_close(q_tri.float(), q_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_tri.float(), k_ref.float(), atol=3e-2, rtol=3e-2)

    q_native = q_ref.clone()
    k_native = k_ref.clone()
    q_fast = q_ref.clone()
    k_fast = k_ref.clone()
    _, torch_graph_ms = _perf_pair(
        "fused_rope_inplace_torch",
        lambda: fused_rope_inplace_torch(q_native, k_native, freqs, positions),
        eager_iters=20,
    )
    _, triton_graph_ms = _perf_pair(
        "fused_rope_inplace_triton",
        lambda: fused_rope_inplace(q_fast, k_fast, freqs, positions),
        eager_iters=20,
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0


def test_scatter_bf16_rows_accuracy_and_perf():
    torch.manual_seed(2)
    device = "cuda"
    rows, cache_rows, head_dim = 2048, 8192, 512
    loc = torch.randperm(cache_rows, device=device, dtype=torch.int64)[:rows].contiguous()
    src = torch.randn(rows, 1, head_dim, device=device, dtype=torch.bfloat16)
    dst_ref = torch.zeros(cache_rows, 1, head_dim, device=device, dtype=torch.bfloat16)
    dst_tri = torch.zeros_like(dst_ref)

    scatter_bf16_rows_torch(dst_ref, loc, src)
    scatter_bf16_rows(dst_tri, loc, src)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst_tri, dst_ref, atol=0, rtol=0)

    _, torch_graph_ms = _perf_pair(
        "scatter_bf16_rows_torch",
        lambda: scatter_bf16_rows_torch(dst_ref, loc, src),
    )
    _, triton_graph_ms = _perf_pair(
        "scatter_bf16_rows_triton",
        lambda: scatter_bf16_rows(dst_tri, loc, src),
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0


def test_scatter_bf16_rows_flat_paged_cache_accuracy():
    torch.manual_seed(22)
    device = "cuda"
    rows, pages, page_size, head_dim = 31, 8, 64, 128
    loc = torch.randperm(pages * page_size, device=device, dtype=torch.int64)[:rows].contiguous()
    src = torch.randn(rows, head_dim, device=device, dtype=torch.bfloat16)
    dst_ref = torch.zeros(pages, page_size * head_dim, device=device, dtype=torch.bfloat16)
    dst_tri = torch.zeros_like(dst_ref)

    scatter_bf16_rows_torch(dst_ref, loc, src)
    scatter_bf16_rows(dst_tri, loc, src)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst_tri, dst_ref, atol=0, rtol=0)


def test_gather_bf16_kv_accuracy_and_perf():
    torch.manual_seed(3)
    device = "cuda"
    q_tokens, total_topk, head_dim = 128, 512, 512
    buffer = torch.randn(16384, head_dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(-8, buffer.shape[0], (q_tokens, total_topk), device=device, dtype=torch.int32)
    lengths = torch.randint(1, total_topk + 1, (q_tokens,), device=device, dtype=torch.int32)

    out_ref, mask_ref = gather_bf16_kv_torch(buffer, indices, lengths, total_topk)
    out_tri, mask_tri = gather_bf16_kv(buffer, indices, lengths, total_topk)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_tri, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(mask_tri, mask_ref, atol=0, rtol=0)

    out_into = torch.empty(q_tokens, total_topk + 7, head_dim, device=device, dtype=torch.bfloat16)
    mask_into = torch.empty(q_tokens, total_topk + 7, device=device, dtype=torch.bool)
    out_into.fill_(1)
    mask_into.fill_(True)
    gather_bf16_kv_into(buffer, indices, lengths, total_topk, out_into, mask_into, 7)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_into[:, 7:], out_ref, atol=0, rtol=0)
    torch.testing.assert_close(mask_into[:, 7:], mask_ref, atol=0, rtol=0)

    _, torch_graph_ms = _perf_pair(
        "gather_bf16_kv_torch",
        lambda: gather_bf16_kv_torch(buffer, indices, lengths, total_topk),
        eager_iters=15,
    )
    _, triton_graph_ms = _perf_pair(
        "gather_bf16_kv_triton",
        lambda: gather_bf16_kv(buffer, indices, lengths, total_topk),
        eager_iters=15,
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0


def test_trim_and_pad_rows_accuracy():
    torch.manual_seed(31)
    device = "cuda"
    idx = torch.randint(-1, 1000, (11, 64), device=device, dtype=torch.int32)
    lengths = torch.randint(1, 65, (11,), device=device, dtype=torch.int32)

    ref_idx, ref_len = trim_and_pad_rows_torch(idx, lengths, 17)
    out_idx, out_len = trim_and_pad_rows(idx, lengths, 17)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_idx, ref_idx, atol=0, rtol=0)
    torch.testing.assert_close(out_len, ref_len, atol=0, rtol=0)

    ref_idx2, ref_len2 = trim_and_pad_rows_torch(idx, lengths, 7)
    out_idx2, out_len2 = trim_and_pad_rows(idx, lengths, 7)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_idx2, ref_idx2, atol=0, rtol=0)
    torch.testing.assert_close(out_len2, ref_len2, atol=0, rtol=0)


def test_direct_sparse_attention_matches_gather_plus_unified():
    torch.manual_seed(32)
    device = "cuda"
    q_tokens, heads, head_dim, topk = 8, 16, 512, 64
    q = torch.randn(q_tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    buffer = torch.randn(2048, head_dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(-1, buffer.shape[0], (q_tokens, topk), device=device, dtype=torch.int32)
    lengths = torch.randint(1, topk + 1, (q_tokens,), device=device, dtype=torch.int32)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    gathered, invalid = gather_bf16_kv(buffer, indices, lengths, topk)

    from dsv4_a100_patch import _TRITON_COMMON

    ref, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        gathered.contiguous(),
        invalid.contiguous(),
        head_dim,
        head_dim**-0.5,
        q_tokens,
        heads,
        topk,
        head_dim,
        attn_sink=attn_sink,
    )
    out, _ = direct_sparse_attention(q, buffer, indices, lengths, head_dim**-0.5, attn_sink=attn_sink)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    _, direct_graph_ms = _perf_pair(
        "direct_sparse_attention_triton",
        lambda: direct_sparse_attention(q, buffer, indices, lengths, head_dim**-0.5, attn_sink=attn_sink),
        eager_iters=10,
    )
    _, gathered_graph_ms = _perf_pair(
        "gather_plus_unified_attention",
        lambda: _TRITON_COMMON.run_unified_attention(
            q.contiguous(),
            gathered.contiguous(),
            invalid.contiguous(),
            head_dim,
            head_dim**-0.5,
            q_tokens,
            heads,
            topk,
            head_dim,
            attn_sink=attn_sink,
        ),
        eager_iters=10,
    )
    assert direct_graph_ms > 0
    assert gathered_graph_ms > 0


def test_direct_sparse_attention_128_matches_gather_plus_unified():
    torch.manual_seed(132)
    device = "cuda"
    q_tokens, heads, head_dim, topk = 8, 16, 128, 64
    q = torch.randn(q_tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    buffer = torch.randn(2048, head_dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(-1, buffer.shape[0], (q_tokens, topk), device=device, dtype=torch.int32)
    lengths = torch.randint(1, topk + 1, (q_tokens,), device=device, dtype=torch.int32)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    gathered, invalid = gather_bf16_kv(buffer, indices, lengths, topk)

    from dsv4_a100_patch import _TRITON_COMMON

    ref, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        gathered.contiguous(),
        invalid.contiguous(),
        head_dim,
        head_dim**-0.5,
        q_tokens,
        heads,
        topk,
        head_dim,
        attn_sink=attn_sink,
    )
    out, _ = direct_sparse_attention(q, buffer, indices, lengths, head_dim**-0.5, attn_sink=attn_sink)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_direct_sparse_attention_masks_out_of_range_indices():
    torch.manual_seed(232)
    device = "cuda"
    q_tokens, heads, head_dim, topk = 4, 8, 128, 16
    q = torch.randn(q_tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    buffer = torch.randn(64, head_dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(-1, buffer.shape[0], (q_tokens, topk), device=device, dtype=torch.int32)
    indices[0, 0] = buffer.shape[0]
    indices[1, 1] = buffer.shape[0] + 17
    lengths = torch.full((q_tokens,), topk, device=device, dtype=torch.int32)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    gathered, invalid = gather_bf16_kv(buffer, indices, lengths, topk)

    from dsv4_a100_patch import _TRITON_COMMON

    ref, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        gathered.contiguous(),
        invalid.contiguous(),
        head_dim,
        head_dim**-0.5,
        q_tokens,
        heads,
        topk,
        head_dim,
        attn_sink=attn_sink,
    )
    out, _ = direct_sparse_attention(q, buffer, indices, lengths, head_dim**-0.5, attn_sink=attn_sink)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_direct_dual_sparse_attention_matches_gather_plus_unified():
    torch.manual_seed(33)
    device = "cuda"
    q_tokens, heads, head_dim = 8, 16, 512
    topk0, topk1 = 48, 24
    q = torch.randn(q_tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    buf0 = torch.randn(2048, head_dim, device=device, dtype=torch.bfloat16)
    buf1 = torch.randn(1024, head_dim, device=device, dtype=torch.bfloat16)
    idx0 = torch.randint(-1, buf0.shape[0], (q_tokens, topk0), device=device, dtype=torch.int32)
    idx1 = torch.randint(-1, buf1.shape[0], (q_tokens, topk1), device=device, dtype=torch.int32)
    len0 = torch.randint(1, topk0 + 1, (q_tokens,), device=device, dtype=torch.int32)
    len1 = torch.randint(1, topk1 + 1, (q_tokens,), device=device, dtype=torch.int32)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    gathered0, invalid0 = gather_bf16_kv(buf0, idx0, len0, topk0)
    gathered1, invalid1 = gather_bf16_kv(buf1, idx1, len1, topk1)
    gathered = torch.cat([gathered0, gathered1], dim=1)
    invalid = torch.cat([invalid0, invalid1], dim=1)

    from dsv4_a100_patch import _TRITON_COMMON

    ref, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        gathered.contiguous(),
        invalid.contiguous(),
        head_dim,
        head_dim**-0.5,
        q_tokens,
        heads,
        topk0 + topk1,
        head_dim,
        attn_sink=attn_sink,
    )
    out, _ = direct_dual_sparse_attention(
        q, buf0, idx0, len0, buf1, idx1, len1, head_dim**-0.5, attn_sink=attn_sink
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    _, direct_graph_ms = _perf_pair(
        "direct_dual_sparse_attention_triton",
        lambda: direct_dual_sparse_attention(
            q, buf0, idx0, len0, buf1, idx1, len1, head_dim**-0.5, attn_sink=attn_sink
        ),
        eager_iters=10,
    )
    _, gathered_graph_ms = _perf_pair(
        "dual_gather_plus_unified_attention",
        lambda: _TRITON_COMMON.run_unified_attention(
            q.contiguous(),
            gathered.contiguous(),
            invalid.contiguous(),
            head_dim,
            head_dim**-0.5,
            q_tokens,
            heads,
            topk0 + topk1,
            head_dim,
            attn_sink=attn_sink,
        ),
        eager_iters=10,
    )
    assert direct_graph_ms > 0
    assert gathered_graph_ms > 0


def test_direct_dual_sparse_attention_128_matches_gather_plus_unified():
    torch.manual_seed(133)
    device = "cuda"
    q_tokens, heads, head_dim = 8, 16, 128
    topk0, topk1 = 48, 24
    q = torch.randn(q_tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    buf0 = torch.randn(2048, head_dim, device=device, dtype=torch.bfloat16)
    buf1 = torch.randn(1024, head_dim, device=device, dtype=torch.bfloat16)
    idx0 = torch.randint(-1, buf0.shape[0], (q_tokens, topk0), device=device, dtype=torch.int32)
    idx1 = torch.randint(-1, buf1.shape[0], (q_tokens, topk1), device=device, dtype=torch.int32)
    len0 = torch.randint(1, topk0 + 1, (q_tokens,), device=device, dtype=torch.int32)
    len1 = torch.randint(1, topk1 + 1, (q_tokens,), device=device, dtype=torch.int32)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    gathered0, invalid0 = gather_bf16_kv(buf0, idx0, len0, topk0)
    gathered1, invalid1 = gather_bf16_kv(buf1, idx1, len1, topk1)
    gathered = torch.cat([gathered0, gathered1], dim=1)
    invalid = torch.cat([invalid0, invalid1], dim=1)

    from dsv4_a100_patch import _TRITON_COMMON

    ref, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        gathered.contiguous(),
        invalid.contiguous(),
        head_dim,
        head_dim**-0.5,
        q_tokens,
        heads,
        topk0 + topk1,
        head_dim,
        attn_sink=attn_sink,
    )
    out, _ = direct_dual_sparse_attention(
        q, buf0, idx0, len0, buf1, idx1, len1, head_dim**-0.5, attn_sink=attn_sink
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_direct_dual_sparse_attention_masks_out_of_range_indices():
    torch.manual_seed(233)
    device = "cuda"
    q_tokens, heads, head_dim = 4, 8, 128
    topk0, topk1 = 16, 12
    q = torch.randn(q_tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    buf0 = torch.randn(64, head_dim, device=device, dtype=torch.bfloat16)
    buf1 = torch.randn(32, head_dim, device=device, dtype=torch.bfloat16)
    idx0 = torch.randint(-1, buf0.shape[0], (q_tokens, topk0), device=device, dtype=torch.int32)
    idx1 = torch.randint(-1, buf1.shape[0], (q_tokens, topk1), device=device, dtype=torch.int32)
    idx0[0, 0] = buf0.shape[0] + 3
    idx1[2, 2] = buf1.shape[0] + 5
    len0 = torch.full((q_tokens,), topk0, device=device, dtype=torch.int32)
    len1 = torch.full((q_tokens,), topk1, device=device, dtype=torch.int32)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    gathered0, invalid0 = gather_bf16_kv(buf0, idx0, len0, topk0)
    gathered1, invalid1 = gather_bf16_kv(buf1, idx1, len1, topk1)
    gathered = torch.cat([gathered0, gathered1], dim=1)
    invalid = torch.cat([invalid0, invalid1], dim=1)

    from dsv4_a100_patch import _TRITON_COMMON

    ref, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        gathered.contiguous(),
        invalid.contiguous(),
        head_dim,
        head_dim**-0.5,
        q_tokens,
        heads,
        topk0 + topk1,
        head_dim,
        attn_sink=attn_sink,
    )
    out, _ = direct_dual_sparse_attention(
        q, buf0, idx0, len0, buf1, idx1, len1, head_dim**-0.5, attn_sink=attn_sink
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_bf16_paged_mqa_logits_accuracy_and_perf():
    torch.manual_seed(4)
    device = "cuda"
    batch, heads, head_dim, max_seq_len = 8, 64, 128, 1024
    block_size = 64
    num_pages = 4096
    pages_per_batch = (max_seq_len + block_size - 1) // block_size
    q = torch.randn(batch, 1, heads, head_dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(num_pages, block_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(batch, heads, device=device, dtype=torch.float32)
    seq_lens = torch.randint(max_seq_len // 2, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    page_table = torch.randint(0, num_pages, (batch, pages_per_batch), device=device, dtype=torch.int32)

    ref = bf16_paged_mqa_logits_torch(q, kv, weight, seq_lens, page_table, None, max_seq_len, False)
    tri = bf16_paged_mqa_logits(q, kv, weight, seq_lens, page_table, None, max_seq_len, False)
    torch.cuda.synchronize()
    torch.testing.assert_close(tri, ref, atol=6e-1, rtol=6e-2)

    _, torch_graph_ms = _perf_pair(
        "bf16_paged_mqa_logits_torch",
        lambda: bf16_paged_mqa_logits_torch(q, kv, weight, seq_lens, page_table, None, max_seq_len, False),
        eager_iters=10,
    )
    _, triton_graph_ms = _perf_pair(
        "bf16_paged_mqa_logits_triton",
        lambda: bf16_paged_mqa_logits(q, kv, weight, seq_lens, page_table, None, max_seq_len, False),
        eager_iters=10,
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0


def test_bf16_indexer_q_accuracy_and_perf():
    torch.manual_seed(40)
    device = "cuda"
    batch, heads, head_dim, seqlen = 64, 64, 128, 4096
    q = torch.randn(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(batch, heads, device=device, dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones((seqlen, head_dim // 4), device=device),
        torch.randn((seqlen, head_dim // 4), device=device),
    )
    positions = torch.randint(0, seqlen, (batch,), device=device, dtype=torch.int32)
    weight_scale = 0.125

    q_ref, w_ref = bf16_indexer_q_torch(q.clone(), weight, weight_scale, freqs, positions)
    q_tri, w_tri = bf16_indexer_q(q.clone(), weight, weight_scale, freqs, positions)
    torch.cuda.synchronize()
    torch.testing.assert_close(q_tri.float(), q_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(w_tri.float(), w_ref.float(), atol=1e-5, rtol=1e-5)

    _, torch_graph_ms = _perf_pair(
        "bf16_indexer_q_torch",
        lambda: bf16_indexer_q_torch(q.clone(), weight, weight_scale, freqs, positions),
        eager_iters=10,
    )
    _, triton_graph_ms = _perf_pair(
        "bf16_indexer_q_triton",
        lambda: bf16_indexer_q(q.clone(), weight, weight_scale, freqs, positions),
        eager_iters=10,
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0

    q_buf = torch.empty_like(q)
    w_buf = torch.empty(batch, heads, 1, device=device, dtype=torch.float32)
    scratch_q = torch.empty_like(q)
    q_tri2, w_tri2 = bf16_indexer_q(
        q.clone(),
        weight,
        weight_scale,
        freqs,
        positions,
        q_out=q_buf,
        weights_out=w_buf,
        scratch_q=scratch_q,
    )
    torch.cuda.synchronize()
    assert q_tri2.data_ptr() == q_buf.data_ptr()
    assert w_tri2.data_ptr() == w_buf.data_ptr()
    torch.testing.assert_close(q_tri2.float(), q_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(w_tri2.float(), w_ref.float(), atol=1e-5, rtol=1e-5)

    q_inplace = q.clone().contiguous()
    q_tri3, w_tri3 = bf16_indexer_q(
        q_inplace,
        weight,
        weight_scale,
        freqs,
        positions,
        q_out=q_buf,
        weights_out=w_buf,
        allow_inplace_input=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(q_tri3.float(), q_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(w_tri3.float(), w_ref.float(), atol=1e-5, rtol=1e-5)


def test_compressor_decode_mask_positions_accuracy_and_perf():
    torch.manual_seed(5)
    device = "cuda"
    rows, dim, ratio = 512, 512, 4
    kv_ref = torch.randn(rows, dim, device=device, dtype=torch.bfloat16)
    kv_tri = kv_ref.clone()
    seq_lens = torch.randint(1, 2048, (rows,), device=device, dtype=torch.int32)
    seq_lens[::3] = (seq_lens[::3] // ratio) * ratio
    plan = _pack_plan(seq_lens)

    pos_ref = compressor_decode_mask_positions_torch(kv_ref, plan, ratio)
    pos_tri = compressor_decode_mask_positions(kv_tri, plan, ratio)
    torch.cuda.synchronize()
    torch.testing.assert_close(kv_tri, kv_ref, atol=0, rtol=0)
    torch.testing.assert_close(pos_tri, pos_ref, atol=0, rtol=0)

    _, torch_graph_ms = _perf_pair(
        "compressor_decode_mask_positions_torch",
        lambda: compressor_decode_mask_positions_torch(kv_ref, plan, ratio),
    )
    _, triton_graph_ms = _perf_pair(
        "compressor_decode_mask_positions_triton",
        lambda: compressor_decode_mask_positions(kv_tri, plan, ratio),
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0


def test_compressor_prefill_metadata_accuracy_and_perf():
    torch.manual_seed(6)
    device = "cuda"
    rows, out_rows, ratio = 4096, 8192, 128
    seq_lens = torch.randint(1, 32768, (rows,), device=device, dtype=torch.int32)
    ragged_ids = torch.randint(0, out_rows, (rows,), device=device, dtype=torch.int32)
    out_loc = torch.randint(0, 1 << 30, (out_rows,), device=device, dtype=torch.int64)
    plan = _pack_plan(seq_lens, ragged_ids)

    pos_ref, loc_ref = compressor_prefill_metadata_torch(plan, out_loc, ratio)
    pos_tri, loc_tri = compressor_prefill_metadata(plan, out_loc, ratio)
    torch.cuda.synchronize()
    torch.testing.assert_close(pos_tri, pos_ref, atol=0, rtol=0)
    torch.testing.assert_close(loc_tri, loc_ref, atol=0, rtol=0)

    _, torch_graph_ms = _perf_pair(
        "compressor_prefill_metadata_torch",
        lambda: compressor_prefill_metadata_torch(plan, out_loc, ratio),
    )
    _, triton_graph_ms = _perf_pair(
        "compressor_prefill_metadata_triton",
        lambda: compressor_prefill_metadata(plan, out_loc, ratio),
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0


def test_compressor_positions_from_plan_accuracy_and_perf():
    torch.manual_seed(7)
    device = "cuda"
    rows, ratio = 4096, 4
    seq_lens = torch.randint(1, 32768, (rows,), device=device, dtype=torch.int32)
    plan = _pack_plan(seq_lens)

    pos_ref = compressor_positions_from_plan_torch(plan, ratio)
    pos_tri = compressor_positions_from_plan(plan, ratio)
    torch.cuda.synchronize()
    torch.testing.assert_close(pos_tri, pos_ref, atol=0, rtol=0)

    _, torch_graph_ms = _perf_pair(
        "compressor_positions_from_plan_torch",
        lambda: compressor_positions_from_plan_torch(plan, ratio),
    )
    _, triton_graph_ms = _perf_pair(
        "compressor_positions_from_plan_triton",
        lambda: compressor_positions_from_plan(plan, ratio),
    )
    assert torch_graph_ms > 0
    assert triton_graph_ms > 0
