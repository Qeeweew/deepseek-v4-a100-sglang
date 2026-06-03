import os
import time

import pytest
import torch

from triton_kernels import (
    bf16_paged_mqa_logits,
    bf16_paged_mqa_logits_torch,
    compressor_decode_mask_positions,
    compressor_decode_mask_positions_torch,
    compressor_prefill_metadata,
    compressor_prefill_metadata_torch,
    compressor_positions_from_plan,
    compressor_positions_from_plan_torch,
    fused_rope_inplace,
    fused_rope_inplace_torch,
    gather_bf16_kv,
    gather_bf16_kv_into,
    gather_bf16_kv_torch,
    scatter_bf16_rows,
    scatter_bf16_rows_torch,
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
    torch_ms = _bench(lambda: fused_rope_inplace_torch(q_native, k_native, freqs, positions), iters=20)
    triton_ms = _bench(lambda: fused_rope_inplace(q_fast, k_fast, freqs, positions), iters=20)
    print(f"perf fused_rope_inplace torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0


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

    torch_ms = _bench(lambda: scatter_bf16_rows_torch(dst_ref, loc, src))
    triton_ms = _bench(lambda: scatter_bf16_rows(dst_tri, loc, src))
    print(f"perf scatter_bf16_rows torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0


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

    torch_ms = _bench(lambda: gather_bf16_kv_torch(buffer, indices, lengths, total_topk), iters=15)
    triton_ms = _bench(lambda: gather_bf16_kv(buffer, indices, lengths, total_topk), iters=15)
    print(f"perf gather_bf16_kv torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0


def test_bf16_paged_mqa_logits_accuracy_and_perf():
    torch.manual_seed(4)
    device = "cuda"
    batch, heads, head_dim, max_seq_len = 8, 64, 128, 1024
    block_size = 64
    num_pages = 4096
    pages_per_batch = (max_seq_len + block_size - 1) // block_size
    q = torch.randn(batch, 1, heads, head_dim, device=device).to(torch.float8_e4m3fn)
    kv = torch.randn(num_pages, block_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(batch, heads, device=device, dtype=torch.float32)
    seq_lens = torch.randint(max_seq_len // 2, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    page_table = torch.randint(0, num_pages, (batch, pages_per_batch), device=device, dtype=torch.int32)

    ref = bf16_paged_mqa_logits_torch(q, kv, weight, seq_lens, page_table, None, max_seq_len, False)
    tri = bf16_paged_mqa_logits(q, kv, weight, seq_lens, page_table, None, max_seq_len, False)
    torch.cuda.synchronize()
    torch.testing.assert_close(tri, ref, atol=6e-1, rtol=6e-2)

    torch_ms = _bench(lambda: bf16_paged_mqa_logits_torch(q, kv, weight, seq_lens, page_table, None, max_seq_len, False), iters=10)
    triton_ms = _bench(lambda: bf16_paged_mqa_logits(q, kv, weight, seq_lens, page_table, None, max_seq_len, False), iters=10)
    print(f"perf bf16_paged_mqa_logits torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0


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

    torch_ms = _bench(lambda: compressor_decode_mask_positions_torch(kv_ref, plan, ratio))
    triton_ms = _bench(lambda: compressor_decode_mask_positions(kv_tri, plan, ratio))
    print(f"perf compressor_decode_mask_positions torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0


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

    torch_ms = _bench(lambda: compressor_prefill_metadata_torch(plan, out_loc, ratio))
    triton_ms = _bench(lambda: compressor_prefill_metadata(plan, out_loc, ratio))
    print(f"perf compressor_prefill_metadata torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0


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

    torch_ms = _bench(lambda: compressor_positions_from_plan_torch(plan, ratio))
    triton_ms = _bench(lambda: compressor_positions_from_plan(plan, ratio))
    print(f"perf compressor_positions_from_plan torch={torch_ms:.3f}ms triton={triton_ms:.3f}ms")
    assert triton_ms > 0
