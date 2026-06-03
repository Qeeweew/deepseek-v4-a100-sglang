import os

os.environ["ENABLE_SGLANG_DSV4_A100_PATCH"] = "1"

import torch

from dsv4_a100_patch import _gather_bf16_kv, apply_patch


def _manual_attention(q, kv, invalid_mask, d_v, sm_scale, attn_sink):
    k = kv[:, :, : q.shape[-1]].float()
    v = kv[:, :, :d_v].float()
    logits = torch.einsum("thd,tkd->thk", q.float(), k) * sm_scale
    logits = logits.masked_fill(invalid_mask[:, None, :], float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.float().view(1, -1, 1).expand(q.shape[0], -1, -1)
        logits = torch.cat([logits, sink], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = probs[:, :, : kv.shape[1]]
    return torch.einsum("thk,tkd->thd", probs, v).to(torch.bfloat16)


def test_patch_imports_and_rope_symbol():
    apply_patch()
    import sglang.jit_kernel.dsv4.elementwise as elementwise
    import sglang.srt.models.deepseek_v4 as model

    assert elementwise.fused_rope_inplace is model.fused_rope_inplace


def test_patch_sets_mxfp4_moe_triton_fallback():
    apply_patch()
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    assert Mxfp4MarlinMoEMethod.apply.__module__ == "dsv4_a100_patch"
    assert (
        Mxfp4MarlinMoEMethod.process_weights_after_loading.__module__
        == "dsv4_a100_patch"
    )
    assert hasattr(Mxfp4MarlinMoEMethod, "_sglang_original_apply")


def test_patch_deepseek_v4_defaults_sets_marlin():
    apply_patch()
    from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
    from sglang.srt.server_args import ServerArgs

    class Args:
        attention_backend = None
        page_size = None
        max_running_requests = None
        kv_cache_dtype = "auto"
        swa_full_tokens_ratio = ServerArgs.swa_full_tokens_ratio
        speculative_algorithm = None
        moe_runner_backend = "auto"

    args = Args()
    apply_deepseek_v4_defaults(args, "DeepseekV4ForCausalLM")
    assert args.attention_backend == "dsv4"
    assert args.page_size == 256
    assert args.kv_cache_dtype == "bf16"
    assert args.moe_runner_backend == "marlin"


def test_bf16_gather_cpu_or_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buf = torch.arange(2 * 4 * 1 * 8, device=device, dtype=torch.float32).to(torch.bfloat16)
    buf = buf.view(2, 4, 1, 8)
    idx = torch.tensor([[0, 1, -1, -1], [4, 7, 2, -1]], device=device, dtype=torch.int32)
    lengths = torch.tensor([2, 3], device=device, dtype=torch.int32)
    gathered, mask = _gather_bf16_kv(buf.squeeze(2), idx, lengths, 4)
    flat = buf.view(-1, 8)
    assert torch.equal(gathered[0, 0], flat[0])
    assert torch.equal(gathered[0, 1], flat[1])
    assert torch.equal(gathered[1, 0], flat[4])
    assert torch.equal(gathered[1, 1], flat[7])
    assert torch.equal(gathered[1, 2], flat[2])
    assert mask.tolist() == [[False, False, True, True], [False, False, False, True]]


def test_triton_unified_attention_matches_torch():
    if not torch.cuda.is_available():
        return
    apply_patch()
    from dsv4_a100_patch import _TRITON_COMMON

    torch.manual_seed(0)
    total_tokens = 3
    h_q = 4
    total_topk = 8
    d_qk = 512
    d_v = 512
    sm_scale = d_qk**-0.5
    q = torch.randn(total_tokens, h_q, d_qk, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(total_tokens, total_topk, d_qk, device="cuda", dtype=torch.bfloat16)
    invalid_mask = torch.zeros(total_tokens, total_topk, device="cuda", dtype=torch.bool)
    invalid_mask[0, -2:] = True
    invalid_mask[1, -1:] = True
    attn_sink = torch.randn(h_q, device="cuda", dtype=torch.float32)

    out, _ = _TRITON_COMMON.run_unified_attention(
        q.contiguous(),
        kv.contiguous(),
        invalid_mask.contiguous(),
        d_v,
        sm_scale,
        total_tokens,
        h_q,
        total_topk,
        d_qk,
        attn_sink=attn_sink,
    )
    ref = _manual_attention(q, kv, invalid_mask, d_v, sm_scale, attn_sink)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_indexer_bf16_torch_fallback_values():
    if not torch.cuda.is_available():
        return
    apply_patch()
    import sglang.srt.layers.attention.dsv4.indexer as indexer

    batch_size = 2
    num_heads = 3
    head_dim = 128
    block_size = 64
    max_seq_len = 96
    q = torch.randn(batch_size, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(4, block_size, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(batch_size, num_heads, 1, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.tensor([[96], [70]], device="cuda", dtype=torch.int32)
    page_table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    logits = indexer.bf16_paged_mqa_logits_torch(
        q, cache, weight, seq_lens, page_table, None, max_seq_len, False
    )
    pages = page_table[:, :2].to(torch.long)
    values = cache.squeeze(2)[pages].reshape(batch_size, 128, head_dim).float()
    scores = torch.einsum("bld,bhd->blh", values, q[:, 0].float())
    ref = (torch.relu(scores) * weight.squeeze(-1)[:, None, :].float()).sum(dim=-1)
    ref = ref[:, :max_seq_len].contiguous()
    pos = torch.arange(max_seq_len, device="cuda").unsqueeze(0)
    ref = ref.masked_fill(pos >= seq_lens.squeeze(-1).long().unsqueeze(1), 0.0)
    torch.testing.assert_close(logits, ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_patch_imports_and_rope_symbol()
    test_bf16_gather_cpu_or_cuda()
    test_triton_unified_attention_matches_torch()
    test_indexer_bf16_torch_fallback_values()
    print("dsv4_a100_patch tests passed")
