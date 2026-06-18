import pytest
import torch

from triton_kernels.mxfp4_int8_moe import (
    _compute_remap_stats,
    mxfp4_int8_dense_forward,
    mxfp4_int8_moe_forward,
    prepare_mxfp4_int8_dense_weight,
    prepare_mxfp4_int8_moe,
    remap_mxfp4_weight_for_int8,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


_LUT_X2 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
)


def _make_random_packed_mxfp4(experts, n, k, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    packed = torch.randint(
        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8, generator=gen
    )
    row_base = torch.randint(
        122, 128, (experts, n, 1), device="cuda", dtype=torch.uint8, generator=gen
    )
    delta = torch.randint(
        0, 4, (experts, n, k // 32), device="cuda", dtype=torch.uint8, generator=gen
    )
    scale = row_base - delta
    return packed, scale


def _dequant_mxfp4(weight, scale):
    u = weight.view(torch.uint8)
    codes = torch.empty((*u.shape[:-1], u.shape[-1] * 2), device=u.device, dtype=torch.uint8)
    codes[..., 0::2] = u & 0x0F
    codes[..., 1::2] = (u >> 4) & 0x0F
    q_x2 = _LUT_X2.to(u.device)[codes.long()]
    exp = scale.view(torch.uint8).to(torch.int16) - 127
    return q_x2.view(*scale.shape, 32) * torch.exp2(exp.float()).unsqueeze(-1) * 0.5


def _mxfp4_moe_reference(
    hidden_states,
    w13,
    s13,
    w2,
    s2,
    topk_ids,
    topk_weights,
    clamp_limit=None,
):
    w13_f = _dequant_mxfp4(w13, s13).reshape(w13.shape[0], w13.shape[1], -1)
    w2_f = _dequant_mxfp4(w2, s2).reshape(w2.shape[0], w2.shape[1], -1)
    m, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    intermediate = w2_f.shape[2]
    out = torch.zeros((m, hidden), device=hidden_states.device, dtype=torch.float32)
    for token in range(m):
        x = hidden_states[token].float()
        for route in range(topk):
            expert = int(topk_ids[token, route])
            gate_up = torch.matmul(x, w13_f[expert].float().t())
            gate, up = gate_up.chunk(2, dim=-1)
            if clamp_limit is not None:
                gate = gate.clamp(max=clamp_limit)
                up = up.clamp(min=-clamp_limit, max=clamp_limit)
            act = torch.nn.functional.silu(gate) * up
            assert act.numel() == intermediate
            out[token] += torch.matmul(act, w2_f[expert].float().t()) * topk_weights[token, route].float()
    return out.to(torch.bfloat16)


def _int4_rowwise_weight(weight_f):
    max_abs = weight_f.abs().amax(dim=-1).clamp_min(1e-12)
    scale = max_abs / 7.0
    q = torch.round(weight_f / scale.unsqueeze(-1)).clamp(-8, 7)
    return q * scale.unsqueeze(-1)


def _int4_moe_reference(hidden_states, w13, s13, w2, s2, topk_ids, topk_weights):
    w13_q = _int4_rowwise_weight(_dequant_mxfp4(w13, s13).reshape(w13.shape[0], w13.shape[1], -1))
    w2_q = _int4_rowwise_weight(_dequant_mxfp4(w2, s2).reshape(w2.shape[0], w2.shape[1], -1))
    m, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    out = torch.zeros((m, hidden), device=hidden_states.device, dtype=torch.float32)
    for token in range(m):
        x = hidden_states[token].float()
        for route in range(topk):
            expert = int(topk_ids[token, route])
            gate, up = torch.matmul(x, w13_q[expert].float().t()).chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            out[token] += torch.matmul(act, w2_q[expert].float().t()) * topk_weights[token, route].float()
    return out.to(torch.bfloat16)


def _relative_l2(actual, expected):
    diff = (actual.float() - expected.float()).norm()
    denom = expected.float().norm().clamp_min(1e-12)
    return float((diff / denom).item())


def test_mxfp4_int8_weight_remap_tracks_original_mxfp4():
    experts, n, k = 3, 96, 128
    weight, scale = _make_random_packed_mxfp4(experts, n, k, seed=20260618)
    _, _, channel_scale = remap_mxfp4_weight_for_int8(weight, scale)
    stats = _compute_remap_stats(weight, scale, headroom_bits=3)

    assert channel_scale.shape == (experts, n)
    assert stats["overflow_count"] == 0
    assert stats["exact_rate_nonzero"] > 0.98
    assert stats["mean_rel_err_nonzero"] < 0.02


def test_triton_repack_matches_torch_repack_bitwise():
    weight, scale = _make_random_packed_mxfp4(3, 19, 128, seed=20260620)
    tri = remap_mxfp4_weight_for_int8(
        weight, scale, headroom_bits=3, use_triton=True
    )
    ref = remap_mxfp4_weight_for_int8(
        weight, scale, headroom_bits=3, use_triton=False
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(tri[0], ref[0], atol=0, rtol=0)
    torch.testing.assert_close(tri[1], ref[1], atol=0, rtol=0)
    torch.testing.assert_close(tri[2], ref[2], atol=0, rtol=0)


def test_mxfp4_int8_dense_matches_original_mxfp4_reference():
    n, k, m = 128, 128, 11
    weight, scale = _make_random_packed_mxfp4(1, n, k, seed=20260619)
    dense_weight = prepare_mxfp4_int8_dense_weight(weight[0], scale[0], headroom_bits=3)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2
    out = mxfp4_int8_dense_forward(x, dense_weight)
    ref_w = _dequant_mxfp4(weight, scale).reshape(1, n, k)[0]
    ref = torch.matmul(x.float(), ref_w.float().t()).to(torch.bfloat16)
    torch.cuda.synchronize()
    assert _relative_l2(out, ref) < 0.04


def test_mxfp4_int8_moe_is_closer_to_original_mxfp4_than_rowwise_int4():
    torch.manual_seed(20260618)

    class Layer(torch.nn.Module):
        pass

    experts, tokens, hidden, intermediate, topk = 4, 17, 128, 128, 2
    w13, s13 = _make_random_packed_mxfp4(experts, 2 * intermediate, hidden, seed=11)
    w2, s2 = _make_random_packed_mxfp4(experts, hidden, intermediate, seed=12)

    layer = Layer().cuda()
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale_inv = torch.nn.Parameter(s13, requires_grad=False)
    layer.w2_weight_scale_inv = torch.nn.Parameter(s2, requires_grad=False)
    prepare_mxfp4_int8_moe(layer, headroom_bits=3)
    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0
    assert layer.w13_weight_scale_inv.numel() == 0
    assert layer.w2_weight_scale_inv.numel() == 0

    hidden_states = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.2
    topk_ids = torch.stack(
        [
            torch.arange(tokens, device="cuda", dtype=torch.int32) % experts,
            (torch.arange(tokens, device="cuda", dtype=torch.int32) * 3 + 1) % experts,
        ],
        dim=1,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device="cuda", dtype=torch.float32), dim=-1
    )

    out = mxfp4_int8_moe_forward(
        hidden_states,
        layer._dsv4_mxfp4_int8_weights,
        topk_ids,
        topk_weights,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    ref = _mxfp4_moe_reference(hidden_states, w13, s13, w2, s2, topk_ids, topk_weights)
    int4_ref = _int4_moe_reference(hidden_states, w13, s13, w2, s2, topk_ids, topk_weights)
    torch.cuda.synchronize()

    int8_rel = _relative_l2(out, ref)
    int4_rel = _relative_l2(int4_ref, ref)
    print(f"mxfp4_int8_rel_l2={int8_rel:.6f} rowwise_int4_rel_l2={int4_rel:.6f}")
    assert int8_rel < 0.08
    assert int8_rel < int4_rel
