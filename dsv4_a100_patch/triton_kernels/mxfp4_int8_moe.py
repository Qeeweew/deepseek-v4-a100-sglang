from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.nn import Parameter

logger = logging.getLogger(__name__)


_E2M1_MAG_X2 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], dtype=torch.float32)
_E2M1_SIGNED_X2 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
)
_MXFP4_INT8_MOE_JIT_AVAILABLE: bool | None = None
_MXFP4_INT8_DENSE_JIT_AVAILABLE: bool | None = None
_VALID_BLOCK_M = {16, 32, 64, 128}
_VALID_BLOCK_N = {32, 64, 128}
_DSV4_A100_MXFP4_INT8_TILE_TABLE: tuple[tuple[int, int, int], ...] = (
    (1, 16, 128),
    (8, 16, 64),
    (1024, 16, 128),
    (2048, 64, 128),
    (1 << 30, 128, 128),
)
_MAX_BLOCK_M = 128


@dataclass
class Mxfp4Int8MoeWeights:
    w13_mxfp4: torch.Tensor
    w13_shift2: torch.Tensor
    w13_channel_scale: torch.Tensor
    w2_mxfp4: torch.Tensor
    w2_shift2: torch.Tensor
    w2_channel_scale: torch.Tensor
    headroom_bits: int


@dataclass
class Mxfp4Int8DenseWeight:
    weight_mxfp4: torch.Tensor
    weight_shift2: torch.Tensor
    weight_channel_scale: torch.Tensor
    headroom_bits: int


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["K", "BLOCK_K", "clamp_abs"],
)
@triton.jit
def _quantize_per_token_kernel(
    a,
    a_q,
    a_scale,
    M,
    K: tl.constexpr,
    clamp_abs: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    values = tl.load(a + m * K + offs, mask=mask, other=0.0).to(tl.float32)
    if clamp_abs > 0.0:
        values = tl.minimum(tl.maximum(values, -clamp_abs), clamp_abs)
    max_abs = tl.max(tl.abs(values), axis=0)
    scale = max_abs / 127.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    q_i32 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [values / scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q_i32 = tl.minimum(tl.maximum(q_i32, -127), 127)
    q = q_i32.to(tl.int8)
    tl.store(a_q + m * K + offs, q, mask=mask)
    tl.store(a_scale + m, scale, mask=m < M)


def quantize_per_token_int8(
    a: torch.Tensor,
    clamp_abs: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != 2:
        raise ValueError(f"a must have shape [M, K], got {tuple(a.shape)}")
    if a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"a must be fp32, fp16, or bf16, got {a.dtype}")
    if not a.is_cuda:
        if clamp_abs > 0.0:
            a_f = a.float().clamp(-float(clamp_abs), float(clamp_abs))
        else:
            a_f = a.float()
        scale = a_f.abs().amax(dim=1) / 127.0
        scale = torch.where(scale == 0.0, torch.ones_like(scale), scale).to(torch.float32)
        q = torch.round(a_f / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return q.contiguous(), scale.contiguous()
    a = a.contiguous()
    m, k = a.shape
    a_q = torch.empty_like(a, dtype=torch.int8)
    a_scale = torch.empty((m,), device=a.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(k)
    _quantize_per_token_kernel[(m,)](
        a,
        a_q,
        a_scale,
        m,
        k,
        float(clamp_abs),
        BLOCK_K=block_k,
    )
    return a_q, a_scale


def _mxfp4_int8_moe_jit_enabled() -> bool:
    env = os.environ.get("SGLANG_DSV4_MXFP4_INT8_USE_JIT", "1").strip().lower()
    return env not in {"0", "false", "no", "off"}


def _try_prewarm_mxfp4_int8_moe_jit(
    *, hidden_size: int, intermediate_size: int, topk: int
) -> None:
    global _MXFP4_INT8_MOE_JIT_AVAILABLE
    if not _mxfp4_int8_moe_jit_enabled():
        _MXFP4_INT8_MOE_JIT_AVAILABLE = False
        return
    if _MXFP4_INT8_MOE_JIT_AVAILABLE is False:
        return
    try:
        from dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_moe import (
            prewarm_mxfp4_int8_moe_jit_modules,
        )

        tiles_env = os.environ.get("SGLANG_DSV4_MXFP4_INT8_JIT_PREWARM_TILES")
        if tiles_env:
            tile_shapes = tuple(_parse_tile_shape(item) for item in tiles_env.replace(";", ",").split(",") if item.strip())
            prewarm_mxfp4_int8_moe_jit_modules(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                topk=topk,
                tile_shapes=tile_shapes,
            )
            _MXFP4_INT8_MOE_JIT_AVAILABLE = True
            return

        block_ms_env = os.environ.get("SGLANG_DSV4_MXFP4_INT8_JIT_PREWARM_BLOCK_M")
        block_n_env = os.environ.get("SGLANG_DSV4_MXFP4_INT8_BLOCK_N")
        if block_ms_env is None:
            selected_block_m = os.environ.get("SGLANG_DSV4_MXFP4_INT8_BLOCK_M")
            if selected_block_m:
                block_ms = (int(selected_block_m),)
            else:
                tile_shapes = tuple(sorted({(tile[1], tile[2]) for tile in _DSV4_A100_MXFP4_INT8_TILE_TABLE}))
                prewarm_mxfp4_int8_moe_jit_modules(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    topk=topk,
                    tile_shapes=tile_shapes,
                )
                _MXFP4_INT8_MOE_JIT_AVAILABLE = True
                return
        else:
            block_ms = tuple(
                int(item) for item in block_ms_env.replace(";", ",").split(",") if item.strip()
            )
        prewarm_mxfp4_int8_moe_jit_modules(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topk=topk,
            block_ms=block_ms,
            block_n=int(block_n_env) if block_n_env else None,
        )
        _MXFP4_INT8_MOE_JIT_AVAILABLE = True
    except Exception:
        _MXFP4_INT8_MOE_JIT_AVAILABLE = False
        logger.exception("MXFP4-int8 MoE JIT prewarm failed")


def _mxfp4_int8_moe_grouped_gemm_nt(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    b_channel_scale: torch.Tensor,
    out: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk: int,
    block_size_m: int,
    mul_topk_weights: bool,
    num_valid_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    block_size_n: int,
) -> None:
    global _MXFP4_INT8_MOE_JIT_AVAILABLE
    if _mxfp4_int8_moe_jit_enabled() and _MXFP4_INT8_MOE_JIT_AVAILABLE is not False:
        try:
            from dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_moe import (
                mxfp4_int8_moe_gemm,
                mxfp4_int8_moe_reduce,
            )

            routed_out = None
            if mul_topk_weights:
                routed_out = torch.empty(
                    (num_valid_tokens, hidden_size),
                    device=out.device,
                    dtype=out.dtype,
                )
            mxfp4_int8_moe_gemm(
                a_q,
                a_scale,
                b_mxfp4,
                b_shift2,
                b_channel_scale,
                out,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                topk=topk,
                block_m=block_size_m,
                block_n=block_size_n,
                source_rows_are_slots=mul_topk_weights,
                num_valid_tokens=num_valid_tokens,
                routed_out=routed_out,
            )
            if mul_topk_weights:
                mxfp4_int8_moe_reduce(
                    routed_out,
                    topk_weights,
                    out,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    topk=topk,
                    block_m=block_size_m,
                    block_n=block_size_n,
                    num_valid_tokens=num_valid_tokens,
                )
            _MXFP4_INT8_MOE_JIT_AVAILABLE = True
            return
        except Exception:
            _MXFP4_INT8_MOE_JIT_AVAILABLE = False
            logger.exception("MXFP4-int8 MoE JIT launch failed")
            raise

    raise RuntimeError("MXFP4-int8 MoE JIT is disabled or unavailable")


def _as_uint8_storage(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.data if hasattr(tensor, "data") else tensor
    if data.dtype == torch.uint8:
        return data.contiguous()
    if data.dtype == torch.int8:
        return data.view(torch.uint8).contiguous()
    if data.dtype == torch.float8_e8m0fnu:
        return data.view(torch.uint8).contiguous()
    if data.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return data.to(torch.float8_e8m0fnu).view(torch.uint8).contiguous()
    raise TypeError(f"unsupported MXFP4 scale dtype: {data.dtype}")


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    u = packed.view(torch.uint8)
    codes = torch.empty((*u.shape[:-1], u.shape[-1] * 2), device=u.device, dtype=torch.uint8)
    codes[..., 0::2] = u & 0x0F
    codes[..., 1::2] = (u >> 4) & 0x0F
    return codes


def _pack_codes_coalesced(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or codes.dim() != 2:
        raise ValueError("codes must be uint8 with shape [N, K]")
    if codes.shape[1] % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {codes.shape[1]}")

    n, k = codes.shape
    n_groups8 = (n + 7) // 8
    padded = torch.zeros((n_groups8 * 8, k), dtype=torch.uint8, device=codes.device)
    padded[:n].copy_(codes)

    packed = torch.empty((k // 32, n_groups8, 128), dtype=torch.uint8, device=codes.device)
    block_codes = padded.view(n_groups8, 8, k // 32, 32).permute(2, 0, 1, 3)
    lanes = packed.view(k // 32, n_groups8, 32, 4)
    byte_indices = torch.tensor(
        [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        dtype=torch.long,
        device=codes.device,
    )

    for lane in range(32):
        group_id = lane >> 2
        thread_id_in_group = lane & 3
        lane_byte_indices = byte_indices[thread_id_in_group * 4 : thread_id_in_group * 4 + 4]
        four_bytes = (
            block_codes[:, :, group_id, 2 * lane_byte_indices] & 0x0F
        ) | ((block_codes[:, :, group_id, 2 * lane_byte_indices + 1] & 0x0F) << 4)
        lanes[:, :, lane, :].copy_(four_bytes)
    return packed.contiguous()


def _pack_shift2(shifts: torch.Tensor) -> torch.Tensor:
    if shifts.dtype != torch.uint8 or shifts.dim() != 2:
        raise ValueError("shifts must be uint8 with shape [N, K_blocks]")
    n, k_blocks = shifts.shape
    stride = (k_blocks + 3) // 4
    n_groups8 = (n + 7) // 8
    out = torch.zeros((stride, n_groups8, 8), dtype=torch.uint8, device=shifts.device)
    padded = torch.zeros((n_groups8 * 8, k_blocks), dtype=torch.uint8, device=shifts.device)
    padded[:n].copy_(shifts & 0x03)
    grouped = padded.view(n_groups8, 8, k_blocks)
    for block in range(k_blocks):
        out[block // 4] |= grouped[:, :, block] << ((block % 4) * 2)
    return out.contiguous()


def _signed_x2_from_codes(codes: torch.Tensor) -> torch.Tensor:
    table = _E2M1_SIGNED_X2.to(device=codes.device)
    return table[codes.long()]


def _nearest_e2m1_codes_x2(values: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    mags = _E2M1_MAG_X2.to(device=values.device)
    abs_values = values.abs().unsqueeze(-1)
    mag_idx = (abs_values - mags).abs().argmin(dim=-1).to(torch.uint8)
    return torch.where(signs < 0, mag_idx | 0x08, mag_idx).to(torch.uint8)


def _summarize_remap(
    original: torch.Tensor,
    remapped: torch.Tensor,
    overflow: torch.Tensor,
    row_span: torch.Tensor,
) -> dict[str, float | int]:
    err = remapped - original
    nonzero = original != 0
    denom = original.abs().clamp_min(1e-12)
    return {
        "num_weights": int(original.numel()),
        "overflow_count": int(overflow.sum().item()),
        "overflow_rate": float(overflow.float().mean().item()),
        "exact_rate_all": float((remapped == original).float().mean().item()),
        "exact_rate_nonzero": float((remapped[nonzero] == original[nonzero]).float().mean().item())
        if bool(nonzero.any())
        else 1.0,
        "mae": float(err.abs().mean().item()),
        "rmse": float(torch.sqrt((err * err).mean()).item()),
        "max_abs_err": float(err.abs().max().item()),
        "mean_rel_err_nonzero": float((err[nonzero].abs() / denom[nonzero]).mean().item())
        if bool(nonzero.any())
        else 0.0,
        "max_rel_err_nonzero": float((err[nonzero].abs() / denom[nonzero]).max().item())
        if bool(nonzero.any())
        else 0.0,
        "row_span_max": int(row_span.max().item()),
    }


@triton.jit
def _remap_mxfp4_channel_kernel(
    scale,
    max_exp_out,
    channel_scale,
    total_rows: tl.constexpr,
    k_blocks: tl.constexpr,
    headroom_bits: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < k_blocks
    scale_u8 = tl.load(scale + row * k_blocks + offs, mask=mask, other=0).to(tl.int32)
    exp = scale_u8 - 127
    exp = tl.where(mask, exp, -32768)
    max_exp = tl.max(exp, axis=0)
    tl.store(max_exp_out + row, max_exp.to(tl.int16), mask=row < total_rows)
    ch_scale = tl.exp2((max_exp - headroom_bits).to(tl.float32)) * 0.5
    tl.store(channel_scale + row, ch_scale, mask=row < total_rows)


@triton.jit
def _pack_shift2_kernel(
    scale,
    max_exp,
    shift2_out,
    total_rows: tl.constexpr,
    n: tl.constexpr,
    k_blocks: tl.constexpr,
    shift_stride: tl.constexpr,
    n_groups8: tl.constexpr,
    headroom_bits: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    e = tl.program_id(0)
    shift_byte = tl.program_id(1)
    n_block = tl.program_id(2)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n
    row = e * n + offs_n
    row_max = tl.load(max_exp + row, mask=mask_n, other=0).to(tl.int32)
    packed = tl.full((BLOCK_N,), 0, tl.uint32)
    for i in tl.static_range(0, 4):
        kb = shift_byte * 4 + i
        valid = mask_n & (kb < k_blocks)
        scale_u8 = tl.load(
            scale + row * k_blocks + kb,
            mask=valid,
            other=0,
        ).to(tl.int32)
        exp = scale_u8 - 127
        delta = row_max - exp
        shift = headroom_bits - delta
        shift = tl.minimum(tl.maximum(shift, 0), 3).to(tl.uint32)
        packed = packed | (tl.where(valid, shift, 0) << (i * 2))
    ng = offs_n // 8
    row8 = offs_n - ng * 8
    out_offsets = ((e * shift_stride + shift_byte) * n_groups8 + ng) * 8 + row8
    tl.store(shift2_out + out_offsets, packed.to(tl.uint8), mask=mask_n)


@triton.jit
def _nearest_e2m1_code_x2(abs_value, sign_bit):
    mag = tl.full(abs_value.shape, 7, tl.uint32)
    mag = tl.where(abs_value <= 10.0, 6, mag)
    mag = tl.where(abs_value <= 7.0, 5, mag)
    mag = tl.where(abs_value <= 5.0, 4, mag)
    mag = tl.where(abs_value <= 3.5, 3, mag)
    mag = tl.where(abs_value <= 2.5, 2, mag)
    mag = tl.where(abs_value <= 1.5, 1, mag)
    mag = tl.where(abs_value <= 0.5, 0, mag)
    return mag | sign_bit


@triton.jit
def _remap_code_nibble(nibble, exp, max_exp, headroom_bits: tl.constexpr):
    mag_code = (nibble & 0x07).to(tl.uint32)
    sign_bit = (nibble & 0x08).to(tl.uint32)
    q_abs = tl.full(mag_code.shape, 12.0, tl.float32)
    q_abs = tl.where(mag_code == 0, 0.0, q_abs)
    q_abs = tl.where(mag_code == 1, 1.0, q_abs)
    q_abs = tl.where(mag_code == 2, 2.0, q_abs)
    q_abs = tl.where(mag_code == 3, 3.0, q_abs)
    q_abs = tl.where(mag_code == 4, 4.0, q_abs)
    q_abs = tl.where(mag_code == 5, 6.0, q_abs)
    q_abs = tl.where(mag_code == 6, 8.0, q_abs)
    delta = max_exp - exp
    shift = headroom_bits - delta
    shift = tl.minimum(tl.maximum(shift, 0), 3)
    target = q_abs * tl.exp2((headroom_bits - delta).to(tl.float32))
    # Matches torch.round for non-negative half values closely enough here:
    # target=0.5 maps to 0 because the nearest E2M1 tie also chooses 0.
    target_i = tl.floor(target + 0.5)
    code_value = target_i * tl.exp2((-shift).to(tl.float32))
    out = _nearest_e2m1_code_x2(code_value, sign_bit)
    return tl.where(mag_code == 0, 0, out).to(tl.uint32)


@triton.jit
def _repack_mxfp4_weight_kernel(
    weight,
    scale,
    max_exp,
    packed_out,
    n: tl.constexpr,
    k_blocks: tl.constexpr,
    n_groups8: tl.constexpr,
    headroom_bits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0)
    kb = tl.program_id(1)
    ng = tl.program_id(2)
    offs = tl.arange(0, BLOCK)
    lane = offs // 4
    lane_byte = offs - lane * 4
    row8 = lane // 4
    tid = lane - row8 * 4
    byte_idx = tid * 2 + (lane_byte & 1) + (lane_byte // 2) * 8
    out_n = ng * 8 + row8
    valid = out_n < n

    row = e * n + out_n
    packed_byte = tl.load(
        weight + row * (k_blocks * 16) + kb * 16 + byte_idx,
        mask=valid,
        other=0,
    ).to(tl.uint32)
    scale_u8 = tl.load(scale + row * k_blocks + kb, mask=valid, other=0).to(tl.int32)
    exp = scale_u8 - 127
    row_max = tl.load(max_exp + row, mask=valid, other=0).to(tl.int32)

    lo = packed_byte & 0x0F
    hi = (packed_byte >> 4) & 0x0F
    remap_lo = _remap_code_nibble(lo, exp, row_max, headroom_bits)
    remap_hi = _remap_code_nibble(hi, exp, row_max, headroom_bits)
    out_byte = remap_lo | (remap_hi << 4)

    out_offsets = ((e * k_blocks + kb) * n_groups8 + ng) * 128 + offs
    tl.store(packed_out + out_offsets, out_byte.to(tl.uint8), mask=valid)


def _remap_mxfp4_weight_for_int8_triton(
    weight: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    headroom_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    e, n, packed_k = weight.shape
    k_blocks = packed_k * 2 // 32
    n_groups8 = triton.cdiv(n, 8)
    shift_stride = triton.cdiv(k_blocks, 4)
    max_exp = torch.empty((e, n), device=weight.device, dtype=torch.int16)
    channel_scale = torch.empty((e, n), device=weight.device, dtype=torch.float32)
    b_mxfp4 = torch.zeros(
        (e, k_blocks, n_groups8, 128), device=weight.device, dtype=torch.uint8
    )
    b_shift2 = torch.zeros(
        (e, shift_stride, n_groups8, 8), device=weight.device, dtype=torch.uint8
    )
    block_k = triton.next_power_of_2(k_blocks)
    _remap_mxfp4_channel_kernel[(e * n,)](
        scale_u8,
        max_exp,
        channel_scale,
        e * n,
        k_blocks,
        headroom_bits,
        BLOCK_K=block_k,
    )
    _pack_shift2_kernel[(e, shift_stride, triton.cdiv(n, 128))](
        scale_u8,
        max_exp,
        b_shift2,
        e * n,
        n,
        k_blocks,
        shift_stride,
        n_groups8,
        headroom_bits,
        BLOCK_N=128,
    )
    _repack_mxfp4_weight_kernel[(e, k_blocks, n_groups8)](
        weight,
        scale_u8,
        max_exp,
        b_mxfp4,
        n,
        k_blocks,
        n_groups8,
        headroom_bits,
        BLOCK=128,
    )
    return b_mxfp4, b_shift2, channel_scale, max_exp


def _compute_remap_stats(
    weight: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    headroom_bits: int,
) -> dict[str, float | int]:
    e, n, packed_k = weight.shape
    k = packed_k * 2
    k_blocks = k // 32
    codes = _unpack_codes(weight).view(e, n, k_blocks, 32)
    q_x2 = _signed_x2_from_codes(codes)
    exp = scale_u8.to(torch.int16) - 127
    max_exp = exp.max(dim=2, keepdim=True).values
    delta = (max_exp - exp).clamp_min(0).to(torch.int16)
    shift = (headroom_bits - delta).clamp(0, 3).to(torch.uint8)
    channel_exp = max_exp.squeeze(2).to(torch.float32) - float(headroom_bits)
    channel_scale = torch.exp2(channel_exp).to(torch.float32) * 0.5
    target_i8 = torch.round(q_x2 * torch.exp2((headroom_bits - delta).float()).unsqueeze(-1))
    overflow = (target_i8 < -128) | (target_i8 > 127)
    target_i8 = target_i8.clamp(-128, 127)
    denom = torch.exp2(shift.float()).unsqueeze(-1)
    target_code_value = target_i8 / denom
    signs = torch.sign(q_x2)
    remapped_codes = _nearest_e2m1_codes_x2(target_code_value, signs)
    remapped_codes = torch.where(q_x2 == 0, torch.zeros_like(remapped_codes), remapped_codes)
    row_span = (exp.max(dim=2).values - exp.min(dim=2).values).to(torch.int16)
    remapped_int = _signed_x2_from_codes(remapped_codes) * denom
    original = q_x2 * torch.exp2(exp.float()).unsqueeze(-1) * 0.5
    remapped = remapped_int * channel_scale[:, :, None, None]
    return _summarize_remap(original, remapped, overflow, row_span)


def remap_mxfp4_weight_for_int8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    headroom_bits: int = 3,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed MXFP4 + UE8M0 scale into mxfp4_int8 persistent format.

    The conversion is intentionally approximate: per-32 MXFP4 scales are folded
    into FP4 code remapping plus a 2-bit integer shift, leaving one FP32 scale
    per output channel for the GEMM epilogue.
    """
    if headroom_bits < 0 or headroom_bits > 7:
        raise ValueError(f"invalid headroom_bits={headroom_bits}")

    w_u8 = weight.data if hasattr(weight, "data") else weight
    w_u8 = w_u8.view(torch.uint8).contiguous()
    scale_u8 = _as_uint8_storage(scale)
    if w_u8.dim() != 3 or scale_u8.dim() != 3:
        raise ValueError(
            f"expected expert weight/scale shapes [E, N, K/2] and [E, N, K/32], "
            f"got {tuple(w_u8.shape)} and {tuple(scale_u8.shape)}"
        )
    if w_u8.shape[0] != scale_u8.shape[0] or w_u8.shape[1] != scale_u8.shape[1]:
        raise ValueError(f"weight/scale shape mismatch: {tuple(w_u8.shape)} vs {tuple(scale_u8.shape)}")
    if w_u8.shape[2] * 2 != scale_u8.shape[2] * 32:
        raise ValueError(f"K mismatch: {tuple(w_u8.shape)} vs {tuple(scale_u8.shape)}")

    if use_triton and w_u8.is_cuda:
        b_mxfp4, b_shift2, channel_scale, _ = _remap_mxfp4_weight_for_int8_triton(
            w_u8,
            scale_u8,
            headroom_bits=headroom_bits,
        )
    else:
        e, n, packed_k = w_u8.shape
        k = packed_k * 2
        k_blocks = k // 32
        codes = _unpack_codes(w_u8).view(e, n, k_blocks, 32)
        q_x2 = _signed_x2_from_codes(codes)
        exp = scale_u8.to(torch.int16) - 127
        max_exp = exp.max(dim=2, keepdim=True).values
        delta = (max_exp - exp).clamp_min(0).to(torch.int16)
        shift = (headroom_bits - delta).clamp(0, 3).to(torch.uint8)
        channel_exp = max_exp.squeeze(2).to(torch.float32) - float(headroom_bits)
        channel_scale = torch.exp2(channel_exp).to(torch.float32) * 0.5
        target_i8 = torch.round(q_x2 * torch.exp2((headroom_bits - delta).float()).unsqueeze(-1))
        denom = torch.exp2(shift.float()).unsqueeze(-1)
        target_code_value = target_i8.clamp(-128, 127) / denom
        signs = torch.sign(q_x2)
        remapped_codes = _nearest_e2m1_codes_x2(target_code_value, signs)
        remapped_codes = torch.where(q_x2 == 0, torch.zeros_like(remapped_codes), remapped_codes)
        packed_experts = []
        packed_shifts = []
        for expert in range(e):
            packed_experts.append(_pack_codes_coalesced(remapped_codes[expert].reshape(n, k)))
            packed_shifts.append(_pack_shift2(shift[expert]))
        b_mxfp4 = torch.stack(packed_experts, dim=0).contiguous()
        b_shift2 = torch.stack(packed_shifts, dim=0).contiguous()
    return b_mxfp4, b_shift2, channel_scale.contiguous()


def prepare_mxfp4_int8_dense_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    headroom_bits: int = 3,
) -> Mxfp4Int8DenseWeight:
    b_mxfp4, b_shift2, channel_scale = remap_mxfp4_weight_for_int8(
        weight.unsqueeze(0),
        scale.unsqueeze(0),
        headroom_bits=headroom_bits,
    )
    return Mxfp4Int8DenseWeight(
        weight_mxfp4=b_mxfp4[0].contiguous(),
        weight_shift2=b_shift2[0].contiguous(),
        weight_channel_scale=channel_scale[0].contiguous(),
        headroom_bits=headroom_bits,
    )


def mxfp4_int8_dense_forward(
    hidden_states: torch.Tensor,
    weight: Mxfp4Int8DenseWeight,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    global _MXFP4_INT8_DENSE_JIT_AVAILABLE
    hidden_states = hidden_states.contiguous()
    a_q, a_scale = quantize_per_token_int8(hidden_states, 0.0)
    out = torch.empty(
        (hidden_states.shape[0], weight.weight_channel_scale.shape[0]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    if (
        bias is None
        and _mxfp4_int8_moe_jit_enabled()
        and _MXFP4_INT8_DENSE_JIT_AVAILABLE is not False
    ):
        try:
            from dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_dense import (
                mxfp4_int8_dense_gemm,
            )

            mxfp4_int8_dense_gemm(
                a_q,
                a_scale,
                weight.weight_mxfp4,
                weight.weight_shift2,
                weight.weight_channel_scale,
                out,
                torch.empty(
                    (
                        4,
                        hidden_states.shape[0],
                        ((weight.weight_channel_scale.shape[0] + 31) // 32) * 32,
                    ),
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                if hidden_states.shape[0] <= 128
                else torch.empty((0,), device=hidden_states.device, dtype=torch.int32),
            )
            _MXFP4_INT8_DENSE_JIT_AVAILABLE = True
            return out
        except Exception:
            _MXFP4_INT8_DENSE_JIT_AVAILABLE = False
            logger.exception("MXFP4-int8 dense JIT launch failed")
            raise

    if bias is not None:
        raise NotImplementedError("MXFP4-int8 dense JIT path does not support bias")
    raise RuntimeError("MXFP4-int8 dense JIT is disabled or unavailable")


def prepare_mxfp4_int8_moe(
    layer: torch.nn.Module,
    *,
    headroom_bits: int = 3,
    topk: int | None = None,
) -> None:
    repack_backend = os.environ.get("SGLANG_DSV4_MXFP4_INT8_REPACK_BACKEND", "triton").strip().lower()
    if repack_backend not in {"triton", "torch"}:
        raise ValueError("SGLANG_DSV4_MXFP4_INT8_REPACK_BACKEND must be 'triton' or 'torch'")
    use_triton = repack_backend == "triton"
    w13_mxfp4, w13_shift2, w13_channel_scale = remap_mxfp4_weight_for_int8(
        layer.w13_weight.data,
        layer.w13_weight_scale_inv.data,
        headroom_bits=headroom_bits,
        use_triton=use_triton,
    )
    w2_mxfp4, w2_shift2, w2_channel_scale = remap_mxfp4_weight_for_int8(
        layer.w2_weight.data,
        layer.w2_weight_scale_inv.data,
        headroom_bits=headroom_bits,
        use_triton=use_triton,
    )
    layer._dsv4_mxfp4_int8_weights = Mxfp4Int8MoeWeights(
        w13_mxfp4=w13_mxfp4,
        w13_shift2=w13_shift2,
        w13_channel_scale=w13_channel_scale,
        w2_mxfp4=w2_mxfp4,
        w2_shift2=w2_shift2,
        w2_channel_scale=w2_channel_scale,
        headroom_bits=headroom_bits,
    )
    if topk is None:
        topk = int(getattr(layer, "top_k", 0) or getattr(layer, "topk", 0) or 8)
    hidden_size = int(w2_channel_scale.shape[1])
    intermediate_size = int(w2_mxfp4.shape[1] * 32)
    _try_prewarm_mxfp4_int8_moe_jit(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        topk=topk,
    )
    empty_u8 = torch.empty(0, device=w13_mxfp4.device, dtype=torch.uint8)
    empty_i8 = empty_u8.view(torch.int8)
    layer.w13_weight = Parameter(empty_i8, requires_grad=False)
    layer.w2_weight = Parameter(empty_i8, requires_grad=False)
    layer.w13_weight_scale_inv = Parameter(empty_u8, requires_grad=False)
    layer.w2_weight_scale_inv = Parameter(empty_u8, requires_grad=False)


def _select_block_size_m(m: int, topk: int, experts: int) -> int:
    return _select_moe_tile_shape(m, topk, experts)[0]


def select_mxfp4_int8_moe_tile_shape(m: int, topk: int, experts: int) -> tuple[int, int]:
    return _select_moe_tile_shape(m, topk, experts)


def max_mxfp4_int8_moe_block_m(m: int, topk: int, experts: int) -> int:
    avg_tokens_per_expert = max(1, (m * topk + max(experts, 1) - 1) // max(experts, 1))
    return min(_MAX_BLOCK_M, max(16, 1 << (avg_tokens_per_expert - 1).bit_length()))


def _parse_tile_shape(text: str) -> tuple[int, int]:
    parts = text.strip().lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"invalid tile shape {text!r}, expected MxN")
    block_m, block_n = int(parts[0]), int(parts[1])
    if block_m not in _VALID_BLOCK_M:
        raise ValueError("block_m must be 16, 32, 64, or 128")
    if block_n not in _VALID_BLOCK_N:
        raise ValueError("block_n must be 32, 64, or 128")
    return block_m, block_n


def _select_moe_tile_shape(m: int, topk: int, experts: int) -> tuple[int, int]:
    max_block_m = max_mxfp4_int8_moe_block_m(m, topk, experts)
    block_m_env = os.environ.get("SGLANG_DSV4_MXFP4_INT8_BLOCK_M", "").strip()
    block_n_env = os.environ.get("SGLANG_DSV4_MXFP4_INT8_BLOCK_N", "").strip()
    if block_m_env or block_n_env:
        block_m = int(block_m_env) if block_m_env else 16
        block_n = int(block_n_env) if block_n_env else 128
        if block_m not in _VALID_BLOCK_M:
            raise ValueError("SGLANG_DSV4_MXFP4_INT8_BLOCK_M must be 16, 32, 64, or 128")
        if block_n not in _VALID_BLOCK_N:
            raise ValueError("SGLANG_DSV4_MXFP4_INT8_BLOCK_N must be 32, 64, or 128")
        if block_m > max_block_m:
            raise ValueError(
                f"SGLANG_DSV4_MXFP4_INT8_BLOCK_M={block_m} exceeds cap "
                f"{max_block_m} for m={m}, topk={topk}, experts={experts}"
            )
        return block_m, block_n
    for max_batch, block_m, block_n in _DSV4_A100_MXFP4_INT8_TILE_TABLE:
        if m <= max_batch and block_m <= max_block_m:
            return block_m, block_n
    for _, block_m, block_n in reversed(_DSV4_A100_MXFP4_INT8_TILE_TABLE):
        if block_m <= max_block_m:
            return block_m, block_n
    return 16, 128


def _moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import triton
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def _apply_swiglu(x: torch.Tensor, clamp_limit: float | None) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    return torch.nn.functional.silu(gate) * up


def mxfp4_int8_moe_forward(
    hidden_states: torch.Tensor,
    weights: Mxfp4Int8MoeWeights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    routed_scaling_factor: float | None = None,
    clamp_limit: float | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "mxfp4_int8 MoE only supports router weights in the W2 reduce path"
        )
    if hidden_states.dtype != torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)
    hidden_states = hidden_states.contiguous()
    topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
    topk_weights_f32 = topk_weights.to(torch.float32).contiguous()

    m = hidden_states.shape[0]
    topk = topk_ids_i32.shape[1]
    experts = weights.w13_mxfp4.shape[0]
    block_size_m, block_size_n = _select_moe_tile_shape(m, topk, experts)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids_i32, block_size_m, experts
    )

    a13_q, a13_scale = quantize_per_token_int8(hidden_states, 0.0)
    gate_up = torch.empty(
        (m * topk, intermediate_size * 2),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    _mxfp4_int8_moe_grouped_gemm_nt(
        a13_q,
        a13_scale,
        weights.w13_mxfp4,
        weights.w13_shift2,
        weights.w13_channel_scale,
        gate_up,
        topk_weights_f32,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk,
        block_size_m,
        False,
        m * topk,
        hidden_size,
        intermediate_size,
        block_size_n,
    )
    intermediate = _apply_swiglu(gate_up, clamp_limit).contiguous()
    a2_q, a2_scale = quantize_per_token_int8(intermediate, 0.0)
    out = torch.empty((m, hidden_size), device=hidden_states.device, dtype=torch.bfloat16)
    _mxfp4_int8_moe_grouped_gemm_nt(
        a2_q,
        a2_scale,
        weights.w2_mxfp4,
        weights.w2_shift2,
        weights.w2_channel_scale,
        out,
        topk_weights_f32,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk,
        block_size_m,
        True,
        m * topk,
        hidden_size,
        intermediate_size,
        block_size_n,
    )
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        out.mul_(routed_scaling_factor)
    return out
