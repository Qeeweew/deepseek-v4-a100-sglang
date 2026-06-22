#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_moe import (
    mxfp4_int8_moe_gemm,
    prewarm_mxfp4_int8_moe_jit_modules,
)
from dsv4_a100_patch.triton_kernels.mxfp4_int8_moe import (
    quantize_per_token_int8,
    select_mxfp4_int8_moe_tile_shape,
)


@dataclass(frozen=True)
class Dsv4Dims:
    hidden_size: int = 4096
    global_intermediate_size: int = 2048
    tensor_parallel_size: int = 8
    intermediate_size: int | None = None
    num_experts: int = 256
    topk: int = 6
    routed_scaling_factor: float = 1.5
    swiglu_limit: float | None = 10.0

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            object.__setattr__(
                self,
                "intermediate_size",
                self.global_intermediate_size // self.tensor_parallel_size,
            )


@dataclass
class Int4Weights:
    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    scalar_type: object


@dataclass
class Mxfp4Int8Weights:
    w13_mxfp4: torch.Tensor
    w13_shift2: torch.Tensor
    w13_scale: torch.Tensor
    w2_mxfp4: torch.Tensor
    w2_shift2: torch.Tensor
    w2_scale: torch.Tensor


def parse_batches(text: str) -> list[int]:
    if ":" in text:
        lo, hi = [int(x) for x in text.split(":", 1)]
        out = []
        x = lo
        while x <= hi:
            out.append(x)
            x *= 2
        return out
    return [int(x) for x in text.split(",") if x.strip()]


def expert_pattern(batch: int, topk: int, num_experts: int, device: str) -> torch.Tensor:
    slots = torch.arange(batch * topk, device=device, dtype=torch.int64)
    return (slots % num_experts).view(batch, topk).contiguous()


def uniform_weights(batch: int, topk: int, device: str) -> torch.Tensor:
    return torch.full((batch, topk), 1.0 / topk, device=device, dtype=torch.float32)


def int4_block_m(batch: int, topk: int, num_experts: int) -> int:
    for block in (8, 16, 32, 48, 64):
        if batch * topk / num_experts / block < 0.9:
            return block
    return 64


def mxfp4_tile_shape(
    batch: int,
    topk: int,
    num_experts: int,
    block_m_override: int,
    block_n_override: int,
) -> tuple[int, int]:
    if block_m_override <= 0 and block_n_override <= 0:
        return select_mxfp4_int8_moe_tile_shape(batch, topk, num_experts)

    if block_m_override > 0:
        block_m = block_m_override
    elif batch <= 1024:
        block_m = 16
    elif batch <= 2048:
        block_m = 64
    else:
        block_m = 128
    if block_m not in (16, 32, 64, 128):
        raise ValueError("--mxfp4-block-m must be one of 16, 32, 64, 128")

    if block_n_override > 0:
        block_n = block_n_override
    elif batch <= 8:
        block_n = 64
    elif batch <= 128:
        block_n = 128
    elif batch == 256:
        block_n = 32
    elif batch <= 512:
        block_n = 64
    elif batch <= 1024:
        block_n = 128
    elif batch <= 2048:
        block_n = 64
    else:
        block_n = 128
    if block_n not in (32, 64, 128):
        raise ValueError("--mxfp4-block-n must be one of 32, 64, 128")
    return block_m, block_n


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
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


def make_marlin_workspace(
    device: torch.device,
    sorted_token_ids: torch.Tensor,
    block_m: int,
    hidden_size: int,
    intermediate_size: int,
) -> torch.Tensor:
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    max_workspace_size = (max(2 * intermediate_size, hidden_size) // 64) * (
        sorted_token_ids.size(0) // block_m
    )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return marlin_make_workspace(device, max_blocks_per_sm=4)[: min(max_workspace_size, sms * 4)]


def make_int4_weights(dims: Dsv4Dims, device: str) -> Int4Weights:
    from sgl_kernel.scalar_type import scalar_types
    from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    int32_min = -(2**31)
    int32_max = 2**31 - 1
    w13_packed = torch.randint(
        int32_min,
        int32_max,
        (dims.num_experts, dims.hidden_size // 8, 2 * dims.intermediate_size),
        dtype=torch.int32,
        device=device,
    ).contiguous()
    w2_packed = torch.randint(
        int32_min,
        int32_max,
        (dims.num_experts, dims.intermediate_size // 8, dims.hidden_size),
        dtype=torch.int32,
        device=device,
    ).contiguous()
    w13_scale = torch.rand(
        (dims.num_experts, dims.hidden_size // 32, 2 * dims.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    w2_scale = torch.rand(
        (dims.num_experts, dims.intermediate_size // 32, dims.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    empty_perm_h = torch.empty((dims.num_experts, 0), dtype=torch.int32, device=device)
    empty_perm_i = torch.empty((dims.num_experts, 0), dtype=torch.int32, device=device)
    w13_qweight = gptq_marlin_moe_repack(
        w13_packed, empty_perm_h, dims.hidden_size, 2 * dims.intermediate_size, 4
    ).contiguous()
    w2_qweight = gptq_marlin_moe_repack(
        w2_packed, empty_perm_i, dims.intermediate_size, dims.hidden_size, 4
    ).contiguous()
    w13_scale = marlin_moe_permute_scales(
        w13_scale, dims.hidden_size, 2 * dims.intermediate_size, 32
    ).contiguous()
    w2_scale = marlin_moe_permute_scales(
        w2_scale, dims.intermediate_size, dims.hidden_size, 32
    ).contiguous()
    torch.cuda.empty_cache()
    return Int4Weights(
        w13_qweight=w13_qweight,
        w2_qweight=w2_qweight,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        scalar_type=scalar_types.uint4b8,
    )


def make_mxfp4_int8_weights(dims: Dsv4Dims, device: str) -> Mxfp4Int8Weights:
    from dsv4_a100_patch.triton_kernels.mxfp4_int8_moe import (
        _pack_codes_coalesced,
        _pack_shift2,
    )

    def make_one(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed_w = []
        packed_s = []
        scales = []
        for expert in range(dims.num_experts):
            gen = torch.Generator(device=device)
            gen.manual_seed(12345 + expert * 1000003 + n * 17 + k)
            codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=device, generator=gen)
            shifts = torch.randint(
                0, 4, (n, k // 32), dtype=torch.uint8, device=device, generator=gen
            )
            packed_w.append(_pack_codes_coalesced(codes))
            packed_s.append(_pack_shift2(shifts))
            scales.append(
                torch.rand((n,), dtype=torch.float32, device=device, generator=gen) * 0.02
                + 0.001
            )
        return (
            torch.stack(packed_w, dim=0).contiguous(),
            torch.stack(packed_s, dim=0).contiguous(),
            torch.stack(scales, dim=0).contiguous(),
        )

    w13_mxfp4, w13_shift2, w13_scale = make_one(2 * dims.intermediate_size, dims.hidden_size)
    w2_mxfp4, w2_shift2, w2_scale = make_one(dims.hidden_size, dims.intermediate_size)
    torch.cuda.empty_cache()
    return Mxfp4Int8Weights(
        w13_mxfp4=w13_mxfp4,
        w13_shift2=w13_shift2,
        w13_scale=w13_scale,
        w2_mxfp4=w2_mxfp4,
        w2_shift2=w2_shift2,
        w2_scale=w2_scale,
    )


def bench_cuda_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
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


def tflops_for_gemm(ms: float, m: int, n: int, k: int) -> float:
    if ms <= 0:
        return 0.0
    return (2.0 * m * n * k) / (ms * 1.0e9)


def swiglu_limit(x: torch.Tensor, limit: float | None) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if limit is not None:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return F.silu(gate) * up


def run_int4(
    weights: Int4Weights,
    batch: int,
    dims: Dsv4Dims,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    from sgl_kernel import moe_sum_reduce
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm

    device = torch.device(args.device)
    hidden = torch.randn((batch, dims.hidden_size), device=device, dtype=torch.bfloat16)
    topk_ids = expert_pattern(batch, dims.topk, dims.num_experts, args.device).to(torch.int32)
    topk_weights = uniform_weights(batch, dims.topk, args.device)
    block_m = int4_block_m(batch, dims.topk, dims.num_experts)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, dims.num_experts
    )
    workspace = make_marlin_workspace(
        device, sorted_token_ids, block_m, dims.hidden_size, dims.intermediate_size
    )
    use_atomic_add = torch.cuda.get_device_capability(device)[0] >= 9
    c13 = torch.empty(
        (batch * dims.topk, 2 * dims.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    act = torch.empty(
        (batch * dims.topk, dims.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )
    c2_slots = torch.empty(
        (batch * dims.topk, dims.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    c2 = torch.empty((batch, dims.hidden_size), dtype=torch.bfloat16, device=device)

    def gemm13() -> torch.Tensor:
        return moe_wna16_marlin_gemm(
            hidden,
            c13,
            weights.w13_qweight,
            None,
            weights.w13_scale,
            None,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_m,
            top_k=dims.topk,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=weights.scalar_type,
            size_m=batch,
            size_n=2 * dims.intermediate_size,
            size_k=dims.hidden_size,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def activation() -> torch.Tensor:
        act.copy_(swiglu_limit(c13, dims.swiglu_limit))
        return act

    def gemm2() -> torch.Tensor:
        return moe_wna16_marlin_gemm(
            act,
            c2_slots,
            weights.w2_qweight,
            None,
            weights.w2_scale,
            None,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type=weights.scalar_type,
            size_m=batch * dims.topk,
            size_n=dims.hidden_size,
            size_k=dims.intermediate_size,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def reduce2() -> torch.Tensor:
        moe_sum_reduce(
            c2_slots.view(batch, dims.topk, dims.hidden_size),
            c2,
            dims.routed_scaling_factor,
        )
        return c2

    def full() -> torch.Tensor:
        gemm13()
        activation()
        gemm2()
        reduce2()
        return c2

    gemm13_ms = bench_cuda_ms(gemm13, args.warmup, args.iters)
    activation_ms = bench_cuda_ms(activation, max(3, args.warmup // 2), args.iters)
    gemm2_ms = bench_cuda_ms(gemm2, args.warmup, args.iters)
    full_ms = bench_cuda_ms(full, args.warmup, args.iters)
    return make_row(
        "int4_marlin",
        batch,
        dims,
        topk_ids,
        block_m,
        0,
        gemm13_ms,
        activation_ms,
        gemm2_ms,
        full_ms,
    )


def run_mxfp4_int8(
    weights: Mxfp4Int8Weights,
    batch: int,
    dims: Dsv4Dims,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    hidden = torch.randn((batch, dims.hidden_size), device=args.device, dtype=torch.bfloat16)
    topk_ids = expert_pattern(batch, dims.topk, dims.num_experts, args.device).to(torch.int32)
    topk_weights = uniform_weights(batch, dims.topk, args.device)
    block_m, block_n = mxfp4_tile_shape(
        batch,
        dims.topk,
        dims.num_experts,
        args.mxfp4_block_m,
        args.mxfp4_block_n,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, dims.num_experts
    )
    c13 = torch.empty(
        (batch * dims.topk, 2 * dims.intermediate_size),
        dtype=torch.bfloat16,
        device=args.device,
    )
    act = torch.empty(
        (batch * dims.topk, dims.intermediate_size),
        dtype=torch.bfloat16,
        device=args.device,
    )
    c2_slots = torch.empty(
        (batch * dims.topk, dims.hidden_size),
        dtype=torch.bfloat16,
        device=args.device,
    )
    c2 = torch.empty((batch, dims.hidden_size), dtype=torch.bfloat16, device=args.device)
    a13_q, a13_scale = quantize_per_token_int8(hidden, 0.0)
    prewarm_mxfp4_int8_moe_jit_modules(
        hidden_size=dims.hidden_size,
        intermediate_size=dims.intermediate_size,
        topk=dims.topk,
        block_ms=(block_m,),
        block_n=block_n,
    )

    def gemm13() -> torch.Tensor:
        mxfp4_int8_moe_gemm(
            a13_q,
            a13_scale,
            weights.w13_mxfp4,
            weights.w13_shift2,
            weights.w13_scale,
            c13,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            hidden_size=dims.hidden_size,
            intermediate_size=dims.intermediate_size,
            topk=dims.topk,
            block_m=block_m,
            block_n=block_n,
            source_rows_are_slots=False,
            num_valid_tokens=batch * dims.topk,
        )
        return c13

    def activation() -> torch.Tensor:
        act.copy_(swiglu_limit(c13, dims.swiglu_limit))
        return act

    a2_q = torch.empty(
        (batch * dims.topk, dims.intermediate_size), dtype=torch.int8, device=args.device
    )
    a2_scale = torch.empty((batch * dims.topk,), dtype=torch.float32, device=args.device)

    def quant2() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal a2_q, a2_scale
        a2_q, a2_scale = quantize_per_token_int8(act, 0.0)
        return a2_q, a2_scale

    def gemm2() -> torch.Tensor:
        mxfp4_int8_moe_gemm(
            a2_q,
            a2_scale,
            weights.w2_mxfp4,
            weights.w2_shift2,
            weights.w2_scale,
            c2,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            hidden_size=dims.hidden_size,
            intermediate_size=dims.intermediate_size,
            topk=dims.topk,
            block_m=block_m,
            block_n=block_n,
            source_rows_are_slots=True,
            num_valid_tokens=batch * dims.topk,
            routed_out=c2_slots,
        )
        return c2

    def full() -> torch.Tensor:
        gemm13()
        activation()
        quant2()
        gemm2()
        if dims.routed_scaling_factor != 1.0:
            c2.mul_(dims.routed_scaling_factor)
        return c2

    gemm13_ms = bench_cuda_ms(gemm13, args.warmup, args.iters)
    activation_ms = bench_cuda_ms(activation, max(3, args.warmup // 2), args.iters)
    quant2()
    gemm2_ms = bench_cuda_ms(gemm2, args.warmup, args.iters)
    full_ms = bench_cuda_ms(full, args.warmup, args.iters)
    return make_row(
        "mxfp4_int8_jit",
        batch,
        dims,
        topk_ids,
        block_m,
        block_n,
        gemm13_ms,
        activation_ms,
        gemm2_ms,
        full_ms,
    )


def make_row(
    backend: str,
    batch: int,
    dims: Dsv4Dims,
    topk_ids: torch.Tensor,
    block_m: int,
    block_n: int,
    gemm13_ms: float,
    activation_ms: float,
    gemm2_ms: float,
    full_ms: float,
) -> dict[str, float | int | str]:
    return {
        "backend": backend,
        "batch": batch,
        "topk": dims.topk,
        "unique_experts": int(torch.unique(topk_ids).numel()),
        "block_m": block_m,
        "block_n": block_n,
        "gemm13_ms": gemm13_ms,
        "activation_ms": activation_ms,
        "gemm2_ms": gemm2_ms,
        "full_ms": full_ms,
        "gemm13_tflops": tflops_for_gemm(
            gemm13_ms, batch * dims.topk, 2 * dims.intermediate_size, dims.hidden_size
        ),
        "gemm2_tflops": tflops_for_gemm(
            gemm2_ms, batch * dims.topk, dims.hidden_size, dims.intermediate_size
        ),
        "full_tflops": tflops_for_gemm(
            full_ms, batch * dims.topk, 3 * dims.intermediate_size, dims.hidden_size
        ),
    }


def print_row(row: dict[str, float | int | str]) -> None:
    print(
        f"{row['backend']:>15} bs={row['batch']:5d} "
        f"block={row['block_m']}x{row['block_n']} "
        f"w13={row['gemm13_ms']:.4f}ms {row['gemm13_tflops']:.1f}TF "
        f"w2={row['gemm2_ms']:.4f}ms {row['gemm2_tflops']:.1f}TF "
        f"full={row['full_ms']:.4f}ms {row['full_tflops']:.1f}TF"
    )


def print_comparison(rows: list[dict[str, float | int | str]]) -> None:
    by_batch: dict[int, dict[str, dict[str, float | int | str]]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch"]), {})[str(row["backend"])] = row
    print("\ncomparison mxfp4_int8_jit / int4_marlin:")
    for batch in sorted(by_batch):
        pair = by_batch[batch]
        if "int4_marlin" not in pair or "mxfp4_int8_jit" not in pair:
            continue
        int4 = pair["int4_marlin"]
        mxfp4 = pair["mxfp4_int8_jit"]
        w2_speedup = float(int4["gemm2_ms"]) / float(mxfp4["gemm2_ms"])
        full_speedup = float(int4["full_ms"]) / float(mxfp4["full_ms"])
        print(
            f"bs={batch:5d} "
            f"w2_speedup={w2_speedup:.2f}x "
            f"full_speedup={full_speedup:.2f}x "
            f"int4_w2={float(int4['gemm2_ms']):.4f}ms "
            f"mxfp4_w2={float(mxfp4['gemm2_ms']):.4f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DeepSeek V4 Flash TP8 INT4 Marlin MoE and MXFP4xINT8 JIT MoE."
    )
    parser.add_argument("--batches", default="1:16384")
    parser.add_argument("--backend", choices=("both", "int4", "mxfp4_int8"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--global-intermediate-size", type=int, default=2048)
    parser.add_argument("--mxfp4-block-m", type=int, default=0)
    parser.add_argument("--mxfp4-block-n", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_results_int4_vs_mxfp4")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.global_intermediate_size % args.tp_size != 0:
        raise ValueError("--global-intermediate-size must be divisible by --tp-size")
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    dims = Dsv4Dims(
        hidden_size=args.hidden_size,
        global_intermediate_size=args.global_intermediate_size,
        tensor_parallel_size=args.tp_size,
    )
    batches = parse_batches(args.batches)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "dims "
        f"H={dims.hidden_size} I_global={dims.global_intermediate_size} "
        f"TP={dims.tensor_parallel_size} I_local={dims.intermediate_size} "
        f"E={dims.num_experts} TOPK={dims.topk} device={args.device}"
    )

    int4_weights = None
    mxfp4_weights = None
    if args.backend in ("both", "int4"):
        print("building synthetic INT4 packed weights and Marlin repacking...")
        int4_weights = make_int4_weights(dims, args.device)
        torch.cuda.synchronize()
    if args.backend in ("both", "mxfp4_int8"):
        print("building synthetic MXFP4xINT8 packed weights...")
        mxfp4_weights = make_mxfp4_int8_weights(dims, args.device)
        torch.cuda.synchronize()

    rows: list[dict[str, float | int | str]] = []
    for batch in batches:
        if int4_weights is not None:
            row = run_int4(int4_weights, batch, dims, args)
            print_row(row)
            rows.append(row)
        if mxfp4_weights is not None:
            row = run_mxfp4_int8(mxfp4_weights, batch, dims, args)
            print_row(row)
            rows.append(row)

    if len({row["backend"] for row in rows}) > 1:
        print_comparison(rows)

    csv_path = output_dir / "int4_vs_mxfp4_int8_moe_microbench.csv"
    json_path = output_dir / "int4_vs_mxfp4_int8_moe_microbench.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"dims": dims.__dict__, "rows": rows}, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
