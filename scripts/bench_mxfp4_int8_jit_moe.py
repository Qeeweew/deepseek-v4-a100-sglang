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
    max_mxfp4_int8_moe_block_m,
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
class Mxfp4Int8Weights:
    w13_mxfp4: torch.Tensor
    w13_shift2: torch.Tensor
    w13_scale: torch.Tensor
    w2_mxfp4: torch.Tensor
    w2_shift2: torch.Tensor
    w2_scale: torch.Tensor


@dataclass
class Mxfp4OgsWeights:
    ogs_weights: object


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


def cutlass_tile_shape(
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
        raise ValueError("--block-m must be one of 16, 32, 64, 128")

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
        raise ValueError("--block-n must be one of 32, 64, 128")
    return block_m, block_n


def candidate_tile_shapes(batch: int, topk: int, num_experts: int) -> list[tuple[int, int]]:
    max_block_m = max_mxfp4_int8_moe_block_m(batch, topk, num_experts)
    return [
        (block_m, block_n)
        for block_m in (16, 32, 64, 128)
        if block_m <= max_block_m
        for block_n in (32, 64, 128)
    ]


def sglang_moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
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


def make_synthetic_weights(dims: Dsv4Dims, device: str) -> Mxfp4Int8Weights:
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

    w13_mxfp4, w13_shift2, w13_scale = make_one(
        2 * dims.intermediate_size, dims.hidden_size
    )
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


def make_synthetic_ogs_weights(dims: Dsv4Dims, device: str) -> Mxfp4OgsWeights:
    from dsv4_a100_patch.triton_kernels import prepare_mxfp4_moe_ogs

    class Layer(torch.nn.Module):
        pass

    layer = Layer().to(device)
    layer.w13_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (dims.num_experts, 2 * dims.intermediate_size, dims.hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (dims.num_experts, dims.hidden_size, dims.intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.randint(
            120,
            128,
            (dims.num_experts, 2 * dims.intermediate_size, dims.hidden_size // 32),
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.randint(
            120,
            128,
            (dims.num_experts, dims.hidden_size, dims.intermediate_size // 32),
            dtype=torch.uint8,
            device=device,
        ),
        requires_grad=False,
    )
    prepare_mxfp4_moe_ogs(layer)
    torch.cuda.empty_cache()
    return Mxfp4OgsWeights(ogs_weights=layer._dsv4_mxfp4_ogs_weights)


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


def run_jit(
    weights: Mxfp4Int8Weights,
    batch: int,
    dims: Dsv4Dims,
    args: argparse.Namespace,
    tile_shape: tuple[int, int] | None = None,
) -> dict[str, float | int | str]:
    hidden = torch.randn((batch, dims.hidden_size), device=args.device, dtype=torch.bfloat16)
    topk_ids = expert_pattern(batch, dims.topk, dims.num_experts, args.device).to(torch.int32)
    topk_weights = uniform_weights(batch, dims.topk, args.device)
    if tile_shape is None:
        block_m, block_n = cutlass_tile_shape(
            batch, dims.topk, dims.num_experts, args.block_m, args.block_n
        )
    else:
        block_m, block_n = tile_shape
    sorted_token_ids, expert_ids, num_tokens_post_padded = sglang_moe_align_block_size(
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

    a2_q = torch.empty((batch * dims.topk, dims.intermediate_size), dtype=torch.int8, device=args.device)
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
    return {
        "backend": "mxfp4_int8_sglang_jit",
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


def run_ogs(
    weights: Mxfp4OgsWeights,
    batch: int,
    dims: Dsv4Dims,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    from dsv4_a100_patch.triton_kernels.mxfp4_moe_ogs import (
        _dsv4_swiglu_fn,
        _make_routing_data,
    )
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs

    hidden = torch.randn((batch, dims.hidden_size), device=args.device, dtype=torch.bfloat16)
    topk_ids = expert_pattern(batch, dims.topk, dims.num_experts, args.device)
    topk_weights = uniform_weights(batch, dims.topk, args.device)
    routing_data, gather_idx, scatter_idx = _make_routing_data(
        topk_ids, topk_weights, dims.num_experts
    )
    intermediate = torch.empty(
        (1, batch * dims.topk, dims.intermediate_size),
        dtype=torch.bfloat16,
        device=args.device,
    )
    output = torch.empty(
        (1, batch, dims.hidden_size),
        dtype=torch.bfloat16,
        device=args.device,
    )
    act = FusedActivation(
        FnSpecs("dsv4_swiglu", _dsv4_swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.0, dims.swiglu_limit),
    )
    gammas = routing_data.gate_scal

    def gemm13() -> torch.Tensor:
        matmul_ogs(
            hidden,
            weights.ogs_weights.w13,
            None,
            routing_data,
            gather_indx=gather_idx,
            precision_config=weights.ogs_weights.w13_precision_config,
            gammas=None,
            fused_activation=act,
            y=intermediate,
        )
        return intermediate

    def gemm2() -> torch.Tensor:
        matmul_ogs(
            intermediate.view(batch * dims.topk, dims.intermediate_size),
            weights.ogs_weights.w2,
            None,
            routing_data,
            scatter_indx=scatter_idx,
            precision_config=weights.ogs_weights.w2_precision_config,
            gammas=gammas,
            y=output,
        )
        return output

    def full() -> torch.Tensor:
        gemm13()
        gemm2()
        if dims.routed_scaling_factor != 1.0:
            output.mul_(dims.routed_scaling_factor)
        return output

    gemm13_ms = bench_cuda_ms(gemm13, args.warmup, args.iters)
    gemm2_ms = bench_cuda_ms(gemm2, args.warmup, args.iters)
    full_ms = bench_cuda_ms(full, args.warmup, args.iters)
    return {
        "backend": "mxfp4_ogs",
        "batch": batch,
        "topk": dims.topk,
        "unique_experts": int(torch.unique(topk_ids).numel()),
        "block_m": 0,
        "block_n": 0,
        "gemm13_ms": gemm13_ms,
        "activation_ms": 0.0,
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
        f"{row['backend']:>22} bs={row['batch']:5d} "
        f"block={row['block_m']}x{row['block_n']} "
        f"w13={row['gemm13_ms']:.4f}ms {row['gemm13_tflops']:.1f}TF "
        f"w2={row['gemm2_ms']:.4f}ms {row['gemm2_tflops']:.1f}TF "
        f"full={row['full_ms']:.4f}ms {row['full_tflops']:.1f}TF"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MXFP4/INT8 JIT and original Triton/OGS MXFP4 MoE paths."
    )
    parser.add_argument("--batches", default="1,8,128,512,2048,4096,8192,16384")
    parser.add_argument("--backend", choices=("jit", "ogs", "both"), default="jit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--global-intermediate-size", type=int, default=2048)
    parser.add_argument("--block-m", type=int, default=0)
    parser.add_argument("--block-n", type=int, default=0)
    parser.add_argument(
        "--autotune-tiles",
        action="store_true",
        help="sweep all supported block_m/block_n candidates and record the fastest full path per batch",
    )
    parser.add_argument("--output-dir", default="benchmark_results")
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
    jit_weights = None
    ogs_weights = None
    if args.backend in ("jit", "both"):
        print("building synthetic MXFP4/INT8 packed weights...")
        jit_weights = make_synthetic_weights(dims, args.device)
        torch.cuda.synchronize()
    if args.backend in ("ogs", "both"):
        print("building synthetic MXFP4 packed weights and OGS swizzling...")
        ogs_weights = make_synthetic_ogs_weights(dims, args.device)
        torch.cuda.synchronize()

    rows = []
    best_rows = []
    for batch in batches:
        if jit_weights is not None and args.autotune_tiles:
            batch_rows = []
            for tile_shape in candidate_tile_shapes(batch, dims.topk, dims.num_experts):
                row = run_jit(jit_weights, batch, dims, args, tile_shape=tile_shape)
                print_row(row)
                rows.append(row)
                batch_rows.append(row)
            best = min(batch_rows, key=lambda item: float(item["full_ms"]))
            best = dict(best)
            best["backend"] = "mxfp4_int8_sglang_jit_best"
            print("best", end=" ")
            print_row(best)
            best_rows.append(best)
        elif jit_weights is not None:
            row = run_jit(jit_weights, batch, dims, args)
            print_row(row)
            rows.append(row)
            best_rows.append(row)
        if ogs_weights is not None:
            row = run_ogs(ogs_weights, batch, dims, args)
            print_row(row)
            rows.append(row)

    csv_path = output_dir / "mxfp4_moe_microbench.csv"
    json_path = output_dir / "mxfp4_moe_microbench.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"dims": dims.__dict__, "rows": rows}, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    if args.autotune_tiles and best_rows:
        best_csv_path = output_dir / "mxfp4_int8_sglang_jit_moe_best_tiles.csv"
        best_json_path = output_dir / "mxfp4_int8_sglang_jit_moe_best_tiles.json"
        with best_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
        best_json_path.write_text(
            json.dumps({"dims": dims.__dict__, "rows": best_rows}, indent=2)
        )
        print(f"wrote {best_csv_path}")
        print(f"wrote {best_json_path}")


if __name__ == "__main__":
    main()
