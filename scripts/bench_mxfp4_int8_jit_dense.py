#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_dense import (
    mxfp4_int8_dense_gemm,
    prewarm_mxfp4_int8_dense_jit_module,
)
from dsv4_a100_patch.triton_kernels.mxfp4_int8_moe import (
    _pack_codes_coalesced,
    _pack_shift2,
    quantize_per_token_int8,
)


def parse_batches(text: str) -> list[int]:
    if ":" in text:
        lo, hi = [int(x) for x in text.split(":", 1)]
        out = []
        value = lo
        while value <= hi:
            out.append(value)
            value *= 2
        return out
    return [int(x) for x in text.split(",") if x.strip()]


def bench_cuda_ms(fn, warmup: int, iters: int) -> float:
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


def capture_cuda_graph(fn, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def bench_cuda_graph_ms(fn, warmup: int, iters: int) -> float:
    graph = capture_cuda_graph(fn, warmup)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def tflops_for_gemm(ms: float, m: int, n: int, k: int) -> float:
    if ms <= 0:
        return 0.0
    return (2.0 * m * n * k) / (ms * 1.0e9)


def make_weight(n: int, k: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(20260619 + n * 17 + k)
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device=device, generator=gen)
    shifts = torch.randint(0, 4, (n, k // 32), dtype=torch.uint8, device=device, generator=gen)
    channel_scale = torch.rand((n,), dtype=torch.float32, device=device, generator=gen) * 0.02 + 0.001
    return (
        _pack_codes_coalesced(codes),
        _pack_shift2(shifts),
        channel_scale.contiguous(),
    )


def run_one(
    batch: int,
    n: int,
    k: int,
    weight_mxfp4: torch.Tensor,
    weight_shift2: torch.Tensor,
    weight_scale: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    hidden = torch.randn((batch, k), device=args.device, dtype=torch.bfloat16)
    out = torch.empty((batch, n), device=args.device, dtype=torch.bfloat16)
    a_q, a_scale = quantize_per_token_int8(hidden, 0.0)
    prewarm_mxfp4_int8_dense_jit_module(k=k, n=n)
    if batch <= 128:
        partial = torch.empty(
            (4, batch, ((n + 31) // 32) * 32),
            device=args.device,
            dtype=torch.int32,
        )
    else:
        partial = torch.empty((0,), device=args.device, dtype=torch.int32)

    def gemm() -> torch.Tensor:
        mxfp4_int8_dense_gemm(
            a_q,
            a_scale,
            weight_mxfp4,
            weight_shift2,
            weight_scale,
            out,
            partial,
        )
        return out

    quant_out_q = torch.empty_like(a_q)
    quant_out_scale = torch.empty_like(a_scale)

    def quant() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal quant_out_q, quant_out_scale
        quant_out_q, quant_out_scale = quantize_per_token_int8(hidden, 0.0)
        return quant_out_q, quant_out_scale

    def quant_gemm() -> torch.Tensor:
        q, scale = quantize_per_token_int8(hidden, 0.0)
        mxfp4_int8_dense_gemm(
            q,
            scale,
            weight_mxfp4,
            weight_shift2,
            weight_scale,
            out,
            partial,
        )
        return out

    if args.cuda_graph:
        quant_ms = bench_cuda_graph_ms(quant, args.warmup, args.iters)
        gemm_ms = bench_cuda_graph_ms(gemm, args.warmup, args.iters)
        full_ms = bench_cuda_graph_ms(quant_gemm, args.warmup, args.iters)
    else:
        quant_ms = bench_cuda_ms(quant, args.warmup, args.iters)
        gemm_ms = bench_cuda_ms(gemm, args.warmup, args.iters)
        full_ms = bench_cuda_ms(quant_gemm, args.warmup, args.iters)
    return {
        "backend": "mxfp4_int8_sglang_jit_dense",
        "batch": batch,
        "n": n,
        "k": k,
        "cuda_graph": int(args.cuda_graph),
        "quant_ms": quant_ms,
        "gemm_ms": gemm_ms,
        "quant_gemm_ms": full_ms,
        "gemm_tflops": tflops_for_gemm(gemm_ms, batch, n, k),
        "quant_gemm_tflops": tflops_for_gemm(full_ms, batch, n, k),
    }


def print_row(row: dict[str, float | int | str]) -> None:
    print(
        "backend={backend} batch={batch} n={n} k={k} "
        "cuda_graph={cuda_graph} quant_ms={quant_ms:.4f} "
        "gemm_ms={gemm_ms:.4f} quant_gemm_ms={quant_gemm_ms:.4f} "
        "gemm_tflops={gemm_tflops:.1f} quant_gemm_tflops={quant_gemm_tflops:.1f}".format(
            **row
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MXFP4/INT8 dense SGLang-JIT GEMM")
    parser.add_argument("--batches", default="1:16384")
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results_dense_jit"))
    parser.set_defaults(cuda_graph=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.index is not None:
        torch.cuda.set_device(device)
    batches = parse_batches(args.batches)
    weight_mxfp4, weight_shift2, weight_scale = make_weight(args.n, args.k, args.device)
    torch.cuda.empty_cache()

    rows = []
    for batch in batches:
        row = run_one(batch, args.n, args.k, weight_mxfp4, weight_shift2, weight_scale, args)
        rows.append(row)
        print_row(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mxfp4_int8_dense_jit_microbench.csv"
    json_path = args.output_dir / "mxfp4_int8_dense_jit_microbench.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
