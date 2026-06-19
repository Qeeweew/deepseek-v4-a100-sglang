# Benchmarks

This page records the current A100 benchmark evidence for the MXFP4/INT8 fused
MoE path. The numbers are hardware- and environment-specific reference points,
not portable guarantees.

## Setup

Validated runtime:

- GPU: A100 / SM80
- Tensor parallel size: 8 for full-model SGLang serving
- SGLang commit: `1c0019da7579db73223195f25b0eed3882dff24e`
- MoE dimensions used by the microbench: `H=4096`, global `I=2048`,
  `TP=8`, local `I=256`, `E=256`, `topk=6`
- Synthetic weights and deterministic synthetic routing; no real checkpoint
  weights are loaded by the microbench.

The current fused-kernel benchmark directly calls the SGLang-JIT integration
path:

```text
dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_moe.mxfp4_int8_moe_gemm
```

The dense MXFP4/INT8 benchmark directly calls:

```text
dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_dense.mxfp4_int8_dense_gemm
```

The baseline is the original Triton/OGS MXFP4 implementation through this
repository's benchmark script:

```text
scripts/bench_mxfp4_int8_jit_moe.py --backend ogs
```

The table below was re-measured against the current SGLang-JIT path with the
same batch list, warmup, and iteration count as the OGS baseline.

## Dense Microbenchmark

The dense benchmark is used to validate the raw packed-weight GEMM path without
MoE routing overhead:

```bash
SGLANG_ROOT=/workspace/sglang \
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
CUDA_VISIBLE_DEVICES=0 \
python scripts/bench_mxfp4_int8_jit_dense.py \
  --n 8192 \
  --k 8192 \
  --batches 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --warmup 10 \
  --iters 30 \
  --output-dir /workspace/monkeypatch/benchmark_results_dense_jit_cudagraph
```

Large-batch dense GEMM is expected to be much closer to A100 tensor-core peak
than routed MoE because it has one regular GEMM grid and no routing, padding,
activation, quantization, workspace, or top-k reduction overhead.

Current SGLang-JIT dense results for `N=8192`, `K=8192`, `warmup=10`,
`iters=30`. Each measured path is captured once and timed with CUDA graph
replay:

| batch | quant ms | GEMM ms | quant+GEMM ms | GEMM TFLOPS | quant+GEMM TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0043 | 0.0495 | 0.0528 | 2.7 | 2.5 |
| 2 | 0.0043 | 0.0494 | 0.0527 | 5.4 | 5.1 |
| 4 | 0.0042 | 0.0496 | 0.0528 | 10.8 | 10.2 |
| 8 | 0.0042 | 0.0499 | 0.0530 | 21.5 | 20.3 |
| 16 | 0.0046 | 0.0484 | 0.0517 | 44.4 | 41.5 |
| 32 | 0.0043 | 0.0645 | 0.0681 | 66.6 | 63.1 |
| 64 | 0.0042 | 0.0731 | 0.0773 | 117.4 | 111.1 |
| 128 | 0.0046 | 0.0931 | 0.0983 | 184.4 | 174.7 |
| 256 | 0.0053 | 0.2053 | 0.2117 | 167.4 | 162.3 |
| 512 | 0.0071 | 0.3313 | 0.3425 | 207.4 | 200.7 |
| 1024 | 0.0100 | 0.5184 | 0.5344 | 265.1 | 257.2 |
| 2048 | 0.0335 | 0.9875 | 1.0148 | 278.3 | 270.9 |
| 4096 | 0.0617 | 1.5474 | 1.6054 | 355.3 | 342.5 |
| 8192 | 0.1177 | 3.1024 | 3.2091 | 354.4 | 342.6 |
| 16384 | 0.2329 | 6.2631 | 6.4852 | 351.1 | 339.1 |

For `M=1`, Nsight Compute confirms the dense path now uses the same small-M
split-K heuristic as the original design: `16x64x128` on-demand GEMM,
split-K `4`, grid `1x128x4`, followed by a bf16 scale/reduce kernel. The
profiled main GEMM launch was about `48.3 us`, with the reduce launch about
`3.6 us`.

## MoE Microbenchmark TP=8

Command used for the current fused path with `TP=8`, `I_local=256`:

```bash
SGLANG_ROOT=/workspace/sglang \
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
CUDA_VISIBLE_DEVICES=0 \
python scripts/bench_mxfp4_int8_jit_moe.py \
  --batches 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --warmup 10 \
  --iters 30 \
  --output-dir /workspace/monkeypatch/benchmark_results_full_jit
```

To sweep all supported JIT tile shapes and generate a best-tile table, add
`--autotune-tiles`:

```bash
SGLANG_ROOT=/workspace/sglang \
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
CUDA_VISIBLE_DEVICES=0 \
python scripts/bench_mxfp4_int8_jit_moe.py \
  --batches 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --autotune-tiles \
  --warmup 10 \
  --iters 30 \
  --output-dir /workspace/monkeypatch/benchmark_results_autotune
```

The sweep writes every candidate to `mxfp4_moe_microbench.csv` and the fastest
full-path JIT tile for each batch to
`mxfp4_int8_sglang_jit_moe_best_tiles.csv`. This is intended as an offline
calibration step. Runtime serving uses a static table instead of online
autotuning, so model startup and CUDA graph capture do not pay the cost of
compiling and timing every variant.

The autotune search caps `block_m` by the average routed-token count per expert:

```text
block_m <= max(16, next_power_of_2(ceil(batch * topk / num_experts)))
```

This avoids selecting a large M tile for small batches where each expert sees
fewer than one minimum-MMA tile of routed tokens on average.

Command used for the OGS baseline with `TP=8`, `I_local=256`:

```bash
SGLANG_ROOT=/workspace/sglang \
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
CUDA_VISIBLE_DEVICES=0 \
python scripts/bench_mxfp4_int8_jit_moe.py \
  --backend ogs \
  --batches 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --warmup 10 \
  --iters 30 \
  --output-dir /workspace/monkeypatch/benchmark_results_full_ogs
```

Capped-autotune full-path results:

| batch | M cap | best tile | OGS full ms | capped JIT full ms | speedup | capped JIT TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16x128 | 0.8203 | 0.1411 | 5.81x | 0.3 |
| 2 | 16 | 16x64 | 0.8112 | 0.1422 | 5.70x | 0.5 |
| 4 | 16 | 16x64 | 0.8101 | 0.1422 | 5.70x | 1.1 |
| 8 | 16 | 16x64 | 0.8176 | 0.1726 | 4.74x | 1.7 |
| 16 | 16 | 16x128 | 0.8217 | 0.2669 | 3.08x | 2.3 |
| 32 | 16 | 16x128 | 1.0593 | 0.4526 | 2.34x | 2.7 |
| 64 | 16 | 16x128 | 1.3114 | 0.4668 | 2.81x | 5.2 |
| 128 | 16 | 16x128 | 1.2877 | 0.4730 | 2.72x | 10.2 |
| 256 | 16 | 16x128 | 1.3244 | 0.4834 | 2.74x | 20.0 |
| 512 | 16 | 16x128 | 1.3683 | 0.5154 | 2.65x | 37.5 |
| 1024 | 32 | 16x128 | 1.4546 | 0.9110 | 1.60x | 42.4 |
| 2048 | 64 | 64x128 | 1.6661 | 1.1552 | 1.44x | 66.9 |
| 4096 | 128 | 128x128 | 3.2062 | 1.7498 | 1.83x | 88.4 |
| 8192 | 128 | 128x128 | 5.1289 | 3.5042 | 1.46x | 88.2 |
| 16384 | 128 | 128x128 | 9.9812 | 5.6643 | 1.76x | 109.2 |

Across this complete batch sweep, the capped-autotuned SGLang-JIT MXFP4/INT8
path is `1.44x-5.81x` faster than the original Triton/OGS MXFP4 path by full
fused-MoE wall time. For larger decode-like batches `4096-16384`, the speedup
is `1.46x-1.83x`.

## MoE Microbenchmark TP=2

TP=2 uses the same global MoE dimensions but a larger local intermediate size:
`H=4096`, global `I=2048`, `TP=2`, local `I=1024`, `E=256`, `topk=6`.
This changes both GEMM shapes and arithmetic intensity, so absolute wall times
are not directly comparable to the TP=8 table.

Command used for the current fused path:

```bash
SGLANG_ROOT=/workspace/sglang \
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
CUDA_VISIBLE_DEVICES=0 \
python scripts/bench_mxfp4_int8_jit_moe.py \
  --backend jit \
  --tp-size 2 \
  --batches 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --autotune-tiles \
  --warmup 10 \
  --iters 30 \
  --output-dir /workspace/monkeypatch/benchmark_results_autotune_capped_tp2
```

Command used for the OGS baseline:

```bash
SGLANG_ROOT=/workspace/sglang \
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
CUDA_VISIBLE_DEVICES=0 \
python scripts/bench_mxfp4_int8_jit_moe.py \
  --backend ogs \
  --tp-size 2 \
  --batches 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
  --warmup 10 \
  --iters 30 \
  --output-dir /workspace/monkeypatch/benchmark_results_ogs_tp2
```

This uses the same OGS swizzle and matmul path from
`dsv4_a100_patch.triton_kernels`.

TP=2 capped-autotune full-path results:

| batch | M cap | best tile | OGS full ms | TP2 JIT full ms | speedup | TP2 JIT TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16x64 | 0.8282 | 0.1429 | 5.80x | 1.1 |
| 2 | 16 | 16x64 | 0.8260 | 0.1597 | 5.17x | 1.9 |
| 4 | 16 | 16x64 | 0.8222 | 0.2490 | 3.30x | 2.4 |
| 8 | 16 | 16x128 | 1.0571 | 0.3427 | 3.08x | 3.5 |
| 16 | 16 | 16x128 | 1.9372 | 0.5911 | 3.28x | 4.1 |
| 32 | 16 | 16x128 | 3.5933 | 1.0775 | 3.33x | 4.5 |
| 64 | 16 | 16x128 | 4.5985 | 1.4011 | 3.28x | 6.9 |
| 128 | 16 | 16x128 | 4.5099 | 1.4113 | 3.20x | 13.7 |
| 256 | 16 | 16x128 | 4.5312 | 1.4353 | 3.16x | 26.9 |
| 512 | 16 | 16x128 | 4.5792 | 1.4897 | 3.07x | 51.9 |
| 1024 | 32 | 32x128 | 4.6738 | 2.7690 | 1.69x | 55.8 |
| 2048 | 64 | 64x128 | 4.9070 | 3.3008 | 1.49x | 93.7 |
| 4096 | 128 | 128x128 | 9.5463 | 4.7023 | 2.03x | 131.5 |
| 8192 | 128 | 128x128 | 14.5545 | 9.9915 | 1.46x | 123.8 |
| 16384 | 128 | 128x128 | 28.5563 | 15.6621 | 1.82x | 158.0 |

Across this TP=2 sweep, the capped-autotuned SGLang-JIT MXFP4/INT8 path is
`1.46x-5.80x` faster than the original Triton/OGS MXFP4 path by full fused-MoE
wall time. For larger decode-like batches `4096-16384`, the speedup is
`1.46x-2.03x`.

## End-to-End Serving

The current SGLang serving benchmark used random IDs and the converted
DeepSeek V4 Flash MXFP4/BF16 checkpoint:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random-ids \
  --num-prompts 32 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency 32
```

Representative current results:

| case | successful requests | duration s | output tok/s | total tok/s | mean TTFT ms | mean TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| `conc=32, input=1024, output=1024` | 32 | 34.74 | 943.18 | 1886.37 | 9450.89 | 24.70 |
| `conc=1, input=65536, output=1024` | 1 | 18.17 | 56.36 | 3663.14 | 2061.68 | 15.73 |

Server logs during the 32-way decode run show CUDA graph enabled and stable
decode throughput around `1.3k tokens/s`:

```text
Decode batch, #running-req: 32, cuda graph: True,
gen throughput (token/s): roughly 1320-1350
```

These serving numbers are representative current throughput for the integrated
path, not a clean OGS-vs-JIT end-to-end delta. Full-server latency also includes
prefill, sparse attention/indexer work, scheduling, request streaming, CUDA
graph warmup, and memory-system effects outside the routed MoE kernel.

## Reproducing

For full-model serving, launch the MXFP4 model with:

```bash
SGLANG_ROOT=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
bash scripts/launch_dsv4_flash_mxfp4_sglang.sh
```

Then run `sglang.bench_serving` with the dimensions listed above.
