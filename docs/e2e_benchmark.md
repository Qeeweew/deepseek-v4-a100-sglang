# End-to-End Benchmark

This page records representative full-server SGLang serving results for the
integrated DeepSeek V4 Flash MXFP4/BF16 path. These numbers are not a clean
kernel-only comparison: full-server latency includes prefill, sparse
attention/indexer work, scheduling, request streaming, CUDA graph warmup, and
memory-system effects outside the routed MoE kernel.

## Setup

The benchmark used random IDs and the converted DeepSeek V4 Flash MXFP4/BF16
checkpoint:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random-ids \
  --num-prompts 32 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency 32
```

## Results

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

## Reproducing

For full-model serving, launch the MXFP4 model with:

```bash
SGLANG_ROOT=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
bash scripts/launch_dsv4_flash_mxfp4_sglang.sh
```

Then run `sglang.bench_serving` with the dimensions listed above.
