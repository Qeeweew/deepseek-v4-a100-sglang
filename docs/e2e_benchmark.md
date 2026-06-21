# End-to-End Benchmark

This page records full-server SGLang serving results for the integrated
DeepSeek V4 Flash MXFP4/BF16 path. These numbers include prefill, decode,
sparse attention/indexer work, scheduling, request streaming, CUDA graph
warmup, and memory-system effects.

## Setup

- Model: `/ssd1/models/DeepSeek-V4-Flash-MoE-MXFP4-BF16`
- GPUs: A100, TP=8
- Server: `scripts/launch_dsv4_flash_mxfp4_sglang.sh`
- Runtime: MXFP4 weights, MXFP4/INT8 fused MoE, indexer query-token CP enabled
  for prefill
- Dataset: random IDs, `--random-range-ratio 1.0`
- Each run used `--flush-cache`

## Decode

Input length is 1024 tokens and output length is 1024 tokens. `batch` is both
`--num-prompts` and `--max-concurrency`.

| batch | successful requests | duration s | output tok/s | total tok/s | mean TTFT ms | mean TPOT ms | mean E2E ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 14.24 | 71.92 | 143.85 | 193.50 | 13.71 | 14219.42 |
| 2 | 2 | 15.31 | 133.78 | 267.57 | 601.06 | 14.36 | 15290.43 |
| 4 | 4 | 16.80 | 243.79 | 487.58 | 1140.61 | 15.29 | 16780.55 |
| 8 | 8 | 19.98 | 410.00 | 819.99 | 2191.22 | 17.37 | 19961.89 |
| 16 | 16 | 23.19 | 706.39 | 1412.78 | 1738.26 | 20.95 | 23174.65 |
| 32 | 32 | 28.19 | 1162.50 | 2325.00 | 2795.12 | 24.80 | 28169.52 |
| 64 | 64 | 37.33 | 1755.82 | 3511.64 | 4513.12 | 32.05 | 37303.10 |

Command template:

```bash
PYTHONPATH=/sgl-workspace/sglang/python${PYTHONPATH:+:$PYTHONPATH} \
python -m sglang.bench_serving \
  --host 127.0.0.1 \
  --port 30002 \
  --backend sglang \
  --model /ssd1/models/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
  --dataset-name random-ids \
  --random-range-ratio 1.0 \
  --num-prompts ${BATCH} \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency ${BATCH} \
  --flush-cache
```

## Prefill

Batch size is 1 and output length is 1024 tokens.

| input tokens | successful requests | duration s | input tok/s | output tok/s | total tok/s | mean TTFT ms | mean TPOT ms | mean E2E ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 1 | 18.35 | 1785.44 | 55.79 | 1841.23 | 3240.98 | 14.75 | 18334.61 |
| 131072 | 1 | 28.77 | 4556.35 | 35.60 | 4591.95 | 11327.39 | 17.03 | 28748.48 |
| 524288 | 1 | 101.69 | 5155.90 | 10.07 | 5165.97 | 74800.13 | 26.27 | 101677.53 |

Command template:

```bash
PYTHONPATH=/sgl-workspace/sglang/python${PYTHONPATH:+:$PYTHONPATH} \
python -m sglang.bench_serving \
  --host 127.0.0.1 \
  --port 30002 \
  --backend sglang \
  --model /ssd1/models/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
  --dataset-name random-ids \
  --random-range-ratio 1.0 \
  --num-prompts 1 \
  --random-input-len ${INPUT_LEN} \
  --random-output-len 1024 \
  --max-concurrency 1 \
  --flush-cache
```
