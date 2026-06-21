# MTP Support

This patch supports DeepSeek V4 MTP speculative decoding on A100 for the
converted `DeepSeek-V4-Flash-MoE-MXFP4-BF16` checkpoint.

## Support Status

- Supported checkpoint layout: `mtp.0` only.
- Supported SGLang algorithm: `EAGLE`.
- Tested configuration:
  - `--speculative-num-steps 3`
  - `--speculative-eagle-topk 1`
  - `--speculative-num-draft-tokens 4`
  - `--speculative-draft-model-quantization fp8`
  - `--speculative-moe-runner-backend marlin`
- Multi-layer EAGLE is not enabled by default. The converted checkpoint used
  here contains only one MTP layer.

The launch script keeps MTP disabled by default. Enable it explicitly:

```bash
MODEL_PATH=/ssd1/models/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
ENABLE_MTP=1 \
CUDA_GRAPH_MAX_BS=8 \
bash scripts/launch_dsv4_flash_mxfp4_sglang.sh
```

## MXFP4 Routing

The MTP draft model must keep `--speculative-draft-model-quantization fp8`.
This value is the SGLang loader entry point for the checkpoint's MXFP4 routed
expert tensors. The A100 patch then marks the DeepSeek V4 NextN quant config as
`is_fp4_experts=True`, so `FusedMoE` is registered through
`Mxfp4MarlinMoEMethod`. That method is patched by this project and its weight
preparation and apply path are replaced with the MXFP4/INT8 implementation.

The expected server log contains one line per TP rank similar to:

```text
Monkey patch: routing DeepSeek V4 MTP MXFP4 experts through Mxfp4MarlinMoEMethod for mxfp4_int8 replacement.
Monkey patch: using mxfp4_int8 MoE for model.decoder.mlp.experts (headroom_bits=3).
```

The MTP checkpoint's `e_proj` and `h_proj` tensors are BF16. They are added to
the FP8 ignored-layer list for the draft model so they remain unquantized. This
prevents A100 from compiling the slow/undesired FP8 weight-only Marlin path:

```text
sgl_kernel_jit_gptq_marlin_bf16_t.../cuda.cu
```

If this JIT path appears during MTP startup, routing is not correct.

## Validation

The current implementation was validated with:

```bash
PYTHONPATH=/workspace/monkeypatch:/workspace/sglang/python \
pytest -q tests/test_dsv4_a100_patch.py
```

Result:

```text
7 passed
```

Runtime checks:

- `/v1/models` returns successfully.
- A short `/v1/completions` request completes.
- MTP draft CUDA graph capture succeeds.
- No `gptq_marlin` or `sgl_kernel_jit_gptq_marlin` compile process appears
  during startup.

## Decode Benchmark

The following results compare MTP against the same server configuration with
`ENABLE_MTP=0`. Input length is 1024 tokens, output length is 1024 tokens, and
`batch` is both `--num-prompts` and `--max-concurrency`.

Setup:

- Model: `/ssd1/models/DeepSeek-V4-Flash-MoE-MXFP4-BF16`
- GPUs: A100, TP=8
- Server script: `scripts/launch_dsv4_flash_mxfp4_sglang.sh`
- CUDA graph max batch size: 8
- Dataset: random IDs, `--random-range-ratio 1.0`

| batch | baseline output tok/s | MTP output tok/s | speedup | baseline TPOT ms | MTP TPOT ms | MTP accept length |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67.03 | 103.57 | 1.55x | 13.69 | 8.40 | 2.06 |
| 2 | 136.17 | 176.15 | 1.29x | 14.33 | 9.42 | 2.14 |
| 4 | 253.89 | 346.33 | 1.36x | 15.26 | 8.98 | 2.34 |
| 8 | 448.62 | 596.86 | 1.33x | 17.32 | 10.22 | 2.41 |

Benchmark command template:

```bash
PYTHONPATH=/workspace/sglang/python \
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
  --output-file /tmp/mtp_e2e_current/mtp/b${BATCH}.jsonl \
  --max-concurrency ${BATCH}
```

## Notes

MTP improves small-batch decode throughput in this setup because the draft
model accepts about 2.1 to 2.4 tokens per target verification step. TTFT may be
higher with MTP because the server performs draft work and verification setup;
the benefit is in sustained decode TPOT and output token throughput.
