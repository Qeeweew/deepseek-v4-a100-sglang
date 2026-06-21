# Runtime Environment

## MXFP4 Launch

Run launch scripts from the repository checkout, or invoke them with an
absolute path. They compute `PROJECT_ROOT` from the script location and prepend
that source tree plus `${SGLANG_ROOT}/python` to `PYTHONPATH`.

```bash
SGLANG_ROOT=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
PORT=30002 \
bash scripts/launch_dsv4_flash_mxfp4_sglang.sh
```

## INT4 Launch

```bash
SGLANG_ROOT=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-MoE-INT4-G32-BF16 \
PORT=30002 \
bash scripts/launch_dsv4_flash_int4_sglang.sh
```

## Key Environment Variables

- `ENABLE_SGLANG_DSV4_A100_PATCH=1`: enables patch injection.
- `SGLANG_DSV4_MXFP4_MOE_BACKEND=mxfp4_int8`: uses the MXFP4/INT8 MoE path.
- `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`: disables unsupported A100 warmup path.
- `SGLANG_OPT_FUSE_WQA_WKV=0`: keeps converted checkpoint tensor names
  compatible with the patch.
- `SGLANG_OPT_USE_TOPK_V2=0`: disables the cluster top-k kernel unsupported on
  A100.
- `SGLANG_TOPK_TRANSFORM_512_TORCH=0`: keeps the CUDA v1 top-k path unless a
  conservative torch fallback is required.
- `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`: avoids the unsupported FP8 paged MQA
  logits path.
- `SGLANG_DSV4_INDEXER_QUERY_CP_PREFILL=0`: optional C4 indexer query-token
  parallelism for prefill.

## Notes

The MXFP4 path uses `--quantization fp8` as the SGLang loader entry point. The
patch then replaces routed expert preparation and execution with the MXFP4/INT8
JIT kernel.

`SGLANG_DSV4_MXFP4_INT8_USE_JIT=1` is the default. The supported path uses this
package's SGLang JIT GEMM and Triton activation quantizer.

The MXFP4/INT8 MoE JIT path selects `(block_m, block_n)` from an A100 static
tile table by batch size. Use the following variables only for measurement or
debugging:

- `SGLANG_DSV4_MXFP4_INT8_BLOCK_M`: force `block_m` to `16`, `32`, `64`, or
  `128`.
- `SGLANG_DSV4_MXFP4_INT8_BLOCK_N`: force `block_n` to `32`, `64`, or `128`.
- `SGLANG_DSV4_MXFP4_INT8_JIT_PREWARM_TILES`: compile explicit tile pairs
  before CUDA graph capture, for example `16x64,16x128,128x128`.
- `SGLANG_DSV4_MXFP4_INT8_JIT_PREWARM_BLOCK_M`: legacy block-M-only prewarm
  override. Prefer `SGLANG_DSV4_MXFP4_INT8_JIT_PREWARM_TILES` when testing
  tuned tile pairs.

The INT4 path uses `--quantization compressed-tensors` and the converted INT4
expert weights.
