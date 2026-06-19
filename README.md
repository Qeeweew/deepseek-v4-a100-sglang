# DeepSeek V4 A100 SGLang Patch

This project packages the A100-specific runtime patches used to serve
DeepSeek V4 Flash checkpoints with SGLang.

The MXFP4/INT8 MoE operator is integrated as an SGLang JIT kernel patch. It is
not exposed as a standalone CUDA extension in this repository. CUTLASS headers
are resolved from the SGLang runtime environment, typically through flashinfer
or deep_gemm.

## What This Provides

- DeepSeek V4 A100 runtime patch for SGLang.
- MXFP4 routed-expert repack and MXFP4/INT8 MoE execution through SGLang JIT.
- INT4 compressed-tensors model path for lower-memory comparison.
- BF16 KV cache and attention/indexer fallbacks for kernels that are not
  supported on A100.
- Conversion scripts for DeepSeek V4 Flash MXFP4 and INT4 serving formats.

## Verified SGLang Version

The current implementation is validated against:

```text
SGLang repository: https://github.com/sgl-project/sglang.git
SGLang commit:     1c0019da7579db73223195f25b0eed3882dff24e
```

Use a matching SGLang source checkout and set `SGLANG_ROOT` when launching.

## Quick Start

Install the patch next to an SGLang source checkout:

```bash
cd /path/to/deepseek-v4-a100
pip install -e .
```

Use the conversion and launch scripts from the source checkout. The Python
package contains the runtime patch and JIT headers; repository-level scripts
and docs are source-tree artifacts.

The MXFP4/INT8 MoE GEMM is loaded through SGLang JIT from this package. The
default MoE path does not require the old standalone experiment extension.
The legacy `mxfp4_int8` extension is only needed for the dense helper and for
forcing the explicit fallback path with `SGLANG_DSV4_MXFP4_INT8_USE_JIT=0`.

Convert the model first:

```bash
python scripts/convert_deepseek_v4_flash_moe_mxfp4_bf16.py \
  --input /path/to/DeepSeek-V4-Flash \
  --output /path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16
```

Then launch with SGLang:

```bash
SGLANG_ROOT=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16 \
bash scripts/launch_dsv4_flash_mxfp4_sglang.sh
```

For the INT4 comparison path:

```bash
python scripts/convert_deepseek_v4_flash_moe_int4.py \
  --input /path/to/DeepSeek-V4-Flash \
  --output /path/to/DeepSeek-V4-Flash-MoE-INT4-G32-BF16 \
  --device cuda:0

SGLANG_ROOT=/path/to/sglang \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-MoE-INT4-G32-BF16 \
bash scripts/launch_dsv4_flash_int4_sglang.sh
```

## Documentation

- `docs/design.md`: runtime architecture and patch scope.
- `docs/mxfp4_int8_moe.md`: MXFP4/INT8 MoE design.
- `docs/model_conversion.md`: MXFP4 and INT4 conversion scripts.
- `docs/sglang_version.md`: SGLang version and integration assumptions.
- `docs/runtime_env.md`: launch environment variables.
- `docs/benchmarks.md`: MoE microbenchmarks and serving throughput notes.

## Notes

MXFP4 repack is not mathematically lossless. Some scale information is folded
into the FP4 code and compact shift metadata so the runtime can decode weights
into INT8 fragments for SM80 tensor cores. The original MXFP4 tensors and UE8M0
scales are replaced after repack to reduce memory usage.

This repository is intended to remain a patch package. Keep SGLang as an
external checkout and record the validated SGLang commit when updating the
integration.
