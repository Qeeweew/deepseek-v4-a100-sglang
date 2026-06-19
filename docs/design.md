# Design Overview

This repository is an SGLang runtime patch for serving DeepSeek V4 Flash on
A100. It is organized around the serving path, not as a standalone kernel
library.

## Scope

The patch handles three A100 gaps:

1. DeepSeek V4 paths that assume newer FP8 or clustered kernels.
2. Routed expert weights stored in packed MXFP4.
3. Attention/indexer/compressor paths that need BF16-compatible cache and
   fallback kernels on SM80.

The patch is injected through `sitecustomize.py` when
`ENABLE_SGLANG_DSV4_A100_PATCH=1` is set. It patches SGLang at import time and
does not require modifying the SGLang source tree.

## Runtime Components

- `dsv4_a100_patch.patch`: SGLang monkeypatch entry point.
- `dsv4_a100_patch.triton_kernels`: A100 Triton kernels and utility fallbacks.
- `dsv4_a100_patch.sglang_jit_patches`: SGLang JIT kernels shipped with this
  patch, including the MXFP4/INT8 MoE CUTLASS-based path.
- `scripts/`: model conversion and launch helpers.

The MXFP4/INT8 routed-expert GEMM is part of `dsv4_a100_patch.sglang_jit_patches`
and is compiled through SGLang's JIT loader. CUTLASS is resolved from the
SGLang runtime environment; this project does not vendor CUTLASS and does not
ship a separate public CUDA operator package.

Per-token INT8 activation quantization is implemented as a Triton kernel in
this package. Dense and routed-expert MXFP4/INT8 GEMMs are compiled from this
package's SGLang JIT headers.

## MXFP4 Serving Path

The converted MXFP4 model keeps routed expert weights in the original packed
MXFP4 format and materializes non-routed FP8 weights as BF16. During SGLang
weight loading, routed expert weights are repacked into the compact runtime
format used by the MXFP4/INT8 MoE JIT kernel. The original packed weight and
UE8M0 scale tensors are then replaced by empty tensors to save memory.

## INT4 Serving Path

The INT4 conversion path materializes non-routed weights as BF16 and converts
routed experts to compressed-tensors INT4 group-size 32. It is useful as a
lower-memory comparison baseline and does not use the MXFP4/INT8 JIT kernel.

## CUDA Graph Capture

CUDA function attributes for MXFP4/INT8 kernels must be initialized once per
worker/device before CUDA graph capture. The JIT module wrapper initializes its
own CUTLASS dynamic shared-memory attributes during prewarm. No attribute setup
should happen inside a CUDA graph capture interval.

## Packaging

The source tree is arranged as an installable Python package:

- `dsv4_a100_patch/`: runtime code and JIT kernel headers.
- `triton_kernels/`: compatibility import shim for older internal paths that
  imported patch kernels from the top-level `triton_kernels` namespace.
- `scripts/`: conversion and launch scripts.
- `docs/`: design, version, and operating notes.

The package includes only patch-owned JIT headers. SGLang, flashinfer/deep_gemm,
Triton, PyTorch, and optional conversion dependencies are installed separately.
