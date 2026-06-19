# SGLang Version

This patch is validated against a specific SGLang source checkout:

```text
repository: https://github.com/sgl-project/sglang.git
commit:     1c0019da7579db73223195f25b0eed3882dff24e
```

Use this version unless you have verified the patch against a newer SGLang
commit.

## Integration Assumptions

The patch relies on:

- SGLang DeepSeek V4 model and quantization loader internals;
- `sglang.jit_kernel.utils` for JIT compilation context and dependency
  resolution;
- SGLang/flashinfer/deep_gemm environment to provide CUTLASS headers;
- SGLang CUDA graph warmup behavior so JIT kernels can be prewarmed before
  capture.

The project does not vendor CUTLASS and does not patch the SGLang source tree.
Set `SGLANG_ROOT` to the SGLang checkout when using the launch scripts.

