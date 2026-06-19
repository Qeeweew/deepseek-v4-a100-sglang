# MXFP4/INT8 MoE Operator Limitations

The current MXFP4/INT8 MoE path is optimized for A100 compatibility and lower
memory use, but it is not expected to reach dense GEMM peak throughput.

## Why Dense GEMM Is Faster

Dense MXFP4/INT8 GEMM has one regular `M x N x K` problem. At large batch, the
kernel can launch many full tiles, reuse a stable tile policy, and use
persistent scheduling when the grid is large enough. This is the case where the
original dense implementation can reach roughly the 350 TFLOPS class on A100,
and the dense SGLang-JIT entry in this repository is intended to preserve that
large-batch path without relying on an external CUDA extension.

MoE GEMM is a grouped sparse routing problem. Even when the total batch is
large, each expert sees only a fraction of the routed tokens:

```text
average expert M ~= batch * topk / num_experts
```

For DeepSeek V4 Flash dimensions with `topk=6` and `E=256`, decode-like
batches frequently leave each expert with far fewer rows than the dense GEMM
tile shape wants. The kernel therefore spends more time on partially filled
tiles, routing metadata, and launch overhead relative to useful MMA work.

## Current Costs

- Per-expert token fragmentation limits tile occupancy.
- `moe_align_block_size` pads each expert independently, so small expert M
  wastes M tile capacity.
- The W13 path writes routed slots, then a separate activation/quantization
  step feeds W2.
- The W2 path writes per-route workspace and reduces top-k routes back to token
  rows.
- The static tile table is tuned for robustness across batch sizes, not for
  every expert-token distribution.
- Persistent scheduling only helps when the grouped grid is large enough; small
  per-expert M does not create the same scheduling shape as dense GEMM.

These effects explain why current MoE microbenchmarks are substantially below
large-batch dense GEMM TFLOPS even though both paths use the same packed MXFP4
weight representation and SM80 tensor cores.

## Practical Implication

Use dense benchmarks to validate the raw packed-weight GEMM path and use MoE
benchmarks to evaluate routed serving performance. A dense 300+ TFLOPS result
does not imply the MoE path should reach the same number, because the routed
operator includes padding, routing, activation, quantization, workspace writes,
and top-k reduction that are absent from dense GEMM.

## Future Work

- Add per-shape autotuned tile tables for W13 and W2 separately.
- Improve grouped persistent scheduling for fragmented expert batches.
- Reduce W2 workspace and top-k reduction overhead.
- Fuse activation and second-stage quantization where practical.
- Explore split-K or stream-K variants for small expert M and large K.
