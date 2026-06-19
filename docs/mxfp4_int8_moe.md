# MXFP4/INT8 MoE JIT Kernel

The MXFP4/INT8 MoE GEMM path is integrated through SGLang JIT. It is not a
separate public CUDA extension in this repository.

## Principle

A100 does not have native FP4 tensor-core MMA. The kernel therefore runs:

```text
INT8 activation x decoded INT8 weight fragment -> INT32 accumulation -> BF16 output
```

The persistent weight remains packed MXFP4 plus compact metadata. The kernel
decodes packed MXFP4 into INT8 fragments inside the CUTLASS mainloop and feeds
those fragments to SM80 INT8 tensor cores.

The activation side is quantized per token to INT8 with one FP32 scale per
token. This is a Triton kernel in `dsv4_a100_patch.triton_kernels`, using
`scale = max(abs(row)) / 127`, zero-row scale `1.0`, and round-to-nearest-even
before clamping to `[-127, 127]`.

## MXFP4 Repack

MXFP4 stores E2M1 FP4 codes plus UE8M0 block scales. Runtime per-block scale
application inside the K loop would be too expensive, so the preprocessing step
remaps the weight:

- packed remapped MXFP4 code
- packed 2-bit shift metadata
- FP32 per-channel scale

The doubled E2M1 integer table is:

```text
{0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}
```

Runtime decode is integer-only:

```text
q_i8 = lut_x2[fp4_code] << shift2
```

The largest value is `12 << 3 = 96`, which fits in signed INT8. The output
epilogue applies activation and channel scales:

```text
out = acc_i32 * activation_scale * channel_scale
```

This repack is not lossless. Some original scale choices cannot be represented
exactly by the compact shift scheme, so the nearest E2M1 code is selected during
remap.

The Triton repack path performs three device-side steps:

1. Compute the maximum UE8M0 exponent for each expert/output row.
2. Pack a 2-bit shift for each K block relative to that row maximum.
3. Remap each FP4 nibble to the nearest representable E2M1 code after folding
   the original block scale into the row scale and compact shift.

`nearest_e2m1_code` is threshold based over the doubled-magnitude table
`[0, 1, 2, 3, 4, 6, 8, 12]`. The midpoints are `0.5, 1.5, 2.5, 3.5, 5, 7, 10`;
the sign bit is restored after choosing the magnitude code. Zero input stays
zero.

## JIT Template Parameters

The JIT kernel templates:

- hidden size
- intermediate size
- top-k
- block M
- block N
- W13 vs W2 source-row semantics

Expert count is intentionally not templated. It has low runtime overhead and
templating it would multiply compile variants for little benefit.

## MoE Flow

W13:

1. Quantize hidden states per token to INT8.
2. Align routed tokens by expert.
3. Run grouped MXFP4/INT8 GEMM to produce gate/up activations.
4. Apply SwiGLU.

W2:

1. Quantize intermediate activations per routed slot.
2. Run grouped MXFP4/INT8 GEMM per slot.
3. Reduce top-k slots with router weights into the final hidden-state output.

`sorted_token_ids.size(0)` is the host-side allocation and maximum launch
capacity. For `num_valid_tokens = M * topk`, `E` experts, and block size
`block_m`, the maximum aligned token count is:

```text
num_valid_tokens < E + 1:
  max_aligned = num_valid_tokens * block_m
otherwise:
  max_aligned = num_valid_tokens + (E + 1) * (block_m - 1)
```

`num_tokens_post_padded[0]` is the device-side actual aligned-token count
written by SGLang's routing alignment kernel. Host launch uses the maximum
capacity above, and device kernels guard real work by `num_tokens_post_padded[0]`.
`expert_ids.size(0)` only needs to cover `ceil(max_aligned / block_m)` M tiles;
it is not a way to derive `sorted_token_ids` capacity.
