DeepSeek V4 A100 Triton kernels
================================

These kernels replace the high-cost PyTorch fallback paths in
`dsv4_a100_patch.py`.

Implemented kernels:

- `fused_rope_inplace`: applies RoPE to Q and optional K in one launch.
- `bf16_paged_mqa_logits`: computes C4 indexer logits directly from BF16 paged KV
  cache, avoiding the paged gather and `einsum` intermediate.
- `scatter_bf16_rows`: stores BF16 KV/indexer rows into flattened cache by token
  locations.
- `gather_bf16_kv`: gathers sparse BF16 KV rows and writes the invalid mask in the
  same launch.
- `compressor_decode_mask_positions`: applies the decode boundary mask in-place
  and emits RoPE positions.
- `compressor_prefill_metadata`: extracts prefill RoPE positions and selected
  output locations from the compressor plan in one launch.

Deliberately left in Python/Torch:

- `_trim_rows`: metadata shape normalization and padding; this is small relative
  to the KV gather and attention work.
- final concatenation shape decisions in the attention path; the gather kernel can
  write at an output offset, so callers can avoid `torch.cat` when needed.
