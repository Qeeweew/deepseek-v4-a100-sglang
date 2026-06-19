from __future__ import annotations

import torch

from triton_kernels import direct_dual_sparse_attention, direct_sparse_attention


def _run_single(q_tokens: int, heads: int, dim: int, topk: int) -> None:
    q = torch.randn(q_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    buf = torch.randn(32768, dim, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(-1, buf.shape[0], (q_tokens, topk), device="cuda", dtype=torch.int32)
    lens = torch.randint(1, topk + 1, (q_tokens,), device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    out, lse = direct_sparse_attention(q, buf, idx, lens, dim**-0.5, attn_sink=sink)
    torch.cuda.synchronize()
    print("single ok", q.shape, idx.shape, out.shape, lse.shape)


def _run_dual(q_tokens: int, heads: int, dim: int, topk0: int, topk1: int) -> None:
    q = torch.randn(q_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    buf0 = torch.randn(32768, dim, device="cuda", dtype=torch.bfloat16)
    buf1 = torch.randn(16384, dim, device="cuda", dtype=torch.bfloat16)
    idx0 = torch.randint(-1, buf0.shape[0], (q_tokens, topk0), device="cuda", dtype=torch.int32)
    idx1 = torch.randint(-1, buf1.shape[0], (q_tokens, topk1), device="cuda", dtype=torch.int32)
    len0 = torch.randint(1, topk0 + 1, (q_tokens,), device="cuda", dtype=torch.int32)
    len1 = torch.randint(1, topk1 + 1, (q_tokens,), device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    out, lse = direct_dual_sparse_attention(
        q, buf0, idx0, len0, buf1, idx1, len1, dim**-0.5, attn_sink=sink
    )
    torch.cuda.synchronize()
    print("dual ok", q.shape, idx0.shape, idx1.shape, out.shape, lse.shape)


def main() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cases = [
        ("single", 1, 128, 512, 64),
        ("single", 24, 128, 512, 64),
        ("single", 24, 96, 512, 64),
        ("dual", 24, 128, 512, 48, 24),
        ("dual", 24, 96, 512, 48, 24),
        ("dual", 32, 128, 512, 64, 32),
    ]

    for case in cases:
        if case[0] == "single":
            _, q_tokens, heads, dim, topk = case
            _run_single(q_tokens, heads, dim, topk)
        else:
            _, q_tokens, heads, dim, topk0, topk1 = case
            _run_dual(q_tokens, heads, dim, topk0, topk1)


if __name__ == "__main__":
    main()
