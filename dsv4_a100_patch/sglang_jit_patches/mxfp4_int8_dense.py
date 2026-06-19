from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api
from dsv4_a100_patch.sglang_jit_patches.jit_loader import load_patch_jit
from dsv4_a100_patch.sglang_jit_patches.mxfp4_int8_moe import (
    _mxfp4_int8_moe_cuda_flags,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _cuda_sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _validate_dense_template_args(k: int, n: int) -> None:
    if k <= 0 or n <= 0:
        raise ValueError(f"k/n must be positive, got {k}/{n}")
    if k % 32 != 0:
        raise ValueError(f"k must be divisible by 32, got {k}")
    if k > 64 and k % 64 != 0:
        raise ValueError(f"k must be <= 64 or divisible by 64, got {k}")


@cache_once
def _jit_mxfp4_int8_dense_module(k: int, n: int) -> Module:
    _validate_dense_template_args(k, n)
    cpp_args = make_cpp_args(k, n)
    module = load_patch_jit(
        "mxfp4_int8_dense",
        str(k),
        str(n),
        cuda_files=[
            "gemm/mxfp4_int8/mxfp4_int8_dense_entry.cuh",
        ],
        cuda_wrappers=[
            (
                "mxfp4_int8_dense_gemm",
                f"Mxfp4Int8DenseGemm<{cpp_args}>::run",
            ),
            (
                "init_mxfp4_int8_dense_attrs",
                f"Mxfp4Int8DenseGemm<{cpp_args}>::init",
            ),
        ],
        extra_cuda_cflags=_mxfp4_int8_moe_cuda_flags(),
        extra_dependencies=["cutlass"],
    )
    module.init_mxfp4_int8_dense_attrs()
    return module


@torch.compiler.disable
def prewarm_mxfp4_int8_dense_jit_module(*, k: int, n: int) -> None:
    """Compile and initialize the dense MXFP4/INT8 JIT module before graph capture."""
    device_index = torch.cuda.current_device()
    _cuda_sm_count(device_index)
    _jit_mxfp4_int8_dense_module(k, n)


@debug_kernel_api
def mxfp4_int8_dense_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    b_channel_scale: torch.Tensor,
    out: torch.Tensor,
    partial: torch.Tensor | None = None,
) -> None:
    if a_q.dim() != 2:
        raise ValueError(f"a_q must have shape [M, K], got {tuple(a_q.shape)}")
    m = int(a_q.shape[0])
    k = int(a_q.shape[1])
    n = int(b_channel_scale.shape[0])
    _validate_dense_template_args(k, n)
    if partial is None:
        if m <= 128:
            acc_ld = ((n + 31) // 32) * 32
            partial = torch.empty((4, m, acc_ld), device=a_q.device, dtype=torch.int32)
        else:
            partial = torch.empty((0,), device=a_q.device, dtype=torch.int32)
    module = _jit_mxfp4_int8_dense_module(k, n)
    module.mxfp4_int8_dense_gemm(
        a_q,
        a_scale,
        b_mxfp4,
        b_shift2,
        b_channel_scale,
        out,
        partial,
        _cuda_sm_count(
            a_q.device.index
            if a_q.device.index is not None
            else torch.cuda.current_device()
        ),
    )
