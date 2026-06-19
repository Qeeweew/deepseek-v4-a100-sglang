from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api
from dsv4_a100_patch.sglang_jit_patches.jit_loader import load_patch_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


def _mxfp4_int8_moe_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
    ]


def _default_block_n(block_m: int) -> int:
    if block_m in (32, 64):
        return 64
    return 128


@cache_once
def _cuda_sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _validate_template_args(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    block_m: int,
    block_n: int,
) -> None:
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError(
            f"hidden_size/intermediate_size must be positive, got "
            f"{hidden_size}/{intermediate_size}"
        )
    if topk <= 0 or topk > 8:
        raise ValueError(f"topk must be in [1, 8], got {topk}")
    if block_m not in {16, 32, 64, 128}:
        raise ValueError(f"block_m must be 16, 32, 64, or 128, got {block_m}")
    if block_n not in {32, 64, 128}:
        raise ValueError(f"block_n must be 32, 64, or 128, got {block_n}")


@cache_once
def _jit_mxfp4_int8_moe_module(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    block_m: int,
    block_n: int,
    source_rows_are_slots: bool,
) -> Module:
    _validate_template_args(
        hidden_size, intermediate_size, topk, block_m, block_n
    )
    cpp_args = make_cpp_args(
        hidden_size,
        intermediate_size,
        topk,
        block_m,
        block_n,
        source_rows_are_slots,
    )
    module = load_patch_jit(
        "mxfp4_int8_moe",
        str(hidden_size),
        str(intermediate_size),
        str(topk),
        str(block_m),
        str(block_n),
        "w2" if source_rows_are_slots else "w13",
        cuda_files=[
            "gemm/mxfp4_int8/mxfp4_int8_moe_entry.cuh",
        ],
        cuda_wrappers=[
            (
                "mxfp4_int8_moe_gemm",
                f"Mxfp4Int8MoeGemm<{cpp_args}>::run",
            ),
            (
                "init_mxfp4_int8_moe_attrs",
                f"Mxfp4Int8MoeGemm<{cpp_args}>::init",
            ),
        ],
        extra_cuda_cflags=_mxfp4_int8_moe_cuda_flags(),
        extra_dependencies=["cutlass"],
    )
    module.init_mxfp4_int8_moe_attrs()
    return module


@torch.compiler.disable
def prewarm_mxfp4_int8_moe_jit_modules(
    *,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    block_ms: tuple[int, ...] = (16, 32, 64),
    block_n: int | None = None,
    tile_shapes: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Compile and initialize MoE JIT modules before CUDA graph capture."""
    device_index = torch.cuda.current_device()
    _cuda_sm_count(device_index)
    if tile_shapes is None:
        tile_shapes = tuple(
            (block_m, _default_block_n(block_m) if block_n is None else block_n)
            for block_m in block_ms
        )
    for block_m, resolved_block_n in tile_shapes:
        for source_rows_are_slots in (False, True):
            _jit_mxfp4_int8_moe_module(
                hidden_size,
                intermediate_size,
                topk,
                block_m,
                resolved_block_n,
                source_rows_are_slots,
            )


@debug_kernel_api
def mxfp4_int8_moe_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_mxfp4: torch.Tensor,
    b_shift2: torch.Tensor,
    b_channel_scale: torch.Tensor,
    out: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    block_m: int,
    source_rows_are_slots: bool,
    num_valid_tokens: int,
    routed_out: torch.Tensor | None = None,
    block_n: int | None = None,
) -> None:
    resolved_block_n = _default_block_n(block_m) if block_n is None else block_n
    _validate_template_args(
        hidden_size, intermediate_size, topk, block_m, resolved_block_n
    )
    if routed_out is None:
        routed_out = out
    module = _jit_mxfp4_int8_moe_module(
        hidden_size,
        intermediate_size,
        topk,
        block_m,
        resolved_block_n,
        source_rows_are_slots,
    )
    module.mxfp4_int8_moe_gemm(
        a_q,
        a_scale,
        b_mxfp4,
        b_shift2,
        b_channel_scale,
        out,
        routed_out,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        _cuda_sm_count(a_q.device.index if a_q.device.index is not None else torch.cuda.current_device()),
    )
