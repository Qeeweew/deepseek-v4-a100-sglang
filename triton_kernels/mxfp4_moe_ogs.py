from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.nn import Parameter

logger = logging.getLogger(__name__)


from .ogs_bitmatrix_patch import patch_oai_bitmatrix_metadata


@triton.jit
def _dsv4_swiglu_fn(input, alpha, limit):
    gate, up = tl.split(
        tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
    )
    if limit is not None:
        gate = tl.minimum(gate, limit)
        up = tl.clamp(up, -limit, limit)
    return gate / (1 + tl.exp(-gate)) * up


@triton.jit
def _pack_bitmatrix_kernel(
    bitmatrix,
    topk_ids,
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offsets = offs_m[:, None] * n_expts_act + offs_k[None, :]
    mask = (offs_m[:, None] < n_rows) & (offs_k[None, :] < n_expts_act)
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    valid = indices >= 0
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        bit_cols = tl.arange(0, BLOCK_K // 32) + i * (BLOCK_K // 32)
        x = tl.where(
            valid[:, :, None] & (div[:, :, None] == bit_cols[None, None, :]),
            (one << rem)[:, :, None],
            0,
        )
        y = tl.reduce_or(x, axis=1)
        ptrs = bitmatrix + offs_m[:, None] * bm_cols + bit_cols[None, :]
        tl.store(ptrs, y, mask=offs_m[:, None] < n_rows)


@dataclass
class OgsMxfp4Weights:
    w13: object
    w2: object
    w13_precision_config: object
    w2_precision_config: object


def _scale_to_e8m0_storage(scale: torch.Tensor) -> torch.Tensor:
    """Return E8M0 scale bytes without materializing persistent FP32 storage."""
    data = scale.data if hasattr(scale, "data") else scale
    if data.dtype == torch.float8_e8m0fnu:
        return data.contiguous()
    if data.dtype == torch.uint8:
        return data.contiguous()
    if data.dtype == torch.int8:
        return data.view(torch.uint8).contiguous()
    if data.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return data.to(torch.float8_e8m0fnu).view(torch.uint8).contiguous()
    raise TypeError(f"unsupported MXFP4 scale dtype: {data.dtype}")


def _bind_scale_parameter(layer: torch.nn.Module, name: str) -> torch.Tensor:
    scale = getattr(layer, name)
    scale_u8 = _scale_to_e8m0_storage(scale)
    if scale_u8.data_ptr() != scale.data.data_ptr() or scale.dtype != torch.uint8:
        setattr(layer, name, Parameter(scale_u8, requires_grad=False))
    getattr(layer, name).format_ue8m0 = True
    return getattr(layer, name)


def _interleave_gate_up_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] % 2 != 0:
        raise ValueError(
            f"DeepSeek MXFP4 w13 rows must be gate/up pairs, got {tuple(tensor.shape)}"
        )
    e, two_i = tensor.shape[:2]
    return (
        tensor.reshape(e, 2, two_i // 2, *tensor.shape[2:])
        .transpose(1, 2)
        .reshape_as(tensor)
        .contiguous()
    )


def _swizzle_mxfp4_for_ogs(weight: torch.Tensor, scale_u8: torch.Tensor, num_warps: int):
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import (
        HopperMXScaleLayout,
        HopperMXValueLayout,
        StridedLayout,
    )

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )

    # A100 has no native MXFP. Force the non-persistent simulated-MX path; the
    # persistent path rejects weight_scale when native MXFP is unavailable. The
    # default sm80 MXFP4 tile shape leaves decode dominated by metadata/tile
    # overhead; this shape is faster for DSV4 small-batch decode on A100.
    opt_flags.update_opt_flags_constraints(
        {"is_persistent": False, "block_m": 64, "block_k": 64, "num_stages": 4}
    )
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
        # Triton 3.6 defaults sm80 MXFP4 values to StridedLayout, which lowers
        # through generic tl.dot_scaled emulation. GPT-OSS uses the Hopper value
        # swizzle even on A100 so _matmul_ogs takes the explicit
        # mxfp4_to_bf16_triton + tl.dot path, which is materially faster. The
        # Hopper scale swizzle avoids strided scale loads in that path.
        value_layout = HopperMXValueLayout
        value_layout_opts = {"mx_axis": 1}
        scale_layout = HopperMXScaleLayout
        scale_layout_opts = {"mx_axis": 1, "num_warps": 4}

    w_tri = convert_layout(
        wrap_torch_tensor(weight.view(torch.uint8).transpose(-2, -1), dtype=FP4),
        value_layout,
        **value_layout_opts,
    )
    scale_tri = convert_layout(
        wrap_torch_tensor(scale_u8.view(torch.float8_e8m0fnu).transpose(-2, -1)),
        scale_layout,
        **scale_layout_opts,
    )
    return w_tri, InFlexData(), scale_tri


def prepare_mxfp4_moe_ogs(layer: torch.nn.Module, num_warps: int = 8) -> None:
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    w13_scale = _bind_scale_parameter(layer, "w13_weight_scale_inv")
    w2_scale = _bind_scale_parameter(layer, "w2_weight_scale_inv")

    w13_tri, w13_flex, w13_scale_tri = _swizzle_mxfp4_for_ogs(
        _interleave_gate_up_rows(layer.w13_weight.data),
        _interleave_gate_up_rows(w13_scale.data),
        num_warps,
    )
    w2_tri, w2_flex, w2_scale_tri = _swizzle_mxfp4_for_ogs(
        layer.w2_weight.data, w2_scale.data, num_warps
    )

    layer._dsv4_mxfp4_ogs_weights = OgsMxfp4Weights(
        w13=w13_tri,
        w2=w2_tri,
        w13_precision_config=PrecisionConfig(
            weight_scale=w13_scale_tri, flex_ctx=FlexCtx(rhs_data=w13_flex)
        ),
        w2_precision_config=PrecisionConfig(
            weight_scale=w2_scale_tri, flex_ctx=FlexCtx(rhs_data=w2_flex)
        ),
    )


def _make_routing_data(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
):
    patch_oai_bitmatrix_metadata()

    from triton_kernels.tensor import BIT, Bitmatrix, SparseMatrix
    from triton_kernels.tensor import make_ragged_tensor_metadata
    from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx

    topk_ids_i16 = topk_ids.to(torch.int16)
    topk_weights_bf16 = topk_weights.to(torch.bfloat16)
    n_rows, topk = topk_ids_i16.shape
    block_m = 512
    block_k = 32
    bm_cols = triton.cdiv(num_local_experts, block_k)
    bitmatrix_raw = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )
    _pack_bitmatrix_kernel[(triton.cdiv(n_rows, block_m),)](
        bitmatrix_raw,
        topk_ids_i16,
        n_rows,
        bm_cols,
        topk,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
    )
    bitmatrix = Bitmatrix(
        bitmatrix_raw,
        dtype=BIT,
        shape=[n_rows, bm_cols * 32],
        shape_max=[n_rows, None],
    )
    topk_weights_bf16 = torch.where(
        topk_ids_i16 == -1,
        torch.full((), -1.0, dtype=torch.bfloat16, device=topk_weights.device),
        topk_weights_bf16,
    )
    sparse = SparseMatrix(indx=topk_ids_i16, vals=topk_weights_bf16, mask=bitmatrix)
    dispatch_idx = sparse.mask_metadata.row_sorted_indx
    combine_idx = sparse.mask_metadata.col_sorted_indx
    ragged = make_ragged_tensor_metadata(
        sparse.mask_metadata.col_sum, dispatch_idx.shape[0]
    )
    gate_scal = sparse.vals.flatten()[combine_idx]
    routing_data = RoutingData(
        gate_scal,
        ragged.slice_sizes,
        num_local_experts,
        topk,
        ragged,
    )
    return routing_data, GatherIndx(combine_idx, dispatch_idx), ScatterIndx(
        dispatch_idx, combine_idx
    )


def mxfp4_moe_forward_ogs(
    hidden_states: torch.Tensor,
    weights: OgsMxfp4Weights,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    routed_scaling_factor: float | None = None,
    clamp_limit: float | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, matmul_ogs

    if hidden_states.dtype != torch.bfloat16:
        hidden_states = hidden_states.to(torch.bfloat16)

    m = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    routing_data, gather_idx, scatter_idx = _make_routing_data(
        topk_ids, topk_weights, weights.w13.shape[0]
    )

    intermediate = torch.empty(
        (1, m * topk, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = torch.empty(
        (1, m, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
    )
    act = FusedActivation(
        FnSpecs(
            "dsv4_swiglu",
            _dsv4_swiglu_fn,
            ("alpha", "limit"),
            reduction_n=2,
        ),
        (1.0, clamp_limit),
    )
    gammas = routing_data.gate_scal
    matmul_ogs(
        hidden_states,
        weights.w13,
        None,
        routing_data,
        gather_indx=gather_idx,
        precision_config=weights.w13_precision_config,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act,
        y=intermediate,
    )
    matmul_ogs(
        intermediate.view(m * topk, intermediate_size),
        weights.w2,
        None,
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=weights.w2_precision_config,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output,
    )
    output = output.view(m, hidden_size)
    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        output.mul_(routed_scaling_factor)
    return output
