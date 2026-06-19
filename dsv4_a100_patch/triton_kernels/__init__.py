import sys
from pathlib import Path

for _path in sys.path:
    _candidate = Path(_path) / "triton_kernels"
    if (
        _candidate.is_dir()
        and _candidate != Path(__file__).resolve().parent
        and (_candidate / "matmul_ogs.py").exists()
    ):
        __path__.append(str(_candidate))
        break

from .attention import direct_dual_sparse_attention, direct_sparse_attention
from .elementwise import (
    compressor_decode_mask_positions,
    compressor_decode_mask_positions_torch,
    compressor_prefill_metadata,
    compressor_prefill_metadata_torch,
    compressor_positions_from_plan,
    compressor_positions_from_plan_torch,
    fused_rope_inplace,
    fused_rope_inplace_torch,
)
from .indexer import (
    bf16_indexer_q,
    bf16_indexer_q_torch,
    bf16_paged_mqa_logits,
    bf16_paged_mqa_logits_torch,
)
from .kv_ops import (
    gather_bf16_kv,
    gather_bf16_kv_into,
    gather_bf16_kv_torch,
    scatter_bf16_rows,
    scatter_bf16_rows_torch,
    trim_and_pad_rows,
    trim_and_pad_rows_torch,
)
from .mxfp4_moe_ogs import mxfp4_moe_forward_ogs, prepare_mxfp4_moe_ogs
from .mxfp4_int8_moe import (
    mxfp4_int8_dense_forward,
    mxfp4_int8_moe_forward,
    prepare_mxfp4_int8_dense_weight,
    prepare_mxfp4_int8_moe,
    quantize_per_token_int8,
    remap_mxfp4_weight_for_int8,
)

__all__ = [
    "bf16_paged_mqa_logits",
    "bf16_paged_mqa_logits_torch",
    "bf16_indexer_q",
    "bf16_indexer_q_torch",
    "compressor_decode_mask_positions",
    "compressor_decode_mask_positions_torch",
    "compressor_prefill_metadata",
    "compressor_prefill_metadata_torch",
    "compressor_positions_from_plan",
    "compressor_positions_from_plan_torch",
    "direct_dual_sparse_attention",
    "direct_sparse_attention",
    "fused_rope_inplace",
    "fused_rope_inplace_torch",
    "gather_bf16_kv",
    "gather_bf16_kv_into",
    "gather_bf16_kv_torch",
    "mxfp4_moe_forward_ogs",
    "mxfp4_int8_dense_forward",
    "mxfp4_int8_moe_forward",
    "prepare_mxfp4_int8_dense_weight",
    "prepare_mxfp4_moe_ogs",
    "prepare_mxfp4_int8_moe",
    "quantize_per_token_int8",
    "remap_mxfp4_weight_for_int8",
    "scatter_bf16_rows",
    "scatter_bf16_rows_torch",
    "trim_and_pad_rows",
    "trim_and_pad_rows_torch",
]
