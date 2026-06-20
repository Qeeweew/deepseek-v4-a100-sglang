#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30002}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SGLANG_ROOT="${SGLANG_ROOT:-${PROJECT_ROOT}/../sglang}"
SGLANG_LOAD_QUANTIZATION="${SGLANG_LOAD_QUANTIZATION:-fp8}"
SGLANG_MOE_ENTRY_BACKEND="${SGLANG_MOE_ENTRY_BACKEND:-marlin}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export ENABLE_SGLANG_DSV4_A100_PATCH="${ENABLE_SGLANG_DSV4_A100_PATCH:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"

# A100 does not support the DeepGEMM HC prenorm kernel used by the default DSV4 warmup path.
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-0}"

# The converted checkpoint stores wq_a and wkv separately; keep SGLang from fusing them into wqkv_a.
export SGLANG_OPT_FUSE_WQA_WKV="${SGLANG_OPT_FUSE_WQA_WKV:-0}"

# A100/sm80 cannot compile the DSV4 topk_v2 cluster kernel (__cluster_dims__/this_cluster).
export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-0}"

# A100 cannot run DeepGEMM's FP8 paged MQA logits API used by the C4 indexer.
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="${SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:-1}"
export SGLANG_DSV4_A100_INT8_INDEXER="${SGLANG_DSV4_A100_INT8_INDEXER:-1}"

# Non-routed weights were materialized as BF16; only routed experts remain packed MXFP4.
export SGLANG_DSV4_FP4_EXPERTS="${SGLANG_DSV4_FP4_EXPERTS:-1}"
export SGLANG_OPT_FP8_WO_A_GEMM="${SGLANG_OPT_FP8_WO_A_GEMM:-0}"
export SGLANG_DSV4_MXFP4_MOE_BACKEND="${SGLANG_DSV4_MXFP4_MOE_BACKEND:-mxfp4_int8}"

# These two CLI values are SGLang loader/dispatch entry points, not the final
# routed expert kernel. The A100 monkeypatch replaces Mxfp4MarlinMoEMethod's
# weight preparation and apply path with mxfp4_int8 by default; set
# SGLANG_DSV4_MXFP4_MOE_BACKEND=ogs to compare against the original MXFP4 OGS path.
# Keep these defaults unless the SGLang quantization registration is changed.
exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization "${SGLANG_LOAD_QUANTIZATION}" \
  --moe-runner-backend "${SGLANG_MOE_ENTRY_BACKEND}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --skip-server-warmup
