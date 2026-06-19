#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import torch
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors import __version__ as compressed_tensors_version
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_scale_zp,
    compress_module,
    initialize_quantized_linear,
)
from safetensors import safe_open
from safetensors.torch import save_file


EXPERT_WEIGHT_RE = re.compile(
    r"^(?:layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.w[123]\.weight$"
)
FP8_SCALE_SUFFIX = ".scale"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepSeek-V4-Flash mixed FP8/MXFP4 checkpoint to BF16 dense "
            "weights with routed MoE experts re-quantized to compressed-tensors "
            "INT4 group-size 32 symmetric RTN."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input HF checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for per-tensor quantization, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Only process the first N safetensors shards; for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--weight-observer",
        default=None,
        choices=[
            "memoryless_minmax",
            "memoryless_mse",
        ],
        help=(
            "Optional llmcompressor weight observer. If unset, the "
            "QuantizationArgs default is used."
        ),
    )
    parser.add_argument(
        "--weight-observer-kwargs",
        default=None,
        help=(
            "JSON object passed to QuantizationArgs.observer_kwargs, for example "
            "'{\"norm\": 2.0}'. If unset, llmcompressor observer defaults are used."
        ),
    )
    return parser.parse_args()


def is_float8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu)


def fp8_block_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or scale.ndim != 2:
        return weight.to(torch.bfloat16)

    rows, cols = weight.shape
    s_rows, s_cols = scale.shape
    if rows % s_rows != 0 or cols % s_cols != 0:
        return weight.to(torch.bfloat16)

    block_m = rows // s_rows
    block_n = cols // s_cols
    w = weight.to(torch.bfloat16).reshape(s_rows, block_m, s_cols, block_n)
    s = scale.to(torch.bfloat16).reshape(s_rows, 1, s_cols, 1)
    return (w * s).reshape(rows, cols).contiguous()


def dequant_mxfp4_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from auto_round_extension.vllm_ext.mxfp4_qdq_utils import to_dtype

    # The checkpoint stores two E2M1 FP4 values per byte. Safetensors reports I8,
    # but the byte pattern should be interpreted as unsigned packed nibbles.
    return to_dtype(
        data_lp=weight.contiguous().view(torch.uint8),
        scale_e8m0=scale.contiguous().view(torch.uint8),
        elem_dtype="fp4_e2m1",
        block_size=32,
        target_dtype=torch.bfloat16,
    ).contiguous()


def make_int4_scheme(
    weight_observer: str | None = None,
    weight_observer_kwargs: dict | None = None,
) -> QuantizationScheme:
    weights = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
        symmetric=True,
        dynamic=False,
        scale_dtype=torch.bfloat16,
    )
    if weight_observer is not None:
        weights.observer = weight_observer
    if weight_observer_kwargs is not None:
        weights.observer_kwargs = weight_observer_kwargs

    return QuantizationScheme(
        targets=["Linear"],
        weights=weights,
    )


def quantize_int4(weight: torch.Tensor, scheme: QuantizationScheme, device: str):
    target_device = torch.device(device)
    with torch.no_grad():
        weight = weight.to(target_device, non_blocking=True)
        module = initialize_quantized_linear(weight, scheme, device)
        calibrate_scale_zp(module)
        compress_module(module)
        state_dict = {
            key: value.detach().cpu()
            for key, value in module.state_dict().items()
        }
    if "weight_shape" in state_dict:
        state_dict["weight_shape"] = state_dict["weight_shape"].to(torch.int32)
    del module, weight
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return state_dict


def copy_non_tensor_files(input_dir: Path, output_dir: Path):
    for root, dirs, files in os.walk(input_dir):
        rel_root = Path(root).relative_to(input_dir)
        if ".cache" in rel_root.parts:
            continue
        for directory in dirs:
            if directory == ".cache":
                continue
            (output_dir / rel_root / directory).mkdir(parents=True, exist_ok=True)
        for file_name in files:
            if file_name.endswith(".safetensors") or file_name.endswith(
                ".safetensors.index.json"
            ):
                continue
            src = Path(root) / file_name
            dst = output_dir / rel_root / file_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def make_regex_ignore() -> list[str]:
    return [
        r"re:.*(?:^|\.)(?:embed|embed_tokens|lm_head|head|shared_head)(?:\.|$).*",
        r"re:.*(?:^|\.)(?:norm|input_layernorm|post_attention_layernorm|enorm|hnorm)(?:\.|$).*",
        r"re:.*(?:^|\.)(?:attn|self_attn)(?:\.|$).*",
        r"re:.*(?:^|\.)(?:compressor)(?:\.|$).*",
        r"re:.*(?:^|\.)(?:mlp|ffn)\.gate(?:\.|$).*",
        r"re:.*shared_experts.*",
        r"re:.*(?:^|\.)(?:e_proj|h_proj)(?:\.|$).*",
        r"re:.*(?:^|\.)(?:hc_head|hc_attn|hc_ffn).*",
    ]


def update_config(output_dir: Path, scheme: QuantizationScheme, ignore: list[str]):
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    scheme.format = "pack-quantized"
    ignore = make_regex_ignore()

    qconfig = QuantizationConfig.model_validate(
        {
            "config_groups": {"config_group_0": scheme},
            "ignore": ignore,
            "quantization_status": QuantizationStatus.COMPRESSED,
        }
    )
    qconfig_data = qconfig.model_dump(exclude=[QUANTIZATION_METHOD_NAME, "format"])
    # The original checkpoint is mixed FP8/MXFP4. After conversion, remaining
    # non-expert tensors are materialized as BF16 and routed experts use
    # compressed-tensors INT4.
    config.pop("quantization_config", None)
    config[QUANTIZATION_CONFIG_NAME] = {
        COMPRESSION_VERSION_NAME: compressed_tensors_version,
        QUANTIZATION_METHOD_NAME: "compressed-tensors",
        SPARSITY_CONFIG_NAME: {},
        TRANSFORM_CONFIG_NAME: {},
        "format": "pack-quantized",
        **qconfig_data,
    }
    config["torch_dtype"] = "bfloat16"
    method = "RTN"
    observer = getattr(scheme.weights, "observer", None) if scheme.weights else None
    observer_kwargs = (
        getattr(scheme.weights, "observer_kwargs", None) if scheme.weights else None
    )
    if observer in ("mse", "memoryless_mse"):
        method = "MSE observer"

    config["llmcompressor_conversion"] = {
        "source": "DeepSeek-V4-Flash",
        "dense_fp8_dequantized_to": "bfloat16",
        "routed_experts": "int4_weight_only_group_32_symmetric",
        "scale_dtype": "bfloat16",
        "method": method,
        "weight_observer": observer,
        "weight_observer_kwargs": observer_kwargs or {},
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_non_tensor_files(input_dir, output_dir)

    with open(input_dir / "model.safetensors.index.json") as f:
        index = json.load(f)

    shard_names = sorted(set(index["weight_map"].values()))
    if args.limit_shards is not None:
        shard_names = shard_names[: args.limit_shards]

    weight_observer_kwargs = None
    if args.weight_observer_kwargs is not None:
        weight_observer_kwargs = json.loads(args.weight_observer_kwargs)
        if not isinstance(weight_observer_kwargs, dict):
            raise SystemExit("--weight-observer-kwargs must be a JSON object")

    scheme = make_int4_scheme(args.weight_observer, weight_observer_kwargs)
    new_weight_map = {}
    total_size = 0
    ignored_modules = set()

    for shard_idx, shard_name in enumerate(shard_names, start=1):
        in_path = input_dir / shard_name
        out_path = output_dir / shard_name
        tensors = {}

        print(f"[{shard_idx}/{len(shard_names)}] processing {shard_name}", flush=True)
        with safe_open(str(in_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            key_set = set(keys)
            skip_scales = set()

            for key in keys:
                if key in skip_scales:
                    continue

                tensor = f.get_tensor(key)
                module_name, param_name = key.rsplit(".", 1) if "." in key else ("", key)

                if EXPERT_WEIGHT_RE.match(key):
                    scale_key = key.removesuffix(".weight") + FP8_SCALE_SUFFIX
                    if scale_key not in key_set:
                        raise RuntimeError(f"Missing MXFP4 scale for {key}")
                    dense = dequant_mxfp4_to_bf16(tensor, f.get_tensor(scale_key))
                    compressed = quantize_int4(dense, scheme, args.device)
                    for suffix, value in compressed.items():
                        out_key = f"{module_name}.{suffix}"
                        tensors[out_key] = value
                    skip_scales.add(scale_key)
                    continue

                if key.endswith(FP8_SCALE_SUFFIX):
                    weight_key = key.removesuffix(FP8_SCALE_SUFFIX) + ".weight"
                    if EXPERT_WEIGHT_RE.match(weight_key):
                        # Routed expert MXFP4 scales are consumed when the
                        # matching packed weight is dequantized and re-quantized.
                        continue
                    if weight_key in key_set and is_float8_tensor(f.get_tensor(weight_key)):
                        # Consumed when visiting the corresponding weight.
                        continue

                if key.endswith(".weight") and is_float8_tensor(tensor):
                    scale_key = key.removesuffix(".weight") + FP8_SCALE_SUFFIX
                    if scale_key in key_set:
                        tensors[key] = fp8_block_dequant(tensor, f.get_tensor(scale_key))
                        skip_scales.add(scale_key)
                    else:
                        tensors[key] = tensor.to(torch.bfloat16)
                    ignored_modules.add(module_name)
                    continue

                if is_float8_tensor(tensor):
                    tensors[key] = tensor.to(torch.bfloat16)
                else:
                    tensors[key] = tensor
                    if param_name == "weight" and not EXPERT_WEIGHT_RE.match(key):
                        ignored_modules.add(module_name)

        save_file(tensors, str(out_path))
        for key, value in tensors.items():
            new_weight_map[key] = shard_name
            total_size += value.numel() * value.element_size()

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": new_weight_map},
            f,
            indent=2,
            sort_keys=True,
        )

    # `model_free_ptq` records non-quantized Linear modules in `ignore`; do the
    # same so compressed-tensors only treats routed experts as INT4.
    update_config(output_dir, scheme, sorted(ignored_modules))

    print(f"done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
