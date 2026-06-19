#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPERT_WEIGHT_RE = re.compile(
    r"^(?:layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.w[123]\.weight$"
)
FP8_SCALE_SUFFIX = ".scale"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepSeek-V4-Flash mixed FP8/MXFP4 checkpoint to BF16 "
            "for non-routed-expert tensors while preserving routed MoE "
            "experts in their original packed MXFP4 layout."
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


def update_config(output_dir: Path):
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    qconfig = dict(config.get("quantization_config") or {})
    qconfig["quant_method"] = "fp8"
    qconfig["activation_scheme"] = qconfig.get("activation_scheme", "dynamic")
    qconfig["fmt"] = qconfig.get("fmt", "e4m3")
    qconfig["scale_fmt"] = qconfig.get("scale_fmt", "ue8m0")
    qconfig["weight_block_size"] = qconfig.get("weight_block_size", [128, 128])
    qconfig["ignored_layers"] = [
        "embed_tokens",
        "lm_head",
        "norm",
        "self_attn",
        "input_layernorm",
        "post_attention_layernorm",
        "gate",
        "shared_experts",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    ]
    qconfig["modules_to_not_convert"] = qconfig["ignored_layers"]

    config["quantization_config"] = qconfig
    config["torch_dtype"] = "bfloat16"
    config["sglang_conversion"] = {
        "source": "DeepSeek-V4-Flash",
        "dense_fp8_dequantized_to": "bfloat16",
        "routed_experts": "preserved_packed_mxfp4_e2m1_group_32",
        "runtime_expectation": (
            "SGLang FP8 quant config with DeepSeek V4 auto-detected "
            "is_fp4_experts=True; non-expert modules are listed in "
            "ignored_layers and load as BF16."
        ),
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

    new_weight_map = {}
    total_size = 0
    preserved_expert_tensors = 0
    dequantized_tensors = 0

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

                if EXPERT_WEIGHT_RE.match(key):
                    scale_key = key.removesuffix(".weight") + FP8_SCALE_SUFFIX
                    if scale_key not in key_set:
                        raise RuntimeError(f"Missing MXFP4 scale for {key}")
                    tensors[key] = tensor
                    tensors[scale_key] = f.get_tensor(scale_key)
                    skip_scales.add(scale_key)
                    preserved_expert_tensors += 2
                    continue

                if key.endswith(FP8_SCALE_SUFFIX):
                    weight_key = key.removesuffix(FP8_SCALE_SUFFIX) + ".weight"
                    if EXPERT_WEIGHT_RE.match(weight_key):
                        continue
                    if weight_key in key_set and is_float8_tensor(f.get_tensor(weight_key)):
                        continue

                if key.endswith(".weight") and is_float8_tensor(tensor):
                    scale_key = key.removesuffix(".weight") + FP8_SCALE_SUFFIX
                    if scale_key in key_set:
                        tensors[key] = fp8_block_dequant(tensor, f.get_tensor(scale_key))
                        skip_scales.add(scale_key)
                    else:
                        tensors[key] = tensor.to(torch.bfloat16)
                    dequantized_tensors += 1
                    continue

                if is_float8_tensor(tensor):
                    tensors[key] = tensor.to(torch.bfloat16)
                    dequantized_tensors += 1
                else:
                    tensors[key] = tensor

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

    update_config(output_dir)

    print(
        "done: "
        f"{output_dir} "
        f"(preserved_expert_tensors={preserved_expert_tensors}, "
        f"dequantized_tensors={dequantized_tensors})",
        flush=True,
    )


if __name__ == "__main__":
    main()
