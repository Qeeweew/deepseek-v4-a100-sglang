# Model Conversion

The patch expects a converted DeepSeek V4 Flash checkpoint. Two conversion paths
are provided.

Both scripts require explicit `--input` and `--output` paths. They do not carry
machine-local defaults, so the generated checkpoints can be reproduced outside
the original development workspace.

## MXFP4/BF16 Path

```bash
python scripts/convert_deepseek_v4_flash_moe_mxfp4_bf16.py \
  --input /path/to/DeepSeek-V4-Flash \
  --output /path/to/DeepSeek-V4-Flash-MoE-MXFP4-BF16
```

This conversion:

- materializes non-routed FP8 tensors as BF16;
- preserves routed MoE expert weights in packed MXFP4 format;
- preserves routed expert UE8M0 scale tensors for load-time repack;
- updates `config.json` so SGLang still enters through the FP8 quantization
  loader path.

At runtime, the SGLang patch repacks routed expert weights into the MXFP4/INT8
kernel format and removes the original packed weight/scale tensors from the
loaded layer objects to reduce memory usage.

The output checkpoint intentionally still contains the routed MXFP4 scale
tensors. They are needed once during model loading because the load-time repack
uses the original UE8M0 block scale to choose row scales, 2-bit shifts, and
nearest E2M1 codes. After repack, the in-memory layer parameters are replaced
with empty tensors.

## INT4/BF16 Path

```bash
python scripts/convert_deepseek_v4_flash_moe_int4.py \
  --input /path/to/DeepSeek-V4-Flash \
  --output /path/to/DeepSeek-V4-Flash-MoE-INT4-G32-BF16 \
  --device cuda:0
```

This conversion:

- materializes non-routed FP8 tensors as BF16;
- dequantizes routed MXFP4 experts to BF16;
- requantizes routed experts to compressed-tensors INT4 group-size 32;
- updates `config.json` for `--quantization compressed-tensors`.

Optional INT4 observer arguments:

```bash
--weight-observer memoryless_mse
--weight-observer-kwargs '{"norm": 2.0}'
```

## Dependencies

The MXFP4 conversion needs PyTorch and safetensors.

The INT4 conversion additionally needs compressed-tensors, llmcompressor, and
the MXFP4 dequant helper from `auto_round_extension`.

The default MXFP4 serving path loads dense and routed expert GEMMs from this
package's SGLang JIT headers and uses the in-package Triton activation
quantizer.
