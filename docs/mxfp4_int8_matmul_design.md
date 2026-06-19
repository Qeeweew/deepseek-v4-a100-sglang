# MXFP4/INT8 量化矩阵乘法设计

本文档说明本项目中 MXFP4/INT8 dense 和 routed MoE GEMM 的设计理念，从
DeepSeek V4 Flash 的 MXFP4 权重格式出发，解释为什么需要近似 repack，以及
当前 CUTLASS/SGLang JIT 实现如何在 A100 上执行。

## 背景

A100 是 SM80 架构，没有 DeepSeek V4 原始 serving 路径依赖的原生 FP4 tensor
core，也不能直接复用面向新架构的 FP8/FP4 kernel。A100 上稳定、可用且吞吐高
的低精度矩阵乘法路径是 INT8 tensor core：

```text
int8 x int8 -> int32 accumulate -> bf16/fp16 output
```

因此这里的目标不是在 kernel 内完整模拟 MXFP4 浮点计算，而是把 DeepSeek V4
的 MXFP4 权重变成适合 INT8 tensor core 消费的形式。

## DeepSeek V4 MXFP4 权重特点

DeepSeek V4 Flash routed expert 权重以 packed MXFP4 E2M1 保存。每个字节保存
两个 4-bit E2M1 code，并且每 32 个 K 元素配一个 UE8M0 scale。逻辑上每个权重
元素可以写成：

```math
w_{n,k} = e2m1(code_{n,k}) \cdot 2^{exp_{n,\lfloor k/32 \rfloor}}
```

其中 `exp` 来自 UE8M0 scale。E2M1 可表示的值很少：

```text
{0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6}
```

实现里使用 doubled integer LUT 表示 E2M1：

```text
q_x2 = {0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}
```

这样 E2M1 的 `0.5` 因子可以放进后续 scale 里处理。

这个格式对存储很紧凑，但对 A100 INT8 GEMM 不直接友好。问题在于 UE8M0 scale
是每个 output channel、每个 K/32 block 一份。如果在 GEMM K loop 中逐 block
做浮点 dequant，就会破坏 INT8 tensor core 的主路径，或者迫使 partial sum 在
K loop 中频繁转换和缩放。

## Repack 的核心思路

repack 的目标是把每个 row 的多份 UE8M0 block scale 转换成：

- 一个 FP32 per-channel scale；
- 每个 K/32 block 一个 2-bit integer shift；
- remapped packed E2M1 code。

运行时 kernel 不再读取原始 UE8M0 scale。主循环只做：

```text
q_i8 = lut_x2[remapped_code] << shift2
```

然后使用 INT8 tensor core 计算：

```text
acc_i32 = act_i8 @ q_i8
out = acc_i32 * activation_scale[token] * channel_scale[channel]
```

设某个 output channel 的最大 block exponent 为 `max_exp`，默认
`headroom_bits=3`。per-channel scale 定义为：

```math
channel\_scale = 2^{max\_exp - headroom\_bits} \cdot 0.5
```

其中 `0.5` 用来补偿 doubled E2M1 LUT。每个 K/32 block 的 exponent 差值为：

```math
delta = max\_exp - exp_{block}
```

2-bit shift 为：

```math
shift2 = clamp(headroom\_bits - delta, 0, 3)
```

如果某个 block 的 scale 与 row 最大 scale 很接近，`shift2` 能精确表示相对比例。
如果差距超过 2-bit shift 的表示范围，则把剩余比例折进 E2M1 code，并选择最近的
E2M1 code。

## 这不是无损转换

MXFP4 原始格式有 per-32 block 的 UE8M0 scale，而 runtime 格式只有 row scale
加 2-bit shift。对于 row 内 scale span 很大的情况，某些 block 不能被精确表示。
repack 会把目标值四舍五入到最近的 E2M1 code。

DeepSeek V4 的 MXFP4 权重并不是任意 scale 分布。为了让 MXFP4 权重能够无损转换到
FP8 表示，同一个 output channel 内的 block scale 差距被限制在 64 倍以内，也就是
exponent span 不超过 6。这个约束给 INT8 repack 提供了基础：默认
`headroom_bits=3` 时，较大的 block 可以通过 2-bit shift 表示，较小的 block 即使
需要折进 E2M1 code，也不会落在完全失控的 scale 区间。

最近 E2M1 code 的选择基于 doubled magnitude 阈值：

```text
0, 1, 2, 3, 4, 6, 8, 12
```

Triton repack kernel 中对应逻辑是按边界选择 magnitude：

```text
<= 0.5  -> 0
<= 1.5  -> 1
<= 2.5  -> 2
<= 3.5  -> 3
<= 5.0  -> 4
<= 7.0  -> 6
<= 10.0 -> 8
else    -> 12
```

符号位从原始 code 保留，原始 zero 仍映射为 zero。实验仓库中的统计脚本进一步
说明，实际 routed expert 权重的 scale 通常比 64 倍上界集中得多。

统计使用实验仓库脚本：

```bash
python /workspace/experiments/mxfp4_int8_design/scripts/analyze_mxfp4_weight_remap.py \
  --headroom-bits 3 \
  --output-dir /workspace/experiments/mxfp4_int8_design/results/headroom3
```

统计口径：

- 模型：`/ssd1/models/DeepSeek-V4-Flash-MoE-MXFP4-BF16`
- 层：`0, 10, 20, 30, 42`
- 每层采样 expert 数：`32`
- projection：`w1, w2, w3`
- 总矩阵数：`5 layers * 32 experts * 3 projections = 480`
- `row span`：同一个 output channel 内，所有 K/32 block 的 `max(exp) - min(exp)`。
- overflow：按 repack 后的目标 INT8 值统计是否超过 `[-128, 127]`。
- exact nonzero：只在原始非零权重上统计 repack 前后的数值是否完全一致。

`headroom_bits=3` 的采样结果：

| 指标 | 数值 | 说明 |
|---|---:|---|
| sampled matrices | 480 | 5 层、每层 32 个 expert、3 个 projection |
| overflow count | 0 | 没有观察到 INT8 溢出 |
| max matrix overflow rate | 0 | 任一矩阵内的最大溢出比例 |
| worst nonzero exact rate | 0.9992876 | 最差矩阵的非零权重精确保留比例 |
| worst mean relative error, nonzero | 0.0005060 | 最差矩阵的非零权重平均相对误差 |
| max remapped zero rate | 0.1551187 | remap 后为 zero 的最大比例 |
| matrix max scale exponent p50 | -5 | 每个矩阵先取 row max exponent，再做分位数 |
| matrix max scale exponent p90 | -4 | 同上 |
| matrix max scale exponent p99 | -2 | 同上 |
| matrix max row span p50 | 2 | 每个矩阵先取 row span 最大值，再做分位数 |
| matrix max row span p90 | 3 | 同上 |
| matrix max row span p99 | 4.21 | 同上 |
| worst matrix row span | 5 | 采样中最差 row span |

最差采样矩阵是 `layer=42, expert=14, projection=w2`：

| 指标 | 数值 |
|---|---:|
| exact_rate_nonzero | 0.9992876 |
| row_span_max | 5 |
| row_span_p99 | 4 |
| mean_rel_err_nonzero | 0.0005060 |

因此 `headroom_bits=3` 在采样 routed expert 权重上没有观察到 INT8 overflow，大部分
非零权重可以精确保留。这是 DeepSeek V4 权重格式约束和实际 scale 分布共同作用的
结果，不是任意 MXFP4 权重都自动成立。它仍然是近似格式，不应该描述为 bit-exact
MXFP4 dequant。

## 权重布局

repack 后的权重不持久展开成完整 INT8。持久格式仍然主要是 packed 4-bit code：

```text
b_mxfp4:         uint8, [E, K/32, ceil(N/8), 128]
b_shift2:        uint8, [E, ceil((K/32)/4), ceil(N/8), 8]
b_channel_scale: fp32,  [E, N]
```

dense 权重使用去掉 expert 维度后的同类布局：

```text
b_mxfp4:         uint8, [K/32, ceil(N/8), 128]
b_shift2:        uint8, [ceil((K/32)/4), ceil(N/8), 8]
b_channel_scale: fp32,  [N]
```

`b_mxfp4` 以 8 个 output channel 为一组，按 MMA 读取需要的 lane 顺序重排。每个
K/32 block 的 16 个原始 packed bytes 会被排成更适合 warp fragment decode 的
128-byte tile。`b_shift2` 每个 byte 存 4 个 K/32 block 的 2-bit shift，并按
`ceil(N/8)` 和 lane 内 row8 排布。

模型加载时，原始 packed MXFP4 weight 和 UE8M0 scale 只用于 repack。repack 完成后，
SGLang layer 上的原始权重和 scale 参数会替换为空 tensor，以节省显存。

## 激活量化

激活侧使用 per-token INT8 quantization。对每个 token row：

```math
activation\_scale_m = \frac{\max_k |x_{m,k}|}{127}
```

```math
act\_i8_{m,k} = round(x_{m,k} / activation\_scale_m)
```

zero row 使用 scale `1.0`。当前实现将 per-token quantization 保持为独立 Triton
kernel，便于在 dense、MoE W13、MoE W2 之间复用，也方便独立 benchmark。GEMM
epilogue 读取 `activation_scale[m]` 和 `channel_scale[n]`，把 INT32 accumulator
写回 BF16。

## CUTLASS 主循环设计

当前 dense/MoE 算子都通过 `dsv4_a100_patch.sglang_jit_patches` 接入 SGLang JIT。
核心 CUDA 头文件在：

```text
dsv4_a100_patch/sglang_jit_patches/csrc/gemm/mxfp4_int8/
```

核心 mainloop 基于 CUTLASS SM80 INT8 tensor core。A operand 是 row-major INT8
激活，B operand 在持久存储中仍是 packed MXFP4。kernel 做以下事情：

1. 使用 `cp.async` 把 A tile、packed B tile、shift2 tile staged 到 shared memory。
2. 在 warp fragment load 阶段解码 packed B：

   ```text
   q_i8 = lookup_mxfp4_x2_byte_perm(packed_nibble, shift2)
   ```

3. 调用 SM80 INT8 MMA，累加到 INT32 fragment。
4. epilogue 应用 per-token activation scale 和 per-channel weight scale。
5. 输出 BF16。

这个设计把浮点 scale 从 K loop 中移走。K loop 内只保留 integer nibble decode、
2-bit shift 和 INT8 MMA，避免每个 K/32 block 做浮点 dequant。

## Dense GEMM 路径

dense JIT wrapper 以 `K` 和 `N` 作为模板参数编译：

```text
Mxfp4Int8DenseGemm<K, N>
```

`M` 是 runtime shape。dense 路径包含两类调度：

- 大 M：使用 fused BF16 epilogue kernel；大 batch 下可以接近 A100 INT8 tensor
  core 的高吞吐区间。
- 小 M：使用 small-M/on-demand B decode 和 split-K/reduce 路径，避免 M 很小时 CTA
  数不足。

small-M split-K 的策略和实验实现保持一致：根据实际 tile count 和 `N` 判断是否
需要 split-K，而不是只看 M。对于 `M<=128`，Python wrapper 会准备 INT32 partial
buffer，CUTLASS main kernel 写 partial，reduce kernel 再缩放并写 BF16。

## MoE Grouped GEMM 路径

MoE 路径编译时把关键 serving shape 作为模板参数：

```text
Mxfp4Int8MoeGemm<
  HiddenSize,
  IntermediateSize,
  TopK,
  BlockM,
  BlockN,
  SourceRowsAreSlots
>
```

其中 `SourceRowsAreSlots=false` 对应 W13，输入来自原始 token row，输出是 routed
slot row；`SourceRowsAreSlots=true` 对应 W2，输入来自 routed slot row，输出回
token row 并执行 top-k weighted reduce。

MoE kernel 直接消费 SGLang `moe_align_block_size` 产生的：

```text
sorted_token_ids
expert_ids
num_tokens_post_padded
```

CUTLASS launch 按实际 `num_tokens_post_padded[0]` 计算 M tile 数。`sorted_token_ids`
和 `expert_ids` 只提供预分配容量，不能用二者互相推导语义。padding slot 在 kernel
内通过 `num_valid_tokens` 和 routing id 过滤。

MoE 的 `BlockM`/`BlockN` 会影响小 batch 下的利用率。本项目使用静态表和离线
autotune 结果选择 tile，并给小 batch 的 `BlockM` 设置上界：

```text
BlockM <= max(16, next_power_of_2(ceil(batch * topk / num_experts)))
```

这样当平均每个 expert 的激活 token 很少时，优先使用 `BlockM=16`，避免大 M tile
造成严重 padding。

## JIT 和 CUDA Graph

本项目不暴露独立 `mxfp4_int8` CUDA extension。dense 和 MoE GEMM 都通过 SGLang
JIT loader 编译，CUTLASS 头文件从 SGLang runtime 环境解析。

JIT module 初始化时会调用对应 `init_*_attrs`，设置 CUTLASS kernel 的 dynamic
shared memory 属性。这个初始化必须发生在 CUDA graph capture 之前；实际 replay
期间只执行已经初始化好的 kernel launch。

`K`、`N`、`TopK`、`BlockM`、`BlockN` 等作为模板参数后，可以减少 runtime 分支和
shape 判断，但会增加编译组合。因此当前没有把 expert 数 `E` 模板化。`E` 主要影响
routing metadata 和权重外层 stride，收益小，而模板化会显著增加编译缓存数量。

## 当前局限

- repack 近似不是原始 MXFP4 的无损表示，精度需要以模型级评测为准。
- per-token activation quantization 有额外 kernel 开销，小 batch 下占比明显。
- MoE grouped GEMM 受 routing 分布、expert padding、W13/W2 两次 GEMM、SwiGLU、
  top-k reduce 影响，不能用 dense 大 M 吞吐直接推断端到端表现。
- 小 M 的性能依赖 tile 和 split-K 策略，需要结合实际 TP、top-k、expert 数和 batch
  分布做 autotune。

## 相关文件

- `dsv4_a100_patch/triton_kernels/mxfp4_int8_moe.py`：repack、per-token quant、
  dense/MoE Python 集成。
- `dsv4_a100_patch/sglang_jit_patches/mxfp4_int8_dense.py`：dense SGLang JIT wrapper。
- `dsv4_a100_patch/sglang_jit_patches/mxfp4_int8_moe.py`：MoE SGLang JIT wrapper。
- `dsv4_a100_patch/sglang_jit_patches/csrc/gemm/mxfp4_int8/mxfp4_cutlass_core.cuh`：
  shared packed-B decode 和 CUTLASS mainloop。
- `dsv4_a100_patch/sglang_jit_patches/csrc/gemm/mxfp4_int8/mxfp4_int8_dense_entry.cuh`：
  dense launch/reduce 入口。
- `dsv4_a100_patch/sglang_jit_patches/csrc/gemm/mxfp4_int8/mxfp4_int8_moe_entry.cuh`：
  grouped MoE launch/reduce 入口。
