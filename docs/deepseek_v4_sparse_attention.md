# DeepSeek V4 Sparse Attention Kernel 设计说明

本文档说明 `/workspace/monkeypatch/triton_kernels/attention.py` 中 A100
BF16 sparse attention kernel 的设计。这里的 kernel 使用 mode-neutral 命名，
不再使用 `decode` 命名，因为同一个 attention kernel 同时服务 decode 和
prefill/extend。两者的区别在 metadata 生成方式，而不是 sparse attention
本身的数学计算。

## 整体执行路径

patched `DeepseekV4AttnBackend.forward` 在 BF16 KV cache 路径下的主要流程是：

1. 如果 `save_kv_cache=True`，先把当前 token 的 K/V 写入 KV cache。
2. 从 `self.forward_metadata.core_attn_metadata` 读取 attention metadata。
3. 把 query 规整成：

   $$Q \in \mathbb{R}^{T \times H_q \times D}$$

   其中 $T$ 是 `total_tokens`，$H_q$ 是本 TP rank 上的 query heads，
   $D$ 是 head dim。

4. 准备 SWA sparse metadata：

   ```text
   swa_page_indices: [T, K_swa]
   swa_topk_lengths: [T]
   ```

5. 当 `compress_ratio=4` 或 `compress_ratio=128` 时，再准备额外 sparse
   source：

   ```text
   c4_sparse_page_indices / c4_sparse_topk_lengths
   c128_page_indices      / c128_topk_lengths_clamp1
   ```

6. 根据 source 数量调用：

   ```text
   direct_sparse_attention       # 单 source
   direct_dual_sparse_attention  # SWA + C4/C128 双 source
   ```

`direct_*` 路径把 sparse gather、QK、online softmax、PV 融合在 Triton kernel
里完成。fallback 路径则会先 materialize gathered KV，再调用 unified attention。

## Decode 和 Prefill 为什么可以共用 Kernel

attention kernel 不需要知道当前是 decode 还是 prefill。它只消费逐行 sparse
metadata：

$$
Q \in \mathbb{R}^{T \times H_q \times D}
$$

$$
\text{Indices} \in \mathbb{Z}^{T \times K}, \qquad
\text{Lengths} \in \mathbb{Z}^{T}
$$

decode 阶段：

```text
T = batch_size
每个 request 对应一个当前 decode token
Lengths[t] 表示该 token 能看到的 sparse KV row 数
```

prefill/extend 阶段：

```text
T = sum(extend_seq_lens)
每个 prompt/extend token 对应一行 query
Lengths[t] 表示该 prefill token 在 causal 约束下能看到的 sparse KV row 数
```

也就是说，causal mask 没有放在 attention kernel 内部处理，而是在 metadata
阶段就已经变成了每个 query row 的 `Indices` 和 `Lengths`。kernel 的任务只是：

```text
对第 t 行 query，只访问 Indices[t, :Lengths[t]] 指定的 KV rows
```

因此 decode 和 prefill 的 sparse attention 数学形式完全一致。

原始 SGLang 的 metadata 路径也是这个结构：

```text
init_forward_metadata_decode:
  seq_lens_casual = 当前每个 request 的 seq_len
  req_pool_indices_repeated = req_pool_indices

init_forward_metadata_prefill:
  expand_prefill_casually(...)
  seq_lens_casual = 每个 prefill query row 对应的 causal seq_len
  req_pool_indices_repeated = 每个 prefill query row 对应的 request id
```

经过这个展开之后，decode 和 prefill 都变成同一个 row-wise sparse attention 问题。

## Attention 数学

对每个 query row $t$ 和 head $h$，单 source sparse attention 是：

$$
s_j = \langle Q_{t,h}, K_{\text{idx}_{t,j}} \rangle \cdot \alpha,
\qquad 0 \le j < L_t
$$

其中：

$$
\alpha = \text{sm\_scale}, \qquad
L_t = \text{Lengths}[t]
$$

softmax 权重为：

$$
p_j = \frac{\exp(s_j)}{\sum_{m=0}^{L_t-1}\exp(s_m)}
$$

输出为：

$$
O_{t,h} = \sum_{j=0}^{L_t-1} p_j V_{\text{idx}_{t,j}}
$$

LSE 为：

$$
\text{LSE}_{t,h} =
\log\left(\sum_{j=0}^{L_t-1}\exp(s_j)\right)
$$

双 source 版本把 SWA 和 C4/C128 两组 sparse rows 拼成一个逻辑上的 sparse
集合。设：

$$
L_t = L^{(0)}_t + L^{(1)}_t
$$

则 softmax 在两个 source 的所有有效 score 上统一归一化：

$$
p_j =
\frac{\exp(s_j)}
{\sum_{m=0}^{L^{(0)}_t-1}\exp(s^{(0)}_m)
 + \sum_{n=0}^{L^{(1)}_t-1}\exp(s^{(1)}_n)}
$$

如果启用 attention sink，分母里还会加入 sink 项：

$$
\text{denom} =
\sum_j \exp(s_j - m) + \exp(s_{\text{sink}} - m)
$$

其中 $m$ 是 online softmax 的 running max。

DSV4 这条 BF16 path 中 K/V 使用同一份 cache row，因此 kernel 从同一个 BF16
row 读取 K 和 V。当前实现支持：

$$
D_{qk} \le 512, \qquad D_v \le 512
$$

内部按 `BLOCK_D=128` 分成最多 4 个 chunk。

## 计算量口径

attention 的主要 dot-product FLOPs 是 QK 和 PV：

$$
\text{FLOPs}_{QK}
= 2 \cdot T \cdot H_q \cdot K_{\text{total}} \cdot D_{qk}
$$

$$
\text{FLOPs}_{PV}
= 2 \cdot T \cdot H_q \cdot K_{\text{total}} \cdot D_v
$$

总 dot FLOPs：

$$
\text{FLOPs}_{\text{dot}}
= 2 T H_q K_{\text{total}} (D_{qk} + D_v)
$$

当 $D_{qk}=D_v=512$ 时：

$$
\text{FLOPs}_{\text{dot}}
= 4 \cdot T \cdot H_q \cdot K_{\text{total}} \cdot 512
$$

softmax、mask、rescale 和 LSE 的标量操作相对 dot-product 较小，但仍在同一个
program 内执行，会影响 latency。性能报告里如果只写 TFLOPS，需要明确说明是否
只统计 QK+PV。

## Triton Non-SplitK Kernel

双 source non-splitK kernel 是：

```text
_direct_dual_sparse_attention_kernel
```

launch grid：

$$
\text{grid} =
\left(
\left\lceil \frac{H_q}{\text{BLOCK\_H}} \right\rceil,
T
\right)
$$

每个 Triton program 负责：

```text
一个 query row t
一个 head block
```

program id 映射：

```text
pid_h = tl.program_id(0)
pid_t = tl.program_id(1)
```

kernel 内部流程：

1. 加载 query tile：

   $$
   Q_{t, h:h+\text{BLOCK\_H}, :}
   $$

2. 对两个 source 循环：

   ```text
   source 0: SWA cache
   source 1: C4 或 C128 compressed cache
   ```

3. 每个 source 内按 `BLOCK_N` 遍历 sparse indices。
4. 用 runtime length 和 index range 生成 valid mask：

   ```text
   valid = offs_n < Lengths[t]
           and row_idx >= 0
           and row_idx < kv_rows
   ```

5. 用 `tl.dot` 计算 QK。
6. 用 online softmax 更新：

   $$
   m_{\text{new}} = \max(m_{\text{old}}, \max(s))
   $$

   $$
   l_{\text{new}} =
   l_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}})
   + \sum_j \exp(s_j - m_{\text{new}})
   $$

7. 用 `tl.dot` 计算 PV 并更新 accumulator：

   $$
   A_{\text{new}} =
   A_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}})
   + \sum_j \exp(s_j - m_{\text{new}}) V_j
   $$

8. 最后归一化并写回：

   $$
   O = \frac{A}{l}
   $$

   同时写出 LSE。

单 source kernel `_direct_sparse_attention_kernel` 是相同设计，只是没有第二个
source 循环。

## Split-K Variant

可选 split-K 路径由两个 kernel 组成：

```text
_direct_dual_sparse_attention_splitk_kernel
_direct_dual_sparse_attention_combine_kernel
```

split-K kernel 的 grid：

$$
\text{grid} =
\left(
\left\lceil \frac{H_q}{\text{BLOCK\_H}} \right\rceil,
T,
\text{split\_k}
\right)
$$

每个 split 只处理 sparse topk 的一个区间，并写出 partial result：

$$
\text{PartialOutput}
\in \mathbb{R}^{S \times T \times H_q \times D_v}
$$

$$
\text{PartialLSE}
\in \mathbb{R}^{S \times T \times H_q}
$$

combine kernel 使用 log-sum-exp rescale 合并 partial results。设第 $s$ 个
split 的 partial LSE 是 $\ell_s$，则：

$$
m = \max_s \ell_s
$$

$$
w_s = \exp(\ell_s - m)
$$

$$
O =
\frac{\sum_s w_s O_s}
{\sum_s w_s}
$$

如果有 attention sink，则分母再加入 sink 项。

环境变量：

```text
SGLANG_DSV4_A100_DUAL_SPLITK=1
```

会强制走 split-K。否则由 `_should_use_dual_splitk` 按 shape 选择。当前 A100
TP8 常见形状 $H_q=64$ 下，heuristic 默认不使用 split-K，因为 decode 和
prefill 单测都显示 non-splitK 更快。

## CUDA Graph 约束

CUDA Graph replay 要求 capture 和 replay 的 launch shape 稳定。因此 kernel
grid 只依赖 shape/bucket：

```text
T
H_q
K_total
```

runtime sparse length 不参与 launch grid，只在 kernel 内作为 mask 使用：

```text
should_compute = n_start < seq_len
valid = offs_n < seq_len and index in range
```

这对 decode 很重要，因为 replay 期间上下文长度会增长；也对 prefill CUDA Graph
重要，因为 prefill graph 是按 `num_tokens` bucket capture 的。只要 tensor
shape 不变，每行的有效 sparse length 可以变化，不需要重新 capture。

## Prefill 单 Kernel 性能

单独 microbench 当前 non-splitK direct dual sparse attention kernel，在 A100 上
使用：

```text
H_q = 64
D_qk = D_v = 512
K_total = 640
```

得到的 QK+PV dot-product 吞吐大约是：

| T | latency | QK+PV TFLOPS |
|---:|---:|---:|
| 1024 | 1.188 ms | 72.28 |
| 4096 | 4.560 ms | 75.35 |
| 8192 | 9.117 ms | 75.37 |
| 16384 | 18.277 ms | 75.20 |

在：

```text
T = 8192
K_total = 1152
```

时约为 77.27 TFLOPS。

这些数字只代表 attention kernel 本身，不包含完整 prefill 路径里的 indexer、
top-k transform、compressor write、MoE、dense GEMM、collective 和 launch
overhead。

## 命名约定

内部 JIT kernel 使用 mode-neutral 名称：

```text
_direct_sparse_attention_kernel
_direct_dual_sparse_attention_kernel
_direct_dual_sparse_attention_splitk_kernel
_direct_dual_sparse_attention_combine_kernel
```

公开 Python wrapper 保持：

```text
direct_sparse_attention
direct_dual_sparse_attention
```

这样 profiler trace 中不会再出现 `decode` kernel 名，避免误解为 prefill
错误地调用了 decode-only kernel。

## Roofline 注意事项

本文档中的 TFLOPS 是基于 shape 的解析计算量：

$$
\text{TFLOPS} =
\frac{\text{FLOPs}_{QK} + \text{FLOPs}_{PV}}
{\text{kernel latency}}
$$

它不能单独证明 kernel 是 memory-bound 或 compute-bound。最终 roofline 判断需要
Nsight Compute counters，例如实际 DRAM bandwidth、tensor pipe utilization、
SM occupancy、memory stall reason 等。
