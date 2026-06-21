# Indexer Query Token CP

本文档说明 DeepSeek V4 A100 patch 中 C4 indexer 的 query-token 并行优化。

## 背景

DeepSeek V4 的 `compress_ratio=4` 层需要先运行 C4 indexer，为每个 query
token 选择 sparse attention 使用的 C4 page indices。原始路径中，每个 TP rank
都会对完整 query token 序列重复执行：

```text
indexer q projection / quantization
int8 paged MQA logits
top-k transform
```

这里的 C4 indexer 权重是 replicated 的，不像主 attention query heads 那样按 TP
切分。因此 prefill 阶段可以把 query token 维度分给不同 TP rank 计算，再把
结果 all-gather 回完整 token 顺序。

## 并行策略

当前实现只切 query token 维度，不切 history/KV 维度：

```text
T = total query tokens
W = attention TP size
chunk = ceil(T / W)

rank r 处理 token [r * chunk, (r + 1) * chunk)
```

每个 rank 对本地 token chunk 仍然读取完整 C4 history，所以每个 token 的 top-k
是在完整历史上计算的，不需要跨 rank merge top-k。

当 `T` 不能被 `W` 整除时，本地 chunk 会 padding 到相同长度。padding 行的
`c4_seq_lens` 置为 0，top-k 输出只用于对齐 all-gather shape，最终会被丢弃。

## 执行路径

开启 `SGLANG_DSV4_INDEXER_QUERY_CP_PREFILL=1` 后，prefill/extend 的 C4 indexer
会走以下路径：

1. 每个 TP rank 选择自己的连续 query token chunk。
2. 对本地 chunk 运行 int8 paged MQA logits。
3. 对本地 logits 运行 `topk_transform_512`。
4. 使用 attention TP group 做 `all_gather_into_tensor`。
5. 由于使用连续分块，gather 后前 `T` 行已经是全局 token 顺序，直接写回
   `core_metadata.c4_sparse_page_indices`。

Decode、mixed mode、top-k v2、indexer capture 和 CUDA graph capture 会回退到
原始完整 token 路径。

## 开关

MXFP4 启动脚本默认开启：

```bash
SGLANG_DSV4_INDEXER_QUERY_CP_PREFILL=1
```

可显式设置为 `0` 回退到原始 indexer 路径。

## 256K Prefill 对比

测试配置：

```text
model: DeepSeek-V4-Flash-MoE-MXFP4-BF16
TP: 8
input tokens: 262144
output tokens: 8
flush cache: true
```

| indexer query CP | TTFT | E2E | input throughput |
|---|---:|---:|---:|
| on | 37148.94 ms | 37285.84 ms | 7027.06 tok/s |
| off | 45955.15 ms | 46091.76 ms | 5685.11 tok/s |

在这组单请求 256K prefill 测试中，开启 query-token CP 后 TTFT 降低约
19.16%，input throughput 提升约 23.60%。

profile trace 中可以看到：

```text
dsv4_a100_patch::c4_indexer_query_cp_select
dsv4_a100_patch::c4_indexer_query_cp_all_gather
dsv4_a100_patch::c4_indexer_query_cp_rerange
nccl:_all_gather_base
```

这些事件证明请求实际走到了 query-token CP 路径。
