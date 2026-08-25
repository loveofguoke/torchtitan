# GLM-5 已确认优化算子

本目录只放已经明确属于 GLM-5 完整 DSA 数据流、保持现有组件接口和数学语义、
可以通过 `--override.imports` 显式启用的实现。默认模型仍使用 `model.py` 中的
PyTorch 参考实现；本目录不会被默认导入。

## 当前实现

| 实现 | 替换对象 | 实际作用 | 训练支持 |
| --- | --- | --- | --- |
| `TritonDSAIndexerTopK` | `DSAIndexerTopK.Config` | 融合 index head 点积、缩放、ReLU、head 权重归约和 mask，直接生成 `[Q,K]` score | indexer 冻结，只需 forward |
| `TritonSparseMLA` | `SparseMLA.Config` | 只读取 top-k compressed KV，计算 `[Q,N,S]` attention；Triton forward/backward | dQ、dKV 均支持 |

这里的 `Q` 是 query token 数，`K` 是全局 key token 数，`N` 是 attention head
数，`S` 是 top-k。SparseMLA 不会创建 `[Q,N,K]` dense attention score。

## 来源和对应关系

实现依据是 GLM-5 的完整 DSA 工程流程，而不是把其他仓库作为运行时依赖：

- 本仓 `torchtitan/models/glm5/model.py::DSAIndexerTopK`、`SparseMLA`：稳定组件契约；
- 本地参考 `D:/yyb/repos/slime/slime_plugins/models/glm5/ops/indexer.py::lighting_indexer`
  和 `IndexerFunction`：Lightning Indexer 的 score/top-k 入口；
- 本地参考 `D:/yyb/repos/slime/slime_plugins/models/glm5/ops/sparse_mla.py::SparseMLA`：
  SparseMLA autograd 边界；
- 本地参考 `tilelang_indexer_fwd.py::indexer_fwd_interface`、
  `tilelang_sparse_mla_fwd.py::sparse_mla_fwd_interface` 和
  `tilelang_sparse_mla_bwd.py::sparse_mla_bwd`：forward/backward 分块和稀疏访问思路；
- 跨层 index 参考 `slime_plugins/models/glm5/glm5.py::source_compute_layer` 及其
  `fused_select_topk` 局部函数；TorchTitan 实现不导入这些符号。

代码已经改写为 Triton，不导入 Slime、TileLang 或其符号。GPU 使用 Triton；NPU
由 TorchTitanTurbo 使用 Triton-Ascend 注册同一数学实现。

## Lightning Indexer

`Glm5DsaIndexer` 负责 Q/K/weight 投影、ComplexRoPE 和归一化；本目录只替换最后的
`DSAIndexerTopK`：

```text
head_score[q,n,k] = relu(dot(q[q,n,:], k[k,:]) * scale)
score[q,k] = sum_n weight[q,n] * head_score[q,n,k] + mask[q,k]
indices[q,:] = topk(score[q,:])
```

Triton kernel 直接写 `[Q,K]`，避免参考实现的 `[N,Q,K]` 中间张量。query 分块限制
单次 score 的生命周期；最终 top-k 仍由 PyTorch 完成，因此选取范围始终是完整的
合法 key 序列，不是块内近似 top-k。

Indexer 参数在当前预训练权重使用方式下冻结，LM loss 不对离散 index 决策提供训练
目标，所以这里不需要伪造 indexer backward。若以后增加 indexer 蒸馏或辅助 loss，
必须先补独立的可导 score/top-k-score 契约。

## SparseMLA forward/backward

`Glm5Attention.Config.inner_attention` 的类型就是 `SparseMLA.Config`；
`TritonSparseMLA` 继承这个组件并保持完全相同的 forward 签名。它不需要继承一个
dense attention 类，因为 Q/KV projection、RoPE、index sharing、`W_V/W_O` 都由外层
`Glm5Attention` 负责。

forward 分成 selected score、FP32 softmax、selected value reduction：

```text
score[q,n,s] = dot(q[q,n,:], kv[index[q,s],:]) * scale
p32 = softmax(score, dtype=fp32)
p = cast(p32, q.dtype)
out[q,n,:C] = sum_s p[q,n,s] * kv[index[q,s],:C]
```

backward 保留两份概率：`p32` 用于精确复现 softmax backward，cast 后的 `p` 用于
value path。两阶段计算为：

```text
dP[q,n,s] = dot(dOut[q,n,:C], kv[index[q,s],:C])
dScore = p32 * (dP - sum_s(p32 * dP)) * scale
dQ = sum_s dScore * selected_K
dKV = atomic_sum(dScore * Q + p * dOut)
```

dKV 使用 FP32 scratch 做跨 query/head 的 atomic accumulation，结束后转换回输入
dtype。该顺序可能和 PyTorch reduction 产生 BF16 尾数差异，因此必须通过 operator
probe、端到端 loss/grad-norm 和 profiler 三重验证，不能只凭能运行就宣称验收。

## 启用方式

```bash
--override.imports \
torchtitan.models.glm5.ops.triton.triton_dsa_indexer,torchtitan.models.glm5.ops.triton.triton_sparse_mla
```

默认不启用。TorchTitanTurbo 的 NPU 注册路径见其 `models/glm5/ops/README.md`。

## 分布式边界

- TP：算子消费 sharding plan 已经产生的 TP-local tensor，不改变 placement；
- EP：attention 位于 dense-region，和 MoE expert mesh 解耦，可与 TP+EP 组合；
- CP：外层 GLM parallelize wrapper 先形成 local-query/global-key 契约，算子不自行通信；
- PP：受跨层共享 index 的 stage 边界约束，算子不负责跨 stage 传递 index。

因此“接口上兼容”不等于“所有拓扑已实机验收”。single、TP、EP、CP、TP+EP 都要
分别跑 smoke、backward、正式精度和 profiler。
