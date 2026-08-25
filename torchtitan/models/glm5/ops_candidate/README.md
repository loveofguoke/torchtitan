# GLM-5 性能优化原型

本目录用于把主流训推框架中的优化思想提炼成可读、可单测的独立原型。它和
`ops/` 的边界是硬约束：

- 不被 `model.py`、配置注册或 override 注册导入；
- 不出现在 torchtitan-test 的可运行优化选项中；
- 不宣称已经改善 GLM-5 性能；
- 只有完成接口、数值、分布式、图模式和 profiler 验收后，才可以迁入 `ops/`。

## 目录层级与参考来源

引用统一写成 `仓库路径::符号`。不固定行号，因为这些上游仓库更新频繁；符号名既能用
`rg` 精确定位，也不会因文件前面插入代码而失效。

| 维度 | 原型 | 主要本地参考 | GLM-5 对应位置 |
| --- | --- | --- | --- |
| 计算 | `compute/triton_residual_rmsnorm.py` | TorchTitan `torchtitan/models/common/nn_modules.py::RMSNorm`；Megatron `megatron/core/fusions/fused_layer_norm.py::FusedLayerNorm` | `model.py::Glm5TransformerBlock` 的 residual + attention/FFN norm |
| 通信 | `communication/overlap_pipeline.py` | TorchTitan `experiments/graph_trainer/ep_eager_chunk.py::maybe_apply_ep_overlap_eager_chunking`、`ep_overlap_pass.py::ep_overlap_schedule_pass`；Megatron `transformer/moe/token_dispatcher.py::MoEAlltoAllTokenDispatcher` | MoE dispatch -> grouped expert compute -> combine |
| 通信/内存 | `communication/bucket_planner.py` | Megatron `distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py::ParamAndGradBuffer`；DeepSpeed `runtime/zero/stage3.py::IPGBucketZ3` | DDP/FSDP gradient reduce 和 parameter gather bucket |
| 通信 | `communication/redistribution_audit.py` | TorchTitan `distributed/spmd_types.py::spmd_redistribute_per_axis`、`models/glm5/parallelize.py::parallelize_glm5` | TP/组合拓扑的多轴 Partial -> Replicate |
| 通信/内存 | `communication/fsdp_layer_grouping.py` | TorchTitan `distributed/fsdp.py::apply_fsdp_to_decoder`；Megatron `param_and_grad_buffer.py::BucketingPolicy` | 相邻兼容 transformer block 的 FSDP 粒度 |
| 流水 | `pipeline/stage_partition.py` | Megatron `pipeline_parallel/schedules.py::forward_backward_pipelining_without_interleaving`；DeepSpeed `runtime/utils.py::partition_balanced` | embedding、block、loss 的 PP stage 划分和 microbatch bubble |
| MoE | `moe/expert_placement.py` | Megatron `transformer/moe/token_dispatcher.py::MoEAlltoAllTokenDispatcher`、`moe_utils.py::permute`；DeepSpeed `moe/ep_tp_dispatch.py::partition_assignments` | expert/rank token payload 的离线均衡假设 |
| 内存 | `memory/workspace.py` | TorchTitan `distributed/linear.py::AllGatherLinear`、`LinearReduceScatter` | index score、SparseMLA 和 fused projection 临时张量 |
| 计算/内存 | `recompute/selective_recompute.py` | TorchTitan `distributed/activation_checkpoint.py::SelectiveAC`、`MemoryBudgetAC`；Megatron `core/recompute.py::checkpointed_forward`；DeepSpeed `runtime/activation_checkpointing/checkpointing.py::partition_activations` | transformer activation，必须保存 top-k/通信输出 |
| 推理内存 | `serving/paged_kv_cache.py` | vLLM `vllm/v1/core/block_pool.py::BlockPool.get_new_blocks`、`touch`、`free_blocks` | 未来 GLM-5 增量解码 compressed KV cache |

## 计算：Triton residual + RMSNorm

原型把 residual add 和 RMSNorm 合成单个 Triton forward，减少一次 residual 写回后的
再次读取以及一次 kernel launch。对应 GLM block 的概念接法：

```python
# Pseudocode only. Do not add this to model.py before validation.
attention_out, residual = fused_residual_rmsnorm(
    attention_out, hidden, attention_norm.weight, eps=eps
)
ffn_out = feed_forward(attention_out)
```

正式化前还缺 backward、任意 stride、autocast、TP/DTensor、图捕获和不同 hidden size
的 autotune。验收看 kernel 数、HBM bytes、norm 时间、loss 和 parameter gradients。

## 通信：分块 dispatch/compute/combine overlap

`run_chunked_overlap` 只描述依赖图：发起 chunk `i+1` 的异步 dispatch，同时计算
chunk `i`，并延后等待 combine。它不持有 NCCL/HCCL stream，也不调用 GLM 模块。

```python
# Pseudocode only.
chunks = split(routed_tokens, num_chunks=2)
result = run_chunked_overlap(
    chunks,
    issue_dispatch=lambda i, x: all_to_all_async(x, stream=comm_stream[i % 2]),
    compute=lambda i, x: grouped_expert_gemm(x, buffer=i % 2),
    issue_combine=lambda i, x: all_to_all_async(x, stream=comm_stream[i % 2]),
    merge=restore_token_order,
)
```

它对应 TorchTitan
`experiments/graph_trainer/ep_eager_chunk.py::maybe_apply_ep_overlap_eager_chunking` 和
`ep_overlap_pass.py::ep_overlap_schedule_pass`，也对应 Megatron
`transformer/moe/token_dispatcher.py::MoEAlltoAllTokenDispatcher`。
正式接入必须解决双缓冲、event、autograd collective、动态 token 数、反向流水和
deadlock。验收重点是 exposed communication 和 end-to-end step critical path。

## 通信/内存：连续 bucket planner

`plan_communication_buckets` 按近似 backward-ready 的逆注册顺序，把参数映射到对齐的
连续区间。它对应 Megatron
`megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py::ParamAndGradBuffer`
和 `Bucket` 的 contiguous grad buffer 与 bucket-ready communication。

```python
# Pseudocode only.
layout = plan_communication_buckets(named_numel, target_numel=40_000_000,
                                    alignment=dp_world_size * 128)
flat_grad = allocate_contiguous(layout[-1].end)
register_post_accumulate_hooks(layout, launch_reduce_scatter_async)
```

GLM-5 的收益点是大量 expert 参数导致的小梯度通信。风险包括 shared/tied parameter、
unused frozen indexer、不同 backward 顺序、TP/EP process group 和 checkpoint view。
验收看 bucket ready 时间、collective 数量、通信暴露时间和 peak memory。

## 通信：多轴 DTensor redistribution 审计

`audit_partial_to_replicate` 不执行 collective，只把一个 transition 的 active partial
mesh axes、目标 Replicate axes 和理论 launch 数写清楚。它直接对应八卡报告中反复出现的
“两个串行 AllReduce”警告：只有多个非 unit partial axes 在同一个 dependency point
全部变成 Replicate，才标记为 flatten candidate。

```python
audit = audit_partial_to_replicate(
    RedistributionTransition(
        "attention_output",
        (MeshAxis("fsdp", 2), MeshAxis("tp", 4)),
        ("fsdp", "tp"),
    )
)
```

它不能证明 process-group flatten 数学等价；正式改动仍需在
`parallelize_glm5`/sharding config 追踪 placement、reduction order 和 autograd。

## 通信/内存：FSDP 相邻层分组

`plan_fsdp_layer_groups` 只合并相邻且 `policy_key` 相同的 layer，避免跨越 dense/MoE
mesh、pipeline stage 或不同 mixed-precision policy。它给出理论 AllGather+
ReduceScatter 次数和 group unsharded bytes，用来筛选 `layers_per_group=1/2/4/8` 的
实验点，不修改 `apply_fsdp_to_decoder`。

真实 profiler 必须同时验证 collective count、exposed communication、HBM 和 overlap；
calls 下降但 peak HBM 或 step time 上升时不晋级。

## 流水：实测代价 stage partition

`plan_contiguous_pipeline_stages` 用每个 unit 的 forward+backward profiler 时间做连续
动态规划，最小化最慢 stage；`ideal_non_interleaved_1f1b_bubble_fraction` 只提供平衡
stage 下的理论 fill/drain 基线。PP8 报告表明 P2P 传输不足 0.43 ms，而等待达到秒级，
所以先做 stage/microbatch/schedule A/B，不做 HCCL kernel。

## MoE：expert/rank 载荷 what-if

`plan_expert_rank_placement` 用实测 per-expert token count 做离线 greedy placement，回答
“静态重新映射最多能把 1.56x payload spread 降到多少”。它不能动态改变训练中的 expert
owner，因为那会破坏 checkpoint、optimizer state 和 process group；真实优化仍应先做
dispatch/compute/combine overlap 和路由统计。

## 内存：stream-local workspace pool

`TensorWorkspacePool` 以 device、dtype、shape 和 stream 为 key，有容量上限，只复用
同 stream 的连续、无梯度临时张量。它对应 TorchTitan
`distributed/linear.py::AllGatherLinear`、`AllGatherLinearMulti` 和
`LinearReduceScatter` 的共享 workspace 生命周期问题。

```python
# Pseudocode only.
score = pool.acquire((query_block, num_keys), dtype=torch.float32, device=device)
try:
    launch_index_score(..., output=score)
    indices = score.topk(k).indices
finally:
    pool.release(score)
```

它目前不接入 indexer，因为 eager 提前释放、异步 consumer、graph capture 和多 stream
都可能让“少一次分配”变成数据竞争或过度 reserved memory。验收同时看 allocation
events、active/reserved memory、fragmentation、step time 和 graph replay。

## 计算/内存：selective recompute

原型先强制保存离散控制流和通信输出，再按 `recompute_cost/output_bytes` 选择保存昂贵
activation。对 GLM5，indexer top-k、MoE expert assignment 和 collective 结果不能被
普通 cost model 随意重算：边界 tie 或通信时序变化会改变后续数据流。

```python
# Pseudocode only.
candidates = profile_forward_activations(model)
policy = plan_selective_recompute(candidates, save_budget_bytes=budget)
checkpoint(block, policy_fn=policy.as_torch_policy())
```

真实接入优先复用 TorchTitan
`distributed/activation_checkpoint.py::SelectiveAC` 或 `MemoryBudgetAC`，原型只帮助
解释策略。验收看 saved bytes、额外 FLOPs、backward time、top-k 一致性和 loss/grad。

## 推理内存：Paged compressed-KV cache

`PagedKVBlockTable` 参考 vLLM
`vllm/v1/core/block_pool.py::BlockPool.get_new_blocks`、`touch`、`free_blocks` 的
block pool/refcount 设计，提供固定 block、fork、free、slot mapping 和部分块
copy-on-write 计划。它只适用于未来推理，不属于当前训练路径。

```python
# Pseudocode only.
plan = table.allocate_tokens(request_id, num_new_tokens)
copy_shared_partial_blocks(plan.copies, compressed_kv_cache)
write_compressed_kv(plan.block_ids, new_tokens)
indices = indexer(query, paged_index_keys)
out = paged_sparse_mla(query, compressed_kv_cache, indices)
```

GLM DSA 还需要 paged index key、全局 position、跨层共享 index 和 prefix-cache hash。
验收看 cache utilization、fragmentation、fork/free 正确性、decode latency 和输出精度。

## 原型晋级规则

原型进入 `ops/` 前必须依次满足：

1. 明确稳定组件契约和 backward；
2. CPU/纯逻辑单测，以及 GPU/NPU operator probe；
3. reference/candidate 端到端精度；
4. single 和目标分布式拓扑；
5. eager/graph 生命周期；
6. profiler 证明目标瓶颈下降；
7. profiler-off 多次运行证明 step time 或 peak memory 实际改善。
