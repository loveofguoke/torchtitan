# GLM-5 performance extension guide

This document separates the readable GLM-5 mathematical model from optional
performance work. The default model remains the correctness reference. Every
optimization must be explicitly selected, numerically compared with that
reference, and measured with a profiler before it is promoted.

The authoritative runnable matrix and promotion decision live in the
[torchtitan-test Full DSA optimization workflow](https://github.com/loveofguoke/torchtitan-test/blob/feat/glm5-full-dsa-test/tests/glm5_2_performance/OPTIMIZATION.md).
This file defines model-side opportunities and acceptance requirements only.

## Boundaries

TorchTitan owns device-independent model structure and generic training
techniques. The GLM-5 directory may provide a readable reference operator and
an optional CUDA implementation, but it must not import TorchTitanTurbo or an
Ascend runtime. NPU implementations belong in TorchTitanTurbo; orchestration
and reports belong in torchtitan-test.

Passing old `glm5_debugmodel` experiments proves that the disabled Full DSA
path remains unchanged. It does not prove index sharing or a kernel is correct.
Treat these as independent factors:

1. per-layer versus shared index semantics;
2. PyTorch reference versus fused operator;
3. single, DP/FSDP, TP, CP, PP, or EP topology;
4. eager versus compiled execution;
5. saved activations, recomputation, or offload.

## Implemented compute optimization

`SparseMLA` is the device-independent reference. It gathers only selected
compressed KV rows and computes `[Q, N, topk]` scores; it does not materialize
dense `[Q, N, K]` attention scores.

The optional CUDA Triton path is selected through overrides:

```bash
--override.imports \
torchtitan.models.glm5.ops.triton.triton_dsa_indexer,torchtitan.models.glm5.ops.triton.triton_sparse_mla
```

It replaces operator boundaries without changing projections, masks, top-k,
residuals, state dict keys, or the sharing schedule. The implementation supports
the debug-model dimensions and bounded production head dimensions; every new
shape still requires the torchtitan-test operator probe before training.

The indexer fuses head dot products, ReLU, head weighting, and mask addition,
avoiding the reference `[N,Q,K]` head-score tensor. SparseMLA avoids the
`[Q,S,H]` selected-KV materialization and supplies Triton dQ/dKV backward.
Expected benefits remain hypotheses until a trace shows lower kernel time or
peak memory.

## Generic TorchTitan optimization points

These facilities already exist in TorchTitan and can be applied through an
external experiment config. No GLM-specific Trainer changes are required.

| Area | Mechanism | Trade-off |
|---|---|---|
| Communication | compiled async TP | overlap TP collectives and GEMM; requires TP and compilation |
| Compute | fused/foreach optimizer and SparseMLA kernels | fewer launches with backend-specific support |
| Memory | chunked LM-head loss | lower logits peak memory, more chunk launches |
| Memory | FSDP CPU offload | lower device memory, more host transfers |
| Compute-memory | full/selective activation checkpointing | fewer saved activations, more backward recomputation |
| Distributed memory | FSDP reshard policy | parameter residency versus repeated all-gathers |

The test repository owns concrete values because they are experiments, not
model semantics.

## Communication optimization

TorchTitan exposes `CompileConfig.enable_async_tensor_parallel`. A TP experiment
uses:

```text
--compile.enable
--compile.components=model
--compile.backend=inductor
--compile.enable_async_tensor_parallel
```

The profiler must show collective work moving under GEMM and lower exposed
communication. A faster isolated collective is not sufficient; compare the
end-to-end critical path.

Full DSA has a future CP opportunity. The current correctness path gathers
global index keys and compressed KV. An index-aware overlap could be:

```python
# Pseudocode only; not implemented in the readable model.
remote_k_future = async_all_gather(index_keys_local)
q = project_queries(hidden_local)
local_scores = index_score(q, index_keys_local)
remote_k = remote_k_future.wait()
global_topk = merge_topk(local_scores, index_score(q, remote_k))

kv_future = fetch_selected_kv(global_topk.remote_indices)
local_partial = sparse_mla(q, kv_local, global_topk.local_indices)
remote_partial = sparse_mla(q, kv_future.wait(), global_topk.remote_indices)
output = merge_softmax_partials(local_partial, remote_partial)
```

This needs stable distributed softmax statistics and autograd-aware
communication. It belongs in a distributed/kernel layer, not an opaque branch
in `Glm5Attention.forward`.

## Memory scheduling and recomputation

The first reproducible memory experiment is chunked loss. Compare
`--loss.num_chunks=1` with `8` or `16`, recording peak active/reserved memory,
step time, and LM-head kernel count. More chunks reduce the largest logits
tensor but may increase launches.

The second is full activation checkpointing. It should preserve loss within
the selected precision standard while reducing saved activations. Measure both
the increase in backward compute and the decrease in peak memory.

Do not add persistent workspaces to the reference model merely to reduce
allocator calls. The unregistered candidate package contains a shape-keyed
workspace prototype:

```python
# Pseudocode for an operator implementation, not model.py.
key = (device, dtype, num_queries, topk, num_heads, head_dim)
workspace = workspace_cache.acquire(key)
try:
    return fused_sparse_mla(q, kv, indices, workspace=workspace)
finally:
    workspace_cache.release(key)
```

The cache needs explicit lifetime, stream safety, a memory cap, and a graph
capture policy. Otherwise fewer allocations can still cause fragmentation or
excess retained memory.

## Mapping common large-model techniques

Megatron-style systems commonly combine sequence parallelism, TP overlap,
distributed optimizers, grouped GEMM, fused normalization/activation, fused
attention, activation recomputation, and bucketed collectives. TorchTitan
expresses many of the same ideas as composable PyTorch features.

| Common technique | GLM-5 target |
|---|---|
| Flash/sparse attention | absorbed SparseMLA over selected compressed KV |
| Grouped GEMM | routed expert computation after dispatch |
| Fused RMSNorm/RoPE | attention and FFN normalization/projection boundaries |
| TP overlap | Q/KV/output projections and dense/MLP GEMMs |
| EP overlap | token dispatch, expert GEMM, and combine |
| Recomputation | transformer blocks while preserving top-k decisions |

Saving top-k is important for selective recomputation: a different tie
decision changes the expert or KV path. TorchTitan's selective checkpoint
policy already treats `topk` as a must-save operation.

## Acceptance rule

Collect reference and candidate with identical model, token plan, checkpoint,
topology, precision, batch, warmup, and profiler window. Require:

1. smoke and backward completion;
2. formal loss/grad-norm acceptance;
3. no new graph break or CPU fallback;
4. lower steady-state step time, peak memory, or exposed communication;
5. the expected kernel/collective change visible in the trace.

An optimization that only helps a synthetic probe remains an operator
prototype, not a training optimization. Confirmed operators and unregistered
ideas are indexed separately in `ops/README.md` and `ops_candidate/README.md`.

The current NPU eight-card evidence and numbered implementation tasks live in
`torchtitan-test/tests/glm5_2_performance/explorations/optimization_backlog.md`.
In particular, TP placement work starts from
`torchtitan.models.glm5.parallelize::parallelize_glm5` and
`torchtitan.distributed.spmd_types::spmd_redistribute_per_axis`; FSDP grouping
starts from `torchtitan.distributed.fsdp::apply_fsdp_to_decoder`; EP overlap is
compared against
`torchtitan.experiments.graph_trainer.ep_eager_chunk::maybe_apply_ep_overlap_eager_chunking`
and `ep_overlap_pass::ep_overlap_schedule_pass`. These are inspection points,
not authorization to modify generic framework code for an NPU-only result.
