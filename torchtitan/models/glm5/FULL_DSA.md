# GLM-5 Full DSA implementation map

## Design boundary

GLM-5 keeps the model definition independent from device kernels:

- `model.py` defines readable PyTorch DSA semantics.
- `ops/` contains optional GPU implementations of the same component contracts.
- TorchTitanTurbo contains NPU implementations.
- `parallelize.py` and `sharding.py` describe distributed execution without
  changing the model equations.

The default configuration never imports `ops/`. Optimized operators are
selected explicitly with `--override.imports`, preserve the stock parameter
layout, and can be compared directly with the PyTorch reference.

`glm5_configs["GLM-5.2"]` records the released non-quantized base decoder,
including 78 layers, hidden size 6144, 64 attention heads, 256 routed experts,
2048-token DSA top-k, frequency-four index sharing with offset three, and the
1,048,576-token/8,000,000-theta RoPE contract. The smaller
`glm5_full_dsa_debugmodel` changes only capacity-related dimensions; it keeps
the same DSA control flow and sharing rule for fast validation. MTP remains a
separate unsupported release feature and is not implied by this model flavor.

## Mathematical pipeline

| DSA stage | TorchTitan implementation | Reference implementation correspondence |
|---|---|---|
| Q-LoRA and KV-LoRA | `Glm5Attention.wq_a/wq_b/wkv_a` | GLM-5 Q/KV down and up projections |
| Index Q/K/weights | `Glm5DsaIndexer` | `DSAIndexer`, including RoPE and FP32 head weights |
| Global score and top-k | `DSAIndexerTopK` | Lightning indexer score equation and exact top-k |
| Cross-layer sharing | `index_source_layer` and the decoder top-k carrier | Full layers publish indices; shared layers reuse them |
| Absorbed query | `Glm5Attention.forward` absorbs `W_K` into no-PE Q | Absorbed MLA query construction |
| Sparse KV access | `SparseMLA.forward` gathers `kv[topk_indices]` | Sparse MLA reads only selected compressed KV rows |
| Sparse attention | `SparseMLA.forward` computes `[Q, N, S]` scores | Top-k attention over latent plus RoPE dimensions |
| Value decode | `W_V` is applied after latent aggregation | Latent output is expanded only after attention |
| Output projection | `Glm5Attention.wo` | MLA output projection |

Here `Q` is the query-token count, `N` is the attention-head count, and `S`
is the selected top-k count. The main attention never creates a `[Q, N, K]`
score tensor. It gathers `[Q, S, C+R]` compressed KV and computes only
`[Q, N, S]` scores.

The PyTorch indexer is deliberately readable and still materializes one
`[Q, K]` index-score matrix. `ops/triton.py` evaluates the same equation in
query blocks with a Triton kernel. It avoids the intermediate `[N,Q,K]`
head-score tensor while preserving exact top-k over the complete legal key
range.

## GPU operator mapping

| Component contract | Reference path | Optional GPU path |
|---|---|---|
| `DSAIndexerTopK.Config` | PyTorch matmul, ReLU, weighted reduction, top-k | `TritonDSAIndexerTopK` score kernel plus PyTorch top-k |
| `SparseMLA.Config` | PyTorch gather, einsum, softmax, latent reduction | Triton selected-score/output forward and dQ/dKV backward |

The optimized path keeps the TorchTitan mathematical contract and imports no
external training framework at runtime. It remains opt-in and is not the
default model implementation.

Enable both GPU operators explicitly:

```bash
python -m torchtitan.train --module glm5 --config <production-config> \
  --override.imports \
  torchtitan.models.glm5.ops.triton.triton_dsa_indexer,torchtitan.models.glm5.ops.triton.triton_sparse_mla
```

The SparseMLA implementation has Triton forward and backward. Hardware
acceptance still requires operator forward/gradient comparison, end-to-end
loss/grad-norm comparison, and profiler evidence on every target backend.

## NPU correspondence

TorchTitanTurbo provides two independent choices: Triton-Ascend registrations
for the shared indexer/SparseMLA implementation, and an Ascend-native
`torch_npu.npu_sparse_flash_attention` SparseMLA override. Tensor layout
adaptation and operator selection remain outside the device-independent model.

## Hugging Face compatibility mode

The absorbed SparseMLA path can reproduce the Hugging Face dense-K/V reference
mathematically. For each head:

```text
q_nope @ (W_K @ c) == (q_nope @ W_K) @ c
sum_j p_j * (W_V @ c_j) == W_V @ (sum_j p_j * c_j)
```

Therefore both implementations produce the same result when they use the same
weights, positions, masks, top-k indices, and arithmetic precision. They may
have small floating-point differences because absorption changes reduction
order.

To reproduce the original HF parity contract:

1. Use `glm5_debugmodel`, or configure `index_topk_freq=1`, so every layer owns
   and executes its own indexer.
2. Do not load GPU or NPU operator overrides.
3. Load the identical HF-converted state dict and fixed input/token plan.
4. Compare FP32 first, then BF16 with the configured numerical tolerances.

`glm5_full_dsa_debugmodel` uses the same dimensions as `glm5_debugmodel` but
enables the production frequency-four index-sharing schedule. It cannot match
an HF run that recomputes an independent index at every layer unless the HF
configuration uses the same sharing schedule.

## Distributed support and limits

The reference path supports DP, FSDP, TP, EP, contiguous CP, and PP layouts
whose stages begin on full-index layers. Current limits are:

- cross-stage top-k transport is not implemented;
- CP load balancing and the `spmd_types` backend are not implemented;
- optimized GPU operators require separate distributed numerical validation;
- KV cache, incremental decoding, MTP, and indexer training objectives are not
  implemented;
- the pretrained indexer remains frozen because language-model loss does not
  train the discrete top-k decision.

## Source correspondence checklist

When synchronizing with the official/HF or Slime definitions, verify:

- projection dimensions and RoPE split;
- `relu(q @ k)`, per-query head weights, and scale placement;
- packed causal legal key ranges;
- top-k sentinel handling;
- full/shared layer schedule and PP boundaries;
- absorbed `W_K`, post-attention `W_V`, and output projection order;
- frozen indexer parameters and checkpoint key omission on shared layers.
