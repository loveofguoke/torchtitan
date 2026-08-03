# GLM-5 Debug Model Design

## Purpose

Add a native, single-device GLM-5 debug model under `torchtitan/models/glm5`.
The first milestone proves model correctness rather than production-scale
performance: a structurally faithful reduced model must run forward, loss, and
backward on CPU and one CUDA GPU, and its CUDA numerics must align with Hugging
Face Transformers `GlmMoeDsaForCausalLM` after weights are converted through the
real TorchTitan state-dict adapter.

The implementation follows the repository's model-extension guidance in
`torchtitan/models/README.md`. It borrows the organization and registration
patterns of `torchtitan/models/deepseek_v3`, but it does not inherit from or
reuse DeepSeek-V3-specific model classes. Shared functionality comes from
`torchtitan/models/common`; GLM-5-specific computation stays in
`torchtitan/models/glm5/model.py`.

The reference architecture is the Transformers `glm_moe_dsa` implementation.
Design analysis used Transformers commit `b3a36037d3feb22e3f0174b3dd4248fcc0f0f722`,
and parity will be exercised against the installed Transformers implementation
(version 5.14.1 at design time).

## Goals

- Register a native TorchTitan model named `glm5` with flavor `debugmodel`.
- Preserve the architectural features that distinguish GLM-5:
  - Q-LoRA and KV-LoRA multi-head latent attention (MLA);
  - interleaved rotary position embeddings;
  - a per-layer DeepSeek Sparse Attention (DSA) indexer;
  - dynamic per-query top-k attention selection;
  - leading dense FFN layers followed by MoE layers;
  - sigmoid routing with an FP32 expert correction bias;
  - routed grouped experts plus shared experts.
- Reuse common TorchTitan components for embeddings, linear layers, norms,
  RoPE, dense feed-forward layers, MoE, experts, and decoder infrastructure.
- Implement a formal `Glm5StateDictAdapter` with both `from_hf()` and `to_hf()`.
- Run CPU unit and forward/backward tests.
- Align single-CUDA-GPU outputs and gradients with Transformers using the same
  reduced configuration, weights, input tokens, positions, and labels.
- Fail clearly when an unsupported distributed mode or malformed architecture
  configuration is requested.

## Non-goals

The first milestone does not implement or claim support for:

- the released 744B parameter configuration as a model flavor;
- loading the full released checkpoint in tests;
- tensor, context, pipeline, expert, fully sharded, or hybrid parallelism;
- cross-layer top-k sharing / IndexCache shared layers;
- KV cache or incremental decoding;
- Flash-MLA or another production DSA kernel;
- long-context performance or memory efficiency;
- MTP / next-token-prediction layers;
- an auxiliary objective for training the DSA indexer from random initialization.

These exclusions are explicit feature boundaries, not implicit claims that the
features work.

## Package Structure

The model package will contain:

```text
torchtitan/models/glm5/
|-- __init__.py
|-- config_registry.py
|-- model.py
|-- parallelize.py
|-- sharding.py
|-- state_dict_adapter.py
`-- README.md
```

The model name will be added to `torchtitan/models/__init__.py`. Tests will live
in the repository-level test tree:

```text
tests/unit_tests/test_glm5_model.py
tests/unit_tests/test_glm5_state_dict_adapter.py
tests/unit_tests/test_glm5_parity.py
```

## Model Boundaries

`model.py` will define four GLM-5-native classes:

- `Glm5DsaIndexer(Module)` computes per-query top-k key indices.
- `Glm5Attention(BaseAttention)` implements GLM-5 MLA, invokes the indexer,
  builds the dynamic sparse mask, and executes the debug eager attention path.
- `Glm5TransformerBlock(TransformerBlock)` composes attention, a dense FFN or
  common MoE, two norms, and residual connections.
- `Glm5Model(Decoder)` supplies the model-level dense causal/document mask and
  the configuration hooks required by TorchTitan.

The classes inherit only common abstractions. In particular,
`Glm5Attention` will not inherit DeepSeek-V3 `Attention`, and
`Glm5StateDictAdapter` will not inherit `DeepSeekV3StateDictAdapter`. This keeps
the DSA data flow, checkpoint mapping, and future optimizations local to GLM-5.

The model will compose these common modules:

- `Embedding` and `Linear`;
- `RMSNorm` for the decoder and MLA low-rank norms;
- `LayerNorm` for the DSA indexer's `k_norm`, including its bias;
- `ComplexRoPE` for interleaved rotary embeddings;
- `FeedForward` for dense and shared FFNs;
- `MoE`, `TokenChoiceTopKRouter`, `RoutedExperts`, and `GroupedExperts`;
- `Decoder` and `TransformerBlock` infrastructure.

All parameters and buffers will be initialized through sub-config `param_init`
entries and the existing recursive `init_states()` mechanism. The model will
not perform manual recursive parameter initialization.

## Debug Configuration

`glm5_debugmodel` will keep the complete GLM-5 block structure while reducing
dimensions, layers, experts, vocabulary, and sequence length:

| Field | Value |
|---|---:|
| `vocab_size` | 2048 |
| `dim` / HF `hidden_size` | 256 |
| `num_hidden_layers` | 4 |
| `num_attention_heads` | 8 |
| `q_lora_rank` | 128 |
| `kv_lora_rank` | 64 |
| `qk_nope_head_dim` | 32 |
| `qk_rope_head_dim` | 32 |
| `v_head_dim` | 64 |
| dense `intermediate_size` | 1024 |
| `moe_intermediate_size` | 256 |
| `n_routed_experts` | 8 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 2 |
| router score function | sigmoid |
| router expert groups / limited groups | 1 / 1 |
| router normalization / scale | true / 2.5 |
| `first_k_dense_replace` | 1 |
| `index_n_heads` | 4 |
| `index_head_dim` | 64 |
| `index_topk` | 8 |
| `max_seq_len` | 128 |
| RoPE theta | 1,000,000 |
| RMSNorm epsilon | 1e-5 |
| attention dropout | 0.0 |

The first layer uses a dense FFN and the remaining three use MoE, so one model
exercises both paths. Every layer owns a full indexer. The HF parity
configuration will explicitly set an all-`full` indexer pattern rather than
depending on a Transformers default.

`Glm5Model.Config` will implement `update_from_config()` and
`get_nparams_and_flops()`. Sequence length updates must resize or rebuild RoPE
state through the common configuration path. FLOP reporting must include the
model's dense/MLA/MoE contribution and document that the debug DSA indexer is an
additional quadratic cost.

## Forward Data Flow

For each block, the model computes:

```text
x
|-- attention_norm
|   `-- Glm5Attention
|       |-- q_a_proj -> q_norm -> q_b_proj
|       |-- kv_a_proj -> kv_norm -> kv_b_proj
|       |-- interleaved ComplexRoPE
|       |-- Glm5DsaIndexer -> topk_indices
|       |-- causal/document mask AND top-k selection
|       `-- eager attention -> o_proj
|-- residual add
|-- ffn_norm
|   `-- dense FeedForward OR common MoE
`-- residual add
```

The attention input is already normalized by `attention_norm`. Its Q-LoRA path
produces a low-rank query residual and then expands it to per-head query states.
Its KV-LoRA path produces compressed KV and the rotary key slice, normalizes the
compressed part, and expands it to per-head non-rotary keys and values. RoPE is
applied only to the rotary Q/K slices before concatenation.

### DSA indexer

The indexer consumes the normalized attention input, the Q-LoRA residual,
positions/RoPE, and the base mask. It computes:

```text
index_q = wq_b(q_resid)
index_k = k_norm(wk(hidden_states))
scores = relu(index_q @ index_k.T) * index_head_dim**-0.5
index_weights = weights_proj(hidden_states) * index_n_heads**-0.5
index_scores = weighted sum of scores over index heads
topk_indices = topk(index_scores, min(index_topk, key_length))
```

The indexer uses interleaved RoPE, matching GLM-5 rather than the non-interleaved
DeepSeek-V3.2 indexer path. Its forward runs under `torch.no_grad()`, matching
Transformers. The language-model loss therefore does not update indexer
parameters. Tests must assert this behavior instead of treating missing indexer
gradients as a failure.

### Debug sparse attention

The debug backend prioritizes mathematical transparency:

```text
attention_scores = Q @ K.T * scale
attention_scores += dense causal/document mask
attention_scores.masked_fill(keys not selected by top-k, -inf)
attention_probs = softmax(attention_scores, dtype=float32)
attention_output = attention_probs @ V
```

The result is cast back to the input dtype before the output projection. This
path is intentionally quadratic and only suitable for the reduced sequence
length. It must not be presented as a production DSA performance implementation.

### Mask semantics

`Glm5Model.get_attention_masks()` will build a dense additive mask with shape
`[B, 1, L, L]`:

- a query may not attend to a future key;
- for packed inputs, a query may only attend to keys in the same document;
- document boundaries use the repository convention `positions == 0`;
- the top-k mask is combined with, and can never reopen, the base mask.

The indexer consumes the corresponding `[B, L, L]` slice. When `index_topk`
exceeds the number of currently valid keys, masked entries may appear in the raw
top-k result, but the base additive mask remains applied to the main attention
and preserves causality/document isolation.

## State-dict Adapter

`Glm5StateDictAdapter` will inherit `MoEStateDictAdapter`, not a concrete model
adapter. It will implement strict bidirectional mappings for embeddings, MLA,
DSA, norms, FFNs, MoE, and the LM head.

### Direct mappings

Representative HF-to-TorchTitan mappings are:

| Hugging Face | TorchTitan |
|---|---|
| `model.embed_tokens.weight` | `tok_embeddings.weight` |
| `self_attn.q_a_proj.weight` | `attention.wq_a.weight` |
| `self_attn.q_a_layernorm.weight` | `attention.q_norm.weight` |
| `self_attn.q_b_proj.weight` | `attention.wq_b.weight` |
| `self_attn.kv_a_proj_with_mqa.weight` | `attention.wkv_a.weight` |
| `self_attn.kv_a_layernorm.weight` | `attention.kv_norm.weight` |
| `self_attn.kv_b_proj.weight` | `attention.wkv_b.weight` |
| `self_attn.o_proj.weight` | `attention.wo.weight` |
| `self_attn.indexer.wq_b.weight` | `attention.indexer.wq_b.weight` |
| `self_attn.indexer.wk.weight` | `attention.indexer.wk.weight` |
| `self_attn.indexer.k_norm.weight` | `attention.indexer.k_norm.weight` |
| `self_attn.indexer.k_norm.bias` | `attention.indexer.k_norm.bias` |
| `self_attn.indexer.weights_proj.weight` | `attention.indexer.weights_proj.weight` |
| `input_layernorm.weight` | `attention_norm.weight` |
| `post_attention_layernorm.weight` | `ffn_norm.weight` |
| `model.norm.weight` | `norm.weight` |
| `lm_head.weight` | `lm_head.weight` |

Dense FFN and shared-expert projections map gate/up/down to `w1`/`w3`/`w2`.
The HF router gate maps to `moe.router.gate`, and
`e_score_correction_bias` maps to TorchTitan `moe.expert_bias_E` without losing
its FP32 buffer dtype.

### Fused expert conversion

Transformers 5.14.1 stores routed expert parameters as:

```text
experts.gate_up_proj  [E, 2F, D]
experts.down_proj     [E, D, F]
```

TorchTitan common grouped experts store:

```text
w1_EFD  [E, F, D]  # gate
w3_EFD  [E, F, D]  # up
w2_EDF  [E, D, F]  # down
```

`from_hf()` splits `gate_up_proj` in half along dimension 1, in gate-then-up
order, and maps `down_proj` directly. `to_hf()` concatenates `w1_EFD` then
`w3_EFD` and maps `w2_EDF` directly. Tests must verify shapes, ordering, values,
and bidirectional round-trip.

### Strictness and ignored keys

Unknown keys are errors. The adapter may ignore only explicitly recognized
recomputable RoPE buffers. If a future full checkpoint exposes the configured
extra MTP layer, the adapter will identify the exact next-layer namespace,
emit a warning that MTP is unsupported, and skip only that namespace. It will
not use a broad pattern that hides unrelated missing mappings.

## Registration and Runtime Configuration

`__init__.py` will expose the debug model configuration dictionary and
`model_registry(flavor)`. The returned `ModelSpec` will name the model `glm5`,
attach `Glm5StateDictAdapter`, and use the normal decoder loss and single-device
training path.

`config_registry.py` will define `glm5_debugmodel() -> Trainer.Config`, selectable
as:

```text
--module glm5 --config glm5_debugmodel
```

The training configuration will use a short sequence and a local/test tokenizer
path appropriate for the existing debug infrastructure. It will not enable a
distributed degree greater than one.

`parallelize.py` and `sharding.py` will satisfy the model-extension interface
without claiming untested distributed support. Runtime configuration validation
will reject TP, CP, PP, EP, FSDP/HSDP, or another multi-rank setup with a clear
`NotImplementedError`. The standard single-device activation-checkpointing and
compile hooks will be applied when requested by the trainer. Parity tests will
disable both and exercise the eager reference path.

`README.md` will describe how to run the debug model, what is tested, the
Transformers reference, the unsupported features, and the staged roadmap. It
will not imply that the released checkpoint or 200K-token execution is already
supported.

## Validation Strategy

### CPU tests

CPU tests must cover:

- debug config construction and model registration;
- configuration validation failures;
- indexer shapes, dtypes, masking, and exact top-k behavior;
- dense causal and packed-document masks;
- dense and MoE block construction;
- complete forward, cross-entropy loss, and backward smoke execution;
- expected absence of indexer gradients;
- state-dict key coverage and strict unknown-key errors;
- fused expert split/concatenate ordering;
- `from_hf()` -> `to_hf()` value-preserving round-trip.

CPU full-model tests are functional rather than the final precision authority.
The common `GroupedExperts` implementation performs its expert matmuls in BF16,
including on CPU, so a nominal FP32 model still contains BF16 expert arithmetic.

### CUDA parity tests

CUDA parity tests construct matching reduced HF and TorchTitan models. They use
the formal adapter to load the HF model's state dict into TorchTitan and then
run identical tokens, positions, labels, and random seeds.

The tests compare:

- DSA top-k indices exactly;
- router expert indices exactly;
- indexer and attention component outputs within dtype-appropriate tolerance;
- block outputs within dtype-appropriate tolerance;
- end-to-end logits and cross-entropy loss;
- representative gradients for trainable attention, FFN/MoE, embedding, and LM
  head parameters after mapping parameter names;
- absence of indexer gradients in both implementations.

FP32 component tests isolate indexer/attention arithmetic. The full MoE model is
compared primarily in BF16 because TorchTitan common grouped experts execute in
BF16. Numerical tolerances will be explicit in the tests and chosen from
observed error bounds with enough margin to avoid hardware-noise flakes while
still catching weight-order, mask, RoPE, and routing errors. CUDA tests skip
cleanly when CUDA is unavailable and use only one visible device.

The parity test requires no network access and no released GLM-5 weights.

### Repository checks

Before completion, run the focused GLM-5 tests, affected existing tests, and
`pre-commit run --all-files`. The full test suite is attempted where the
environment permits; unrelated network, device, or distributed test constraints
must be reported rather than represented as model failures.

## Error Handling

Use `ValueError` for invalid user-facing architecture settings, including:

- non-positive Q-LoRA rank (the GLM-5 indexer requires the Q-LoRA residual);
- incompatible Q/K/RoPE head dimensions;
- `index_head_dim < qk_rope_head_dim`;
- non-positive `index_topk`;
- expert count not divisible by the configured group count;
- router top-k greater than the number of routed experts;
- dense layer count outside `[0, num_hidden_layers]`.

Use `NotImplementedError` for explicitly out-of-scope runtime modes such as
distributed parallelism, shared indexer layers, cache-based decoding, or a
non-debug sparse kernel request. Use assertions only for internal invariants
that cannot be caused by user configuration.

## Future Roadmap

After the single-device correctness milestone, later work may add:

1. FSDP and expert parallelism for reduced-model distributed training.
2. Tensor parallelism with distributed index-head reduction and replicated
   global top-k indices.
3. Context/pipeline parallelism and, when needed, cross-layer top-k transfer.
4. A production index-aware DSA/Flash-MLA kernel for long sequences.
5. KV cache and incremental decoding.
6. Cross-layer IndexCache/shared indexer patterns.
7. MTP layers and released-checkpoint coverage.
8. An explicit auxiliary/distillation objective for training indexers from
   random initialization.

Each item requires its own correctness and numerics design; none is implied by
the debug milestone.

## Acceptance Criteria

The milestone is accepted when all of the following are true:

- `glm5_debugmodel` builds through the normal TorchTitan model registry.
- The reduced model runs forward, loss, and backward on CPU.
- The reduced model runs on one CUDA GPU.
- HF weights load through `Glm5StateDictAdapter` with complete mapped-key
  coverage.
- Adapter round-trip preserves every supported HF state-dict tensor.
- CUDA parity tests meet explicit tolerances for components, logits, loss, and
  representative gradients.
- DSA top-k and MoE router expert selections are exactly equal to Transformers.
- Indexer parameters remain gradient-free in both models.
- Unsupported capabilities fail clearly and are documented.
- Focused tests and formatting/lint checks pass, with any environment-limited
  broader checks reported accurately.
