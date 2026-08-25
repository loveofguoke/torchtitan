# GLM-5 debug model

`glm5_debugmodel` is a native TorchTitan implementation of a small but
structurally faithful GLM-5 configuration. On the default SPMD backend it
supports single-device execution, DDP/HSDP/FSDP, CP, TP, PP, EP, and their
valid combinations. It is intended for model development, CPU training-step
coverage, and numerical comparison with Transformers; it is not a production
configuration and is not the released 744B model.

The numerical reference is Hugging Face Transformers'
`src/transformers/models/glm_moe_dsa` implementation, specifically
`GlmMoeDsaForCausalLM`. The TorchTitan model uses a strict
`Glm5StateDictAdapter` to move the same weights between the two layouts.
The default flavor keeps an indexer on every layer for compatibility with the
existing HF parity suite. `glm5_shared_index_debugmodel` exercises the released
cross-layer index-sharing behavior.

## Run the debug configuration

The registered trainer configuration is selected with `--module glm5 --config
glm5_debugmodel`:

```bash
python -m torchtitan.train --module glm5 --config glm5_debugmodel
```

For the repository launcher, request exactly one process/device:

```bash
NGPU=1 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

The shared-index flavor uses the same dimensions and a frequency-three
`FFSSFSSF` schedule:

```bash
NGPU=1 MODULE=glm5 CONFIG=glm5_shared_index_debugmodel ./run_train.sh
```

`glm5_full_dsa_debugmodel` has exactly the same reduced dimensions as
`glm5_debugmodel`, but uses the production frequency-four cross-layer index
sharing schedule. It is the fast correctness flavor for the complete DSA data
flow; the original debug flavor remains the parity baseline.

```bash
NGPU=1 MODULE=glm5 CONFIG=glm5_full_dsa_debugmodel ./run_train.sh
```

The model registry also exposes `model_registry("GLM-5.2")`, which defines the
released, non-quantized 78-layer base decoder: hidden size 6144, 64 attention
heads, Q/KV LoRA ranks 2048/512, 256 routed experts, 2048-token sparse
attention, frequency-four index sharing, and a 1,048,576-token RoPE cache
contract. It is an architecture/checkpoint definition rather than a local
debug trainer configuration; production data, tokenizer, checkpoint, and
parallelism settings must be supplied by the deployment. The release's MTP
head remains outside this DSA-focused model flavor.

Data-parallel runs over 8 GPUs use the same launcher with the parallelism
degrees set explicitly:

```bash
# DDP/HSDP: replicated weights, gradient all-reduce over 8 GPUs
NGPU=8 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --parallelism.data_parallel_replicate_degree 8 --parallelism.data_parallel_shard_degree 1

# FSDP: sharded weights over 8 GPUs
NGPU=8 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree 8

# CP: two contiguous sequence shards with local queries and global DSA keys
NGPU=2 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --parallelism.context_parallel_degree 2
```

The debug config disables CP load balancing because GLM DSA global key indices
must retain the original token order. A CP run therefore requires
`context_parallel_load_balancer=None`; head-tail and PTRR layouts are rejected.

The configuration uses the local test tokenizer assets and the `c4_test`
dataset. For a CPU-only functional check, use the unit test below rather than
the distributed launcher.

## Debug-model shape

| Field | Value |
| --- | ---: |
| Vocabulary | 2048 |
| Hidden size / layers / attention heads | 256 / 8 / 8 |
| Q-LoRA / KV-LoRA rank | 128 / 64 |
| QK non-RoPE / RoPE / value head dimensions | 32 / 32 / 64 |
| Dense FFN hidden size | 1024 |
| MoE hidden size / routed experts / shared experts / top-k | 256 / 8 / 1 / 2 |
| Leading dense layers | 1 |
| Indexer heads / head dimension / top-k | 4 / 64 / 8 |
| Maximum sequence length / RoPE theta | 128 / 1,000,000 |
| RMSNorm epsilon / attention dropout | 1e-5 / 0.0 |

The first decoder layer has a dense FFN and the remaining seven have grouped,
sigmoid-routed MoE plus a shared expert. Every layer has a full DSA indexer.
The model retains GLM-5's Q-LoRA/KV-LoRA multi-head latent attention (MLA),
interleaved RoPE, per-query DSA top-k selection, and the FP32 router correction
path, while reducing the dimensions and expert count above.

The GLM-5-specific absorbed MLA, DSA indexer, and `SparseMLA` PyTorch reference
live in `model.py`. Unlike the earlier dense reference, the main attention
gathers only selected compressed KV rows and never expands full per-head K/V.
Embeddings,
linear layers, RMSNorm/LayerNorm, interleaved RoPE, dense FFNs, routing,
routed/shared MoE experts, decoder blocks, and parameter initialization reuse
`torchtitan/models/common`.

## Verification

CPU forward, causal language-model loss, and backward are covered by the
native GLM-5 unit tests. Run the focused CPU and optional parity suite with:

```bash
python -m pytest \
  tests/unit_tests/test_glm5.py \
  tests/unit_tests/test_glm5_parity.py \
  tests/unit_tests/test_config_manager.py -v
```

On a host with one CUDA GPU and Transformers installed, the same parity module
executes the FP32 component comparisons and the BF16 end-to-end forward/loss/
gradient comparison after state-dict conversion:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  tests/unit_tests/test_glm5_parity.py -v
```

The CPU tests pass in the development environment. CUDA parity is an
acceptance gate, not a CPU substitute: it is skipped on a host without CUDA
and must be run on a one-GPU CUDA host before claiming GPU numerical
acceptance.

## Optional TileLang operators

The GLM-5 `ops` package contains optional TileLang indexer and SparseMLA
implementations. They reuse the same model component contracts and state-dict
layout, but require CUDA, BF16, TileLang, and the production operator geometry.
The debug flavors exercise the same data flow but do not satisfy those kernel
shapes.

For a compatible production trainer configuration, enable the operators with:

```bash
--override.imports \
torchtitan.models.glm5.ops.tilelang.tilelang_dsa_indexer,torchtitan.models.glm5.ops.tilelang.tilelang_sparse_mla
```

The default PyTorch implementation remains device-independent and is the
correctness path for GPU and NPU. An NPU package can replace either component
config independently without patching the GLM model; the current Turbo path
replaces SparseMLA and retains the reference indexer.

See [FULL_DSA.md](FULL_DSA.md) for the mathematical path, HF compatibility
mode, GPU/NPU operator mapping, and distributed boundaries.

## Current boundaries

The PyTorch indexer still materializes its complete score matrix before top-k,
so index selection remains quadratic without the optional fused indexer. The
main attention itself is sparse and scales with the selected top-k.

The current flavor explicitly does not support:

- KV cache or incremental decoding;
- MTP layers or an auxiliary indexer training objective;
- CP load balancing or a communication-overlapped production DSA kernel;
- the `spmd_types` backend.

The correctness-first CP path gathers positions inside the GLM forward wrapper
and builds a local-query/global-key dense mask without changing the shared
Trainer. It then gathers the indexer's key projection before global top-k and
gathers compressed KV before sparse attention. This preserves GLM DSA semantics
but does not yet overlap global-key communication. Pipeline stages that contain
shared-index layers must begin on a full-index layer because top-k tensors are
not transferred across PP stage boundaries. The runtime accepts
DDP/HSDP/FSDP, CP, TP, PP, and EP on the `partial_dtensor` backend and rejects the
unsupported layouts above. The indexer runs under `torch.no_grad()` by design,
so language-model loss does not train its parameters. Its pretrained parameters
are explicitly frozen and excluded from optimizer state, while remaining part
of the model state dict and checkpoints.

## Roadmap

Before beginning these later stages, run the recorded CUDA parity suite on a
one-GPU CUDA host. It remains the numerical acceptance gate for this milestone
and is currently pending because this host has no CUDA device.

After the single-device correctness milestone, later work may add:

1. A communication-overlapped, index-aware SparseMLA CP kernel.
2. Validation and enablement of the `spmd_types` backend.
3. KV cache and incremental decoding.
4. Cross-stage IndexCache transport for arbitrary PP boundaries.
5. MTP layers and released-checkpoint coverage.
6. An explicit auxiliary or distillation objective for training indexers from
   random initialization.

Each stage requires its own correctness and numerical-validation design; none
is implied by the debug milestone.
