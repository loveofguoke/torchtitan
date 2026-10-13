# GLM-5 model

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

The cross-repository interfaces and mandatory downstream checks are documented
in [DEPENDENCY_AUDIT.md](DEPENDENCY_AUDIT.md).

## Components

- `model.py`: decoder, transformer block, MLA projections, and IndexShare data
  flow.
- `dsa.py`: DSA indexer, causal top-k selection, BlockMask construction, and
  the FlexAttention reference path.
- `sharding.py`: declarative TP, SP, CP, EP, FSDP, and DTensor placements.
- `parallelize.py`: model parallelization order and the correctness-first CP
  gather path.
- `state_dict_adapter.py`: Hugging Face/TorchTitan checkpoint key and expert
  layout conversion.
- `config_registry.py`: runnable TorchTitan trainer configurations.

## Registered architecture flavors

`glm5_configs` contains both reduced development shapes and the released
GLM-5.2 language-model decoder shape:

| Flavor | Purpose | Source |
| --- | --- | --- |
| `debugmodel` | Small independent-indexer correctness model | TorchTitan test configuration |
| `shared_dsa_debugmodel` | Small cross-layer IndexShare correctness model | TorchTitan test configuration |
| `glm5_2_debugmodel` | Small model preserving the released dense/MoE and IndexShare layer patterns | Derived from `zai-org/GLM-5.2` |
| `glm5_2_scaled_debugmodel` | Approximately 420M-parameter long-run validation model that fits the single-device and artifact-transfer contracts | Scaled from `zai-org/GLM-5.2` |
| `744B-A40B` | Released 78-layer GLM-5.2 decoder architecture | `zai-org/GLM-5.2` |

The released flavor uses vocabulary 154880, hidden size 6144, 78 layers,
64 attention heads, Q-LoRA rank 2048, KV-LoRA rank 512, three leading dense
layers, 256 routed experts with top-8 routing, one shared expert, and a
1,048,576-token maximum context. Its indexer has 32 heads of width 128 and
selects up to 2048 keys. RoPE uses width 64 and theta 8,000,000.

The cross-layer index schedule is also taken from the released config:
layers 0, 1, and 2 compute indices; layers 3--5 reuse layer 2; layer 6
computes; layers 7--9 reuse layer 6; this four-layer cadence continues through
layer 77. This is represented by `Glm5Model.Config.index_sources`.

Primary configuration references:

- [zai-org/GLM-5.2 config.json](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [Transformers GLM-MoE-DSA configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py)
- [Transformers GLM-MoE-DSA model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py)
- [slime GLM-5.2 744B-A40B recipe](https://github.com/THUDM/slime/blob/main/docs/en/examples/glm5.2-744B-A40B.md)
- [slime model arguments](https://github.com/THUDM/slime/blob/main/scripts/models/glm5.2-744B-A40B.sh)

The flavor records the public architecture; it is not yet a claim of complete
released-checkpoint training support. The checkpoint contains one additional
MTP layer, which this decoder does not yet construct. In addition, the current
TorchTitan layer builder still materializes an unused indexer on IndexShare
consumer layers, while the released Transformers and slime implementations
omit those weights. These two differences must be closed before strict full
checkpoint compatibility can be claimed.

The BF16 `zai-org/GLM-5.2` and FP8 `zai-org/GLM-5.2-FP8` releases share this
architecture. FP8 is a weight/compute representation and converter concern,
not a second model shape, so it is intentionally not registered as a duplicate
architecture flavor.

## Supported features

| Feature | Current scope |
| --- | --- |
| FSDP / HSDP / DDP | Decoder-layer data parallelism is implemented. |
| Tensor Parallelism | MLA heads and dense/MoE projections are sharded; the correctness-first indexer is replicated. |
| Expert Parallelism | Routed experts use TorchTitan's common token-choice MoE path. |
| Context Parallelism | Local queries gather global index keys and attention K/V; contiguous token shards only. |
| Pipeline Parallelism | Decoder stages and cross-stage IndexShare payloads are implemented. |
| Activation checkpointing | Uses the common decoder activation-checkpoint path. |
| Checkpoint conversion | BF16 HF/TorchTitan key conversion exists; strict released-checkpoint coverage is pending MTP and shared-layer cleanup. |
| `torch.compile` | Not yet claimed as an upstream-supported GLM configuration. |
| FP8 | The public FP8 checkpoint shares the architecture, but its converter is not implemented here. |
| Long-context production kernel | Not available; the current DSA path is a correctness reference. |

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

The default configuration uses TorchTitan's standard PyTorch all-to-all token
dispatcher. A separate opt-in configuration exercises the GPU DeepEP v2
backend with EP=4:

```bash
NGPU=4 MODULE=glm5 CONFIG=glm5_debugmodel_deepep ./run_train.sh
```

This requires a compatible `deep_ep` v2 installation and supported NVIDIA
hardware. It does not select the separate DeepEP-Ascend prototype.

The shared-index DSA debug configuration reuses one Top-K selection across
each adjacent pair of layers:

```bash
NGPU=1 MODULE=glm5 CONFIG=glm5_shared_dsa_debugmodel ./run_train.sh
```

For upstream CI and GLM-5.2-specific regression coverage, use the structurally
faithful debug flavor. It keeps the small dimensions but uses three leading
dense layers followed by five MoE layers and the released IndexShare cadence:

```bash
NGPU=1 MODULE=glm5 CONFIG=glm5_2_debugmodel ./run_train.sh
```

For longer numerical and graph-mode validation, use the separate scaled debug
flavor. It leaves the CI-oriented debug flavors unchanged:

```bash
NGPU=1 MODULE=glm5 CONFIG=glm5_2_scaled_debugmodel ./run_train.sh
```

The scaled flavor has 16 layers so PP8 owns two layers per stage. It retains
three leading dense layers, the released four-layer IndexShare cadence, Top-8
routing, the released MLA/DSA head dimensions, RoPE theta, normalization
epsilon, and a routed-expert count divisible by EP8. Hidden size, vocabulary,
Q/KV LoRA ranks, expert width, and total routed experts are reduced so the
unsharded optimizer state, compilation workspace, and activations fit the
smaller single-device validation accelerator. Its dimensions are 768 hidden,
8 attention heads, 24 routed experts, 384 expert hidden, vocabulary 32768,
and maximum context length 4096. Its uncompressed FP32 parameters remain below
2 GiB so the canonical checkpoint can use the existing GPU/NPU artifact path
without multi-part transport. This is a validation workload, not a claim
that its throughput or convergence predicts the released 744B model.

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
| Leading dense layers | 1 (`debugmodel`) or 3 (`glm5_2_debugmodel`) |
| Indexer heads / head dimension / top-k | 4 / 64 / 8 |
| Maximum sequence length / RoPE theta | 128 / 1,000,000 |
| RMSNorm epsilon / attention dropout | 1e-5 / 0.0 |

In the legacy `debugmodel`, the first decoder layer has a dense FFN and the
remaining seven have grouped, sigmoid-routed MoE plus a shared expert. The
`glm5_2_debugmodel` changes that split to three dense plus five MoE layers.
Both retain GLM-5's Q-LoRA/KV-LoRA multi-head latent attention (MLA),
interleaved RoPE, per-query DSA top-k selection, and the FP32 router correction
path, while reducing the dimensions and expert count above.

The GLM-5-specific MLA, DSA indexer, and `DSAInnerAttention` reference kernel
live in `model.py`. Embeddings,
linear layers, RMSNorm/LayerNorm, interleaved RoPE, dense FFNs, routing,
routed/shared MoE experts, decoder blocks, and parameter initialization reuse
`torchtitan/models/common`.

`glm5_2_debugmodel` uses the same reduced dimensions and therefore remains a
small CI model. Unlike the legacy debug flavor, layers 0--2 are dense and
layers 3--7 are MoE, while its index sources are
`(0, 1, 2, 2, 2, 2, 6, 6)`. This covers the released layer transitions without
requiring the 744B model in CI.

Debug-model throughput, MFU, communication ratio, and kernel distribution are
not production-performance estimates. Small hidden sizes underfill matrix
engines, collectives have different message sizes, and fixed framework/kernel
overheads occupy a much larger fraction of a step. Performance claims for the
released model require a representative scaled configuration or the official
shape on target hardware; debug results are only correctness and code-path
evidence.

## Verification

CPU forward, causal language-model loss, backward, CP helpers, state-dict
conversion, and FLOP accounting are covered by the native GLM-5 unit tests:

```bash
python -m pytest \
  tests/unit_tests/test_glm5.py \
  tests/unit_tests/test_state_dict_adapter.py -v
```

The MFU estimate follows TorchTitan's common MoE convention, but charges the
frozen DSA indexer for forward work only and includes its query/key score and
per-head score-reduction terms. Top-k selection and elementwise operations are
still excluded, as they are in the framework's other model-level estimates.

Exploratory Titan/HF and GPU/NPU parity lives in the separate
`torchtitan-test` repository. From an adjacent source checkout, run its stable
pytest entry or scenario workflow instead of looking for parity code here:

```bash
cd ../torchtitan-test
python -m pytest \
  tests/unit_tests/test_glm5_2_parity_artifacts.py \
  tests/unit_tests/test_glm5_2_parity_workflow.py -q

python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --data
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --actual-capture
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --expected-capture
python tests/glm5_2_parity/titan_hf_gpu_fp32_random.py --compare
```

The exact GPU/NPU commands, fixture transport, rerun lifecycle, and report
semantics are maintained in `torchtitan-test/tests/glm5_2_parity/README.md`.
CPU tests do not replace accelerator parity; rerun the affected scenario after
model mathematics, state conversion, routing, precision, or parallel input
semantics change.

### Evidence required for an upstream submission

Like the Qwen3.5 and Kimi model READMEs, the final upstream README should report
measured numerical evidence rather than only list test files. The following
results must be generated from a fixed commit and retained with the PR:

| Experiment | Result to publish | Current status |
| --- | --- | --- |
| Hugging Face vs TorchTitan, one GPU, FP32 | logits max/mean absolute error, cosine similarity, KL divergence, top-1/top-5 match | Pending official-checkpoint run |
| Hugging Face vs TorchTitan, BF16 | loss and grad-norm error over fixed tokens | Pending official-checkpoint run |
| No parallelism vs FSDP/TP/CP/EP/PP | maximum logits/loss/gradient difference for each representative topology | Test harness exists; publish CUDA results before PR |
| HF -> TorchTitan -> HF checkpoint | missing/unexpected keys and tensor-wise maximum error | Pending strict released-checkpoint coverage |
| Save/resume | uninterrupted vs resumed loss and grad-norm sequence | Pending upstream integration recipe |
| Debug-model convergence | fixed dataset, token budget, loss curve, and final loss | Pending representative CUDA run |
| Performance | tokens/s, TFLOPS, MFU, peak memory, sequence length, and topology | Optional for correctness PR; required before a production-performance claim |

Do not paste NPU-only smoke results into these numerical-parity rows. NPU
results may be listed separately as downstream portability evidence, while the
upstream correctness result must be reproducible on TorchTitan's supported
CUDA/ROCm CI environment. A result row must include the model flavor, checkpoint,
dtype, sequence length, batch/token count, topology, device type, software
revisions, seed, and command; a bare "passed" is not sufficient.

## Current boundaries

DSA uses an eager dense score matrix followed by a top-k mask. Its time and
memory are quadratic in sequence length, so this implementation is suitable
only for the reduced debug sequence length and is not a Flash-MLA or dedicated
DSA-kernel implementation.

The current implementation explicitly does not support:

- KV cache or incremental decoding;
- Flash-MLA or a production specialized DSA kernel;
- MTP layers or an auxiliary indexer training objective;
- CP load balancing or a communication-overlapped production DSA kernel;
- the `spmd_types` backend.

The correctness-first CP path gathers positions inside the GLM forward wrapper
and builds a local-query/global-key dense mask without changing the shared
Trainer. It then gathers the indexer's key projection before global top-k and
gathers attention K/V before sparse attention. This preserves GLM DSA semantics
but does not yet reduce global-key communication or storage. The runtime accepts
DDP/HSDP/FSDP, CP, TP, PP, and EP on the `partial_dtensor` backend and rejects the
unsupported layouts above. The indexer runs under `torch.no_grad()` by design,
so language-model loss does not train its parameters. Its pretrained parameters
are explicitly frozen and excluded from optimizer state, while remaining part
of the model state dict and checkpoints.

## Roadmap

Later work may add:

1. A communication-overlapped, index-aware DSA or Flash-MLA CP kernel.
2. Validation and enablement of the `spmd_types` backend.
3. KV cache and incremental decoding.
4. Remove unused indexer modules from IndexShare consumer layers.
5. MTP layers and strict released-checkpoint coverage.
6. An explicit auxiliary or distillation objective for training indexers from
   random initialization.

Each stage requires its own correctness and numerical-validation design; none
is implied by the debug milestone.
