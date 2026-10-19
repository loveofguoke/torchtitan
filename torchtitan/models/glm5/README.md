# GLM-5

This directory implements the GLM-5.2 decoder architecture for TorchTitan.
The implementation follows TorchTitan's config-first model contract and uses
shared PyTorch-native components for distributed training.

## Model support

- Multi-head latent attention (MLA)
- DeepSeek sparse attention (DSA) with a frozen learned indexer
- Dense-to-MoE layer transition and shared experts
- Cross-layer reuse of DSA index selections
- One full NextN multi-token-prediction (MTP) block for the released model
- Data, tensor, context, pipeline, and expert parallel configuration through
  TorchTitan's common model lifecycle
- Hugging Face checkpoint conversion

Available flavors are defined in `flavors.py`:

- `debugmodel`: small development model
- `dsa_debugmodel`: alias for the DSA development model
- `shared_dsa_debugmodel`: development model exercising index sharing
- `glm5_2_debugmodel`: small model preserving the released layer schedule
- `glm5_2_debugmodel_mtp`: small model with one trainable NextN/MTP objective
- `glm5_2_scaled_debugmodel`: larger validation model
- `744B-A40B`: released GLM-5.2 architecture

Runnable development recipes live in
`torchtitan_recipes/tests/models/glm5.py`. The released flavor describes the
full architecture and is not intended for single-host smoke tests.

## Files

- `flavors.py`: model flavors and nested config construction
- `model.py`: decoder, transformer block, MLA, and DSA integration
- `dsa.py`: learned indexer and sparse FlexAttention mask construction
- `sharding.py`: GLM-specific declarative SPMD layouts
- `state_dict_adapter.py`: Hugging Face checkpoint conversion
- `mtp.py`: GLM-specific full-DSA MTP block built on the common MTP lifecycle

## Multi-token prediction

The released GLM-5.2 checkpoint declares 78 target decoder layers and one
``num_nextn_predict_layers`` block.  The extra block is stored as
``model.layers.78`` and contains the complete DSA attention and MoE sublayers,
plus ``enorm``, ``hnorm``, ``eh_proj``, and ``shared_head.norm``.  It reuses the
model's token embedding and top-level language-model head rather than storing a
second vocabulary projection.

TorchTitan therefore models training-time MTP as a real auxiliary decoder
depth, not as an inference-only flag.  The dataloader prepares a shifted token
and label sequence, the GLM block builds the corresponding DSA mask, and
``MTPLoss`` adds the weighted auxiliary cross-entropy objective.  See
[`MTP.md`](MTP.md) for the data, model, checkpoint, parallelism, and validation
contracts.

The checkpoint field ``index_share_for_mtp_iteration`` belongs to iterative
speculative decoding: draft step zero computes a sparse Top-K selection and
later draft iterations may reuse it.  A training batch executes the single
NextN block once per objective, so that serving optimization is deliberately
not exposed as a training configuration or a silent fallback here.

## Validation

Run the CPU model tests with:

```bash
pytest tests/unit_tests/cpu/models/glm5 -q
```

For numerical changes, compare loss and gradient norm with identical seeds,
parallelism, and deterministic settings as described in the repository
development guide.
