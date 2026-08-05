# GLM-5 Parity Test Isolation and Failure Diagnostics Design

## Goal

Make the GLM-5 Transformers parity tests distinguish harmless floating-point
differences from real Indexer or routed-MoE divergence, while keeping successful
test output quiet and making failures actionable.

## Scope

The change is limited to `tests/unit_tests/test_glm5_parity.py`. It does not
change the GLM-5 model, state-dict adapter, numerical kernels, or acceptance
tolerances for final BF16 outputs.

## FP32 Indexer Isolation

The existing component test computes the HF and TorchTitan query residuals
independently and requires exact equality before calling either Indexer. The two
RMSNorm implementations use different CUDA paths and can differ by one FP32
ULP even when they are numerically equivalent.

The revised test will:

1. Compare the independently computed query residuals with `rtol=1e-6` and
   `atol=1e-7`.
2. Use the HF query residual as the common input to both Indexers.
3. Continue to require exact equality of the returned integer top-k indices.

This separates the Linear/RMSNorm parity contract from the Indexer parity
contract. It does not alter the BF16 end-to-end path, where each model must
continue computing its own query residuals.

## BF16 Failure Diagnostics

The BF16 test will collect comparable intermediate results during the normal HF
and TorchTitan forward passes without printing during successful runs. If the
final logits assertion fails, its failure message will include compact
diagnostics for each decoder layer:

- maximum absolute block-output difference;
- maximum absolute block-output difference by sequence position;
- DSA Indexer top-k mismatch count and affected sequence positions;
- routed-MoE expert-index mismatch count and affected sequence positions;
- the first layer and position showing a discrete selection mismatch.

Forward hooks will be registered only for the duration of the diagnostic
forward pass and removed in a `finally` block. Captured tensors will be detached
so the diagnostic path does not retain autograd graphs. Diagnostics must not
change model inputs, outputs, routing decisions, or random-number state.

The existing `rtol=5e-2, atol=5e-2` logits acceptance threshold will remain
unchanged. A large mismatch will therefore stay a failure instead of being
hidden by a wider tolerance.

## Test Structure and Error Handling

Small test-only helpers will format mismatch counts and per-position maximum
absolute errors. Helpers will validate that corresponding HF and TorchTitan
records have compatible shapes and report missing records explicitly rather
than silently omitting them.

The end-to-end test will still compare block output, logits, loss, gradients,
and Indexer gradient exclusion. Diagnostic collection is supplemental and will
only affect the assertion message for a logits failure.

## Verification

Verification will cover:

1. A focused CPU unit test for diagnostic formatting and mismatch accounting.
2. The CUDA Transformers component parity test, demonstrating that the relaxed
   query-residual comparison passes while exact Indexer top-k parity remains
   enforced.
3. The CUDA BF16 parity test, demonstrating either a pass or a failure containing
   the new layer, position, Indexer, and MoE diagnostics.
4. The full `tests/unit_tests/test_glm5_parity.py` suite.

If CUDA or Transformers are unavailable in the implementation workspace, CPU
tests and static checks will still run, and the unavailable CUDA verification
will be reported explicitly rather than described as passing.

## Version Control

The design document and implementation will be kept in separate Git commits.
No unrelated working-tree changes will be staged or committed.
