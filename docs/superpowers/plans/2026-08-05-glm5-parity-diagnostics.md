# GLM-5 Parity Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate FP32 Indexer parity from harmless RMSNorm rounding and attach layer, position, Indexer, and MoE diagnostics to BF16 logits failures.

**Architecture:** Keep all changes in the optional parity test module. Pure helpers summarize detached tensors; temporary hooks capture corresponding HF and TorchTitan intermediates during the existing end-to-end forwards, and diagnostics are formatted only when the logits assertion fails.

**Tech Stack:** Python 3.12, PyTorch, `unittest`, pytest, Hugging Face Transformers.

## Global Constraints

- Modify only `tests/unit_tests/test_glm5_parity.py` plus this plan.
- Keep BF16 logits tolerance at `rtol=5e-2, atol=5e-2`.
- Keep successful test output quiet.
- Do not change model, adapter, kernel, or random-number behavior.
- Detach captured tensors and remove temporary hooks in a `finally` block.
- Do not stage unrelated changes.

---

### Task 1: Pure diagnostic summaries

**Files:**
- Modify: `tests/unit_tests/test_glm5_parity.py`
- Test: `tests/unit_tests/test_glm5_parity.py`

**Interfaces:**
- Produces: `_format_parity_diagnostics(records: dict[str, dict[int, torch.Tensor]], positions_BL: torch.Tensor) -> str`.
- Consumes record keys `hf_blocks`, `titan_blocks`, `hf_indexer`, `titan_indexer`, `hf_router`, and `titan_router`.

- [x] **Step 1: Write the failing CPU test**

Add `TestGlm5ParityDiagnostics.test_format_reports_layer_position_and_discrete_mismatches`. Use two synthetic layers of block outputs and selection indices. Assert the result contains:

```python
"layer 0: block_max_abs=0"
"layer 1: block_max_abs=0.5"
"block_position_max_abs=[0:0, 1:0.5, 2:0]"
"indexer_mismatched_queries=1 positions=[2]"
"router_mismatched_queries=1 positions=[1]"
"first_discrete_mismatch=layer 1 position 1 source=router"
```

Include an Indexer difference that selects only a future token for an early query, proving causally invalid selections are ignored.

- [x] **Step 2: Run the test and verify RED**

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/h50064611/torchtitan/.miniconda3/envs/torchtitan-dev/bin/python -m pytest tests/unit_tests/test_glm5_parity.py::TestGlm5ParityDiagnostics::test_format_reports_layer_position_and_discrete_mismatches -v
```

Expected: FAIL because `_format_parity_diagnostics` is undefined.

- [x] **Step 3: Implement the pure helpers**

Add:

```python
def _selection_mismatch_positions(
    actual_BLK: torch.Tensor,
    expected_BLK: torch.Tensor,
    positions_BL: torch.Tensor | None = None,
) -> list[int]:
    """Return query positions whose selected sets differ."""


def _format_parity_diagnostics(
    records: dict[str, dict[int, torch.Tensor]],
    positions_BL: torch.Tensor,
) -> str:
    """Format layer-local block, Indexer, and router parity evidence."""
```

For Indexer selections, ignore indices greater than the causal position and compare the remaining sets. For router selections, compare sorted expert sets. Validate corresponding shapes. Emit all per-position block differences and the earliest discrete mismatch ordered by layer, position, then source.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: PASS.

---

### Task 2: Isolate FP32 Indexer parity

**Files:**
- Modify: `tests/unit_tests/test_glm5_parity.py:228-250`
- Test: `tests/unit_tests/test_glm5_parity.py::TestGlm5TransformersComponentParity::test_indexer_topk_matches_transformers_exactly`

**Interfaces:**
- Consumes the existing HF and TorchTitan query residuals.
- Produces a tolerant upstream check and exact same-input Indexer check.

- [x] **Step 1: Preserve the observed RED evidence**

Use the supplied CUDA failure as the red result: 220/4096 elements differ, maximum absolute difference `2.384185791015625e-07`, with HF `MulBackward0` and TorchTitan `FusedRmsNormBackward0`.

- [x] **Step 2: Apply the minimal isolation change**

```python
torch.testing.assert_close(
    titan_q_resid,
    hf_q_resid,
    rtol=1e-6,
    atol=1e-7,
)
```

Pass `hf_q_resid` to both Indexers. Keep exact integer parity using `torch.testing.assert_close(titan_topk, hf_topk, rtol=0, atol=0)` for useful mismatch details.

- [x] **Step 3: Run the focused CUDA test when available**

```bash
CUDA_VISIBLE_DEVICES=4 python -m pytest tests/unit_tests/test_glm5_parity.py::TestGlm5TransformersComponentParity::test_indexer_topk_matches_transformers_exactly -v
```

Expected: PASS. Report an unavailable CUDA device or Transformers dependency explicitly.

---

### Task 3: Capture BF16 layer and routing evidence

**Files:**
- Modify: `tests/unit_tests/test_glm5_parity.py:341-462`
- Test: `tests/unit_tests/test_glm5_parity.py::TestGlm5TransformersBfloat16Parity::test_end_to_end_bfloat16_output_loss_moe_and_gradients`

**Interfaces:**
- Consumes `_format_parity_diagnostics` from Task 1.
- Produces detached records under the six documented record keys.

- [x] **Step 1: Add temporary capture hooks**

Before the full model calls, register decoder-layer and Indexer hooks. For routed layers, capture HF gate output index `2` and TorchTitan router output index `1`. Capture HF decoder output index `0` and the direct TorchTitan decoder tensor. Each hook stores `tensor.detach()` and returns nothing.

```python
try:
    hf_outputs = self.hf_model(...)
    titan_logits = self.titan_model(...)
finally:
    for handle in hook_handles:
        handle.remove()
```

- [x] **Step 2: Attach diagnostics only to logits failure**

```python
try:
    torch.testing.assert_close(
        titan_logits,
        hf_logits,
        rtol=5e-2,
        atol=5e-2,
    )
except AssertionError as error:
    diagnostics = _format_parity_diagnostics(records, self.positions)
    raise AssertionError(
        f"{error}\n\nGLM-5 parity diagnostics:\n{diagnostics}"
    ) from error
```

Do not alter loss, backward, gradient, or Indexer-gradient assertions.

- [x] **Step 3: Run the BF16 CUDA test when available**

```bash
CUDA_VISIBLE_DEVICES=4 python -m pytest tests/unit_tests/test_glm5_parity.py::TestGlm5TransformersBfloat16Parity::test_end_to_end_bfloat16_output_loss_moe_and_gradients -v
```

Expected: PASS or the existing logits failure augmented with every diagnostic field. A diagnostic failure remains a parity failure.

---

### Task 4: Verify and commit

**Files:**
- Modify: `tests/unit_tests/test_glm5_parity.py`
- Create: `docs/superpowers/plans/2026-08-05-glm5-parity-diagnostics.md`

**Interfaces:**
- Consumes all earlier tasks.
- Produces one implementation commit without unrelated files.

- [x] **Step 1: Run static validation**

```bash
git diff --check
python -m compileall -q tests/unit_tests/test_glm5_parity.py
```

Expected: both commands exit 0.

- [x] **Step 2: Run the full parity module**

```bash
python -m pytest tests/unit_tests/test_glm5_parity.py -v -rs
```

Expected locally: CPU diagnostics pass; CUDA classes pass or skip for a documented missing dependency/device. On the target CUDA host, FP32 Indexer passes and any remaining BF16 failure includes diagnostics.

- [x] **Step 3: Review scope**

```bash
git diff -- tests/unit_tests/test_glm5_parity.py docs/superpowers/plans/2026-08-05-glm5-parity-diagnostics.md
git status --short
```

Expected: only the test and plan differ from the design commit.

- [x] **Step 4: Commit**

```bash
git add tests/unit_tests/test_glm5_parity.py docs/superpowers/plans/2026-08-05-glm5-parity-diagnostics.md
git commit -m "test: improve GLM-5 parity diagnostics"
```

- [x] **Step 5: Verify repository state**

```bash
git status --short --branch
git log -2 --oneline
```

Expected: clean tree with separate design and implementation commits.
