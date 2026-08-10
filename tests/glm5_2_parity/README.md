# Portable GLM-5.2 parity workflow

The GLM-5.2 parity suite supports three execution modes:

- `paired` (default): run two endpoints in one process and immediately report.
- `capture`: run one endpoint and write a portable, checksummed artifact.
- `compare`: compare two artifacts on CPU and write the same HTML diagnostics.

Artifacts contain a versioned JSON manifest, exact fixture tensors, module
activation and gradient traces, discrete routing selections, logits, loss,
canonical parameters, and canonical gradients. Tensor data is stored in
checksummed safetensors shards. Python stdout/stderr is attached as
`attachments/runtime.log`. A completed artifact is an immutable directory. If
execution fails after capture starts, the partial observations, runtime log,
and exception are written with `status=failed`; failed runs cannot be compared.
Because parameters and gradients are retained for offline comparison, budget at
least two model-state sizes per artifact, plus activation traces and fixtures.

## Existing paired command

torchtitan glm5.2 vs hf glm5.2
- gpu
- fp32

Existing commands remain valid; `GLM5_PARITY_MODE=paired` is implicit:

```bash
CUDA_VISIBLE_DEVICES=7 \
GLM5_PARITY_ACTUAL=titan:fp32 \
GLM5_PARITY_EXPECTED=hf:fp32 \
GLM5_PARITY_HF_ROUTED_EXPERT_COMPUTE=model \
GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE=fp32 \
GLM5_PARITY_LAYERS=all \
GLM5_PARITY_COMPONENTS=all \
GLM5_PARITY_DATA_CASE=random \
GLM5_PARITY_REPORT_DIR=parity_reports \
python -m pytest \
tests/unit_tests/test_glm5_parity.py::TestGlm5Parity::test_configured_precision_suite \
-s \
> parity_reports/glm5_parity_test.log 2>&1
```

## Offline GPU/NPU comparison

torchtitan glm5.2 gpu vs npu
- fp32&bf16

All capture commands in one comparison must use the same model, data,
component, layer, and compute-mode settings. Start from a clean, identical Git
commit on both servers.

First capture the TorchTitan GPU baseline. Each capture independently creates
the model weights and test batches on CPU from the model seed, case seed, test
ordinal, and effective configuration. The artifact stores their checksums plus
all local intermediate results. No HF model or reference artifact participates
in this workflow:

```bash
CUDA_VISIBLE_DEVICES=7 \
GLM5_PARITY_MODE=capture \
GLM5_PARITY_ENDPOINT=titan:fp32 \
GLM5_PARITY_ARTIFACT=parity_artifacts/titan-gpu-fp32 \
GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE=model \
GLM5_PARITY_LAYERS=all \
GLM5_PARITY_COMPONENTS=all \
GLM5_PARITY_DATA_CASE=random \
python -m pytest \
tests/unit_tests/test_glm5_parity.py::TestGlm5Parity::test_configured_precision_suite \
-s \
> parity_reports/glm5_parity_test.log 2>&1
```

Capture TorchTitan NPU on Ascend server:

```bash
ASCEND_RT_VISIBLE_DEVICES=4 \
GLM5_PARITY_DEVICE=npu \
GLM5_PARITY_MODE=capture \
GLM5_PARITY_ENDPOINT=titan:fp32 \
GLM5_PARITY_ARTIFACT=parity_artifacts/titan-npu-fp32 \
GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE=model \
GLM5_PARITY_LAYERS=all \
GLM5_PARITY_COMPONENTS=all \
GLM5_PARITY_DATA_CASE=random \
python -m pytest \
tests/unit_tests/test_glm5_parity.py::TestGlm5Parity::test_configured_precision_suite \
-s \
> parity_reports/glm5_parity_test.log 2>&1
```

After copying the NPU artifact back, compare Titan NPU with Titan GPU. This step
does not construct a model and does not require an accelerator:

```bash
GLM5_PARITY_MODE=compare \
GLM5_PARITY_ACTUAL_ARTIFACT=parity_artifacts/titan-npu-fp32 \
GLM5_PARITY_EXPECTED_ARTIFACT=parity_artifacts/titan-gpu-fp32 \
GLM5_PARITY_REPORT=parity_reports/titan-npu-vs-gpu-fp32.html \
python -m pytest \
tests/unit_tests/test_glm5_parity.py::TestGlm5Parity::test_configured_precision_suite \
-s \
> parity_reports/glm5_parity_test.log 2>&1
```

Repeat the workflow with `bf16` endpoints. For BF16, use
`GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE=model`; the explicit `fp32` routed
expert experiment is intentionally restricted to an FP32 Titan endpoint.

Comparison rejects different test plans, fixture tensors, effective
configuration, Git commits, incomplete artifacts, and corrupt shards. Dirty
source trees are rejected by default. `GLM5_PARITY_ALLOW_DIRTY=1` exists only
for exploratory development and weakens source reproducibility.

## Framework self-check

`tests/unit_tests/test_glm5_2_parity_artifacts.py` is not part of capture or
compare execution. It is a fast CPU regression suite for the artifact protocol:
dtype-preserving round trips, checksums, incomplete or failed runs, configuration
and fixture mismatch rejection, offline tensor comparison, the report contents
links, and the default model-size budget. Run it after changing the framework:

```bash
python -m pytest tests/unit_tests/test_glm5_2_parity_artifacts.py -q
```
