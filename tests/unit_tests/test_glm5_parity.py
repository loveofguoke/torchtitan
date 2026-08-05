# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable single-device GLM-5 parity tests.

The module separates five concerns:

* ``ParityDataFactory`` creates repeatable inputs and masks.
* ``ParityModelPair`` owns the HF/TorchTitan model pair and layer lookup.
* ``ParityRecorder`` stores scalar and discrete comparison results.
* ``LayerTrace`` captures reusable layer, indexer, and router observations.
* ``RecursiveModuleTrace`` captures every tensor leaf and its gradient under selected layers and top-level modules.
* ``TestGlm5Parity`` selects implementation x precision endpoints at runtime.
* The single test class declares the component, layers, precision, and data.

The reference model is optional and all HF comparisons are CUDA-gated.  The
native router precision tests remain CPU-safe.  Reports include canonical
module paths, mismatch positions, and endpoint precision labels.
"""

from __future__ import annotations

import os
import re
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from html import escape
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.glm5 import Glm5StateDictAdapter, glm5_configs

_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
try:
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
except Exception as exc:  # pragma: no cover - environment dependent
    _TRANSFORMERS_IMPORT_ERROR = exc
    GlmMoeDsaConfig = None
    GlmMoeDsaForCausalLM = None


@dataclass(frozen=True)
class PrecisionPolicy:
    """Numerical policy for one parity run."""

    name: str
    dtype: torch.dtype
    rtol: float
    atol: float
    exact_discrete: bool = True


FP32 = PrecisionPolicy("fp32", torch.float32, 1e-4, 1e-5)
BF16 = PrecisionPolicy("bf16", torch.bfloat16, 5e-2, 5e-2)
TOP_LEVEL_COMPONENTS = {"tok_embeddings", "norm", "lm_head"}


@dataclass
class ParityBatch:
    """All inputs shared by a component or end-to-end comparison."""

    hidden_states: torch.Tensor
    positions: torch.Tensor
    causal_mask: torch.Tensor
    hf_position_embeddings: tuple[torch.Tensor, torch.Tensor]
    tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class ModelEndpoint:
    """One model implementation at one numerical precision."""

    implementation: str
    precision: PrecisionPolicy

    @property
    def label(self) -> str:
        return f"{self.implementation}:{self.precision.name}"


@dataclass(frozen=True)
class ComparisonSpec:
    """The two endpoints and tolerance used by one comparison."""

    actual: ModelEndpoint
    expected: ModelEndpoint
    scope: str
    component: str
    rtol: float
    atol: float

    @property
    def label(self) -> str:
        return f"{self.actual.label} vs {self.expected.label}"


class ParityDataFactory:
    """Build repeatable batches without embedding data assumptions in tests."""

    @staticmethod
    def make(
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 2,
        sequence_length: int = 16,
        seed: int = 41,
        vocab_size: int = 2048,
        hidden_size: int = 256,
        data_case: str = "random",
        make_tokens: bool = False,
    ) -> ParityBatch:
        data_case = data_case.lower()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        # Generate deterministic synthetic data for repeatable comparisons.
        if data_case == "zeros":
            hidden_states = torch.zeros(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype,
            )
        elif data_case == "ones":
            hidden_states = torch.ones(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype,
            )
        elif data_case == "extreme":
            hidden_states = 20.0 * torch.randn(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype, generator=generator,
            )
        elif data_case == "alternating":
            values = torch.tensor(
                (-1.0, 1.0), device=device, dtype=dtype
            )
            pattern = torch.arange(
                batch_size * sequence_length * hidden_size,
                device=device,
            ).reshape(batch_size, sequence_length, hidden_size)
            hidden_states = values[(pattern % 2).long()]
        elif data_case == "random":
            hidden_states = torch.randn(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype, generator=generator,
            )
        else:
            raise ValueError(f"unsupported parity data case: {data_case}")
        positions = torch.arange(
            sequence_length, device=device, dtype=torch.long
        ).expand(batch_size, -1)
        causal_mask = _causal_mask(positions, dtype)
        if make_tokens:
            if data_case == "zeros":
                tokens = torch.zeros(
                    batch_size, sequence_length, device=device, dtype=torch.long
                )
            elif data_case == "ones":
                tokens = torch.full(
                    (batch_size, sequence_length), vocab_size - 1,
                    device=device, dtype=torch.long,
                )
            elif data_case == "alternating":
                tokens = torch.arange(
                    batch_size * sequence_length, device=device
                ).reshape(batch_size, sequence_length) % vocab_size
            else:
                tokens = torch.randint(
                    vocab_size,
                    (batch_size, sequence_length),
                    device=device,
                    dtype=torch.long,
                    generator=generator,
                )
        else:
            tokens = None
        return ParityBatch(
            hidden_states=hidden_states,
            positions=positions,
            causal_mask=causal_mask,
            hf_position_embeddings=(
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            ),
            tokens=tokens,
        )

    @staticmethod
    def attach_hf_position_embeddings(
        batch: ParityBatch, hf_model: torch.nn.Module
    ) -> ParityBatch:
        batch.hf_position_embeddings = hf_model.model.rotary_emb(
            batch.hidden_states, position_ids=batch.positions
        )
        return batch

    @staticmethod
    def cast(batch: ParityBatch, *, model: torch.nn.Module, dtype: torch.dtype) -> ParityBatch:
        """Cast numeric inputs for one endpoint and recompute its RoPE cache."""
        hidden_states = batch.hidden_states.to(dtype=dtype)
        positions = batch.positions
        cast_batch = ParityBatch(
            hidden_states=hidden_states,
            positions=positions,
            causal_mask=batch.causal_mask.to(dtype=dtype),
            hf_position_embeddings=(
                torch.empty(0, device=hidden_states.device),
                torch.empty(0, device=hidden_states.device),
            ),
            tokens=batch.tokens,
        )
        if hasattr(model, "model"):
            cast_batch.hf_position_embeddings = model.model.rotary_emb(
                hidden_states, position_ids=positions
            )
        return cast_batch


@dataclass
class ParityModelPair:
    """A weight-identical HF/TorchTitan model pair."""

    hf: torch.nn.Module
    titan: torch.nn.Module
    device: torch.device
    precision: PrecisionPolicy

    def hf_layer(self, layer_index: int) -> torch.nn.Module:
        return self.hf.model.layers[layer_index]

    def titan_layer(self, layer_index: int) -> torch.nn.Module:
        return self.titan.layers[str(layer_index)]

    def convert(self, precision: PrecisionPolicy) -> None:
        self.hf.to(device=self.device, dtype=precision.dtype)
        self.titan.to(device=self.device, dtype=precision.dtype)
        for module in self.hf.modules():
            indexer = getattr(module, "indexer", None)
            if indexer is not None:
                indexer.weights_proj.float()
        for model in (self.hf, self.titan):
            indexer_weights = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.endswith("indexer.weights_proj.weight")
            ]
            expected = len(getattr(model, "model", model).layers)
            if len(indexer_weights) != expected:
                raise AssertionError(
                    "unexpected number of GLM-5 indexer weights: "
                    f"expected {expected}, got {len(indexer_weights)}"
                )
            for name, parameter in indexer_weights:
                if parameter.dtype is not torch.float32:
                    raise AssertionError(
                        f"{name} must remain FP32, got {parameter.dtype}"
                    )
        self.precision = precision


def _hf_config() -> Any:
    assert GlmMoeDsaConfig is not None
    return GlmMoeDsaConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=1024,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        index_topk=8,
        index_head_dim=64,
        index_n_heads=4,
        first_k_dense_replace=1,
        indexer_types=["full", "full", "full", "full"],
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        use_cache=False,
        _attn_implementation="eager",
    )


def _build_pair(
    device: torch.device,
    *,
    precision: PrecisionPolicy = FP32,
    seed: int = 41,
) -> ParityModelPair:
    assert GlmMoeDsaForCausalLM is not None
    torch.manual_seed(seed)
    # Build both implementations from one HF state dict so every comparison
    # tests execution differences rather than unrelated random weights.
    hf_model = GlmMoeDsaForCausalLM(_hf_config()).float()
    titan_config = glm5_configs["debugmodel"]()
    titan_model = titan_config.build()
    titan_model.init_states()  # Initialize decoder runtime state.
    adapter = Glm5StateDictAdapter(titan_config, hf_assets_path=None)
    # The adapter is the single source of truth for HF -> TorchTitan names.
    titan_model.load_state_dict(
        adapter.from_hf(hf_model.state_dict()), strict=True
    )
    pair = ParityModelPair(
        hf=hf_model.to(device).eval(),
        titan=titan_model.to(device).eval(),
        device=device,
        precision=FP32,
    )
    if precision.dtype is not torch.float32:
        pair.convert(precision)
    return pair


def _build_models(
    device: torch.device, *, seed: int = 41
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Backward-compatible FP32 constructor used by older parity callers."""
    pair = _build_pair(device, precision=FP32, seed=seed)
    return pair.hf, pair.titan


def _convert_models_to_bfloat16(
    hf_model: torch.nn.Module,
    titan_model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Backward-compatible conversion that preserves FP32 indexer weights."""
    pair = ParityModelPair(hf_model, titan_model, device, FP32)
    pair.convert(BF16)


def _selection_mismatch_positions(
    actual_BLK: torch.Tensor,
    expected_BLK: torch.Tensor,
    positions_BL: torch.Tensor | None = None,
) -> list[int]:
    """Return query positions whose selected sets differ."""
    if actual_BLK.shape != expected_BLK.shape:
        raise ValueError(
            "selection shapes differ: "
            f"actual={tuple(actual_BLK.shape)}, expected={tuple(expected_BLK.shape)}"
        )
    if actual_BLK.ndim != 3:
        raise ValueError(
            f"selections must have shape [B, L, K], got {actual_BLK.shape}"
        )
    if positions_BL is not None and positions_BL.shape != actual_BLK.shape[:2]:
        raise ValueError(
            "positions must match the selection batch and sequence dimensions: "
            f"positions={tuple(positions_BL.shape)}, selections={tuple(actual_BLK.shape)}"
        )
    actual_cpu = actual_BLK.detach().cpu()
    expected_cpu = expected_BLK.detach().cpu()
    positions_cpu = None if positions_BL is None else positions_BL.detach().cpu()
    mismatched_positions: list[int] = []
    for batch_index in range(actual_cpu.shape[0]):
        for sequence_index in range(actual_cpu.shape[1]):
            causal_position = (
                None
                if positions_cpu is None
                else int(positions_cpu[batch_index, sequence_index])
            )
            actual = {
                int(index)
                for index in actual_cpu[batch_index, sequence_index]
                if causal_position is None or int(index) <= causal_position
            }
            expected = {
                int(index)
                for index in expected_cpu[batch_index, sequence_index]
                if causal_position is None or int(index) <= causal_position
            }
            if actual != expected:
                mismatched_positions.append(sequence_index)
    return mismatched_positions


def _causal_mask(positions_BL: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch_size, sequence_length = positions_BL.shape
    token_indices = torch.arange(sequence_length, device=positions_BL.device)
    allowed = token_indices[None, :, None] >= token_indices[None, None, :]
    return torch.zeros(
        batch_size,
        1,
        sequence_length,
        sequence_length,
        dtype=dtype,
        device=positions_BL.device,
    ).masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


@dataclass
class ComparisonResult:
    scope: str
    component: str
    precision: str
    layer: int | str
    samples: int
    passed: bool
    max_abs: float | None = None
    max_rel: float | None = None
    mismatch_count: int = 0
    mismatch_positions: list[int] = field(default_factory=list)
    peak_position: int | None = None
    module_path: str = ""
    detail: str = ""
    parent_path: str = ""
    level: int = 0
    node_kind: str = "checkpoint"
    checkpoint: bool = True


class ParityRecorder:
    """Collect comparable scalar rows and render a compact report."""

    def __init__(self, precision: PrecisionPolicy, *, precision_label: str | None = None):
        self.precision = precision
        self.precision_label = precision_label or precision.name
        self.results: list[ComparisonResult] = []

    @staticmethod
    def _cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    def tensor(
        self,
        *,
        scope: str,
        component: str,
        layer: int | str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float | None = None,
        atol: float | None = None,
        module_path: str = "",
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "checkpoint",
        checkpoint: bool = True,
    ) -> ComparisonResult:
        actual_cpu = self._cpu(actual)
        expected_cpu = self._cpu(expected)
        rtol = self.precision.rtol if rtol is None else rtol
        atol = self.precision.atol if atol is None else atol
        if actual_cpu.shape != expected_cpu.shape:
            result = ComparisonResult(
                scope, component, self.precision_label, layer, 0, False,
                module_path=module_path,
                detail=f"shape {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}",
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
            )
        else:
            diff = (actual_cpu - expected_cpu).abs()
            scale = expected_cpu.abs().clamp_min(torch.finfo(torch.float32).tiny)
            peak_position = None
            mismatch_positions: list[int] = []
            detail = ""
            if diff.ndim >= 3:
                position_max = diff.amax(dim=tuple([0] + list(range(2, diff.ndim))))
                peak_position = int(position_max.argmax())
                mismatch_mask = diff > atol + rtol * expected_cpu.abs()
                position_mismatch = mismatch_mask.any(dim=tuple([0] + list(range(2, diff.ndim))))
                mismatch_positions = [
                    index for index, value in enumerate(position_mismatch.tolist()) if value
                ]
                detail = f"peak_position={peak_position}"
            mismatch_count = int(
                (diff > atol + rtol * expected_cpu.abs()).sum().item()
            )
            result = ComparisonResult(
                scope,
                component,
                self.precision_label,
                layer,
                actual_cpu.numel(),
                bool(torch.all(diff <= atol + rtol * expected_cpu.abs())),
                float(diff.max()) if diff.numel() else 0.0,
                float((diff / scale).max()) if diff.numel() else 0.0,
                mismatch_count=mismatch_count,
                mismatch_positions=mismatch_positions,
                peak_position=peak_position,
                module_path=module_path,
                detail=detail,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
            )
        self.results.append(result)
        return result

    def discrete(
        self,
        *,
        scope: str,
        component: str,
        layer: int | str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        positions: torch.Tensor | None = None,
        module_path: str = "",
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "checkpoint",
        checkpoint: bool = True,
    ) -> ComparisonResult:
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        mismatch_positions: set[int] = set()
        mismatch_count = 0
        detail = ""
        if actual_cpu.shape != expected_cpu.shape:
            passed = False
            detail = f"shape {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}"
        elif actual_cpu.ndim < 2:
            passed = bool(torch.equal(actual_cpu, expected_cpu))
            mismatch_count = 0 if passed else 1
        elif positions is not None and actual_cpu.ndim == 3:
            # Indexer selections are causal.  Differences in future slots are
            # not observable at that query and must not be reported as fails.
            mismatch_positions = set(
                _selection_mismatch_positions(actual_cpu, expected_cpu, positions)
            )
            mismatch_count = len(mismatch_positions)
            passed = mismatch_count == 0
            if mismatch_positions:
                detail = "ordered causal selections differ"
        else:
            passed = True
            for batch_index in range(actual_cpu.shape[0]):
                for sequence_index in range(actual_cpu.shape[1]):
                    left = actual_cpu[batch_index, sequence_index].reshape(-1)
                    right = expected_cpu[batch_index, sequence_index].reshape(-1)
                    if not torch.equal(left, right):
                        passed = False
                        mismatch_count += 1
                        mismatch_positions.add(sequence_index)
            if mismatch_positions:
                detail = "ordered indices differ"
        result = ComparisonResult(
            scope,
            component,
            self.precision_label,
            layer,
            int(actual_cpu.numel()),
            passed,
            mismatch_count=mismatch_count,
            mismatch_positions=sorted(mismatch_positions),
            module_path=module_path,
            detail=detail,
            parent_path=parent_path,
            level=level,
            node_kind=node_kind,
            checkpoint=checkpoint,
        )
        self.results.append(result)
        return result

    def missing(
        self,
        *,
        scope: str,
        component: str,
        layer: int | str,
        module_path: str,
        detail: str,
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "activation",
        checkpoint: bool = True,
    ) -> ComparisonResult:
        """Record a missing hook/output instead of silently dropping a row."""
        result = ComparisonResult(
            scope=scope,
            component=component,
            precision=self.precision_label,
            layer=layer,
            samples=0,
            passed=not checkpoint,
            module_path=module_path,
            detail=detail,
            parent_path=parent_path,
            level=level,
            node_kind=node_kind,
            checkpoint=checkpoint,
        )
        self.results.append(result)
        return result

    @property
    def failed(self) -> list[ComparisonResult]:
        return [
            result for result in self.results
            if result.checkpoint and not result.passed
        ]

    @staticmethod
    def _status(result: ComparisonResult) -> str:
        if not result.checkpoint:
            return "TRACE"
        return "PASS" if result.passed else "FAIL"

    def table(self, *, color: bool = False) -> str:
        headers = (
            "scope", "level", "parent", "component", "node_kind", "checkpoint",
            "module_path", "precision", "layer", "samples", "status", "max_abs",
            "max_rel", "mismatches", "positions", "detail",
        )
        lines = [" | ".join(headers), "-|-|-|-|-|-|-|-|-|-|-|-|-|-|-| "]
        for result in self.results:
            status = self._status(result)
            if color:
                status = (
                    f"\033[32m{status}\033[0m"
                    if status == "PASS"
                    else f"\033[31m{status}\033[0m"
                    if status == "FAIL"
                    else f"\033[36m{status}\033[0m"
                )
            lines.append(
                " | ".join(
                    (
                        result.scope,
                        str(result.level),
                        result.parent_path,
                        result.component,
                        result.node_kind,
                        "yes" if result.checkpoint else "no",
                        result.module_path,
                        result.precision,
                        str(result.layer),
                        str(result.samples),
                        status,
                        "-" if result.max_abs is None else f"{result.max_abs:.6g}",
                        "-" if result.max_rel is None else f"{result.max_rel:.6g}",
                        str(result.mismatch_count),
                        str(result.mismatch_positions),
                        result.detail
                        + ("; " if result.detail and result.peak_position is not None else "")
                        + (
                            f"peak_position={result.peak_position}"
                            if result.peak_position is not None
                            and "peak_position=" not in result.detail
                            else ""
                        ),
                    )
                )
            )
        checkpoint_total = sum(result.checkpoint for result in self.results)
        trace_total = len(self.results) - checkpoint_total
        passed = checkpoint_total - len(self.failed)
        rate = (
            100.0
            if checkpoint_total == 0
            else 100.0 * passed / checkpoint_total
        )
        lines.append("")
        lines.append(
            f"summary: precision={self.precision_label} rows={len(self.results)} "
            f"checkpoints={checkpoint_total} trace_rows={trace_total} "
            f"passed={passed} failed={len(self.failed)} pass_rate={rate:.1f}%"
        )
        return "\n".join(lines)

    def write(self, path: str | None = None) -> str:
        report = self.table(color=False)
        output_path = path or os.environ.get("GLM5_PARITY_REPORT")
        if output_path:
            if output_path.lower().endswith(".html"):
                rows = []
                for result in self.results:
                    status = self._status(result)
                    status_class = (
                        "pass" if status == "PASS"
                        else "fail" if status == "FAIL" else "trace"
                    )
                    rows.append(
                        "<tr>"
                        f"<td>{escape(result.scope)}</td>"
                        f"<td>{result.level}</td>"
                        f"<td>{escape(result.parent_path)}</td>"
                        f"<td>{escape(result.component)}</td>"
                        f"<td>{escape(result.node_kind)}</td>"
                        f"<td>{'yes' if result.checkpoint else 'no'}</td>"
                        f"<td>{escape(result.module_path)}</td>"
                        f"<td>{escape(result.precision)}</td>"
                        f"<td>{escape(str(result.layer))}</td>"
                        f"<td>{result.samples}</td>"
                        f"<td class='{status_class}'>{status}</td>"
                        f"<td>{'-' if result.max_abs is None else f'{result.max_abs:.6g}'}</td>"
                        f"<td>{'-' if result.max_rel is None else f'{result.max_rel:.6g}'}</td>"
                        f"<td>{result.mismatch_count}</td>"
                        f"<td>{escape(str(result.mismatch_positions))}</td>"
                        f"<td>{escape(result.detail)}</td>"
                        "</tr>"
                    )
                html_report = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<style>body{font-family:monospace}table{border-collapse:collapse}"
                    "th,td{border:1px solid #bbb;padding:4px 8px}"
                    ".pass{color:#087f23;font-weight:bold}.fail{color:#b00020;font-weight:bold}"
                    ".trace{color:#007c91;font-weight:bold}"
                    "</style></head><body>"
                    f"<h2>GLM-5 parity report ({escape(self.precision_label)})</h2>"
                    "<table><tr><th>scope</th><th>level</th><th>parent</th>"
                    "<th>component</th><th>node_kind</th><th>checkpoint</th>"
                    "<th>module_path</th><th>precision</th>"
                    "<th>layer</th><th>samples</th><th>status</th><th>max_abs</th>"
                    "<th>max_rel</th><th>mismatches</th><th>positions</th><th>detail</th></tr>"
                    + "".join(rows)
                    + "</table>"
                    f"<p>{escape(report.splitlines()[-1])}</p></body></html>\n"
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(html_report)
            else:
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(report + "\n")
        return report

    def assert_all_passed(self) -> None:
        if self.failed:
            raise AssertionError(self.table(color=False))


def _format_parity_diagnostics(
    records: dict[str, dict[int, torch.Tensor | None]],
    positions_BL: torch.Tensor,
) -> str:
    """Format layer-local block, indexer, and router parity evidence."""
    expected_record_names = (
        "hf_blocks", "titan_blocks", "hf_indexer", "titan_indexer",
        "hf_router", "titan_router",
    )
    missing_record_names = [
        name for name in expected_record_names if name not in records
    ]
    if missing_record_names:
        return f"missing record groups: {missing_record_names}"
    recorder = ParityRecorder(BF16)
    for layer in sorted(set(records.get("hf_blocks", {})) | set(records.get("titan_blocks", {}))):
        hf_block = records.get("hf_blocks", {}).get(layer)
        titan_block = records.get("titan_blocks", {}).get(layer)
        if hf_block is not None and titan_block is not None:
            recorder.tensor(
                scope="e2e", component="block", layer=layer,
                actual=titan_block, expected=hf_block,
            )
        for component in ("indexer", "router"):
            hf_value = records.get(f"hf_{component}", {}).get(layer)
            titan_value = records.get(f"titan_{component}", {}).get(layer)
            if hf_value is not None and titan_value is not None:
                recorder.discrete(
                    scope="e2e", component=component, layer=layer,
                    actual=titan_value, expected=hf_value,
                    positions=positions_BL,
                )
    lines = []
    layers = sorted(
        set(records.get("hf_blocks", {}))
        | set(records.get("titan_blocks", {}))
        | set(records.get("hf_indexer", {}))
        | set(records.get("titan_indexer", {}))
        | set(records.get("hf_router", {}))
        | set(records.get("titan_router", {}))
    )
    for layer in layers:
        hf_block = records.get("hf_blocks", {}).get(layer)
        titan_block = records.get("titan_blocks", {}).get(layer)
        if hf_block is None or titan_block is None:
            lines.append(
                f"layer {layer}: block_records=missing"
                f"(hf={hf_block is not None}, titan={titan_block is not None})"
            )
        elif hf_block.ndim != 3 or titan_block.ndim != 3:
            lines.append(
                f"layer {layer}: block_shape_error=expected [B, L, D] compatible with positions"
            )
        elif hf_block.shape != titan_block.shape:
            lines.append(
                f"layer {layer}: block_shape_mismatch="
                f"hf{tuple(hf_block.shape)}!=titan{tuple(titan_block.shape)}"
            )
        else:
            difference = (hf_block.float() - titan_block.float()).abs()
            position_max = difference.amax(dim=(0, 2)).detach().cpu()
            values = ", ".join(
                f"{position}:{float(value):.6g}"
                for position, value in enumerate(position_max)
            )
            lines.append(
                f"layer {layer}: block_max_abs={float(position_max.max()):.6g} "
                f"block_position_max_abs=[{values}]"
            )
    for result in recorder.results:
        if result.component == "block":
            continue
        else:
            lines.append(
                f"layer {result.layer}: {result.component}_mismatched_queries="
                f"{result.mismatch_count} positions={result.mismatch_positions}"
            )
    for component in ("indexer", "router"):
        hf_records = records.get(f"hf_{component}", {})
        titan_records = records.get(f"titan_{component}", {})
        for layer in sorted(set(hf_records) | set(titan_records)):
            if hf_records.get(layer) is None or titan_records.get(layer) is None:
                lines.append(
                    f"layer {layer}: {component}_records=missing"
                    f"(hf={hf_records.get(layer) is not None}, "
                    f"titan={titan_records.get(layer) is not None})"
                )
    failures = [
        result
        for result in recorder.results
        if not result.passed and result.component in {"indexer", "router"}
    ]
    first_failure = min(
        failures,
        key=lambda result: (
            int(result.layer) if isinstance(result.layer, int) else 10**9,
            result.mismatch_positions[0] if result.mismatch_positions else 10**9,
            result.component,
        ),
        default=None,
    )
    if first_failure is None:
        lines.append("first_discrete_mismatch=none")
    else:
        lines.append(
            f"first_discrete_mismatch=layer {first_failure.layer} "
            f"position {first_failure.mismatch_positions[0]} "
            f"source={first_failure.component}"
        )
    return "\n".join(lines)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"module output has no tensor: {type(output)!r}")


def _tensor_leaves(output: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Return every tensor in a module output, including tuple branches."""
    if isinstance(output, torch.Tensor):
        return [(prefix, output)]
    if isinstance(output, (tuple, list)):
        leaves: list[tuple[str, torch.Tensor]] = []
        for index, value in enumerate(output):
            branch = f"{prefix}[{index}]"
            leaves.extend(_tensor_leaves(value, branch))
        return leaves
    if isinstance(output, dict):
        leaves = []
        for key, value in output.items():
            leaves.extend(_tensor_leaves(value, f"{prefix}.{key}"))
        return leaves
    return []


@dataclass
class LayerTrace:
    blocks_hf: dict[int, torch.Tensor] = field(default_factory=dict)
    blocks_titan: dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_hf: dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_titan: dict[int, torch.Tensor] = field(default_factory=dict)
    router_hf: dict[int, torch.Tensor] = field(default_factory=dict)
    router_titan: dict[int, torch.Tensor] = field(default_factory=dict)

    @staticmethod
    @contextmanager
    def install(pair: ParityModelPair, layer_indices: list[int]) -> Iterator["LayerTrace"]:
        trace = LayerTrace()
        handles = []

        def save(
            target: dict[int, torch.Tensor],
            layer: int,
            output: Any,
            index: int | None = None,
        ) -> None:
            value = output[index] if index is not None else output
            target[layer] = _first_tensor(value).detach().cpu()

        for layer_index in layer_indices:
            hf_layer = pair.hf_layer(layer_index)
            titan_layer = pair.titan_layer(layer_index)
            handles.append(hf_layer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.blocks_hf, layer, o, 0)
            ))
            handles.append(titan_layer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.blocks_titan, layer, o)
            ))
            handles.append(hf_layer.self_attn.indexer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.indexer_hf, layer, o)
            ))
            handles.append(titan_layer.attention.indexer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.indexer_titan, layer, o)
            ))
            if getattr(titan_layer, "moe_enabled", False):
                def save_hf_router(
                    _module, inputs, output, *, layer=layer_index
                ) -> None:
                    batch_size, sequence_length = inputs[0].shape[:2]
                    value = output[2].view(batch_size, sequence_length, -1)
                    trace.router_hf[layer] = value.detach().cpu()

                handles.append(hf_layer.mlp.gate.register_forward_hook(save_hf_router))
                handles.append(titan_layer.moe.router.register_forward_hook(
                    lambda _m, _i, o, layer=layer_index: save(trace.router_titan, layer, o, 1)
                ))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class EndpointTrace:
    """Layer observations keyed by endpoint label for matrix comparisons."""

    blocks: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    indexer: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    router: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)

    @staticmethod
    @contextmanager
    def install(
        endpoints: dict[ModelEndpoint, torch.nn.Module],
        layer_indices: list[int],
    ) -> Iterator["EndpointTrace"]:
        trace = EndpointTrace()
        handles = []

        for endpoint, model in endpoints.items():
            label = endpoint.label
            trace.blocks[label] = {}
            trace.indexer[label] = {}
            trace.router[label] = {}
            layers = model.model.layers if endpoint.implementation == "hf" else model.layers
            for layer_index in layer_indices:
                layer = (
                    layers[layer_index]
                    if endpoint.implementation == "hf"
                    else layers[str(layer_index)]
                )

                def capture_block(
                    _module,
                    _inputs,
                    output,
                    *,
                    label=label,
                    layer_index=layer_index,
                    implementation=endpoint.implementation,
                ) -> None:
                    value = output[0] if implementation == "hf" else output
                    trace.blocks[label][layer_index] = _first_tensor(value).detach().cpu()

                def capture_indexer(
                    _module, _inputs, output, *, label=label, layer_index=layer_index
                ) -> None:
                    trace.indexer[label][layer_index] = _first_tensor(output).detach().cpu()

                handles.append(layer.register_forward_hook(capture_block))
                indexer = (
                    layer.self_attn.indexer
                    if endpoint.implementation == "hf"
                    else layer.attention.indexer
                )
                handles.append(indexer.register_forward_hook(capture_indexer))
                if endpoint.implementation == "hf" and hasattr(layer.mlp, "gate"):
                    def capture_hf_router(
                        _module, inputs, output, *, label=label, layer_index=layer_index
                    ) -> None:
                        batch_size, sequence_length = inputs[0].shape[:2]
                        trace.router[label][layer_index] = (
                            output[2].view(batch_size, sequence_length, -1)
                            .detach()
                            .cpu()
                        )

                    handles.append(layer.mlp.gate.register_forward_hook(capture_hf_router))
                elif getattr(layer, "moe_enabled", False):
                    def capture_titan_router(
                        _module, _inputs, output, *, label=label, layer_index=layer_index
                    ) -> None:
                        trace.router[label][layer_index] = output[1].detach().cpu()

                    handles.append(layer.moe.router.register_forward_hook(capture_titan_router))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class RecursiveModuleTrace:
    """Capture module outputs and tensor gradients under selected layers."""

    activations: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    gradients: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    @staticmethod
    @contextmanager
    def install(
        endpoints: dict[ModelEndpoint, torch.nn.Module],
        layer_indices: list[int],
        *,
        include_model_modules: bool = False,
    ) -> Iterator["RecursiveModuleTrace"]:
        trace = RecursiveModuleTrace()
        handles = []
        for endpoint, model in endpoints.items():
            label = endpoint.label
            trace.activations[label] = {}
            trace.gradients[label] = {}
            layers = model.model.layers if endpoint.implementation == "hf" else model.layers
            for layer_index in layer_indices:
                layer = (
                    layers[layer_index]
                    if endpoint.implementation == "hf"
                    else layers[str(layer_index)]
                )
                root = (
                    f"model.layers.{layer_index}"
                    if endpoint.implementation == "hf"
                    else f"layers.{layer_index}"
                )
                for suffix, module in layer.named_modules():
                    path = root if not suffix else f"{root}.{suffix}"

                    def capture(
                        _module,
                        _inputs,
                        output,
                        *,
                        label=label,
                        path=path,
                    ) -> None:
                        for suffix, value in _tensor_leaves(output):
                            trace.activations[label][path + suffix] = value.detach().cpu()
                            if value.requires_grad:
                                value.register_hook(
                                    lambda gradient,
                                    label=label,
                                    path=path + suffix: trace.gradients[label].__setitem__(
                                        path, gradient.detach().cpu()
                                    )
                                )

                    handles.append(module.register_forward_hook(capture))
            if include_model_modules:
                layer_prefix = (
                    "model.layers."
                    if endpoint.implementation == "hf"
                    else "layers."
                )
                for suffix, module in model.named_modules():
                    if (
                        not suffix
                        or suffix in {"model", "layers"}
                        or suffix.startswith(layer_prefix)
                    ):
                        continue
                    path = suffix

                    def capture_model_module(
                        _module,
                        _inputs,
                        output,
                        *,
                        label=label,
                        path=path,
                    ) -> None:
                        for branch, value in _tensor_leaves(output):
                            trace.activations[label][path + branch] = value.detach().cpu()
                            if value.requires_grad:
                                value.register_hook(
                                    lambda gradient,
                                    label=label,
                                    path=path + branch: trace.gradients[label].__setitem__(
                                        path, gradient.detach().cpu()
                                    )
                                )

                    handles.append(module.register_forward_hook(capture_model_module))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


def _run_causal_lm_endpoint(
    model: torch.nn.Module,
    endpoint: ModelEndpoint,
    batch: ParityBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one endpoint with a common explicit causal mask."""
    assert batch.tokens is not None
    labels = batch.tokens
    if endpoint.implementation == "hf":
        output = model(
            input_ids=batch.tokens,
            position_ids=batch.positions,
            attention_mask=batch.causal_mask,
            labels=labels,
            use_cache=False,
        )
        return output.logits, output.loss
    logits = model(
        batch.tokens,
        positions=batch.positions,
        attention_masks=batch.causal_mask,
    )
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
    )
    return logits, loss


class ComponentParityMixin:
    pair: ParityModelPair
    batch: ParityBatch
    precision: PrecisionPolicy

    def _recorder(self) -> ParityRecorder:
        return ParityRecorder(
            self.precision,
            precision_label=(
                f"titan:{self.precision.name} vs hf:{self.precision.name}"
            ),
        )

    def _record_component_trace(
        self,
        component: str,
        layer_indices: list[int],
        recorder: ParityRecorder,
    ) -> None:
        """Trace the complete selected layers for a component-level test."""
        actual = ModelEndpoint("titan", self.precision)
        expected = ModelEndpoint("hf", self.precision)
        endpoints = {
            actual: self.pair.titan,
            expected: self.pair.hf,
        }
        spec = ComparisonSpec(
            actual=actual,
            expected=expected,
            scope="component_trace",
            component=component,
            rtol=self.precision.rtol,
            atol=self.precision.atol,
        )
        with RecursiveModuleTrace.install(endpoints, layer_indices) as trace:
            with torch.no_grad():
                for layer_index in layer_indices:
                    hf_layer = self.pair.hf_layer(layer_index)
                    titan_layer = self.pair.titan_layer(layer_index)
                    hf_layer(
                        self.batch.hidden_states,
                        attention_mask=self.batch.causal_mask,
                        position_ids=self.batch.positions,
                        position_embeddings=self.batch.hf_position_embeddings,
                        use_cache=False,
                    )
                    titan_layer(
                        self.batch.hidden_states,
                        self.batch.causal_mask,
                        self.batch.positions,
                    )
        self._record_recursive_trace(
            spec, trace, recorder, layer_indices=layer_indices
        )

    def _normalized(self, layer_index: int) -> torch.Tensor:
        return self.pair.hf_layer(layer_index).input_layernorm(
            self.batch.hidden_states
        )

    def _normalized_hidden_states(self, layer_index: int) -> torch.Tensor:
        """Compatibility alias for the original component-test helper."""
        return self._normalized(layer_index)

    @staticmethod
    def _module_path(layer_index: int, component: str) -> str:
        """Render both canonical module paths for one logical component."""
        paths = {
            "block": (f"model.layers.{layer_index}", f"layers.{layer_index}"),
            "attention": (
                f"model.layers.{layer_index}.self_attn",
                f"layers.{layer_index}.attention",
            ),
            "indexer": (
                f"model.layers.{layer_index}.self_attn.indexer",
                f"layers.{layer_index}.attention.indexer",
            ),
            "router": (
                f"model.layers.{layer_index}.mlp.gate",
                f"layers.{layer_index}.moe.router",
            ),
            "input_norm": (
                f"model.layers.{layer_index}.input_layernorm",
                f"layers.{layer_index}.attention_norm",
            ),
            "ffn_norm": (
                f"model.layers.{layer_index}.post_attention_layernorm",
                f"layers.{layer_index}.ffn_norm",
            ),
        }
        hf_path, titan_path = paths.get(
            component,
            (
                f"model.layers.{layer_index}.{component}",
                f"layers.{layer_index}.{component}",
            ),
        )
        return f"hf:{hf_path} <-> titan:{titan_path}"

    @staticmethod
    def _normalize_component_name(layer_index: int, component: str) -> str:
        """Normalize a user component selector to a layer-relative path."""
        name = component.strip().strip(".")
        for prefix in (
            f"model.layers.{layer_index}.",
            f"layers.{layer_index}.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        name = re.sub(r"^(?:model\.)?layers\.\d+\.", "", name)
        if name in {"decoder_block", "layer", "block", ""}:
            return ""
        aliases = {
            "dense_ffn": "feed_forward",
            "ffn": "feed_forward",
            "self_attn": "attention",
            "indexer": "attention.indexer",
            "router": "moe.router",
            "gate": "moe.router",
        }
        return aliases.get(name, name)

    def _resolve_component_module(
        self,
        endpoint: ModelEndpoint,
        layer_index: int,
        component: str,
    ) -> torch.nn.Module:
        """Resolve any named module through the canonical report hierarchy."""
        target = self._normalize_component_name(layer_index, component)
        layer = (
            self._model(endpoint).model.layers[layer_index]
            if endpoint.implementation == "hf"
            else self._model(endpoint).layers[str(layer_index)]
        )
        if target == "mlp":
            is_moe = (
                hasattr(layer, "moe")
                if endpoint.implementation == "titan"
                else hasattr(layer.mlp, "gate")
            )
            target = "moe" if is_moe else "feed_forward"
        if not target:
            return layer
        endpoint_root = (
            f"model.layers.{layer_index}"
            if endpoint.implementation == "hf"
            else f"layers.{layer_index}"
        )
        for suffix, module in layer.named_modules():
            endpoint_path = endpoint_root if not suffix else f"{endpoint_root}.{suffix}"
            logical_path = self._logical_activation_path(endpoint, endpoint_path)
            canonical = logical_path.removeprefix(f"layers.{layer_index}.")
            if canonical == target:
                return module
        available = sorted(
            self._logical_activation_path(
                endpoint,
                endpoint_root if not suffix else f"{endpoint_root}.{suffix}",
            ).removeprefix(f"layers.{layer_index}.")
            for suffix, _module in layer.named_modules()
        )
        raise ValueError(
            f"component {component!r} is not present in layer {layer_index}; "
            f"available paths include {available[:20]}"
        )

    def get_component(
        self, layer_index: int, component: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Return the HF and TorchTitan modules for any canonical component."""
        return (
            self._resolve_component_module(
                ModelEndpoint("hf", self.precision), layer_index, component
            ),
            self._resolve_component_module(
                ModelEndpoint("titan", self.precision), layer_index, component
            ),
        )

    def get_model_component(
        self, component: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Return a top-level component such as embeddings, norm, or head."""
        aliases = {
            "embedding": "tok_embeddings",
            "embeddings": "tok_embeddings",
            "embed_tokens": "tok_embeddings",
            "model.embed_tokens": "tok_embeddings",
            "model.norm": "norm",
        }
        target = aliases.get(component.strip().strip("."), component.strip().strip("."))

        def resolve(endpoint: ModelEndpoint) -> torch.nn.Module:
            model = self._model(endpoint)
            for suffix, module in model.named_modules():
                if not suffix or suffix.startswith(
                    "model.layers." if endpoint.implementation == "hf" else "layers."
                ):
                    continue
                logical = self._logical_activation_path(endpoint, suffix)
                if logical == target:
                    return module
            raise ValueError(f"top-level component {component!r} was not found")

        return (
            resolve(ModelEndpoint("hf", self.precision)),
            resolve(ModelEndpoint("titan", self.precision)),
        )

    def component_modules(
        self, layer_index: int, component: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Return corresponding HF/TorchTitan modules by logical component name."""
        hf_layer = self.pair.hf_layer(layer_index)
        titan_layer = self.pair.titan_layer(layer_index)
        paths = {
            "block": (hf_layer, titan_layer),
            "attention": (hf_layer.self_attn, titan_layer.attention),
            "indexer": (hf_layer.self_attn.indexer, titan_layer.attention.indexer),
            "input_norm": (hf_layer.input_layernorm, titan_layer.attention_norm),
            "ffn_norm": (hf_layer.post_attention_layernorm, titan_layer.ffn_norm),
            "q_norm": (hf_layer.self_attn.q_a_layernorm, titan_layer.attention.q_norm),
            "kv_norm": (hf_layer.self_attn.kv_a_layernorm, titan_layer.attention.kv_norm),
        }
        if component == "router":
            if not hasattr(titan_layer, "moe"):
                raise ValueError(f"layer {layer_index} does not contain an MoE router")
            return hf_layer.mlp.gate, titan_layer.moe.router
        if component in {"dense_ffn", "feed_forward"}:
            if not hasattr(titan_layer, "feed_forward"):
                raise ValueError(
                    f"layer {layer_index} does not contain a dense feed-forward module"
                )
            return hf_layer.mlp, titan_layer.feed_forward
        if component == "moe":
            if not hasattr(titan_layer, "moe"):
                raise ValueError(f"layer {layer_index} does not contain an MoE module")
            return hf_layer.mlp, titan_layer.moe
        if component in paths:
            return paths[component]
        return self.get_component(layer_index, component)

    def compare_model_component_trace(
        self,
        component: str,
        *,
        spec: ComparisonSpec | None = None,
    ) -> ParityRecorder:
        """Compare a top-level module activation such as ``lm_head``."""
        if spec is None:
            actual = ModelEndpoint("titan", self.precision)
            expected = ModelEndpoint("hf", self.precision)
            spec = ComparisonSpec(
                actual=actual,
                expected=expected,
                scope="model_component_trace",
                component=component,
                rtol=self.precision.rtol,
                atol=self.precision.atol,
            )
        endpoints = {
            spec.actual: self._model(spec.actual),
            spec.expected: self._model(spec.expected),
        }
        recorder = ParityRecorder(
            spec.actual.precision, precision_label=spec.label
        )
        with RecursiveModuleTrace.install(
            endpoints, [], include_model_modules=True
        ) as trace:
            for endpoint in (spec.actual, spec.expected):
                with torch.no_grad():
                    _run_causal_lm_endpoint(
                        endpoints[endpoint], endpoint, self._batch_for(endpoint)
                    )
        normalized = self._normalize_model_component(component)
        self._record_recursive_trace(
            spec,
            trace,
            recorder,
            layer_indices=[],
            component_filter=normalized,
        )
        if not recorder.results:
            raise ValueError(
                f"component {component!r} produced no trace rows for the selected model"
            )
        return recorder

    @staticmethod
    def _normalize_model_component(component: str) -> str:
        aliases = {
            "embedding": "tok_embeddings",
            "embeddings": "tok_embeddings",
            "embed_tokens": "tok_embeddings",
            "model.embed_tokens": "tok_embeddings",
            "model.norm": "norm",
        }
        return aliases.get(component.strip().strip("."), component.strip().strip("."))

    def compare_component_trace(
        self,
        component: str,
        layer_indices: list[int],
        *,
        spec: ComparisonSpec | None = None,
        sequential: bool = False,
    ) -> ParityRecorder:
        """Compare an arbitrary module and its descendants across layers.

        The selected module is a checkpoint; descendant outputs remain trace
        rows unless they are composition nodes.  ``sequential=True`` feeds
        each selected layer into the next one; the default probes each layer
        from the same input.  This gives callers a generic path for new
        modules without adding a bespoke runner.
        """
        if self._normalize_model_component(component) in TOP_LEVEL_COMPONENTS:
            return self.compare_model_component_trace(component, spec=spec)
        if not layer_indices:
            raise ValueError("at least one layer is required for a component trace")
        if spec is None:
            actual = ModelEndpoint("titan", self.precision)
            expected = ModelEndpoint("hf", self.precision)
            spec = ComparisonSpec(
                actual=actual,
                expected=expected,
                scope="component_trace",
                component=component,
                rtol=self.precision.rtol,
                atol=self.precision.atol,
            )
        endpoints = {
            spec.actual: self._model(spec.actual),
            spec.expected: self._model(spec.expected),
        }
        recorder = ParityRecorder(
            spec.actual.precision, precision_label=spec.label
        )
        with RecursiveModuleTrace.install(endpoints, layer_indices) as trace:
            for endpoint in (spec.actual, spec.expected):
                batch = self._batch_for(endpoint)
                model = endpoints[endpoint]
                layers = (
                    model.model.layers
                    if endpoint.implementation == "hf"
                    else model.layers
                )
                current = batch.hidden_states
                for layer_index in layer_indices:
                    layer = (
                        layers[layer_index]
                        if endpoint.implementation == "hf"
                        else layers[str(layer_index)]
                    )
                    if endpoint.implementation == "hf":
                        output = layer(
                            current,
                            attention_mask=batch.causal_mask,
                            position_ids=batch.positions,
                            position_embeddings=batch.hf_position_embeddings,
                            use_cache=False,
                        )
                    else:
                        output = layer(
                            current,
                            batch.causal_mask,
                            batch.positions,
                        )
                    if sequential:
                        value = (
                            output[0]
                            if endpoint.implementation == "hf"
                            and isinstance(output, (tuple, list))
                            else output
                        )
                        current = _first_tensor(value)
                    elif endpoint.implementation == "hf":
                        current = batch.hidden_states
        normalized = self._normalize_component_name(layer_indices[0], component)
        self._record_recursive_trace(
            spec,
            trace,
            recorder,
            layer_indices=layer_indices,
            component_filter=normalized,
        )
        if not recorder.results:
            raise ValueError(
                f"component {component!r} produced no trace rows for the selected layers"
            )
        return recorder

    def compare_component(
        self,
        component: str,
        layer_indices: list[int] | None = None,
        *,
        spec: ComparisonSpec | None = None,
        sequential: bool = False,
    ) -> ParityRecorder:
        """Public component entry point for unit and multi-layer checks."""
        selected = self._selected_layers() if layer_indices is None else layer_indices
        return self.compare_components(
            component, selected, spec=spec, sequential=sequential
        )

    def compare_components(
        self,
        component: str,
        layer_indices: list[int],
        *,
        spec: ComparisonSpec | None = None,
        sequential: bool = False,
    ) -> ParityRecorder:
        """Dispatch a reusable comparison by component and arbitrary layers."""
        runners: dict[str, Callable[[list[int]], ParityRecorder]] = {
            "indexer": self.compare_indexer,
            "router": self.compare_router,
            "attention": self.compare_attention,
            "block": self.compare_blocks,
        }
        if component in runners and spec is None and not sequential:
            return runners[component](layer_indices)
        return self.compare_component_trace(
            component, layer_indices, spec=spec, sequential=sequential
        )

    def compare_indexer(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            hf_attention = self.pair.hf_layer(layer_index).self_attn
            titan_attention = self.pair.titan_layer(layer_index).attention
            hidden_states = self._normalized(layer_index)
            hf_q_resid = hf_attention.q_a_layernorm(hf_attention.q_a_proj(hidden_states))
            titan_q_resid = titan_attention.q_norm(titan_attention.wq_a(hidden_states))
            recorder.tensor(
                scope="component", component="q_residual", layer=layer_index,
                actual=titan_q_resid, expected=hf_q_resid,
                rtol=1e-6 if self.precision is FP32 else self.precision.rtol,
                atol=1e-7 if self.precision is FP32 else self.precision.atol,
                module_path=(
                    f"hf:model.layers.{layer_index}.self_attn.q_a_layernorm"
                    f" <-> titan:layers.{layer_index}.attention.q_norm"
                ),
                parent_path=f"layers.{layer_index}.attention",
                level=4,
                node_kind="activation_checkpoint",
            )
            common_q_resid = hf_q_resid
            hf_topk = hf_attention.indexer(
                hidden_states, common_q_resid, self.batch.hf_position_embeddings,
                self.batch.causal_mask[:, 0], self.batch.positions,
            )
            titan_topk = titan_attention.indexer(
                hidden_states, common_q_resid, self.batch.positions,
                self.batch.causal_mask[:, 0],
            )
            recorder.discrete(
                scope="component", component="indexer", layer=layer_index,
                actual=titan_topk, expected=hf_topk, positions=self.batch.positions,
                module_path=self._module_path(layer_index, "indexer"),
                parent_path=f"layers.{layer_index}.attention",
                level=4,
                node_kind="discrete_checkpoint",
            )
        self._record_component_trace("indexer", layer_indices, recorder)
        return recorder

    def compare_router(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            hf_layer = self.pair.hf_layer(layer_index)
            titan_moe = self.pair.titan_layer(layer_index).moe
            normalized = self._normalized(layer_index)
            _, hf_weights, hf_indices = hf_layer.mlp.gate(normalized)
            titan_weights, titan_indices, _ = titan_moe.router(
                normalized, titan_moe.expert_bias_E
            )
            recorder.discrete(
                scope="component", component="router_indices", layer=layer_index,
                actual=titan_indices.view_as(hf_indices), expected=hf_indices,
                module_path=self._module_path(layer_index, "router"),
                parent_path=f"layers.{layer_index}.moe",
                level=4,
                node_kind="discrete_checkpoint",
            )
            recorder.tensor(
                scope="component", component="router_weights", layer=layer_index,
                actual=titan_weights,
                expected=hf_weights.view_as(titan_weights),
                rtol=1e-6,
                atol=1e-7,
                module_path=self._module_path(layer_index, "router"),
                parent_path=f"layers.{layer_index}.moe",
                level=4,
                node_kind="activation_checkpoint",
            )
        self._record_component_trace("router", layer_indices, recorder)
        return recorder

    def compare_attention(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            normalized = self._normalized(layer_index)
            hf_output = self.pair.hf_layer(layer_index).self_attn(
                hidden_states=normalized,
                position_embeddings=self.batch.hf_position_embeddings,
                attention_mask=self.batch.causal_mask,
                position_ids=self.batch.positions,
            )[0]
            titan_output = self.pair.titan_layer(layer_index).attention(
                normalized, self.batch.causal_mask, self.batch.positions
            )
            recorder.tensor(
                scope="component", component="attention", layer=layer_index,
                actual=titan_output, expected=hf_output,
                module_path=self._module_path(layer_index, "attention"),
                parent_path=f"layers.{layer_index}",
                level=3,
                node_kind="attention_checkpoint",
            )
        self._record_component_trace("attention", layer_indices, recorder)
        return recorder

    def compare_blocks(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            hf_output = self.pair.hf_layer(layer_index)(
                self.batch.hidden_states,
                attention_mask=self.batch.causal_mask,
                position_ids=self.batch.positions,
                position_embeddings=self.batch.hf_position_embeddings,
                use_cache=False,
            )[0]
            titan_output = self.pair.titan_layer(layer_index)(
                self.batch.hidden_states,
                self.batch.causal_mask,
                self.batch.positions,
            )
            recorder.tensor(
                scope="composition", component="decoder_block", layer=layer_index,
                actual=titan_output, expected=hf_output,
                module_path=self._module_path(layer_index, "block"),
                parent_path="layers",
                level=2,
                node_kind="layer_checkpoint",
            )
        self._record_component_trace("block", layer_indices, recorder)
        return recorder


class _ParityDiagnostics:
    def _check_format_reports_layer_position_and_discrete_mismatches(self) -> None:
        zero = torch.zeros(1, 3, 2)
        divergent = zero.clone()
        divergent[0, 1, 0] = 0.5
        indices = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
        changed = torch.tensor([[[0, 1], [0, 1], [0, 2]]])
        records = {
            "hf_blocks": {0: zero, 1: zero},
            "titan_blocks": {0: zero, 1: divergent},
            "hf_indexer": {0: indices, 1: indices},
            "titan_indexer": {0: indices, 1: changed},
            "hf_router": {},
            "titan_router": {},
        }
        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1, 2]]))
        self.assertIn("layer 1: block_max_abs=0.5", diagnostics)
        self.assertIn("block_position_max_abs=[0:0, 1:0.5, 2:0]", diagnostics)
        self.assertIn("indexer_mismatched_queries=1 positions=[2]", diagnostics)

    def _check_format_reports_missing_expected_records(self) -> None:
        records = {
            "hf_blocks": {0: torch.zeros(1, 2, 2)},
            "titan_blocks": {0: None},
            "hf_indexer": {0: None},
            "titan_indexer": {0: None},
            "hf_router": {1: torch.zeros(1, 2, 2)},
            "titan_router": {1: None},
        }
        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1]]))
        self.assertIn("block_records=missing(hf=True, titan=False)", diagnostics)
        self.assertIn("indexer_records=missing(hf=False, titan=False)", diagnostics)
        self.assertIn("router_records=missing(hf=True, titan=False)", diagnostics)

    def _check_format_reports_incompatible_block_shape(self) -> None:
        malformed = torch.zeros(2, 2)
        records = {
            "hf_blocks": {0: malformed},
            "titan_blocks": {0: malformed},
            "hf_indexer": {},
            "titan_indexer": {},
            "hf_router": {},
            "titan_router": {},
        }
        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1]]))
        self.assertIn(
            "block_shape_error=expected [B, L, D] compatible with positions",
            diagnostics,
        )

    def _check_recorder_reports_tensor_and_discrete_rows(self) -> None:
        recorder = ParityRecorder(BF16)
        recorder.tensor(
            scope="component", component="block", layer=1,
            actual=torch.tensor([1.0, 1.5]),
            expected=torch.tensor([1.0, 1.0]),
            module_path="layers.1.moe.routed_experts.7.w1",
        )
        recorder.discrete(
            scope="component", component="indexer", layer=1,
            actual=torch.tensor([[[0, 2]]]), expected=torch.tensor([[[0, 1]]]),
        )
        report = recorder.table()
        self.assertIn("block", report)
        self.assertIn("layers.1.moe.routed_experts.7.w1", report)
        self.assertIn("indexer", report)
        self.assertIn("failed=2", report)

    def _check_recorder_writes_plain_text_report(self) -> None:
        recorder = ParityRecorder(FP32)
        recorder.tensor(
            scope="unit", component="identity", layer="-",
            actual=torch.ones(2), expected=torch.ones(2),
        )
        self.assertIn("pass_rate=100.0%", recorder.write())


class _ParityRouterPrecision:
    def _check_router_gate_is_evaluated_in_float32(self) -> None:
        titan_model = glm5_configs["debugmodel"]().build()
        titan_model.init_states()
        titan_model.bfloat16()
        router = titan_model.layers["1"].moe.router
        hidden_states = torch.randn(1, 16, 256, dtype=torch.bfloat16)
        _, _, scores = router(hidden_states, titan_model.layers["1"].moe.expert_bias_E)
        expected_scores = torch.sigmoid(
            F.linear(hidden_states.float(), router.gate.weight.float())
        )
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)

    def _check_router_uses_gate_forward_for_fp32_biased_computation(self) -> None:
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=Linear.Config(in_features=3, out_features=4, bias=True),
            top_k=2,
            score_func="sigmoid",
        ).build()
        router.bfloat16()
        hidden_states = torch.randn(1, 2, 3, dtype=torch.bfloat16)
        calls = 0

        def count_gate_calls(*_args) -> None:
            nonlocal calls
            calls += 1

        hook = router.gate.register_forward_hook(count_gate_calls)
        try:
            _, _, scores = router(hidden_states)
        finally:
            hook.remove()
        expected_scores = torch.sigmoid(
            F.linear(hidden_states.float(), router.gate.weight.float(), router.gate.bias.float())
        )
        self.assertEqual(calls, 1)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
        scores.sum().backward()
        self.assertIsNotNone(router.gate.weight.grad)
        self.assertIsNotNone(router.gate.bias.grad)


class _ParityComponentTests(ComponentParityMixin):
    def _assert_recorder(self, recorder: ParityRecorder) -> None:
        report_path = os.environ.get("GLM5_PARITY_REPORT")
        if report_path:
            root, extension = os.path.splitext(report_path)
            safe_name = self.id().split(".")[-1]
            report_path = f"{root}__{safe_name}{extension or '.txt'}"
        recorder.write(report_path)
        recorder.assert_all_passed()

    def _check_indexer_topk_matches_transformers(self) -> None:
        self._assert_recorder(self.compare_indexer(self._selected_layers()))

    def _check_router_selection_and_weights(self) -> None:
        layers = [
            layer
            for layer in self._selected_layers()
            if self.pair.titan_layer(layer).moe_enabled
        ]
        self._assert_recorder(self.compare_router(layers))

    def _check_attention_output(self) -> None:
        self._assert_recorder(self.compare_attention(self._selected_layers()))

    def _check_dense_block_output(self) -> None:
        self._assert_recorder(self.compare_blocks(self._selected_layers()))


class _ParityTrainingTests:
    """Reusable training-path checks for the selected runtime precision."""

    def _check_end_to_end_output_loss_moe_and_gradients(self) -> None:
        assert self.batch.tokens is not None
        labels = self.batch.tokens.clone()
        layer_indices = list(range(len(self.pair.hf.model.layers)))
        recorder = ParityRecorder(self.precision)
        self.pair.hf.zero_grad(set_to_none=True)
        self.pair.titan.zero_grad(set_to_none=True)
        # Keep the original routed-MoE block probe in addition to the full
        # model trace.  It isolates the first MoE layer from later error
        # accumulation.
        with torch.no_grad():
            moe_input = self.pair.hf.model.embed_tokens(self.batch.tokens)
            hf_moe_position_embeddings = self.pair.hf.model.rotary_emb(
                moe_input, position_ids=self.batch.positions
            )
            hf_moe_output = self.pair.hf.model.layers[1](
                moe_input,
                attention_mask=self.batch.causal_mask,
                position_ids=self.batch.positions,
                position_embeddings=hf_moe_position_embeddings,
                use_cache=False,
            )[0]
            titan_moe_output = self.pair.titan.layers["1"](
                moe_input, self.batch.causal_mask, self.batch.positions
            )
        recorder.tensor(
            scope="component",
            component="moe_block_direct",
            layer=1,
            actual=titan_moe_output,
            expected=hf_moe_output,
            module_path=(
                "hf:model.layers.1.mlp <-> titan:layers.1.moe"
            ),
        )
        with LayerTrace.install(self.pair, layer_indices) as trace:
            hf_outputs = self.pair.hf(
                input_ids=self.batch.tokens,
                position_ids=self.batch.positions,
                attention_mask=self.batch.causal_mask,
                labels=labels,
                use_cache=False,
            )
            titan_logits = self.pair.titan(
                self.batch.tokens, positions=self.batch.positions,
                attention_masks=self.batch.causal_mask,
            )

        for layer_index in layer_indices:
            if layer_index in trace.blocks_hf and layer_index in trace.blocks_titan:
                recorder.tensor(
                    scope="e2e", component="decoder_block", layer=layer_index,
                    actual=trace.blocks_titan[layer_index],
                    expected=trace.blocks_hf[layer_index],
                    module_path=f"layers.{layer_index}",
                )
            if layer_index in trace.indexer_hf and layer_index in trace.indexer_titan:
                recorder.discrete(
                    scope="e2e", component="indexer", layer=layer_index,
                    actual=trace.indexer_titan[layer_index],
                    expected=trace.indexer_hf[layer_index],
                    positions=self.batch.positions,
                    module_path=f"layers.{layer_index}.attention.indexer",
                )
            if layer_index in trace.router_hf and layer_index in trace.router_titan:
                recorder.discrete(
                    scope="e2e", component="router", layer=layer_index,
                    actual=trace.router_titan[layer_index],
                    expected=trace.router_hf[layer_index],
                    module_path=f"layers.{layer_index}.moe.router",
                )

        hf_loss = hf_outputs.loss
        titan_loss = F.cross_entropy(
            titan_logits[:, :-1].float().reshape(-1, titan_logits.shape[-1]),
            labels[:, 1:].reshape(-1),
        )
        recorder.tensor(
            scope="e2e", component="logits", layer="all",
            actual=titan_logits, expected=hf_outputs.logits,
            module_path="lm_head",
        )
        recorder.tensor(
            scope="e2e", component="loss", layer="all",
            actual=titan_loss, expected=hf_loss,
            module_path="loss",
        )

        hf_loss.backward()
        titan_loss.backward()
        hf_layer0 = self.pair.hf.model.layers[0]
        titan_layer0 = self.pair.titan.layers["0"]
        hf_layer1 = self.pair.hf.model.layers[1]
        titan_layer1 = self.pair.titan.layers["1"]
        expert_width = self.pair.hf.config.moe_intermediate_size
        gradient_pairs = (
            (
                "embedding",
                self.pair.hf.model.embed_tokens.weight.grad,
                self.pair.titan.tok_embeddings.weight.grad,
                "hf:model.embed_tokens.weight <-> titan:tok_embeddings.weight",
            ),
            (
                "q_a_proj",
                hf_layer0.self_attn.q_a_proj.weight.grad,
                titan_layer0.attention.wq_a.weight.grad,
                "hf:model.layers.0.self_attn.q_a_proj.weight <-> "
                "titan:layers.0.attention.wq_a.weight",
            ),
            (
                "dense_gate_proj",
                hf_layer0.mlp.gate_proj.weight.grad,
                titan_layer0.feed_forward.w1.weight.grad,
                "hf:model.layers.0.mlp.gate_proj.weight <-> titan:layers.0.feed_forward.w1.weight",
            ),
            (
                "router_gate",
                hf_layer1.mlp.gate.weight.grad,
                titan_layer1.moe.router.gate.weight.grad,
                "hf:model.layers.1.mlp.gate.weight <-> titan:layers.1.moe.router.gate.weight",
            ),
            (
                "routed_gate_up",
                hf_layer1.mlp.experts.gate_up_proj.grad[:, :expert_width],
                titan_layer1.moe.routed_experts.inner_experts.w1_EFD.grad,
                "hf:model.layers.1.mlp.experts.gate_up_proj[:, :expert_width] <-> "
                "titan:layers.1.moe.routed_experts.inner_experts.w1_EFD",
            ),
            (
                "routed_up",
                hf_layer1.mlp.experts.gate_up_proj.grad[:, expert_width:],
                titan_layer1.moe.routed_experts.inner_experts.w3_EFD.grad,
                "hf:model.layers.1.mlp.experts.gate_up_proj[:, expert_width:] <-> "
                "titan:layers.1.moe.routed_experts.inner_experts.w3_EFD",
            ),
            (
                "routed_down",
                hf_layer1.mlp.experts.down_proj.grad,
                titan_layer1.moe.routed_experts.inner_experts.w2_EDF.grad,
                "hf:model.layers.1.mlp.experts.down_proj <-> "
                "titan:layers.1.moe.routed_experts.inner_experts.w2_EDF",
            ),
            (
                "lm_head",
                self.pair.hf.lm_head.weight.grad,
                self.pair.titan.lm_head.weight.grad,
                "hf:lm_head.weight <-> titan:lm_head.weight",
            ),
        )
        for name, hf_gradient, titan_gradient, module_path in gradient_pairs:
            self.assertIsNotNone(hf_gradient, f"HF {name} gradient is missing")
            self.assertIsNotNone(titan_gradient, f"TorchTitan {name} gradient is missing")
            assert hf_gradient is not None and titan_gradient is not None
            recorder.tensor(
                scope="gradient", component=name, layer="all",
                actual=titan_gradient, expected=hf_gradient,
                module_path=module_path,
            )
        for model in (self.pair.hf, self.pair.titan):
            for name, parameter in model.named_parameters():
                if ".indexer." in name:
                    self.assertIsNone(parameter.grad, f"indexer gradient found: {name}")

        report_path = os.environ.get("GLM5_PARITY_REPORT")
        report = recorder.write(report_path)
        if recorder.failed:
            raise AssertionError(report)


class TestGlm5Parity(
    unittest.TestCase,
    _ParityDiagnostics,
    _ParityRouterPrecision,
    _ParityComponentTests,
    _ParityTrainingTests,
):
    """Single configurable GLM-5 parity suite.

    The suite owns one runtime endpoint selection.  By default it compares
    TorchTitan FP32 with Transformers FP32.  Set ``GLM5_PARITY_PRECISION`` to
    ``bf16`` for the same implementation comparison at BF16, or set the two
    endpoint variables for a cross-precision run:

    ``GLM5_PARITY_ACTUAL=titan:bf16``
    ``GLM5_PARITY_EXPECTED=hf:fp32``
    ``GLM5_PARITY_LAYERS=0,1`` (or ``all``)
    ``GLM5_PARITY_COMPONENTS=indexer,router,gradient``
    ``GLM5_PARITY_DATA_CASE=random|zeros|ones|extreme|alternating``
    ``GLM5_PARITY_COMPONENT_EXECUTION=independent|sequential``

    The test methods do not construct a precision-specific class.  They use
    the models and batches prepared here.  Fixed runners preserve the original
    checks, while ``compare_component`` can trace any named module path.
    """

    ACTUAL_ENDPOINT = os.environ.get("GLM5_PARITY_ACTUAL", "titan:fp32")
    EXPECTED_ENDPOINT = os.environ.get("GLM5_PARITY_EXPECTED", "hf:fp32")
    PRECISION_OVERRIDE = os.environ.get("GLM5_PARITY_PRECISION")
    RUN_COMPONENTS = os.environ.get("GLM5_PARITY_COMPONENTS", "all")
    LAYER_INDICES = os.environ.get("GLM5_PARITY_LAYERS", "all")
    DATA_CASE = os.environ.get("GLM5_PARITY_DATA_CASE", "random")
    DATA_SEED = int(os.environ.get("GLM5_PARITY_DATA_SEED", "61"))
    BATCH_SIZE = int(os.environ.get("GLM5_PARITY_BATCH_SIZE", "2"))
    SEQUENCE_LENGTH = int(os.environ.get("GLM5_PARITY_SEQUENCE_LENGTH", "16"))
    COMPONENT_EXECUTION = os.environ.get(
        "GLM5_PARITY_COMPONENT_EXECUTION", "independent"
    )

    @staticmethod
    def _endpoint(value: str) -> ModelEndpoint:
        implementation, _, precision_name = value.lower().partition(":")
        if not precision_name:
            precision_name = precision_name or "fp32"
        if implementation not in {"hf", "titan"}:
            raise ValueError(f"unsupported GLM-5 implementation: {implementation}")
        policies = {FP32.name: FP32, BF16.name: BF16, "bfloat16": BF16}
        try:
            precision = policies[precision_name]
        except KeyError as error:
            raise ValueError(f"unsupported GLM-5 precision: {precision_name}") from error
        return ModelEndpoint(implementation, precision)

    @classmethod
    def _configured_endpoints(cls) -> tuple[ModelEndpoint, ModelEndpoint]:
        if cls.PRECISION_OVERRIDE:
            precision = cls.PRECISION_OVERRIDE.lower()
            return (
                cls._endpoint(f"titan:{precision}"),
                cls._endpoint(f"hf:{precision}"),
            )
        return cls._endpoint(cls.ACTUAL_ENDPOINT), cls._endpoint(cls.EXPECTED_ENDPOINT)

    @classmethod
    def setUpClass(cls) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            cls.gpu_ready = False
            cls.gpu_skip_reason = f"Transformers unavailable: {_TRANSFORMERS_IMPORT_ERROR!r}"
            return
        if not torch.cuda.is_available():
            cls.gpu_ready = False
            cls.gpu_skip_reason = "GLM-5 parity requires CUDA"
            return

        cls.gpu_ready = True
        cls.device = torch.device("cuda")
        cls.actual_endpoint, cls.expected_endpoint = cls._configured_endpoints()
        cls.precision = cls.actual_endpoint.precision

        # Build the endpoint table once.  Test methods only select from this
        # table and never perform ad hoc dtype conversion.
        precisions = {
            FP32.name,
            BF16.name,
            cls.actual_endpoint.precision.name,
            cls.expected_endpoint.precision.name,
        }
        cls.models: dict[tuple[str, str], torch.nn.Module] = {}
        cls.pairs: dict[str, ParityModelPair] = {}
        for precision_name in sorted(precisions):
            precision = BF16 if precision_name == BF16.name else FP32
            pair = _build_pair(cls.device, precision=precision, seed=61)
            cls.pairs[precision.name] = pair
            cls.models[("hf", precision.name)] = pair.hf
            cls.models[("titan", precision.name)] = pair.titan

        # Component methods operate on a same-precision HF/TorchTitan pair.
        cls.pair = cls.pairs[cls.actual_endpoint.precision.name]
        cls.hf_model = cls.pair.hf
        cls.titan_model = cls.pair.titan
        cls.batch = ParityDataFactory.make(
            device=cls.device,
            dtype=cls.actual_endpoint.precision.dtype,
            batch_size=cls.BATCH_SIZE,
            sequence_length=cls.SEQUENCE_LENGTH,
            hidden_size=cls.pair.hf.config.hidden_size,
            vocab_size=cls.pair.hf.config.vocab_size,
            seed=cls.DATA_SEED,
            data_case=cls.DATA_CASE,
            make_tokens=True,
        )
        cls.batch = ParityDataFactory.attach_hf_position_embeddings(
            cls.batch, cls.pair.hf
        )
        cls.hidden_states = cls.batch.hidden_states
        cls.positions = cls.batch.positions
        cls.causal_mask = cls.batch.causal_mask
        cls.hf_position_embeddings = cls.batch.hf_position_embeddings
        cls.base_batch = ParityDataFactory.make(
            device=cls.device,
            dtype=torch.float32,
            batch_size=cls.BATCH_SIZE,
            sequence_length=cls.SEQUENCE_LENGTH,
            hidden_size=cls.pair.hf.config.hidden_size,
            vocab_size=cls.pair.hf.config.vocab_size,
            seed=cls.DATA_SEED,
            data_case=cls.DATA_CASE,
            make_tokens=True,
        )
        cls.adapter = Glm5StateDictAdapter(
            glm5_configs["debugmodel"](), hf_assets_path=None
        )

    def _model(self, endpoint: ModelEndpoint) -> torch.nn.Module:
        return self.models[(endpoint.implementation, endpoint.precision.name)]

    def _batch_for(self, endpoint: ModelEndpoint) -> ParityBatch:
        return ParityDataFactory.cast(
            self.base_batch,
            model=self._model(endpoint),
            dtype=endpoint.precision.dtype,
        )

    def _configured_spec(self) -> ComparisonSpec:
        actual, expected = self.actual_endpoint, self.expected_endpoint
        is_fp32 = actual.precision is FP32 and expected.precision is FP32
        tolerance = (1e-4, 1e-5) if is_fp32 else (5e-2, 5e-2)
        return ComparisonSpec(
            actual=actual,
            expected=expected,
            scope="configured",
            component="model",
            rtol=tolerance[0],
            atol=tolerance[1],
        )

    def _component_enabled(self, name: str) -> bool:
        selected = {
            item.strip().lower()
            for item in self.RUN_COMPONENTS.split(",")
            if item.strip()
        }
        return not selected or "all" in selected or name in selected

    def _selected_layers(self) -> list[int]:
        if self.LAYER_INDICES.strip().lower() == "all":
            return list(range(len(self.pair.hf.model.layers)))
        return [
            int(value.strip())
            for value in self.LAYER_INDICES.split(",")
            if value.strip()
        ]

    @staticmethod
    def _endpoint_path(endpoint: ModelEndpoint, layer: int, component: str) -> str:
        base = "model.layers" if endpoint.implementation == "hf" else "layers"
        suffix = {
            "decoder_block": "",
            "indexer": ".self_attn.indexer"
            if endpoint.implementation == "hf"
            else ".attention.indexer",
            "router": ".mlp.gate"
            if endpoint.implementation == "hf"
            else ".moe.router",
        }[component]
        return f"{endpoint.label}:{base}.{layer}{suffix}"

    def _logical_activation_path(
        self, endpoint: ModelEndpoint, endpoint_path: str
    ) -> str:
        """Map HF/TorchTitan module names to one report hierarchy."""
        if endpoint.implementation == "titan":
            return endpoint_path
        path = (
            endpoint_path.replace("model.layers.", "layers.")
            .replace("model.embed_tokens", "tok_embeddings")
            .replace("model.norm", "norm")
        )
        layer_match = re.match(r"layers\.(\d+)", path)
        layer_is_moe = False
        if layer_match:
            layer = self._model(endpoint).model.layers[int(layer_match.group(1))]
            layer_is_moe = hasattr(layer.mlp, "gate")
        replacements = (
            (".self_attn.indexer", ".attention.indexer"),
            (".self_attn", ".attention"),
            (".input_layernorm", ".attention_norm"),
            (".post_attention_layernorm", ".ffn_norm"),
            (".q_a_proj", ".wq_a"),
            (".q_a_layernorm", ".q_norm"),
            (".q_b_proj", ".wq_b"),
            (".kv_a_proj_with_mqa", ".wkv_a"),
            (".kv_a_layernorm", ".kv_norm"),
            (".kv_b_proj", ".wkv_b"),
            (".o_proj", ".wo"),
            (".mlp.shared_experts.gate_proj", ".moe.shared_experts.w1"),
            (".mlp.shared_experts.up_proj", ".moe.shared_experts.w3"),
            (".mlp.shared_experts.down_proj", ".moe.shared_experts.w2"),
            (".mlp.experts.gate_up_proj", ".moe.routed_experts"),
            (".mlp.experts.down_proj", ".moe.routed_experts"),
            (".mlp.gate", ".moe.router"),
            (".mlp.gate_proj", ".feed_forward.w1"),
            (".mlp.up_proj", ".feed_forward.w3"),
            (".mlp.down_proj", ".feed_forward.w2"),
        )
        for source, target in replacements:
            path = path.replace(source, target)
        if ".mlp" in path:
            path = path.replace(".mlp", ".moe" if layer_is_moe else ".feed_forward")
        return path

    def _record_recursive_trace(
        self,
        spec: ComparisonSpec,
        trace: RecursiveModuleTrace,
        recorder: ParityRecorder,
        *,
        layer_indices: list[int] | None = None,
        component_filter: str | None = None,
    ) -> None:
        """Add every captured module activation to the hierarchical report."""
        selected_layers = (
            self._selected_layers() if layer_indices is None else layer_indices
        )
        normalized_filter = (
            component_filter.strip().strip(".")
            if component_filter is not None
            else None
        )
        if component_filter is None:
            filter_paths = set()
        elif normalized_filter in TOP_LEVEL_COMPONENTS:
            filter_paths = {normalized_filter}
        else:
            filter_paths = {
                (
                    f"layers.{layer}.{normalized_filter}"
                    if normalized_filter
                    else f"layers.{layer}"
                )
                for layer in selected_layers
            }
        leaf_filter = bool(
            normalized_filter
            and normalized_filter not in TOP_LEVEL_COMPONENTS
            and "." not in normalized_filter
        )
        logical: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
        for endpoint in (spec.actual, spec.expected):
            label = endpoint.label
            for endpoint_path, value in trace.activations[label].items():
                logical_path = self._logical_activation_path(endpoint, endpoint_path)
                logical.setdefault(logical_path, {})[label] = (endpoint_path, value)

        explicit_paths = {
            f"layers.{layer}"
            for layer in selected_layers
        }
        if normalized_filter is None:
            explicit_paths.update(
                path
                for layer in selected_layers
                for path in (
                    f"layers.{layer}.attention.indexer",
                    f"layers.{layer}.moe.router",
                )
            )
        for logical_path in sorted(logical):
            base_logical = logical_path.split("[", 1)[0]
            if base_logical in explicit_paths and base_logical not in filter_paths:
                continue
            selected_match = bool(
                filter_paths
                and any(
                    base_logical == path or base_logical.startswith(f"{path}.")
                    for path in filter_paths
                )
            )
            if leaf_filter:
                selected_match = selected_match or base_logical.endswith(
                    f".{normalized_filter}"
                )
            if filter_paths and not selected_match:
                continue
            values = logical[logical_path]
            layer_match = re.match(r"layers\.(\d+)", base_logical)
            layer: int | str = (
                int(layer_match.group(1)) if layer_match else "global"
            )
            parent_path = base_logical.rpartition(".")[0]
            level = len(logical_path.split(".")) - 1
            component = logical_path.rsplit(".", 1)[-1]
            checkpoint = base_logical.endswith(
                (".attention", ".moe", ".feed_forward")
            ) or base_logical in filter_paths or (
                leaf_filter and base_logical.endswith(f".{normalized_filter}")
            )
            node_kind = "composition_checkpoint" if checkpoint else "activation"
            actual_value = values.get(spec.actual.label)
            expected_value = values.get(spec.expected.label)
            module_path = (
                f"{spec.actual.label}:{actual_value[0] if actual_value else logical_path}"
                f" <-> {spec.expected.label}:"
                f"{expected_value[0] if expected_value else logical_path}"
            )
            if (
                actual_value is not None
                and spec.actual.label == spec.expected.label
            ):
                expected_value = actual_value
            if actual_value is None or expected_value is None:
                recorder.missing(
                    scope="trace",
                    component=component,
                    layer=layer,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind=node_kind,
                    checkpoint=False,
                    detail=(
                        f"trace-only missing actual={actual_value is not None}, "
                        f"expected={expected_value is not None}"
                    ),
                )
                continue
            if base_logical.endswith(".attention.indexer"):
                recorder.discrete(
                    scope="trace",
                    component=component,
                    layer=layer,
                    actual=actual_value[1],
                    expected=expected_value[1],
                    positions=self._batch_for(spec.actual).positions,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind="discrete_checkpoint",
                    checkpoint=checkpoint,
                )
                continue
            recorder.tensor(
                scope="trace",
                component=component,
                layer=layer,
                actual=actual_value[1],
                expected=expected_value[1],
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
            )

    def _record_recursive_gradient_trace(
        self,
        spec: ComparisonSpec,
        trace: RecursiveModuleTrace,
        recorder: ParityRecorder,
        *,
        layer_indices: list[int],
        component_filter: str | None = None,
    ) -> None:
        """Add gradients of traced module outputs to the same hierarchy."""
        normalized_filter = (
            component_filter.strip().strip(".")
            if component_filter is not None
            else None
        )
        if component_filter is None:
            filter_paths = set()
        elif normalized_filter in TOP_LEVEL_COMPONENTS:
            filter_paths = {normalized_filter}
        else:
            filter_paths = {
                (
                    f"layers.{layer}.{normalized_filter}"
                    if normalized_filter
                    else f"layers.{layer}"
                )
                for layer in layer_indices
            }
        leaf_filter = bool(
            normalized_filter
            and normalized_filter not in TOP_LEVEL_COMPONENTS
            and "." not in normalized_filter
        )
        logical: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
        for endpoint in (spec.actual, spec.expected):
            label = endpoint.label
            for endpoint_path, value in trace.gradients[label].items():
                logical_path = self._logical_activation_path(endpoint, endpoint_path)
                logical.setdefault(logical_path, {})[label] = (endpoint_path, value)
        for logical_path in sorted(logical):
            base_logical = logical_path.split("[", 1)[0]
            selected_match = bool(
                filter_paths
                and any(
                    base_logical == path or base_logical.startswith(f"{path}.")
                    for path in filter_paths
                )
            )
            if leaf_filter:
                selected_match = selected_match or base_logical.endswith(
                    f".{normalized_filter}"
                )
            if filter_paths and not selected_match:
                continue
            values = logical[logical_path]
            layer_match = re.match(r"layers\.(\d+)", base_logical)
            layer: int | str = (
                int(layer_match.group(1)) if layer_match else "global"
            )
            parent_path = base_logical.rpartition(".")[0]
            level = len(logical_path.split(".")) - 1
            component = logical_path.rsplit(".", 1)[-1]
            checkpoint = base_logical.endswith(
                (".attention", ".moe", ".feed_forward")
            ) or base_logical in filter_paths or (
                leaf_filter and base_logical.endswith(f".{normalized_filter}")
            )
            node_kind = "gradient_checkpoint" if checkpoint else "gradient_activation"
            actual_value = values.get(spec.actual.label)
            expected_value = values.get(spec.expected.label)
            module_path = (
                f"{spec.actual.label}:{actual_value[0] if actual_value else logical_path}.grad"
                f" <-> {spec.expected.label}:"
                f"{expected_value[0] if expected_value else logical_path}.grad"
            )
            if (
                actual_value is not None
                and spec.actual.label == spec.expected.label
            ):
                expected_value = actual_value
            if actual_value is None or expected_value is None:
                recorder.missing(
                    scope="activation_gradient",
                    component=component,
                    layer=layer,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind=node_kind,
                    checkpoint=False,
                    detail=(
                        f"gradient trace missing actual={actual_value is not None}, "
                        f"expected={expected_value is not None}"
                    ),
                )
                continue
            recorder.tensor(
                scope="activation_gradient",
                component=component,
                layer=layer,
                actual=actual_value[1],
                expected=expected_value[1],
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
            )

    def compare_end_to_end(self, spec: ComparisonSpec) -> ParityRecorder:
        """Compare every decoder layer plus final logits and loss."""
        actual_model = self._model(spec.actual)
        expected_model = self._model(spec.expected)
        actual_batch = self._batch_for(spec.actual)
        expected_batch = self._batch_for(spec.expected)
        endpoints = {
            spec.actual: actual_model,
            spec.expected: expected_model,
        }
        recorder = ParityRecorder(spec.actual.precision, precision_label=spec.label)
        actual_layers = (
            actual_model.model.layers
            if spec.actual.implementation == "hf"
            else actual_model.layers
        )
        layer_indices = self._selected_layers()
        if any(layer < 0 or layer >= len(actual_layers) for layer in layer_indices):
            raise ValueError(
                f"GLM5_PARITY_LAYERS is outside [0, {len(actual_layers) - 1}]: "
                f"{layer_indices}"
            )
        with (
            EndpointTrace.install(endpoints, layer_indices) as trace,
            RecursiveModuleTrace.install(
                endpoints,
                layer_indices,
                include_model_modules=True,
            ) as module_trace,
        ):
            actual_logits, actual_loss = _run_causal_lm_endpoint(
                actual_model, spec.actual, actual_batch
            )
            expected_logits, expected_loss = _run_causal_lm_endpoint(
                expected_model, spec.expected, expected_batch
            )
            actual_model.zero_grad(set_to_none=True)
            expected_model.zero_grad(set_to_none=True)
            actual_loss.backward()
            if expected_model is not actual_model:
                expected_loss.backward()

        self._record_recursive_trace(
            spec, module_trace, recorder, layer_indices=layer_indices
        )
        self._record_recursive_gradient_trace(
            spec, module_trace, recorder, layer_indices=layer_indices
        )

        for layer in layer_indices:
            actual_block = trace.blocks[spec.actual.label].get(layer)
            expected_block = trace.blocks[spec.expected.label].get(layer)
            block_path = (
                f"{self._endpoint_path(spec.actual, layer, 'decoder_block')} <-> "
                f"{self._endpoint_path(spec.expected, layer, 'decoder_block')}"
            )
            if actual_block is None or expected_block is None:
                recorder.missing(
                    scope=spec.scope,
                    component="decoder_block",
                    layer=layer,
                    module_path=block_path,
                    detail=(
                        f"missing trace actual={actual_block is not None}, "
                        f"expected={expected_block is not None}"
                    ),
                )
            else:
                recorder.tensor(
                    scope=spec.scope,
                    component="decoder_block",
                    layer=layer,
                    actual=actual_block,
                    expected=expected_block,
                    rtol=spec.rtol,
                    atol=spec.atol,
                    module_path=block_path,
                    parent_path="layers",
                    level=2,
                    node_kind="layer_checkpoint",
                )

            actual_indexer = trace.indexer[spec.actual.label].get(layer)
            expected_indexer = trace.indexer[spec.expected.label].get(layer)
            indexer_path = (
                f"{self._endpoint_path(spec.actual, layer, 'indexer')} <-> "
                f"{self._endpoint_path(spec.expected, layer, 'indexer')}"
            )
            if actual_indexer is None or expected_indexer is None:
                recorder.missing(
                    scope=spec.scope,
                    component="indexer",
                    layer=layer,
                    module_path=indexer_path,
                    parent_path=f"layers.{layer}.attention",
                    level=4,
                    node_kind="discrete_checkpoint",
                    detail=(
                        f"missing trace actual={actual_indexer is not None}, "
                        f"expected={expected_indexer is not None}"
                    ),
                )
            else:
                recorder.discrete(
                    scope=spec.scope,
                    component="indexer",
                    layer=layer,
                    actual=actual_indexer,
                    expected=expected_indexer,
                    positions=actual_batch.positions,
                    module_path=indexer_path,
                    parent_path=f"layers.{layer}.attention",
                    level=4,
                    node_kind="discrete_checkpoint",
                )

            actual_router = trace.router[spec.actual.label].get(layer)
            expected_router = trace.router[spec.expected.label].get(layer)
            if actual_router is None and expected_router is None:
                continue
            router_path = (
                f"{self._endpoint_path(spec.actual, layer, 'router')} <-> "
                f"{self._endpoint_path(spec.expected, layer, 'router')}"
            )
            if actual_router is None or expected_router is None:
                recorder.missing(
                    scope=spec.scope,
                    component="router",
                    layer=layer,
                    module_path=router_path,
                    parent_path=f"layers.{layer}.moe",
                    level=4,
                    node_kind="discrete_checkpoint",
                    detail=(
                        f"missing trace actual={actual_router is not None}, "
                        f"expected={expected_router is not None}"
                    ),
                )
            else:
                recorder.discrete(
                    scope=spec.scope,
                    component="router",
                    layer=layer,
                    actual=actual_router,
                    expected=expected_router,
                    module_path=router_path,
                    parent_path=f"layers.{layer}.moe",
                    level=4,
                    node_kind="discrete_checkpoint",
                )

        recorder.tensor(
            scope=spec.scope,
            component="logits",
            layer="all",
            actual=actual_logits,
            expected=expected_logits,
            rtol=spec.rtol,
            atol=spec.atol,
            module_path=(
                f"{spec.actual.label}:lm_head <-> {spec.expected.label}:lm_head"
            ),
            parent_path="model",
            level=1,
            node_kind="logits_checkpoint",
        )
        recorder.tensor(
            scope=spec.scope,
            component="loss",
            layer="all",
            actual=actual_loss,
            expected=expected_loss,
            rtol=spec.rtol,
            atol=spec.atol,
            module_path=f"{spec.actual.label}:loss <-> {spec.expected.label}:loss",
            parent_path="model",
            level=1,
            node_kind="loss_checkpoint",
        )
        self._record_parameter_tree(spec, recorder)
        self._record_gradient_tree(spec, recorder)
        return recorder

    def _canonical_state(
        self, endpoint: ModelEndpoint
    ) -> dict[str, torch.Tensor]:
        # Parameters are the stable adapter surface; transient buffers such
        # as rotary caches are intentionally excluded from this tree.
        state = dict(self._model(endpoint).named_parameters())
        if endpoint.implementation == "hf":
            return self.adapter.from_hf(state)
        return state

    def _canonical_gradient(
        self, endpoint: ModelEndpoint
    ) -> tuple[dict[str, torch.Tensor], set[str]]:
        """Map gradients through the same HF/TorchTitan adapter surface."""
        model = self._model(endpoint)
        if endpoint.implementation == "titan":
            gradients = {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            missing = {
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is None
            }
            return gradients, missing

        gradients: dict[str, torch.Tensor] = {}
        missing: set[str] = set()
        for name, parameter in model.named_parameters():
            is_missing = parameter.grad is None
            value = torch.zeros_like(parameter) if is_missing else parameter.grad
            mapped = self.adapter.from_hf({name: value})
            gradients.update(mapped)
            if is_missing:
                missing.update(mapped)
        return gradients, missing

    def _record_gradient_tree(
        self, spec: ComparisonSpec, recorder: ParityRecorder
    ) -> None:
        actual_state, actual_missing = self._canonical_gradient(spec.actual)
        expected_state, expected_missing = self._canonical_gradient(spec.expected)
        for key in sorted(
            set(actual_state)
            | set(expected_state)
            | actual_missing
            | expected_missing
        ):
            actual = actual_state.get(key)
            expected = expected_state.get(key)
            layer_match = self.adapter._TITAN_LAYER_KEY.fullmatch(key)
            layer = int(layer_match.group(1)) if layer_match else "global"
            actual_key = (
                self._hf_path_for_titan_key(key)
                if spec.actual.implementation == "hf"
                else key
            )
            expected_key = (
                self._hf_path_for_titan_key(key)
                if spec.expected.implementation == "hf"
                else key
            )
            module_path = (
                f"{spec.actual.label}:{actual_key}.grad <-> "
                f"{spec.expected.label}:{expected_key}.grad"
            )
            parent_path = key.rpartition(".")[0]
            level = len(key.split(".")) - 1
            if actual is None or expected is None:
                recorder.missing(
                    scope="gradient",
                    component="gradient",
                    layer=layer,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind="gradient",
                    checkpoint=not (
                        key in actual_missing and key in expected_missing
                    ),
                    detail=(
                        f"gradient missing actual={actual is not None}, "
                        f"expected={expected is not None}"
                    ),
                )
                continue
            routed_projection = {
                self.adapter._TITAN_GATE: "w1",
                self.adapter._TITAN_UP: "w3",
                self.adapter._TITAN_DOWN: "w2",
            }
            expert_name = next(
                (
                    projection
                    for suffix, projection in routed_projection.items()
                    if key.endswith(suffix)
                ),
                None,
            )
            if (
                expert_name is not None
                and actual.ndim >= 1
                and expected.ndim >= 1
                and actual.shape == expected.shape
            ):
                for expert in range(actual.shape[0]):
                    actual_expert_path = f"{actual_key}[expert={expert}]"
                    expected_expert_path = f"{expected_key}[expert={expert}]"
                    expert_parent = (
                        f"layers.{layer}.moe.routed_experts.{expert}"
                    )
                    recorder.tensor(
                        scope="gradient",
                        component=f"expert_{expert_name}_gradient",
                        layer=layer,
                        actual=actual[expert],
                        expected=expected[expert],
                        rtol=spec.rtol,
                        atol=spec.atol,
                        module_path=(
                            f"{spec.actual.label}:{actual_expert_path}.grad <-> "
                            f"{spec.expected.label}:{expected_expert_path}.grad"
                        ),
                        parent_path=expert_parent,
                        level=len(expert_parent.split(".")) - 1,
                        node_kind="gradient",
                        checkpoint=True,
                    )
                continue
            recorder.tensor(
                scope="gradient",
                component="gradient",
                layer=layer,
                actual=actual,
                expected=expected,
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind="gradient",
                checkpoint=True,
            )

    def _hf_path_for_titan_key(self, titan_key: str) -> str:
        if titan_key in self.adapter._to_hf_top_level:
            return self.adapter._to_hf_top_level[titan_key]
        match = self.adapter._TITAN_LAYER_KEY.fullmatch(titan_key)
        if match is None:
            return titan_key
        layer, suffix = match.groups()
        if suffix in {
            self.adapter._TITAN_GATE,
            self.adapter._TITAN_UP,
        }:
            return f"model.layers.{layer}.mlp.experts.gate_up_proj"
        if suffix == self.adapter._TITAN_DOWN:
            return f"model.layers.{layer}.mlp.experts.down_proj"
        hf_suffix = self.adapter._to_hf_layer.get(suffix, suffix)
        return f"model.layers.{layer}.{hf_suffix}"

    def _record_parameter_tree(
        self, spec: ComparisonSpec, recorder: ParityRecorder
    ) -> None:
        """Compare every adapter-visible state key, including packed experts."""
        actual_state = self._canonical_state(spec.actual)
        expected_state = self._canonical_state(spec.expected)
        routed_paths = {
            self.adapter._TITAN_GATE: ("w1", "gate"),
            self.adapter._TITAN_UP: ("w3", "up"),
            self.adapter._TITAN_DOWN: ("w2", "down"),
        }
        for key in sorted(set(actual_state) | set(expected_state)):
            actual = actual_state.get(key)
            expected = expected_state.get(key)
            layer_match = self.adapter._TITAN_LAYER_KEY.fullmatch(key)
            layer = int(layer_match.group(1)) if layer_match else "global"
            actual_key = (
                self._hf_path_for_titan_key(key)
                if spec.actual.implementation == "hf"
                else key
            )
            expected_key = (
                self._hf_path_for_titan_key(key)
                if spec.expected.implementation == "hf"
                else key
            )
            path = (
                f"{spec.actual.label}:{actual_key} <-> "
                f"{spec.expected.label}:{expected_key}"
            )
            if actual is None or expected is None:
                recorder.missing(
                    scope="parameters",
                    component="state_dict",
                    layer=layer,
                    module_path=path,
                    parent_path=key.rpartition(".")[0],
                    level=len(key.split(".")) - 1,
                    node_kind="parameter",
                    checkpoint=False,
                    detail=(
                        f"missing key actual={actual is not None}, "
                        f"expected={expected is not None}"
                    ),
                )
                continue
            routed_suffix = None
            for suffix, value in routed_paths.items():
                if key.endswith(suffix):
                    routed_suffix = value
                    break
            if (
                routed_suffix is not None
                and actual.ndim >= 1
                and expected.ndim >= 1
                and actual.shape == expected.shape
            ):
                expert_name, hf_projection = routed_suffix
                for expert in range(actual.shape[0]):
                    def expert_path(endpoint: ModelEndpoint) -> str:
                        if endpoint.implementation == "titan":
                            return (
                                f"layers.{layer}.moe.routed_experts."
                                f"{expert}.{expert_name}"
                            )
                        if hf_projection == "gate":
                            projection = "gate_up_proj[:, :expert_width]"
                        elif hf_projection == "up":
                            projection = "gate_up_proj[:, expert_width:]"
                        else:
                            projection = "down_proj"
                        return (
                            f"model.layers.{layer}.mlp.experts.{projection}"
                            f"[expert={expert}]"
                        )

                    module_path = (
                        f"{spec.actual.label}:{expert_path(spec.actual)} <-> "
                        f"{spec.expected.label}:{expert_path(spec.expected)}"
                    )
                    recorder.tensor(
                        scope="parameters",
                        component="expert_parameter",
                        layer=layer,
                        actual=actual[expert],
                        expected=expected[expert],
                        rtol=spec.rtol,
                        atol=spec.atol,
                        module_path=module_path,
                        parent_path=(
                            f"layers.{layer}.moe.routed_experts.{expert}"
                        ),
                        level=len(
                            f"layers.{layer}.moe.routed_experts.{expert}".split(".")
                        ) - 1,
                        node_kind="parameter",
                        checkpoint=False,
                    )
                continue
            recorder.tensor(
                scope="parameters",
                component="state_dict",
                layer=layer,
                actual=actual,
                expected=expected,
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=path,
                parent_path=key.rpartition(".")[0],
                level=len(key.split(".")) - 1,
                node_kind="parameter",
                checkpoint=False,
            )

    def test_format_reports_layer_position_and_discrete_mismatches(self) -> None:
        self._check_format_reports_layer_position_and_discrete_mismatches()

    def test_format_reports_missing_expected_records(self) -> None:
        self._check_format_reports_missing_expected_records()

    def test_format_reports_incompatible_block_shape(self) -> None:
        self._check_format_reports_incompatible_block_shape()

    def test_recorder_reports_tensor_and_discrete_rows(self) -> None:
        self._check_recorder_reports_tensor_and_discrete_rows()

    def test_recorder_writes_plain_text_report(self) -> None:
        self._check_recorder_writes_plain_text_report()

    def test_router_gate_is_evaluated_in_float32(self) -> None:
        self._check_router_gate_is_evaluated_in_float32()

    def test_router_uses_gate_forward_for_fp32_biased_computation(self) -> None:
        self._check_router_uses_gate_forward_for_fp32_biased_computation()

    def test_indexer_topk_matches_transformers_exactly(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._check_indexer_topk_matches_transformers()

    def test_router_selection_and_weights_match_transformers_exactly(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._check_router_selection_and_weights()

    def test_attention_output_matches_transformers_fp32(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._check_attention_output()

    def test_dense_block_output_matches_transformers_fp32(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._check_dense_block_output()

    def test_end_to_end_bfloat16_output_loss_moe_and_gradients(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        spec = ComparisonSpec(
            actual=ModelEndpoint("titan", BF16),
            expected=ModelEndpoint("hf", BF16),
            scope="legacy_bf16",
            component="model",
            rtol=BF16.rtol,
            atol=BF16.atol,
        )
        recorder = self.compare_end_to_end(spec)
        report_path = os.environ.get("GLM5_PARITY_REPORT")
        if report_path:
            root, extension = os.path.splitext(report_path)
            report_path = f"{root}__legacy_bf16{extension or '.txt'}"
        report = recorder.write(report_path)
        if recorder.failed:
            raise AssertionError(report)

    def test_configured_precision_suite(self) -> None:
        """Run the selected precision pair and all applicable comparisons."""
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        spec = self._configured_spec()
        recorder = self.compare_end_to_end(spec)

        same_precision = spec.actual.precision is spec.expected.precision
        cross_implementation = (
            spec.actual.implementation != spec.expected.implementation
        )
        if same_precision and cross_implementation:
            if self._component_enabled("indexer"):
                self._check_indexer_topk_matches_transformers()
            if self._component_enabled("router"):
                self._check_router_selection_and_weights()
            if self._component_enabled("attention"):
                self._check_attention_output()
            if self._component_enabled("block"):
                self._check_dense_block_output()
        selected_components = {
            item.strip().lower()
            for item in self.RUN_COMPONENTS.split(",")
            if item.strip() and item.strip().lower() != "all"
        }
        fixed_components = {
            "indexer",
            "router",
            "attention",
            "block",
            "gradient",
            "parameters",
            "logits",
            "loss",
            "model",
            "e2e",
        }
        for component in sorted(selected_components - fixed_components):
            component_recorder = self.compare_component_trace(
                component,
                self._selected_layers(),
                spec=spec,
                sequential=self.COMPONENT_EXECUTION.lower() == "sequential",
            )
            component_report_path = None
            configured_report = os.environ.get("GLM5_PARITY_REPORT")
            if configured_report:
                root, extension = os.path.splitext(configured_report)
                safe_component = re.sub(r"[^A-Za-z0-9_.-]+", "_", component)
                component_report_path = (
                    f"{root}__component_{safe_component}"
                    f"{extension or '.txt'}"
                )
            component_report = component_recorder.write(component_report_path)
            if component_recorder.failed:
                raise AssertionError(component_report)
        report_path = os.environ.get("GLM5_PARITY_REPORT")
        if report_path:
            root, extension = os.path.splitext(report_path)
            safe_label = spec.label.replace(":", "_").replace(" ", "_")
            report_path = f"{root}__{safe_label}{extension or '.txt'}"
        report = recorder.write(report_path)
        if recorder.failed:
            raise AssertionError(report)
