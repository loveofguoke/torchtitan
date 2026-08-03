# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Strict single-device GLM-5 checkpoint conversion.

Transformers stores a routed-expert gate and up projection in one tensor,
whereas TorchTitan's ``GroupedExperts`` owns three explicit grouped tensors.
This adapter deliberately keeps that conversion local and refuses keys outside
the supported debug-model checkpoint surface.
"""

import re
import warnings
from typing import Any

import torch

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.utils import MoEStateDictAdapter

from .model import Glm5Model


class Glm5StateDictAdapter(MoEStateDictAdapter):
    """Bidirectional adapter for Transformers GLM-MoE-DSA checkpoints."""

    HF_TO_TITAN_TOP_LEVEL = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    HF_TO_TITAN_LAYER = {
        "self_attn.q_a_proj.weight": "attention.wq_a.weight",
        "self_attn.q_a_layernorm.weight": "attention.q_norm.weight",
        "self_attn.q_b_proj.weight": "attention.wq_b.weight",
        "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a.weight",
        "self_attn.kv_a_layernorm.weight": "attention.kv_norm.weight",
        "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.indexer.wq_b.weight": "attention.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight": "attention.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight": "attention.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias": "attention.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight": "attention.indexer.weights_proj.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "mlp.gate.weight": "moe.router.gate.weight",
        "mlp.gate.e_score_correction_bias": "moe.expert_bias_E",
        "mlp.shared_experts.gate_proj.weight": "moe.shared_experts.w1.weight",
        "mlp.shared_experts.up_proj.weight": "moe.shared_experts.w3.weight",
        "mlp.shared_experts.down_proj.weight": "moe.shared_experts.w2.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    _IGNORED_HF_KEYS = {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }
    _HF_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    _TITAN_LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.+)$")
    _FUSED_GATE_UP = "mlp.experts.gate_up_proj"
    _FUSED_DOWN = "mlp.experts.down_proj"
    _TITAN_GATE = "moe.routed_experts.inner_experts.w1_EFD"
    _TITAN_UP = "moe.routed_experts.inner_experts.w3_EFD"
    _TITAN_DOWN = "moe.routed_experts.inner_experts.w2_EDF"

    def __init__(
        self,
        model_config: Glm5Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map = {
            **self.HF_TO_TITAN_TOP_LEVEL,
            **{
                f"model.layers.{{}}.{hf_suffix}": f"layers.{{}}.{titan_suffix}"
                for hf_suffix, titan_suffix in self.HF_TO_TITAN_LAYER.items()
            },
        }
        self._to_hf_top_level = {v: k for k, v in self.HF_TO_TITAN_TOP_LEVEL.items()}
        self._to_hf_layer = {v: k for k, v in self.HF_TO_TITAN_LAYER.items()}

    def _layer_config(self, layer_index: int):
        if not 0 <= layer_index < len(self.model_config.layers):
            raise KeyError(f"unmapped HF key: layer index {layer_index} is unsupported")
        return self.model_config.layers[layer_index]

    @staticmethod
    def _norm_shape(normalized_shape: int | tuple[int, ...]) -> tuple[int, ...]:
        return (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )

    @staticmethod
    def _linear_shape(config: Any) -> tuple[int, int]:
        return (config.out_features, config.in_features)

    def _expected_shape(self, titan_key: str) -> tuple[int, ...]:
        if titan_key == "tok_embeddings.weight":
            return (
                self.model_config.tok_embeddings.num_embeddings,
                self.model_config.tok_embeddings.embedding_dim,
            )
        if titan_key == "norm.weight":
            return self._norm_shape(self.model_config.norm.normalized_shape)
        if titan_key == "lm_head.weight":
            return self._linear_shape(self.model_config.lm_head)

        match = self._TITAN_LAYER_KEY.fullmatch(titan_key)
        if match is None:
            raise KeyError(f"unmapped TorchTitan key: {titan_key}")
        layer_index = int(match.group(1))
        suffix = match.group(2)
        layer = self._layer_config(layer_index)
        attention = layer.attention

        linear_configs = {
            "attention.wq_a.weight": attention.wq_a,
            "attention.wq_b.weight": attention.wq_b,
            "attention.wkv_a.weight": attention.wkv_a,
            "attention.wkv_b.weight": attention.wkv_b,
            "attention.wo.weight": attention.wo,
            "attention.indexer.wq_b.weight": attention.indexer.wq_b,
            "attention.indexer.wk.weight": attention.indexer.wk,
            "attention.indexer.weights_proj.weight": attention.indexer.weights_proj,
        }
        if suffix in linear_configs:
            return self._linear_shape(linear_configs[suffix])

        norm_configs = {
            "attention.q_norm.weight": attention.q_norm,
            "attention.kv_norm.weight": attention.kv_norm,
            "attention.indexer.k_norm.weight": attention.indexer.k_norm,
            "attention.indexer.k_norm.bias": attention.indexer.k_norm,
            "attention_norm.weight": layer.attention_norm,
            "ffn_norm.weight": layer.ffn_norm,
        }
        if suffix in norm_configs:
            return self._norm_shape(norm_configs[suffix].normalized_shape)

        if suffix.startswith("feed_forward.") and layer.feed_forward is not None:
            projection = suffix.removeprefix("feed_forward.").removesuffix(".weight")
            if projection in {"w1", "w2", "w3"}:
                return self._linear_shape(getattr(layer.feed_forward, projection))

        if layer.moe is not None:
            moe = layer.moe
            if suffix == "moe.router.gate.weight":
                return self._linear_shape(moe.router.gate)
            if suffix == "moe.expert_bias_E":
                return (moe.num_experts,)
            if suffix in {self._TITAN_GATE, self._TITAN_UP}:
                experts = moe.routed_experts.inner_experts
                return (experts.num_experts, experts.hidden_dim, experts.dim)
            if suffix == self._TITAN_DOWN:
                experts = moe.routed_experts.inner_experts
                return (experts.num_experts, experts.dim, experts.hidden_dim)
            if suffix.startswith("moe.shared_experts."):
                projection = suffix.removeprefix("moe.shared_experts.").removesuffix(
                    ".weight"
                )
                if projection in {"w1", "w2", "w3"}:
                    return self._linear_shape(getattr(moe.shared_experts, projection))

        raise KeyError(f"unmapped TorchTitan key: {titan_key}")

    def _validate_tensor_shape(self, titan_key: str, value: Any) -> None:
        expected = self._expected_shape(titan_key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(
                f"invalid shape for {titan_key}: expected {expected}, got {actual}"
            )

    def _validate_fused_experts(self, layer_index: int, value: Any) -> None:
        layer = self._layer_config(layer_index)
        if layer.moe is None:
            raise KeyError(
                f"unmapped HF key: layer {layer_index} has no routed GLM-5 experts"
            )
        experts = layer.moe.routed_experts.inner_experts
        expected = (experts.num_experts, 2 * experts.hidden_dim, experts.dim)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(
                f"invalid fused expert shape: expected {expected}, got {actual}"
            )

    def _validate_fused_down(self, layer_index: int, value: Any) -> None:
        titan_key = f"layers.{layer_index}.{self._TITAN_DOWN}"
        self._validate_tensor_shape(titan_key, value)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        self._validate_hf_rope_config(ComplexRoPE.Config)
        state_dict: dict[str, Any] = {}

        for hf_key, value in hf_state_dict.items():
            if hf_key in self._IGNORED_HF_KEYS:
                continue
            if hf_key in self.HF_TO_TITAN_TOP_LEVEL:
                titan_key = self.HF_TO_TITAN_TOP_LEVEL[hf_key]
                self._validate_tensor_shape(titan_key, value)
                state_dict[titan_key] = value
                continue

            match = self._HF_LAYER_KEY.fullmatch(hf_key)
            if match is None:
                raise KeyError(f"unmapped HF key: {hf_key}")
            layer_index = int(match.group(1))
            suffix = match.group(2)
            if layer_index == len(self.model_config.layers):
                warnings.warn(
                    "Skipping unsupported GLM-5 next MTP layer namespace "
                    f"model.layers.{layer_index}.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            self._layer_config(layer_index)

            if suffix == self._FUSED_GATE_UP:
                self._validate_fused_experts(layer_index, value)
                gate, up = value.chunk(2, dim=1)
                gate_key = f"layers.{layer_index}.{self._TITAN_GATE}"
                up_key = f"layers.{layer_index}.{self._TITAN_UP}"
                state_dict[gate_key] = gate
                state_dict[up_key] = up
                continue
            if suffix == self._FUSED_DOWN:
                self._validate_fused_down(layer_index, value)
                state_dict[f"layers.{layer_index}.{self._TITAN_DOWN}"] = value
                continue

            titan_suffix = self.HF_TO_TITAN_LAYER.get(suffix)
            if titan_suffix is None:
                raise KeyError(f"unmapped HF key: {hf_key}")
            titan_key = f"layers.{layer_index}.{titan_suffix}"
            self._validate_tensor_shape(titan_key, value)
            state_dict[titan_key] = value

        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        fused_experts: dict[int, dict[str, Any]] = {}

        for titan_key, value in state_dict.items():
            hf_key = self._to_hf_top_level.get(titan_key)
            if hf_key is not None:
                self._validate_tensor_shape(titan_key, value)
                hf_state_dict[hf_key] = value
                continue

            match = self._TITAN_LAYER_KEY.fullmatch(titan_key)
            if match is None:
                raise KeyError(f"unmapped TorchTitan key: {titan_key}")
            layer_index = int(match.group(1))
            suffix = match.group(2)
            self._layer_config(layer_index)

            if suffix in {self._TITAN_GATE, self._TITAN_UP, self._TITAN_DOWN}:
                self._validate_tensor_shape(titan_key, value)
                fused_experts.setdefault(layer_index, {})[suffix] = value
                continue

            hf_suffix = self._to_hf_layer.get(suffix)
            if hf_suffix is None:
                raise KeyError(f"unmapped TorchTitan key: {titan_key}")
            self._validate_tensor_shape(titan_key, value)
            hf_state_dict[f"model.layers.{layer_index}.{hf_suffix}"] = value

        for layer_index, weights in fused_experts.items():
            required = {self._TITAN_GATE, self._TITAN_UP, self._TITAN_DOWN}
            missing = required.difference(weights)
            if missing:
                raise KeyError(
                    "incomplete routed expert mapping for layer "
                    f"{layer_index}: missing {sorted(missing)}"
                )
            hf_state_dict[
                f"model.layers.{layer_index}.{self._FUSED_GATE_UP}"
            ] = torch.cat((weights[self._TITAN_GATE], weights[self._TITAN_UP]), dim=1)
            hf_state_dict[f"model.layers.{layer_index}.{self._FUSED_DOWN}"] = weights[
                self._TITAN_DOWN
            ]

        return hf_state_dict
