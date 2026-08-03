# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional CUDA component-parity checks for the GLM-5 debug model.

These checks intentionally exercise the real Transformers implementation and
the state-dict adapter.  They remain CUDA-gated because Task 8 owns the
single-GPU end-to-end BF16 and gradient acceptance work.
"""

import unittest

import torch

from torchtitan.models.glm5 import glm5_configs, Glm5StateDictAdapter

try:
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
except Exception as _TRANSFORMERS_IMPORT_ERROR:  # pragma: no cover - environment dependent
    GlmMoeDsaConfig = None
    GlmMoeDsaForCausalLM = None
else:
    _TRANSFORMERS_IMPORT_ERROR = None


def _hf_config():
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


def _build_models(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Construct exact FP32 peers and load the HF tensors through the adapter."""
    assert GlmMoeDsaForCausalLM is not None
    torch.manual_seed(41)
    hf_model = GlmMoeDsaForCausalLM(_hf_config()).float()

    titan_config = glm5_configs["debugmodel"]()
    titan_model = titan_config.build()
    titan_model.init_states()
    adapter = Glm5StateDictAdapter(titan_config, hf_assets_path=None)
    titan_model.load_state_dict(adapter.from_hf(hf_model.state_dict()), strict=True)

    hf_model = hf_model.to(device).eval()
    titan_model = titan_model.to(device).eval()
    return hf_model, titan_model


def _causal_mask(positions_BL: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    B, L = positions_BL.shape
    token_indices_L = torch.arange(L, device=positions_BL.device)
    allowed_BLL = token_indices_L[None, :, None] >= token_indices_L[None, None, :]
    return torch.zeros(B, 1, L, L, dtype=dtype, device=positions_BL.device).masked_fill(
        ~allowed_BLL.unsqueeze(1), torch.finfo(dtype).min
    )


class TestGlm5TransformersComponentParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                "Transformers GLM-MoE-DSA is unavailable: "
                f"{_TRANSFORMERS_IMPORT_ERROR!r}"
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "GLM-5 Transformers parity requires one CUDA device"
            )

        cls.device = torch.device("cuda")
        cls.hf_model, cls.titan_model = _build_models(cls.device)
        cls.batch_size = 2
        cls.sequence_length = 16
        torch.manual_seed(41)
        cls.hidden_states = torch.randn(
            cls.batch_size,
            cls.sequence_length,
            256,
            device=cls.device,
            dtype=torch.float32,
        )
        cls.positions = torch.arange(
            cls.sequence_length, device=cls.device, dtype=torch.long
        ).expand(cls.batch_size, -1)
        cls.causal_mask = _causal_mask(cls.positions, cls.hidden_states.dtype)
        cls.hf_position_embeddings = cls.hf_model.model.rotary_emb(
            cls.hidden_states, position_ids=cls.positions
        )

    def _normalized_hidden_states(self, layer_index: int) -> torch.Tensor:
        return self.hf_model.model.layers[layer_index].input_layernorm(
            self.hidden_states
        )

    def test_indexer_topk_matches_transformers_exactly(self) -> None:
        layer_index = 0
        hf_attention = self.hf_model.model.layers[layer_index].self_attn
        titan_attention = self.titan_model.layers[str(layer_index)].attention
        hidden_states = self._normalized_hidden_states(layer_index)
        hf_q_resid = hf_attention.q_a_layernorm(hf_attention.q_a_proj(hidden_states))
        titan_q_resid = titan_attention.q_norm(titan_attention.wq_a(hidden_states))
        torch.testing.assert_close(titan_q_resid, hf_q_resid, rtol=0, atol=0)

        hf_topk = hf_attention.indexer(
            hidden_states,
            hf_q_resid,
            self.hf_position_embeddings,
            self.causal_mask[:, 0],
            self.positions,
        )
        titan_topk = titan_attention.indexer(
            hidden_states,
            titan_q_resid,
            self.positions,
            self.causal_mask[:, 0],
        )
        self.assertTrue(torch.equal(titan_topk, hf_topk))

    def test_router_selection_and_weights_match_transformers_exactly(self) -> None:
        layer_index = 1
        hf_layer = self.hf_model.model.layers[layer_index]
        titan_moe = self.titan_model.layers[str(layer_index)].moe
        normalized_hidden_states = self._normalized_hidden_states(layer_index)

        _, hf_weights_RK, hf_indices_RK = hf_layer.mlp.gate(normalized_hidden_states)
        titan_weights_BLK, titan_indices_BLK, _ = titan_moe.router(
            normalized_hidden_states, titan_moe.expert_bias_E
        )
        hf_weights_BLK = hf_weights_RK.view_as(titan_weights_BLK)
        hf_indices_BLK = hf_indices_RK.view_as(titan_indices_BLK)
        self.assertTrue(torch.equal(titan_indices_BLK, hf_indices_BLK))
        torch.testing.assert_close(
            titan_weights_BLK, hf_weights_BLK, rtol=1e-6, atol=1e-7
        )

    def test_attention_output_matches_transformers_fp32(self) -> None:
        layer_index = 0
        hf_attention = self.hf_model.model.layers[layer_index].self_attn
        titan_attention = self.titan_model.layers[str(layer_index)].attention
        normalized_hidden_states = self._normalized_hidden_states(layer_index)

        hf_output_BLD = hf_attention(
            hidden_states=normalized_hidden_states,
            position_embeddings=self.hf_position_embeddings,
            attention_mask=self.causal_mask,
            position_ids=self.positions,
        )[0]
        titan_output_BLD = titan_attention(
            normalized_hidden_states, self.causal_mask, self.positions
        )
        torch.testing.assert_close(
            titan_output_BLD, hf_output_BLD, rtol=1e-4, atol=1e-5
        )

    def test_dense_block_output_matches_transformers_fp32(self) -> None:
        layer_index = 0
        hf_layer = self.hf_model.model.layers[layer_index]
        titan_layer = self.titan_model.layers[str(layer_index)]

        hf_output_BLD = hf_layer(
            self.hidden_states,
            attention_mask=self.causal_mask,
            position_ids=self.positions,
            position_embeddings=self.hf_position_embeddings,
            use_cache=False,
        )[0]
        titan_output_BLD = titan_layer(
            self.hidden_states, self.causal_mask, self.positions
        )
        torch.testing.assert_close(
            titan_output_BLD, hf_output_BLD, rtol=1e-4, atol=1e-5
        )
