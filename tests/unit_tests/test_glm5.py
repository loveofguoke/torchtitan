# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import torchtitan.models.glm5 as glm5

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.common import (
    ComplexRoPE,
    FlexAttention,
    LayerNorm,
    Linear,
    RMSNorm,
)
from torchtitan.models.glm5 import build_glm5_layers, glm5_configs, Glm5StateDictAdapter
from torchtitan.models.glm5.config_registry import glm5_debugmodel
from torchtitan.models.glm5.model import (
    Glm5Attention,
    Glm5DsaIndexer,
    Glm5Model,
    Glm5TransformerBlock,
)


def _indexer_config() -> Glm5DsaIndexer.Config:
    return Glm5DsaIndexer.Config(
        dim=16,
        q_lora_rank=8,
        n_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        index_topk=3,
        wq_b=Linear.Config(in_features=8, out_features=16),
        wk=Linear.Config(in_features=16, out_features=8),
        k_norm=LayerNorm.Config(normalized_shape=8),
        weights_proj=Linear.Config(in_features=16, out_features=2),
        rope=ComplexRoPE.Config(dim=4, max_seq_len=8, theta=1_000_000),
    )


def _attention_config() -> Glm5Attention.Config:
    return Glm5Attention.Config(
        dim=16,
        n_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        attention_dropout=0.0,
        wq_a=Linear.Config(in_features=16, out_features=8),
        q_norm=RMSNorm.Config(normalized_shape=8),
        wq_b=Linear.Config(in_features=8, out_features=16),
        wkv_a=Linear.Config(in_features=16, out_features=8),
        kv_norm=RMSNorm.Config(normalized_shape=4),
        wkv_b=Linear.Config(in_features=4, out_features=16),
        wo=Linear.Config(in_features=8, out_features=16),
        rope=ComplexRoPE.Config(dim=4, max_seq_len=8, theta=1_000_000),
        indexer=_indexer_config(),
        inner_attention=FlexAttention.Config(),
    )


def _build_debug_model() -> Glm5Model:
    model = glm5_configs["debugmodel"]().build()
    model.init_states()
    model.eval()
    return model


def _debug_layer_kwargs() -> dict:
    config = glm5_configs["debugmodel"]()
    attention = config.layers[0].attention
    moe = config.layers[1].moe
    assert moe is not None
    return {
        "n_layers": len(config.layers),
        "n_dense_layers": 1,
        "dim": config.dim,
        "n_heads": attention.n_heads,
        "q_lora_rank": attention.q_lora_rank,
        "kv_lora_rank": attention.kv_lora_rank,
        "qk_nope_head_dim": attention.qk_nope_head_dim,
        "qk_rope_head_dim": attention.qk_rope_head_dim,
        "v_head_dim": attention.v_head_dim,
        "dense_hidden_dim": config.layers[0].feed_forward.w1.out_features,
        "moe_hidden_dim": moe.routed_experts.inner_experts.hidden_dim,
        "num_experts": moe.num_experts,
        "num_shared_experts": (
            moe.shared_experts.w1.out_features
            // moe.routed_experts.inner_experts.hidden_dim
        ),
        "router_top_k": moe.router.top_k,
        "router_num_expert_groups": moe.router.num_expert_groups,
        "router_num_limited_groups": moe.router.num_limited_groups,
        "router_route_scale": moe.router.route_scale,
        "index_n_heads": attention.indexer.n_heads,
        "index_head_dim": attention.indexer.head_dim,
        "index_topk": attention.indexer.index_topk,
        "attention_dropout": attention.attention_dropout,
        "rope": attention.rope,
    }


def _dense_causal_mask(
    positions_BL: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, L = positions_BL.shape
    key_positions_11L = torch.arange(L, device=positions_BL.device)[None, None, :]
    return torch.zeros(B, 1, L, L, dtype=dtype, device=positions_BL.device).masked_fill(
        key_positions_11L > positions_BL.unsqueeze(-1).unsqueeze(1),
        float("-inf"),
    )


def _reference_sparse_attention(
    attention: Glm5Attention,
    x_BLD: torch.Tensor,
    attention_masks_B1LL: torch.Tensor,
    positions_BL: torch.Tensor,
    topk_indices_BLK: torch.Tensor,
) -> torch.Tensor:
    B, L, _ = x_BLD.shape
    q_resid_BLR = _reference_rms_norm(
        F.linear(x_BLD, attention.wq_a.weight, attention.wq_a.bias),
        attention.q_norm,
    )
    q_BLNH = F.linear(q_resid_BLR, attention.wq_b.weight, attention.wq_b.bias).view(
        B, L, attention.n_heads, attention.qk_head_dim
    )
    q_nope_BLNP, q_rope_BLNR = torch.split(
        q_BLNH,
        [attention.qk_nope_head_dim, attention.qk_rope_head_dim],
        dim=-1,
    )
    compressed_kv_BLC = F.linear(x_BLD, attention.wkv_a.weight, attention.wkv_a.bias)
    kv_BLR, k_rope_BL1R = torch.split(
        compressed_kv_BLC,
        [attention.kv_lora_rank, attention.qk_rope_head_dim],
        dim=-1,
    )
    kv_BLR = _reference_rms_norm(kv_BLR, attention.kv_norm)
    k_rope_BL1R = k_rope_BL1R.unsqueeze(2)
    rope_cache_BL1R2 = attention.rope.cache[positions_BL].unsqueeze(2)
    q_rope_complex_BLNR2 = torch.view_as_complex(
        q_rope_BLNR.float().reshape(B, L, attention.n_heads, -1, 2)
    )
    k_rope_complex_BL1R2 = torch.view_as_complex(
        k_rope_BL1R.float().reshape(B, L, 1, -1, 2)
    )
    q_rope_BLNR = (
        torch.view_as_real(q_rope_complex_BLNR2 * rope_cache_BL1R2)
        .flatten(3)
        .type_as(q_rope_BLNR)
    )
    k_rope_BL1R = (
        torch.view_as_real(k_rope_complex_BL1R2 * rope_cache_BL1R2)
        .flatten(3)
        .type_as(k_rope_BL1R)
    )
    q_BLNH = torch.cat((q_nope_BLNP, q_rope_BLNR), dim=-1)
    kv_BLNX = F.linear(kv_BLR, attention.wkv_b.weight, attention.wkv_b.bias).view(
        B,
        L,
        attention.n_heads,
        attention.qk_nope_head_dim + attention.v_head_dim,
    )
    k_nope_BLNP, v_BLNV = torch.split(
        kv_BLNX, [attention.qk_nope_head_dim, attention.v_head_dim], dim=-1
    )
    k_BLNH = torch.cat(
        (k_nope_BLNP, k_rope_BL1R.expand(-1, -1, attention.n_heads, -1)),
        dim=-1,
    )
    selected_BLL = torch.zeros(B, L, L, dtype=torch.bool, device=x_BLD.device).scatter(
        -1, topk_indices_BLK.long(), True
    )
    sparse_mask_B1LL = attention_masks_B1LL.masked_fill(
        ~selected_BLL.unsqueeze(1), torch.finfo(x_BLD.dtype).min
    )
    scores_BNLL = (
        torch.matmul(q_BLNH.transpose(1, 2), k_BLNH.transpose(1, 2).transpose(-1, -2))
        * attention.softmax_scale
    )
    scores_BNLL = scores_BNLL + sparse_mask_B1LL
    probs_BNLL = F.softmax(scores_BNLL, dim=-1, dtype=torch.float32).to(q_BLNH.dtype)
    output_BLNV = torch.matmul(probs_BNLL, v_BLNV.transpose(1, 2)).transpose(1, 2)
    return F.linear(
        output_BLNV.contiguous().view(B, L, -1), attention.wo.weight, attention.wo.bias
    )


def _reference_rms_norm(x_BLD: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    eps = torch.finfo(x_BLD.dtype).eps if norm.eps is None else norm.eps
    return (
        x_BLD
        * torch.rsqrt(x_BLD.square().mean(dim=-1, keepdim=True) + eps)
        * norm.weight
    )


def _reference_indexer_topk(
    indexer: Glm5DsaIndexer,
    hidden_states_BLD: torch.Tensor,
    q_resid_BLR: torch.Tensor,
    positions_BL: torch.Tensor,
    attention_mask_BLL: torch.Tensor | None,
) -> torch.Tensor:
    B, L, _ = hidden_states_BLD.shape
    q_BLNH = F.linear(
        q_resid_BLR,
        indexer.wq_b.weight,
        indexer.wq_b.bias,
    ).view(B, L, indexer.n_heads, indexer.head_dim)
    q_rot_BLNR, q_pass_BLNP = torch.split(
        q_BLNH,
        [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim],
        dim=-1,
    )
    k_BLH = F.layer_norm(
        F.linear(hidden_states_BLD, indexer.wk.weight, indexer.wk.bias),
        indexer.k_norm.normalized_shape,
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        indexer.k_norm.eps,
    )
    k_BL1H = k_BLH.unsqueeze(2)
    k_rot_BL1R, k_pass_BL1P = torch.split(
        k_BL1H,
        [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim],
        dim=-1,
    )
    rope_cache_BL1R2 = indexer.rope.cache[positions_BL].unsqueeze(2)
    q_rot_complex_BLNR2 = torch.view_as_complex(
        q_rot_BLNR.float().reshape(B, L, indexer.n_heads, -1, 2)
    )
    k_rot_complex_BL1R2 = torch.view_as_complex(
        k_rot_BL1R.float().reshape(B, L, 1, -1, 2)
    )
    q_rot_BLNR = (
        torch.view_as_real(q_rot_complex_BLNR2 * rope_cache_BL1R2)
        .flatten(3)
        .type_as(q_rot_BLNR)
    )
    k_rot_BL1R = (
        torch.view_as_real(k_rot_complex_BL1R2 * rope_cache_BL1R2)
        .flatten(3)
        .type_as(k_rot_BL1R)
    )
    q_BLNH = torch.cat((q_rot_BLNR, q_pass_BLNP), dim=-1)
    k_BLH = torch.cat((k_rot_BL1R, k_pass_BL1P), dim=-1).squeeze(2)

    scores_BNLL = (
        torch.matmul(
            q_BLNH.float().transpose(1, 2),
            k_BLH.float().transpose(1, 2).unsqueeze(1),
        )
        * indexer.softmax_scale
    )
    scores_BNLL = F.relu(scores_BNLL)
    weights_BLN = F.linear(
        hidden_states_BLD.to(indexer.weights_proj.weight.dtype),
        indexer.weights_proj.weight,
        indexer.weights_proj.bias,
    ).float() * (indexer.n_heads**-0.5)
    index_scores_BLL = torch.matmul(
        weights_BLN.unsqueeze(-2), scores_BNLL.transpose(1, 2)
    ).squeeze(-2)
    if attention_mask_BLL is not None:
        index_scores_BLL = index_scores_BLL + attention_mask_BLL.float()
    else:
        key_positions_11L = torch.arange(L, device=positions_BL.device)[None, None, :]
        index_scores_BLL = index_scores_BLL.masked_fill(
            key_positions_11L > positions_BL.unsqueeze(-1), float("-inf")
        )
    topk = min(indexer.index_topk, index_scores_BLL.shape[-1])
    return index_scores_BLL.topk(topk, dim=-1).indices.to(torch.int32)


class TestGlm5DsaIndexer(unittest.TestCase):
    def test_indexer_returns_masked_int32_topk(self):
        indexer = _indexer_config().build()
        indexer.init_states()
        hidden_states_BLD = torch.randn(2, 5, 16)
        q_resid_BLR = torch.randn(2, 5, 8)
        positions_BL = torch.arange(5).expand(2, -1)
        attention_mask_BLL = torch.full((2, 5, 5), float("-inf"))
        attention_mask_BLL.masked_fill_(
            torch.ones(5, 5, dtype=torch.bool).tril().unsqueeze(0), 0.0
        )

        topk_indices_BLK = indexer(
            hidden_states_BLD,
            q_resid_BLR,
            positions_BL,
            attention_mask_BLL,
        )

        self.assertEqual(topk_indices_BLK.dtype, torch.int32)
        self.assertEqual(topk_indices_BLK.shape, (2, 5, 3))
        full_topk_queries_B = positions_BL >= topk_indices_BLK.shape[-1] - 1
        self.assertTrue(
            torch.all(
                topk_indices_BLK[full_topk_queries_B]
                <= positions_BL[full_topk_queries_B].unsqueeze(-1)
            )
        )
        self.assertTrue(
            torch.all(
                torch.any(
                    topk_indices_BLK[:, 0] > positions_BL[:, 0].unsqueeze(-1),
                    dim=-1,
                )
            )
        )

    def test_indexer_matches_independent_reference(self):
        torch.manual_seed(17)
        indexer = _indexer_config().build()
        indexer.init_states()
        hidden_states_BLD = torch.randn(1, 4, 16)
        q_resid_BLR = torch.randn(1, 4, 8)
        positions_BL = torch.arange(4).unsqueeze(0)
        attention_mask_BLL = torch.zeros(1, 4, 4).masked_fill(
            ~torch.ones(4, 4, dtype=torch.bool).tril().unsqueeze(0),
            float("-inf"),
        )

        actual_BLK = indexer(
            hidden_states_BLD,
            q_resid_BLR,
            positions_BL,
            attention_mask_BLL,
        )
        expected_BLK = _reference_indexer_topk(
            indexer,
            hidden_states_BLD,
            q_resid_BLR,
            positions_BL,
            attention_mask_BLL,
        )
        self.assertTrue(torch.equal(actual_BLK, expected_BLK))

    def test_indexer_is_no_grad_and_keeps_weights_projection_fp32(self):
        indexer = _indexer_config().build()
        indexer.init_states()
        indexer.bfloat16()
        self.assertEqual(indexer.weights_proj.weight.dtype, torch.float32)
        out_BLK = indexer(
            torch.randn(1, 4, 16, dtype=torch.bfloat16),
            torch.randn(1, 4, 8, dtype=torch.bfloat16),
            torch.arange(4).unsqueeze(0),
            torch.zeros(1, 4, 4, dtype=torch.bfloat16),
        )
        self.assertFalse(out_BLK.requires_grad)


class TestGlm5Attention(unittest.TestCase):
    def test_attention_topk_cannot_reopen_causal_mask(self):
        attention = _attention_config().build()
        attention.init_states()
        x_BLD = torch.randn(1, 4, 16)
        positions_BL = torch.arange(4).unsqueeze(0)
        base_mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
        future_selecting_topk_BLK = torch.tensor(
            [[[3, 2], [3, 2], [3, 2], [3, 2]]], dtype=torch.int32
        )
        with mock.patch.object(
            attention.indexer,
            "forward",
            return_value=future_selecting_topk_BLK,
        ):
            actual_BLD = attention(x_BLD, base_mask_B1LL, positions_BL)
        expected_BLD = _reference_sparse_attention(
            attention,
            x_BLD,
            base_mask_B1LL,
            positions_BL,
            future_selecting_topk_BLK,
        )
        torch.testing.assert_close(actual_BLD, expected_BLD, rtol=1e-5, atol=1e-6)

    def test_attention_output_shape_and_backward(self):
        attention = _attention_config().build()
        attention.init_states()
        x_BLD = torch.randn(2, 5, 16, requires_grad=True)
        positions_BL = torch.arange(5).expand(2, -1)
        mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
        output_BLD = attention(x_BLD, mask_B1LL, positions_BL)
        self.assertEqual(output_BLD.shape, x_BLD.shape)
        output_BLD.square().mean().backward()
        self.assertIsNotNone(attention.wq_a.weight.grad)
        self.assertIsNotNone(attention.wo.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in attention.indexer.parameters())
        )

    def test_attention_rejects_malformed_additive_masks(self):
        attention = _attention_config().build()
        attention.init_states()
        x_BLD = torch.randn(2, 4, 16)
        positions_BL = torch.arange(4).expand(2, -1)
        mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
        invalid_masks = {
            "rank": mask_B1LL[:, 0],
            "channels": torch.zeros(2, 2, 4, 4),
            "batch": torch.zeros(1, 1, 4, 4),
            "length": torch.zeros(2, 1, 4, 3),
            "dtype": torch.zeros(2, 1, 4, 4, dtype=torch.int32),
            "device": torch.zeros(2, 1, 4, 4, device="meta"),
        }
        for case, invalid_mask_B1LL in invalid_masks.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "attention_masks"):
                    attention(x_BLD, invalid_mask_B1LL, positions_BL)

    def test_attention_config_rejects_zero_mla_dimensions(self):
        invalid_configs = {
            "kv_lora_rank": ({"kv_lora_rank": 0}, "kv_lora_rank"),
            "qk_nope_head_dim": ({"qk_nope_head_dim": 0}, "qk_nope_head_dim"),
            "qk_rope_head_dim": ({"qk_rope_head_dim": -4}, "qk_rope_head_dim"),
            "v_head_dim": ({"v_head_dim": 0}, "v_head_dim"),
        }
        for case, (changes, message) in invalid_configs.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, message):
                    dataclasses.replace(_attention_config(), **changes)

    def test_attention_config_rejects_dimension_consistent_zero_rope(self):
        config = _attention_config()
        zero_rope_indexer = dataclasses.replace(
            config.indexer,
            head_dim=4,
            qk_rope_head_dim=0,
            wq_b=Linear.Config(in_features=8, out_features=8),
            wk=Linear.Config(in_features=16, out_features=4),
            k_norm=LayerNorm.Config(normalized_shape=4),
            rope=dataclasses.replace(config.indexer.rope, dim=0),
        )
        with self.assertRaisesRegex(ValueError, "qk_rope_head_dim"):
            dataclasses.replace(
                config,
                qk_rope_head_dim=0,
                wq_b=Linear.Config(in_features=8, out_features=8),
                wkv_a=Linear.Config(in_features=16, out_features=4),
                rope=dataclasses.replace(config.rope, dim=0),
                indexer=zero_rope_indexer,
            )


class TestGlm5Model(unittest.TestCase):
    def test_layer_builder_rejects_infeasible_grouped_routing(self):
        kwargs = _debug_layer_kwargs()
        invalid_groupings = (
            {
                "router_num_expert_groups": 4,
                "router_num_limited_groups": 1,
                "router_top_k": 3,
            },
            {"router_num_expert_groups": 8},
        )
        for overrides in invalid_groupings:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "experts_per_group|top_k"):
                    build_glm5_layers(**(kwargs | overrides))

    def test_debug_model_config_has_approved_architecture(self):
        config = glm5_configs["debugmodel"]()

        self.assertEqual(config.vocab_size, 2048)
        self.assertEqual(config.dim, 256)
        self.assertEqual(len(config.layers), 4)
        self.assertEqual(config.max_seq_len, 128)
        self.assertEqual(config.norm.eps, 1e-5)
        for layer_id, layer_config in enumerate(config.layers):
            self.assertIsInstance(layer_config, Glm5TransformerBlock.Config)
            attention = layer_config.attention
            self.assertEqual(attention.n_heads, 8)
            self.assertEqual(attention.q_lora_rank, 128)
            self.assertEqual(attention.kv_lora_rank, 64)
            self.assertEqual(attention.qk_nope_head_dim, 32)
            self.assertEqual(attention.qk_rope_head_dim, 32)
            self.assertEqual(attention.v_head_dim, 64)
            self.assertEqual(attention.attention_dropout, 0.0)
            self.assertEqual(attention.rope.max_seq_len, 128)
            self.assertEqual(attention.rope.theta, 1_000_000)
            self.assertEqual(attention.rope.scaling, "none")
            self.assertEqual(attention.q_norm.eps, 1e-6)
            self.assertEqual(attention.kv_norm.eps, 1e-6)
            self.assertEqual(attention.indexer.n_heads, 4)
            self.assertEqual(attention.indexer.head_dim, 64)
            self.assertEqual(attention.indexer.index_topk, 8)
            self.assertEqual(attention.indexer.k_norm.eps, 1e-6)
            self.assertIsNotNone(attention.indexer)
            if layer_id == 0:
                self.assertIsNotNone(layer_config.feed_forward)
                self.assertIsNone(layer_config.moe)
                self.assertEqual(layer_config.feed_forward.w1.out_features, 1024)
            else:
                self.assertIsNone(layer_config.feed_forward)
                self.assertIsNotNone(layer_config.moe)
                self.assertEqual(layer_config.moe.num_experts, 8)
                self.assertEqual(layer_config.moe.router.top_k, 2)
                self.assertEqual(layer_config.moe.router.num_expert_groups, 1)
                self.assertEqual(layer_config.moe.router.num_limited_groups, 1)
                self.assertEqual(layer_config.moe.router.score_func, "sigmoid")
                self.assertEqual(layer_config.moe.router.route_scale, 2.5)
                self.assertTrue(layer_config.moe.router.route_norm)
                self.assertEqual(
                    layer_config.moe.routed_experts.inner_experts.hidden_dim, 256
                )
                self.assertEqual(layer_config.moe.shared_experts.w1.out_features, 256)

    def test_dense_mask_enforces_causality_and_document_boundaries(self):
        model = _build_debug_model()
        positions_BL = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)

        mask_B1LL = model.get_attention_masks(positions_BL)

        min_value = torch.finfo(mask_B1LL.dtype).min
        self.assertEqual(mask_B1LL.shape, (1, 1, 5, 5))
        self.assertEqual(mask_B1LL[0, 0, 4, 3].item(), 0.0)
        self.assertEqual(mask_B1LL[0, 0, 4, 1].item(), min_value)
        self.assertEqual(mask_B1LL[0, 0, 1, 2].item(), min_value)

    def test_debug_model_forward_shape(self):
        model = _build_debug_model()
        tokens_BL = torch.randint(0, 2048, (2, 12))
        positions_BL = torch.arange(12).expand(2, -1)

        logits_BLV = model(tokens_BL, positions=positions_BL)

        self.assertEqual(logits_BLV.shape, (2, 12, 2048))


class TestGlm5Registration(unittest.TestCase):
    def test_model_registry_has_single_device_glm5_hooks(self):
        spec = glm5.model_registry("debugmodel")

        self.assertEqual(spec.name, "glm5")
        self.assertEqual(spec.flavor, "debugmodel")
        self.assertIsInstance(spec.model, Glm5Model.Config)
        self.assertIs(spec.state_dict_adapter, Glm5StateDictAdapter)
        self.assertIsNone(spec.pipelining_fn)
        self.assertIs(spec.post_optimizer_build_fn, register_moe_load_balancing_hook)

    def test_debug_training_config_uses_single_device_defaults(self):
        config = glm5_debugmodel()

        self.assertEqual(config.training.local_batch_size, 2)
        self.assertEqual(config.training.seq_len, 128)
        self.assertEqual(config.training.steps, 10)
        self.assertEqual(config.metrics.log_freq, 1)
        self.assertEqual(config.checkpoint.interval, 10)
        self.assertEqual(config.dataloader.dataset, "c4_test")
        self.assertEqual(config.hf_assets_path, "./tests/assets/tokenizer")
        self.assertEqual(config.optimizer.param_groups[0].optimizer_kwargs["lr"], 8e-4)
        self.assertFalse(config.compile.enable)
        self.assertIsNone(config.activation_checkpoint)
        self.assertEqual(config.parallelism, ParallelismConfig())

    def test_parallelism_allows_only_one_unresolved_single_device_layout(self):
        validate = glm5.validate_glm5_parallelism

        self.assertIsNone(validate(ParallelismConfig()))
        single_rank_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        self.assertIsNone(validate(ParallelismConfig(), single_rank_dims))

    def test_parallelism_rejects_each_multi_rank_mode(self):
        validate = glm5.validate_glm5_parallelism
        invalid_configs = {
            "TP": ParallelismConfig(tensor_parallel_degree=2),
            "CP": ParallelismConfig(context_parallel_degree=2),
            "PP": ParallelismConfig(pipeline_parallel_degree=2),
            "EP": ParallelismConfig(expert_parallel_degree=2),
            "DP replicate": ParallelismConfig(data_parallel_replicate_degree=2),
            "DP shard": ParallelismConfig(data_parallel_shard_degree=2),
            "SPMD backend": ParallelismConfig(spmd_backend="full_dtensor"),
        }

        for mode, parallelism in invalid_configs.items():
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(NotImplementedError, mode):
                    validate(parallelism)

    def test_parallelism_reports_every_offending_mode_together(self):
        validate = glm5.validate_glm5_parallelism

        with self.assertRaisesRegex(NotImplementedError, "TP.*CP.*DP shard"):
            validate(
                ParallelismConfig(
                    tensor_parallel_degree=2,
                    context_parallel_degree=2,
                    data_parallel_shard_degree=2,
                )
            )

    def test_parallelism_rejects_resolved_multi_rank_world(self):
        validate = glm5.validate_glm5_parallelism
        multi_rank_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
        )

        with self.assertRaisesRegex(NotImplementedError, "DP shard.*world_size"):
            validate(ParallelismConfig(), multi_rank_dims)


class TestGlm5StateDictAdapter(unittest.TestCase):
    def _adapter(self) -> Glm5StateDictAdapter:
        return Glm5StateDictAdapter(glm5_configs["debugmodel"](), hf_assets_path=None)

    def test_fused_expert_gate_up_order(self):
        adapter = self._adapter()
        E, F, D = 8, 256, 256
        gate_EFD = torch.arange(E * F * D).reshape(E, F, D)
        up_EFD = gate_EFD + gate_EFD.numel()
        down_EDF = torch.arange(E * D * F).reshape(E, D, F)
        hf_state = {
            "model.layers.1.mlp.experts.gate_up_proj": torch.cat(
                (gate_EFD, up_EFD), dim=1
            ),
            "model.layers.1.mlp.experts.down_proj": down_EDF,
        }

        titan_state = adapter.from_hf(hf_state)

        self.assertTrue(
            torch.equal(
                titan_state["layers.1.moe.routed_experts.inner_experts.w1_EFD"],
                gate_EFD,
            )
        )
        self.assertTrue(
            torch.equal(
                titan_state["layers.1.moe.routed_experts.inner_experts.w3_EFD"],
                up_EFD,
            )
        )
        self.assertTrue(
            torch.equal(
                titan_state["layers.1.moe.routed_experts.inner_experts.w2_EDF"],
                down_EDF,
            )
        )

    def test_direct_indexer_and_fp32_expert_bias_mappings(self):
        adapter = self._adapter()
        indexer_wq_b = torch.randn(256, 128)
        indexer_wk = torch.randn(64, 256)
        indexer_norm_weight = torch.randn(64)
        indexer_norm_bias = torch.randn(64)
        indexer_weights_proj = torch.randn(4, 256)
        expert_bias = torch.randn(8, dtype=torch.float32)

        titan_state = adapter.from_hf(
            {
                "model.layers.1.self_attn.indexer.wq_b.weight": indexer_wq_b,
                "model.layers.1.self_attn.indexer.wk.weight": indexer_wk,
                "model.layers.1.self_attn.indexer.k_norm.weight": indexer_norm_weight,
                "model.layers.1.self_attn.indexer.k_norm.bias": indexer_norm_bias,
                "model.layers.1.self_attn.indexer.weights_proj.weight": indexer_weights_proj,
                "model.layers.1.mlp.gate.e_score_correction_bias": expert_bias,
            }
        )

        expected_indexer = {
            "layers.1.attention.indexer.wq_b.weight": indexer_wq_b,
            "layers.1.attention.indexer.wk.weight": indexer_wk,
            "layers.1.attention.indexer.k_norm.weight": indexer_norm_weight,
            "layers.1.attention.indexer.k_norm.bias": indexer_norm_bias,
            "layers.1.attention.indexer.weights_proj.weight": indexer_weights_proj,
        }
        for titan_key, hf_value in expected_indexer.items():
            self.assertIs(titan_state[titan_key], hf_value)
        self.assertEqual(titan_state["layers.1.moe.expert_bias_E"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(titan_state["layers.1.moe.expert_bias_E"], expert_bias)
        )

    def test_full_state_dict_roundtrip(self):
        config = glm5_configs["debugmodel"]()
        model = config.build()
        model.init_states()
        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        original = model.state_dict()

        restored = adapter.from_hf(adapter.to_hf(original))

        self.assertEqual(set(restored), set(original))
        for key in original:
            self.assertTrue(torch.equal(restored[key], original[key]), key)

    def test_unknown_hf_key_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unmapped HF key"):
            self._adapter().from_hf({"model.layers.0.unexpected.weight": torch.ones(1)})

    def test_only_next_mtp_layer_namespace_is_warned_and_skipped(self):
        with self.assertWarnsRegex(UserWarning, "MTP"):
            titan_state = self._adapter().from_hf(
                {"model.layers.4.some_mtp_weight": torch.ones(1)}
            )
        self.assertEqual(titan_state, {})

    def test_later_layer_namespace_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unmapped HF key"):
            self._adapter().from_hf({"model.layers.5.some_mtp_weight": torch.ones(1)})

    def test_to_hf_rejects_incomplete_fused_expert_mapping(self):
        with self.assertRaisesRegex(KeyError, "incomplete"):
            self._adapter().to_hf(
                {
                    "layers.1.moe.routed_experts.inner_experts.w1_EFD": torch.ones(
                        8, 256, 256
                    )
                }
            )
