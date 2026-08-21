# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: 测试torchtitan glm的实现是否完全符合数学定义，以及模型配置、非法行为、精度规范、边界行为、错误处理等
# NOTE: 保证所有行为合理，遵循自我规范

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
import torchtitan.models.glm5 as glm5

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _split_module,
    pipeline_llm,
)
from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear, RMSNorm
from torchtitan.models.glm5 import build_glm5_layers, glm5_configs, Glm5StateDictAdapter
from torchtitan.models.glm5.config_registry import glm5_debugmodel
from torchtitan.models.glm5.model import (
    DSAIndexerTopK,
    DSAInnerAttention,
    Glm5Attention,
    Glm5DsaIndexer,
    Glm5Model,
    Glm5TransformerBlock,
)
from torchtitan.models.glm5.parallelize import apply_glm5_cp_to_forward


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
        rope=ComplexRoPE.Config(dim=4, max_context_length=8, theta=1_000_000),
        topk=DSAIndexerTopK.Config(index_topk=3, softmax_scale=8**-0.5),
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
        wq_a=Linear.Config(in_features=16, out_features=8),
        q_norm=RMSNorm.Config(normalized_shape=8),
        wq_b=Linear.Config(in_features=8, out_features=16),
        wkv_a=Linear.Config(in_features=16, out_features=8),
        kv_norm=RMSNorm.Config(normalized_shape=4),
        wkv_b=Linear.Config(in_features=4, out_features=16),
        wo=Linear.Config(in_features=8, out_features=16),
        rope=ComplexRoPE.Config(dim=4, max_context_length=8, theta=1_000_000),
        indexer=_indexer_config(),
        inner_attention=DSAInnerAttention.Config(attention_dropout=0.0),
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
        "attention_dropout": attention.inner_attention.attention_dropout,
        "rope": attention.rope,
    }


def _dense_causal_mask(
    positions_T: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    T = positions_T.shape[0]
    key_positions_K = torch.arange(T, device=positions_T.device)
    return torch.zeros(1, T, T, dtype=dtype, device=positions_T.device).masked_fill(
        key_positions_K > positions_T.unsqueeze(-1).unsqueeze(0),
        float("-inf"),
    )


def _reference_sparse_attention(
    attention: Glm5Attention,
    x_BLD: torch.Tensor,
    attention_masks_B1LL: torch.Tensor,
    positions_BL: torch.Tensor,
    topk_indices_BLK: torch.Tensor,
) -> torch.Tensor:
    input_was_token_first = x_BLD.ndim == 2
    if input_was_token_first:
        x_BLD = x_BLD.unsqueeze(0)
        attention_masks_B1LL = attention_masks_B1LL.unsqueeze(0)
        positions_BL = positions_BL.unsqueeze(0)
        topk_indices_BLK = topk_indices_BLK.unsqueeze(0)
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
    output = F.linear(
        output_BLNV.contiguous().view(B, L, -1), attention.wo.weight, attention.wo.bias
    )
    return output.squeeze(0) if input_was_token_first else output


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
    input_was_token_first = hidden_states_BLD.ndim == 2
    if input_was_token_first:
        hidden_states_BLD = hidden_states_BLD.unsqueeze(0)
        q_resid_BLR = q_resid_BLR.unsqueeze(0)
        positions_BL = positions_BL.unsqueeze(0)
        if attention_mask_BLL is not None:
            attention_mask_BLL = attention_mask_BLL.unsqueeze(0)
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
    output = index_scores_BLL.topk(topk, dim=-1).indices.to(torch.int32)
    return output.squeeze(0) if input_was_token_first else output


# DSA Indexer模块测试
class TestGlm5DsaIndexer(unittest.TestCase):
    def test_topk_supports_local_queries_against_global_keys(self):
        topk = DSAIndexerTopK.Config(index_topk=2, softmax_scale=0.5).build()
        q_BQNH = torch.randn(2, 3, 4)
        k_BKH = torch.randn(5, 4)
        weights_BQN = torch.randn(2, 3)
        attention_mask_BQK = torch.zeros(2, 5)
        attention_mask_BQK[0, 4] = float("-inf")

        actual_BQK = topk(
            q_BQNH,
            k_BKH,
            weights_BQN,
            attention_mask_BQK,
        )
        scores_BNQK = F.relu(
            torch.matmul(q_BQNH.float().transpose(0, 1), k_BKH.float().T) * 0.5
        )
        expected_scores_BQK = (
            torch.matmul(
                weights_BQN.unsqueeze(-2),
                scores_BNQK.transpose(0, 1),
            ).squeeze(-2)
            + attention_mask_BQK
        )
        expected_BQK = expected_scores_BQK.topk(2, dim=-1).indices.to(torch.int32)

        self.assertEqual(actual_BQK.shape, (2, 2))
        self.assertTrue(torch.equal(actual_BQK, expected_BQK))

    # 测试torchtitan indexer自身的基本契约
    def test_indexer_returns_masked_int32_topk(self):
        indexer = _indexer_config().build()
        indexer.init_states()
        # 构造输入数据, 完全随机, 只测试indexer的行为
        hidden_states_BLD = torch.randn(10, 16)
        q_resid_BLR = torch.randn(10, 8)
        positions_BL = torch.arange(5).repeat(2)
        attention_mask_BLL = torch.full((10, 10), float("-inf"))
        attention_mask_BLL.masked_fill_(
            torch.block_diag(*[torch.ones(5, 5, dtype=torch.bool).tril()] * 2), 0.0
        )

        topk_indices_BLK = indexer(
            hidden_states_BLD,
            q_resid_BLR,
            positions_BL,
            attention_mask_BLL,
        )

        # 输出 dtype 是 torch.int32
        self.assertEqual(topk_indices_BLK.dtype, torch.int32)
        # 输出 shape 是 [B, S, topk]
        self.assertEqual(topk_indices_BLK.shape, (10, 3))
        # top-k 不超过配置的 index_topk
        full_topk_queries_B = positions_BL >= topk_indices_BLK.shape[-1] - 1
        self.assertTrue(
            torch.all(
                topk_indices_BLK[full_topk_queries_B]
                <= positions_BL[full_topk_queries_B].unsqueeze(-1)
            )
        )
        # causal 条件下不能选择未来 token
        self.assertTrue(
            torch.all(
                torch.any(
                    topk_indices_BLK[positions_BL == 0] > 0,
                    dim=-1,
                )
            )
        )

    # 测试torchtitan indexer的行为与独立实现的reference indexer一致
    def test_indexer_matches_independent_reference(self):
        torch.manual_seed(17)
        indexer = _indexer_config().build()
        indexer.init_states()
        # 构造输入数据, 完全随机, 只测试indexer的行为
        hidden_states_BLD = torch.randn(4, 16)
        q_resid_BLR = torch.randn(4, 8)
        positions_BL = torch.arange(4)
        attention_mask_BLL = torch.zeros(4, 4).masked_fill(
            ~torch.ones(4, 4, dtype=torch.bool).tril(),
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
        # 检查 TorchTitan indexer 的代码实现是否与数学 reference 一致
        self.assertTrue(torch.equal(actual_BLK, expected_BLK))

    # 测试torchtitan indexer 符合 glm5 indexer的特殊设计
    # 1. indexer 是离散选择模块
    # 2. indexer 参数不参与 LM loss
    # 3. indexer head weight 保持 FP32
    def test_indexer_is_no_grad_and_keeps_weights_projection_fp32(self):
        indexer = _indexer_config().build()
        indexer.init_states()
        self.assertTrue(all(not param.requires_grad for param in indexer.parameters()))
        self.assertIn("wq_b.weight", indexer.state_dict())
        indexer.bfloat16()
        # 测试 BF16 转换后 weights_proj.weight 仍是 FP32
        self.assertEqual(indexer.weights_proj.weight.dtype, torch.float32)
        out_BLK = indexer(
            torch.randn(4, 16, dtype=torch.bfloat16),
            torch.randn(4, 8, dtype=torch.bfloat16),
            torch.arange(4),
            torch.zeros(4, 4, dtype=torch.bfloat16),
        )
        # 测试 indexer 输出不需要梯度
        self.assertFalse(out_BLK.requires_grad)


# glm5 MLA+DSA Attention模块测试
class TestGlm5Attention(unittest.TestCase):
    def test_dsa_inner_attention_rejects_invalid_dropout(self):
        with self.assertRaisesRegex(ValueError, "attention_dropout"):
            DSAInnerAttention.Config(attention_dropout=1.0)

    def test_dsa_inner_attention_supports_local_queries_and_global_kv(self):
        inner_attention = DSAInnerAttention.Config(attention_dropout=0.0).build()
        q_BQNH = torch.randn(2, 2, 4)
        k_BKNH = torch.randn(5, 2, 4)
        v_BKNV = torch.randn(5, 2, 3)
        attention_mask_B1QK = torch.zeros(1, 2, 5)
        topk_indices_BQT = torch.tensor([[0, 3], [1, 4]], dtype=torch.int32)

        actual_BQNV = inner_attention(
            q_BQNH,
            k_BKNH,
            v_BKNV,
            attention_mask_B1QK,
            topk_indices_BQT,
            scale=0.5,
        )
        selected_BQK = torch.zeros(2, 5, dtype=torch.bool).scatter(
            -1,
            topk_indices_BQT.long(),
            True,
        )
        sparse_mask_B1QK = attention_mask_B1QK.masked_fill(
            ~selected_BQK.unsqueeze(0),
            torch.finfo(q_BQNH.dtype).min,
        )
        scores_BNQK = (
            torch.matmul(
                q_BQNH.transpose(0, 1),
                k_BKNH.transpose(0, 1).transpose(-1, -2),
            )
            * 0.5
            + sparse_mask_B1QK
        )
        probs_BNQK = F.softmax(scores_BNQK, dim=-1, dtype=torch.float32)
        expected_BQNV = torch.matmul(
            probs_BNQK,
            v_BKNV.transpose(0, 1),
        ).transpose(0, 1)

        self.assertEqual(actual_BQNV.shape, (2, 2, 3))
        torch.testing.assert_close(actual_BQNV, expected_BQNV)

    def test_attention_allows_independent_index_head_dimension(self):
        config = _attention_config()
        index_head_dim = 12
        indexer = dataclasses.replace(
            config.indexer,
            head_dim=index_head_dim,
            wq_b=Linear.Config(
                in_features=config.q_lora_rank,
                out_features=config.indexer.n_heads * index_head_dim,
            ),
            wk=Linear.Config(
                in_features=config.dim,
                out_features=index_head_dim,
            ),
            k_norm=LayerNorm.Config(normalized_shape=index_head_dim),
            topk=dataclasses.replace(
                config.indexer.topk,
                softmax_scale=index_head_dim**-0.5,
            ),
        )

        updated = dataclasses.replace(config, indexer=indexer)

        self.assertEqual(updated.qk_head_dim, 8)
        self.assertEqual(updated.indexer.head_dim, index_head_dim)

    def test_attention_topk_cannot_reopen_causal_mask(self):
        attention = _attention_config().build()
        attention.init_states()
        x_BLD = torch.randn(4, 16)
        positions_BL = torch.arange(4)
        base_mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
        future_selecting_topk_BLK = torch.tensor(
            [[3, 2], [3, 2], [3, 2], [3, 2]], dtype=torch.int32
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
        x_BLD = torch.randn(10, 16, requires_grad=True)
        positions_BL = torch.arange(5).repeat(2)
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
        x_BLD = torch.randn(8, 16)
        positions_BL = torch.arange(4).repeat(2)
        mask_B1LL = _dense_causal_mask(positions_BL, dtype=x_BLD.dtype)
        invalid_masks = {
            "rank": mask_B1LL[0],
            "channels": torch.zeros(2, 8, 8),
            "length": torch.zeros(1, 8, 7),
            "dtype": torch.zeros(1, 8, 8, dtype=torch.int32),
            "device": torch.zeros(1, 8, 8, device="meta"),
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
            topk=dataclasses.replace(
                config.indexer.topk,
                softmax_scale=4**-0.5,
            ),
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
        self.assertEqual(len(config.layers), 8)
        self.assertEqual(config.max_context_length, 128)
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
            self.assertIsInstance(attention.inner_attention, DSAInnerAttention.Config)
            self.assertEqual(attention.inner_attention.attention_dropout, 0.0)
            self.assertEqual(attention.rope.max_context_length, 128)
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
        positions_BL = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

        mask_B1LL = model.get_attention_masks(positions_BL)

        min_value = torch.finfo(mask_B1LL.dtype).min
        self.assertEqual(mask_B1LL.shape, (1, 5, 5))
        self.assertEqual(mask_B1LL[0, 4, 3].item(), 0.0)
        self.assertEqual(mask_B1LL[0, 4, 1].item(), min_value)
        self.assertEqual(mask_B1LL[0, 1, 2].item(), min_value)

    def test_dense_mask_accepts_decoder_positions_keyword(self):
        model = _build_debug_model()
        positions_BL = torch.arange(4)

        mask_B1LL = model.get_attention_masks(positions=positions_BL)

        self.assertEqual(mask_B1LL.shape, (1, 4, 4))

    def test_non_first_pp_stage_builds_dense_mask_without_embeddings(self):
        # Every PP stage containing attention builds its own mask. A non-first
        # stage is pruned by _split_module (tok_embeddings=None, layers keys
        # keep original indices), so get_attention_masks must not touch
        # tok_embeddings.
        config = glm5_configs["debugmodel"]()
        model = config.build()
        model.init_states()
        fqn_per_stage = _generate_llm_fqn_per_model_part(2, len(config.layers), 1, 1)
        stage = _split_module(model, fqn_per_stage[1])
        self.assertIsNone(stage.tok_embeddings)

        mask_B1LL = stage.get_attention_masks(torch.arange(5))

        self.assertEqual(mask_B1LL.shape, (1, 5, 5))
        self.assertTrue(torch.isfinite(mask_B1LL).all())

    def test_output_only_pp_stage_does_not_build_attention_mask(self):
        config = glm5_configs["debugmodel"]()
        model = config.build()
        model.init_states()
        fqn_per_stage = _generate_llm_fqn_per_model_part(8, len(config.layers), 1, 1)
        self.assertEqual(fqn_per_stage[-1], ["norm", "lm_head"])

        stage = _split_module(model, fqn_per_stage[-1])
        self.assertEqual(len(stage.layers), 0)
        self.assertIsNone(stage.tok_embeddings)
        self.assertIsNotNone(stage.norm)
        self.assertIsNotNone(stage.lm_head)

        positions_BL = torch.arange(5)
        self.assertIsNone(stage.get_attention_masks(positions_BL))

        hidden_BLD = torch.randn(5, config.dim)
        logits_BLV = stage(hidden_BLD, positions=positions_BL)
        self.assertEqual(logits_BLV.shape, (5, config.vocab_size))

    def test_debug_model_forward_shape(self):
        model = _build_debug_model()
        tokens_BL = torch.randint(0, 2048, (24,))
        positions_BL = torch.arange(12).repeat(2)

        logits_BLV = model(tokens_BL, positions=positions_BL)

        self.assertEqual(logits_BLV.shape, (24, 2048))

    def test_debug_model_cpu_forward_loss_backward(self):
        torch.manual_seed(29)
        config = glm5_configs["debugmodel"]()
        model = config.build()
        model.init_states()
        model.train()
        tokens_BL = torch.randint(0, config.vocab_size, (32,))
        positions_BL = torch.arange(16).repeat(2)
        labels_BL = torch.randint(0, config.vocab_size, (32,))
        logits_BLV = model(tokens_BL, positions=positions_BL)
        loss = F.cross_entropy(logits_BLV.float(), labels_BL)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.tok_embeddings.weight.grad)
        self.assertIsNotNone(model.layers["0"].feed_forward.w1.weight.grad)
        self.assertIsNotNone(
            model.layers["1"].moe.routed_experts.inner_experts.w1_EFD.grad
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for layer in model.layers.values()
                for parameter in layer.attention.indexer.parameters()
            )
        )

    def test_debug_model_reports_exact_positive_nparams_and_flops(self):
        config = glm5_configs["debugmodel"]()
        model = config.build()
        model.init_states()

        nparams, flops = config.get_nparams_and_flops(model, seq_len=16)

        self.assertIsInstance(nparams, int)
        self.assertIsInstance(flops, int)
        self.assertGreater(nparams, 0)
        self.assertGreater(flops, 0)
        self.assertEqual(
            nparams, sum(parameter.numel() for parameter in model.parameters())
        )


class TestGlm5Registration(unittest.TestCase):
    def test_model_config_defaults_match_official_glm5(self):
        config_fields = {
            config_field.name: config_field
            for config_field in dataclasses.fields(Glm5Model.Config)
        }
        self.assertEqual(config_fields["dim"].default, 6144)
        self.assertEqual(config_fields["vocab_size"].default, 154880)

    def test_model_registry_has_single_device_glm5_hooks(self):
        spec = glm5.model_registry("debugmodel")

        self.assertEqual(spec.name, "glm5")
        self.assertEqual(spec.flavor, "debugmodel")
        self.assertIsInstance(spec.model, Glm5Model.Config)
        self.assertIs(spec.state_dict_adapter, Glm5StateDictAdapter)
        self.assertIs(spec.pipelining_fn, pipeline_llm)
        self.assertIs(spec.post_optimizer_build_fn, register_moe_load_balancing_hook)

    def test_debug_training_config_uses_single_device_defaults(self):
        config = glm5_debugmodel()

        self.assertEqual(config.training.num_tokens_per_microbatch_per_dp_rank, 256)
        self.assertEqual(config.training.max_context_length, 128)
        self.assertEqual(config.training.steps, 10)
        self.assertEqual(config.metrics.log_freq, 1)
        self.assertEqual(config.checkpoint.interval, 10)
        self.assertFalse(config.dataloader.shuffle)
        self.assertEqual(config.hf_assets_path, "./tests/assets/tokenizer")
        self.assertEqual(config.optimizer.param_groups[0].optimizer_kwargs["lr"], 8e-4)
        self.assertFalse(config.compile.enable)
        self.assertIsNone(config.activation_checkpoint)
        self.assertEqual(
            config.parallelism,
            ParallelismConfig(
                enable_sequence_parallel=True,
                context_parallel_load_balancer=None,
                pipeline_parallel_last_stage_less_layers=0,
                spmd_backend="partial_dtensor",
            ),
        )

    def test_parallelism_allows_only_one_unresolved_single_device_layout(self):
        validate = glm5.validate_glm5_parallelism

        self.assertIsNone(
            validate(ParallelismConfig(spmd_backend="partial_dtensor"))
        )
        single_rank_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        self.assertIsNone(
            validate(
                ParallelismConfig(spmd_backend="partial_dtensor"),
                single_rank_dims,
            )
        )

    def test_parallelism_rejects_each_unsupported_layout(self):
        validate = glm5.validate_glm5_parallelism
        invalid_configs = {
            "CP load balancing": ParallelismConfig(context_parallel_degree=2),
            "SPMD backend": ParallelismConfig(spmd_backend="spmd_types"),
        }

        for mode, parallelism in invalid_configs.items():
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(NotImplementedError, mode):
                    validate(parallelism)

    def test_parallelism_allows_cp_with_contiguous_sequence_shards(self):
        self.assertIsNone(
            glm5.validate_glm5_parallelism(
                ParallelismConfig(
                    context_parallel_degree=2,
                    context_parallel_load_balancer=None,
                    spmd_backend="partial_dtensor",
                )
            )
        )

        resolved_cp_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
        )
        with self.assertRaisesRegex(NotImplementedError, "CP load balancing"):
            glm5.validate_glm5_parallelism(
                ParallelismConfig(spmd_backend="partial_dtensor"),
                resolved_cp_dims,
            )

    def test_parallelism_allows_tp_pp_ep(self):
        validate = glm5.validate_glm5_parallelism

        # TP/PP/EP are supported independently and combined with SP.
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    tensor_parallel_degree=2,
                    enable_sequence_parallel=True,
                    spmd_backend="partial_dtensor",
                )
            )
        )
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    pipeline_parallel_degree=2,
                    spmd_backend="partial_dtensor",
                )
            )
        )
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    expert_parallel_degree=2,
                    spmd_backend="partial_dtensor",
                )
            )
        )
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    tensor_parallel_degree=2,
                    pipeline_parallel_degree=2,
                    expert_parallel_degree=2,
                    enable_sequence_parallel=True,
                    spmd_backend="partial_dtensor",
                )
            )
        )

    def test_parallelism_allows_dp_configs(self):
        validate = glm5.validate_glm5_parallelism

        self.assertIsNone(
            validate(
                ParallelismConfig(
                    data_parallel_replicate_degree=8,
                    data_parallel_shard_degree=1,
                    spmd_backend="partial_dtensor",
                )
            )
        )
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    data_parallel_replicate_degree=1,
                    data_parallel_shard_degree=8,
                    spmd_backend="partial_dtensor",
                )
            )
        )

    def test_parallelism_reports_every_offending_mode_together(self):
        validate = glm5.validate_glm5_parallelism

        with self.assertRaisesRegex(NotImplementedError, "CP.*SPMD backend"):
            validate(
                ParallelismConfig(
                    tensor_parallel_degree=2,
                    context_parallel_degree=2,
                    enable_sequence_parallel=True,
                    spmd_backend="spmd_types",
                )
            )

    def test_parallelism_allows_resolved_multi_rank_dp_world(self):
        validate = glm5.validate_glm5_parallelism

        # DDP: replicate=8, shard=1 on 8 ranks.
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    data_parallel_replicate_degree=8,
                    data_parallel_shard_degree=1,
                    spmd_backend="partial_dtensor",
                ),
                ParallelDims(
                    dp_replicate=8,
                    dp_shard=1,
                    cp=1,
                    tp=1,
                    pp=1,
                    ep=1,
                    world_size=8,
                ),
            )
        )

        # FSDP: shard=8, replicate=1 on 8 ranks.
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    data_parallel_replicate_degree=1,
                    data_parallel_shard_degree=8,
                    spmd_backend="partial_dtensor",
                ),
                ParallelDims(
                    dp_replicate=1,
                    dp_shard=8,
                    cp=1,
                    tp=1,
                    pp=1,
                    ep=1,
                    world_size=8,
                ),
            )
        )

        # A resolved sequence-parallel layout is supported.
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    tensor_parallel_degree=2,
                    spmd_backend="partial_dtensor",
                ),
                ParallelDims(
                    dp_replicate=1,
                    dp_shard=4,
                    cp=1,
                    tp=2,
                    pp=1,
                    ep=1,
                    world_size=8,
                ),
            )
        )

        # A resolved TP+PP+EP layout (SP off) is allowed. EP borrows ranks
        # from dp_shard x cp x tp, so it does not multiply world_size: here
        # dp_shard(2) * tp(2) * pp(2) = world_size(8), and the routed experts
        # span efsdp(2) x ep(2) = 4 ranks.
        self.assertIsNone(
            validate(
                ParallelismConfig(
                    tensor_parallel_degree=2,
                    pipeline_parallel_degree=2,
                    expert_parallel_degree=2,
                    enable_sequence_parallel=True,
                    spmd_backend="partial_dtensor",
                ),
                ParallelDims(
                    dp_replicate=1,
                    dp_shard=2,
                    cp=1,
                    tp=2,
                    pp=2,
                    ep=2,
                    world_size=8,
                ),
            )
        )

    def test_sharding_config_populated_for_tp_ep(self):
        import spmd_types as spmd
        from torchtitan.models.common.decoder_sharding import (
            dense_param_placement,
            dense_sequence_parallel_placement,
        )
        from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
        from torchtitan.models.glm5.sharding import _GROUPED_EXPERTS_PARAM_LAYOUT

        trainer_config = glm5_debugmodel()
        trainer_config.parallelism = ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
            enable_sequence_parallel=True,
            spmd_backend="partial_dtensor",
        )
        model_config = trainer_config.model_spec.model
        self.assertIsInstance(model_config, Glm5Model.Config)
        model_config.update_from_config(config=trainer_config)

        # Root-level decoder configs (tok_embeddings / norm / lm_head).
        self.assertIsNotNone(model_config.tok_embeddings.sharding_config)
        self.assertIsNotNone(model_config.norm.sharding_config)
        self.assertIsNotNone(model_config.lm_head.sharding_config)

        dense_layer = model_config.layers[0]
        moe_layer = model_config.layers[1]

        # Attention MLA: input boundary, RoPE cache, low-rank projections and
        # norms all populated.
        attention = dense_layer.attention
        self.assertIsNotNone(attention.sharding_config)
        sequence_layout = dense_sequence_parallel_placement()
        self.assertEqual(
            attention.sharding_config.in_src_shardings["x_TD"], sequence_layout
        )
        self.assertEqual(
            attention.wo.sharding_config.out_dst_shardings,
            sequence_layout,
        )
        self.assertIsNotNone(attention.rope.sharding_config)
        self.assertIsNotNone(attention.wq_a.sharding_config)
        self.assertIsNotNone(attention.q_norm.sharding_config)
        self.assertIsNotNone(attention.wkv_a.sharding_config)
        self.assertIsNotNone(attention.kv_norm.sharding_config)
        self.assertIsNotNone(attention.wq_b.sharding_config)
        self.assertIsNotNone(attention.wkv_b.sharding_config)
        self.assertIsNotNone(attention.wo.sharding_config)
        self.assertIsNotNone(attention.inner_attention.sharding_config)
        self.assertIsNotNone(attention.inner_attention.sharding_config.local_map)
        inner_grad_layouts = (
            attention.inner_attention.sharding_config.local_map.in_grad_placements
        )
        self.assertIsNotNone(inner_grad_layouts)
        self.assertEqual(len(inner_grad_layouts), 5)
        self.assertIsNotNone(inner_grad_layouts[3])
        self.assertIsNotNone(inner_grad_layouts[4])

        # DSA indexer: every projection is Replicate on TP (correctness-first:
        # a head-shard would leave topk on a partial score tensor).
        indexer = attention.indexer
        self.assertIsNotNone(indexer.sharding_config)
        self.assertIsNotNone(indexer.wq_b.sharding_config)
        self.assertIsNotNone(indexer.wk.sharding_config)
        self.assertIsNotNone(indexer.k_norm.sharding_config)
        self.assertIsNotNone(indexer.weights_proj.sharding_config)
        self.assertIsNotNone(indexer.rope.sharding_config)
        self.assertIsNotNone(indexer.topk.sharding_config)
        self.assertIsNotNone(indexer.topk.sharding_config.local_map)
        replicate_placement = dense_param_placement(tp=spmd.R)
        for proj in (indexer.wq_b, indexer.wk, indexer.weights_proj):
            self.assertEqual(
                proj.sharding_config.state_shardings["weight"],
                replicate_placement,
            )

        # Norms + dense FFN on the dense layer.
        self.assertIsNotNone(dense_layer.attention_norm.sharding_config)
        self.assertIsNotNone(dense_layer.ffn_norm.sharding_config)
        self.assertIsNotNone(dense_layer.feed_forward.sharding_config)
        self.assertIsNone(dense_layer.moe)

        # MoE layer: wrapper + router + shared experts + routed experts filled.
        # With EP on, expert weights use the sparse layout keyed by the
        # grouped-experts param layout names.
        moe = moe_layer.moe
        self.assertIsNotNone(moe.sharding_config)
        self.assertTrue(moe.seq_dim_tp_sharded)
        self.assertIsNotNone(moe.router.gate.sharding_config)
        self.assertIsNotNone(moe.shared_experts.sharding_config)
        self.assertIsNotNone(moe.routed_experts.inner_experts.sharding_config)
        expert_state_shardings = (
            moe.routed_experts.inner_experts.sharding_config.state_shardings
        )
        self.assertEqual(
            set(expert_state_shardings), set(_GROUPED_EXPERTS_PARAM_LAYOUT)
        )
        for name in _GROUPED_EXPERTS_PARAM_LAYOUT:
            self.assertEqual(
                expert_state_shardings[name], expert_param_placement_sparse()
            )
        self.assertIsNone(moe_layer.feed_forward)


class TestGlm5ContextParallel(unittest.TestCase):
    def test_cp_wrapper_skips_collectives_for_output_only_pp_stage(self):
        observed = {}

        def model_forward(tokens, positions, attention_masks):
            observed["positions"] = positions
            observed["attention_masks"] = attention_masks
            return tokens

        get_attention_masks = mock.Mock()
        model = SimpleNamespace(
            layers={},
            forward=model_forward,
            get_attention_masks=get_attention_masks,
        )
        cp_mesh = mock.Mock()
        cp_mesh.get_group.return_value = object()
        tokens_BLD = torch.randn(4, 8)
        positions_BL = torch.arange(4)

        with (
            mock.patch(
                "torchtitan.models.glm5.parallelize.dist._get_process_group_name",
                return_value="cp_group",
            ),
            mock.patch(
                "torchtitan.models.glm5.parallelize.dist.all_gather"
            ) as all_gather,
        ):
            apply_glm5_cp_to_forward(model, cp_mesh)
            output = model.forward(tokens_BLD, positions_BL)

        self.assertIs(output, tokens_BLD)
        self.assertIs(observed["positions"], positions_BL)
        self.assertIsNone(observed["attention_masks"])
        get_attention_masks.assert_not_called()
        all_gather.assert_not_called()

    def test_cp_wrapper_builds_mask_and_gathers_indexer_keys_and_attention_kv(self):
        attention = _attention_config().build()
        observed = {}

        def model_forward(tokens, positions, attention_masks):
            observed["model_positions"] = positions
            observed["attention_masks"] = attention_masks
            return tokens

        def get_attention_masks(global_positions):
            observed["global_positions"] = global_positions
            return torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

        model = SimpleNamespace(
            layers={"0": SimpleNamespace(attention=attention)},
            forward=model_forward,
            get_attention_masks=get_attention_masks,
        )
        cp_mesh = mock.Mock()
        cp_mesh.size.return_value = 2
        cp_mesh.get_local_rank.return_value = 1
        process_group = object()
        cp_mesh.get_group.return_value = process_group

        def topk_forward(q, k, weights, attention_mask):
            observed["indexer_k"] = k
            return torch.zeros(q.shape[:1] + (2,), dtype=torch.int32)

        def inner_forward(q, k, v, attention_mask, topk_indices, *, scale):
            observed["attention_k"] = k
            observed["attention_v"] = v
            return torch.zeros(q.shape[:-1] + (v.shape[-1],))

        attention.indexer.topk.forward = topk_forward
        attention.inner_attention.forward = inner_forward

        def fake_all_gather(outputs, local_tensor, *, group):
            self.assertIs(group, process_group)
            outputs[0].copy_(local_tensor)
            outputs[1].copy_(local_tensor + 10)

        def fake_flex_allgather(k, v, sequence_dim, process_group_name):
            self.assertEqual(sequence_dim, 0)
            self.assertEqual(process_group_name, "cp_group")
            return torch.cat((k, k + 10), dim=0), torch.cat((v, v + 20), dim=0)

        with (
            mock.patch(
                "torchtitan.models.glm5.parallelize.dist._get_process_group_name",
                return_value="cp_group",
            ),
            mock.patch(
                "torchtitan.models.glm5.parallelize.dist.all_gather",
                side_effect=fake_all_gather,
            ),
            mock.patch(
                "torchtitan.models.glm5.parallelize.flex_cp_allgather",
                side_effect=fake_flex_allgather,
            ),
        ):
            apply_glm5_cp_to_forward(model, cp_mesh)
            local_positions = torch.tensor([2, 3])
            local_tokens = torch.tensor([7, 8])
            model.forward(local_tokens, local_positions)

            q_BQNH = torch.randn(2, 2, 4)
            k_BKH = torch.randn(2, 4)
            weights_BQN = torch.randn(2, 2)
            mask_BQK = torch.zeros(2, 4)
            attention.indexer.topk(q_BQNH, k_BKH, weights_BQN, mask_BQK)

            k_BLNH = torch.randn(2, 2, 4)
            v_BLNH = torch.randn(2, 2, 3)
            attention.inner_attention(
                q_BQNH,
                k_BLNH,
                v_BLNH,
                torch.zeros(1, 2, 4),
                torch.zeros(2, 2, dtype=torch.int32),
                scale=0.5,
            )

        self.assertTrue(
            torch.equal(observed["global_positions"], torch.tensor([2, 3, 12, 13]))
        )
        self.assertIs(observed["model_positions"], local_positions)
        self.assertTrue(
            torch.equal(
                observed["attention_masks"],
                torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)[:, 2:4, :],
            )
        )
        self.assertEqual(observed["indexer_k"].shape[0], 4)
        self.assertTrue(torch.equal(observed["indexer_k"][2:], k_BKH + 10))
        self.assertEqual(observed["attention_k"].shape[0], 4)
        self.assertEqual(observed["attention_v"].shape[0], 4)


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
                {"model.layers.8.some_mtp_weight": torch.ones(1)}
            )
        self.assertEqual(titan_state, {})

    def test_later_layer_namespace_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unmapped HF key"):
            self._adapter().from_hf({"model.layers.9.some_mtp_weight": torch.ones(1)})

    def test_to_hf_rejects_incomplete_fused_expert_mapping(self):
        with self.assertRaisesRegex(KeyError, "incomplete"):
            self._adapter().to_hf(
                {
                    "layers.1.moe.routed_experts.inner_experts.w1_EFD": torch.ones(
                        8, 256, 256
                    )
                }
            )
