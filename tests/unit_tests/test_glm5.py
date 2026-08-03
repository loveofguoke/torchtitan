# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from torchtitan.models.common import ComplexRoPE, FlexAttention, LayerNorm, Linear
from torchtitan.models.glm5.model import Glm5Attention, Glm5DsaIndexer


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
        q_norm=LayerNorm.Config(normalized_shape=8),
        wq_b=Linear.Config(in_features=8, out_features=16),
        wkv_a=Linear.Config(in_features=16, out_features=8),
        kv_norm=LayerNorm.Config(normalized_shape=4),
        wkv_b=Linear.Config(in_features=4, out_features=16),
        wo=Linear.Config(in_features=8, out_features=16),
        rope=ComplexRoPE.Config(dim=4, max_seq_len=8, theta=1_000_000),
        indexer=_indexer_config(),
        inner_attention=FlexAttention.Config(),
    )


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
    q_resid_BLR = F.layer_norm(
        F.linear(x_BLD, attention.wq_a.weight, attention.wq_a.bias),
        attention.q_norm.normalized_shape,
        attention.q_norm.weight,
        attention.q_norm.bias,
        attention.q_norm.eps,
    )
    q_BLNH = F.linear(q_resid_BLR, attention.wq_b.weight, attention.wq_b.bias).view(
        B, L, attention.n_heads, attention.qk_head_dim
    )
    q_nope_BLNP, q_rope_BLNR = torch.split(
        q_BLNH,
        [attention.qk_nope_head_dim, attention.qk_rope_head_dim],
        dim=-1,
    )
    compressed_kv_BLC = F.linear(
        x_BLD, attention.wkv_a.weight, attention.wkv_a.bias
    )
    kv_BLR, k_rope_BL1R = torch.split(
        compressed_kv_BLC,
        [attention.kv_lora_rank, attention.qk_rope_head_dim],
        dim=-1,
    )
    kv_BLR = F.layer_norm(
        kv_BLR,
        attention.kv_norm.normalized_shape,
        attention.kv_norm.weight,
        attention.kv_norm.bias,
        attention.kv_norm.eps,
    )
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
    scores_BNLL = torch.matmul(
        q_BLNH.transpose(1, 2), k_BLNH.transpose(1, 2).transpose(-1, -2)
    ) * attention.softmax_scale
    scores_BNLL = scores_BNLL + sparse_mask_B1LL
    probs_BNLL = F.softmax(scores_BNLL, dim=-1, dtype=torch.float32).to(q_BLNH.dtype)
    output_BLNV = torch.matmul(probs_BNLL, v_BLNV.transpose(1, 2)).transpose(1, 2)
    return F.linear(
        output_BLNV.contiguous().view(B, L, -1), attention.wo.weight, attention.wo.bias
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
