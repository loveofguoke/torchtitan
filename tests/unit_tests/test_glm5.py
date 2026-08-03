# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.models.glm5.model import Glm5DsaIndexer


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
