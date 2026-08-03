# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.protocols.module import Module


class Glm5DsaIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        q_lora_rank: int
        n_heads: int
        head_dim: int
        qk_rope_head_dim: int
        index_topk: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: ComplexRoPE.Config

        def __post_init__(self) -> None:
            if self.q_lora_rank <= 0:
                raise ValueError("GLM-5 DSA requires q_lora_rank > 0.")
            if self.head_dim < self.qk_rope_head_dim:
                raise ValueError("index_head_dim must be >= qk_rope_head_dim.")
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even.")
            if self.index_topk <= 0:
                raise ValueError("index_topk must be > 0.")

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = config.head_dim**-0.5
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build().float()
        self.rope = config.rope.build()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.weights_proj.float()
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states_BLD: torch.Tensor,
        q_resid_BLR: torch.Tensor,
        positions_BL: torch.Tensor,
        attention_mask_BLL: torch.Tensor | None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states_BLD.shape
        q_BLNH = self.wq_b(q_resid_BLR).view(B, L, self.n_heads, self.head_dim)
        q_rot_BLNR, q_pass_BLNP = torch.split(
            q_BLNH,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_BL1H = self.k_norm(self.wk(hidden_states_BLD)).unsqueeze(2)
        k_rot_BL1R, k_pass_BL1P = torch.split(
            k_BL1H,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_rot_BLNR, k_rot_BL1R = self.rope(q_rot_BLNR, k_rot_BL1R, positions_BL)
        q_BLNH = torch.cat((q_rot_BLNR, q_pass_BLNP), dim=-1)
        k_BLH = torch.cat((k_rot_BL1R, k_pass_BL1P), dim=-1).squeeze(2)

        scores_BNLL = (
            torch.matmul(
                q_BLNH.float().transpose(1, 2),
                k_BLH.float().transpose(1, 2).unsqueeze(1),
            )
            * self.softmax_scale
        )
        scores_BNLL = F.relu(scores_BNLL)
        weights_BLN = self.weights_proj(
            hidden_states_BLD.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads**-0.5)
        index_scores_BLL = torch.matmul(
            weights_BLN.unsqueeze(-2), scores_BNLL.transpose(1, 2)
        ).squeeze(-2)
        if attention_mask_BLL is not None:
            index_scores_BLL = index_scores_BLL + attention_mask_BLL.float()
        else:
            key_positions_11L = torch.arange(L, device=positions_BL.device)[
                None, None, :
            ]
            index_scores_BLL = index_scores_BLL.masked_fill(
                key_positions_11L > positions_BL.unsqueeze(-1), float("-inf")
            )
        topk = min(self.index_topk, index_scores_BLL.shape[-1])
        return index_scores_BLL.topk(topk, dim=-1).indices.to(torch.int32)
