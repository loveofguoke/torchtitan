# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from torchtitan.models.common import ComplexRoPE, FlexAttention, LayerNorm, Linear
from torchtitan.models.common.attention import BaseAttention
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


class Glm5Attention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        q_lora_rank: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        attention_dropout: float
        wq_a: Linear.Config
        q_norm: LayerNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: LayerNorm.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        rope: ComplexRoPE.Config
        indexer: Glm5DsaIndexer.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

        @property
        def qk_head_dim(self) -> int:
            return self.qk_nope_head_dim + self.qk_rope_head_dim

        def __post_init__(self) -> None:
            if self.q_lora_rank <= 0:
                raise ValueError("GLM-5 MLA requires q_lora_rank > 0.")
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("GLM-5 MLA requires an even qk_rope_head_dim.")
            if self.n_heads <= 0:
                raise ValueError("GLM-5 MLA requires n_heads > 0.")
            if not 0.0 <= self.attention_dropout < 1.0:
                raise ValueError("attention_dropout must be in [0, 1).")
            if (
                self.wq_a.in_features != self.dim
                or self.wq_a.out_features != self.q_lora_rank
            ):
                raise ValueError("wq_a must project dim to q_lora_rank.")
            if self.q_norm.normalized_shape != self.q_lora_rank:
                raise ValueError("q_norm normalized_shape must equal q_lora_rank.")
            if (
                self.wq_b.in_features != self.q_lora_rank
                or self.wq_b.out_features != self.n_heads * self.qk_head_dim
            ):
                raise ValueError("wq_b must project q_lora_rank to all query heads.")
            if (
                self.wkv_a.in_features != self.dim
                or self.wkv_a.out_features != self.kv_lora_rank + self.qk_rope_head_dim
            ):
                raise ValueError(
                    "wkv_a must project dim to compressed KV and RoPE key."
                )
            if self.kv_norm.normalized_shape != self.kv_lora_rank:
                raise ValueError("kv_norm normalized_shape must equal kv_lora_rank.")
            if (
                self.wkv_b.in_features != self.kv_lora_rank
                or self.wkv_b.out_features
                != self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
            ):
                raise ValueError("wkv_b must project compressed KV to all KV heads.")
            if (
                self.wo.in_features != self.n_heads * self.v_head_dim
                or self.wo.out_features != self.dim
            ):
                raise ValueError("wo must project all value heads to dim.")
            if self.rope.dim != self.qk_rope_head_dim:
                raise ValueError("rope dim must equal qk_rope_head_dim.")
            if (
                self.indexer.dim != self.dim
                or self.indexer.q_lora_rank != self.q_lora_rank
                or self.indexer.n_heads != self.n_heads
                or self.indexer.head_dim != self.qk_head_dim
                or self.indexer.qk_rope_head_dim != self.qk_rope_head_dim
            ):
                raise ValueError("indexer dimensions must match GLM-5 MLA dimensions.")

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.attention_dropout = config.attention_dropout
        self.softmax_scale = config.qk_head_dim**-0.5
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.rope = config.rope.build()
        self.indexer = config.indexer.build()

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions_BL: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not isinstance(attention_masks, torch.Tensor)
            or attention_masks.layout != torch.strided
        ):
            raise ValueError("GLM-5 eager attention requires dense attention_masks.")

        B, L, _ = x_BLD.shape
        if positions_BL is None:
            positions_BL = torch.arange(L, device=x_BLD.device).unsqueeze(0).expand(
                B, -1
            )

        q_resid_BLR = self.q_norm(self.wq_a(x_BLD))
        q_BLNH = self.wq_b(q_resid_BLR).view(B, L, self.n_heads, self.qk_head_dim)
        q_nope_BLNP, q_rope_BLNR = torch.split(
            q_BLNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv_BLC = self.wkv_a(x_BLD)
        kv_BLR, k_rope_BL1R = torch.split(
            compressed_kv_BLC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_BLR = self.kv_norm(kv_BLR)
        k_rope_BL1R = k_rope_BL1R.unsqueeze(2)
        q_rope_BLNR, k_rope_BL1R = self.rope(
            q_rope_BLNR, k_rope_BL1R, positions_BL
        )
        q_BLNH = torch.cat((q_nope_BLNP, q_rope_BLNR), dim=-1)
        kv_BLNX = self.wkv_b(kv_BLR).view(
            B, L, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_BLNP, v_BLNV = torch.split(
            kv_BLNX, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_BLNH = torch.cat(
            (k_nope_BLNP, k_rope_BL1R.expand(-1, -1, self.n_heads, -1)),
            dim=-1,
        )
        topk_indices_BLK = self.indexer(
            x_BLD,
            q_resid_BLR,
            positions_BL,
            attention_masks[:, 0],
        )
        selected_BLL = torch.zeros(
            B, L, L, dtype=torch.bool, device=x_BLD.device
        ).scatter(-1, topk_indices_BLK.long(), True)
        min_value = torch.finfo(x_BLD.dtype).min
        sparse_mask_B1LL = attention_masks.masked_fill(
            ~selected_BLL.unsqueeze(1), min_value
        )
        scores_BNLL = torch.matmul(
            q_BLNH.transpose(1, 2), k_BLNH.transpose(1, 2).transpose(-1, -2)
        ) * self.softmax_scale
        scores_BNLL = scores_BNLL + sparse_mask_B1LL
        probs_BNLL = F.softmax(scores_BNLL, dim=-1, dtype=torch.float32).to(
            q_BLNH.dtype
        )
        probs_BNLL = F.dropout(
            probs_BNLL, p=self.attention_dropout, training=self.training
        )
        output_BLNV = torch.matmul(probs_BNLL, v_BLNV.transpose(1, 2)).transpose(1, 2)
        return self.wo(output_BLNV.contiguous().view(B, L, -1))
