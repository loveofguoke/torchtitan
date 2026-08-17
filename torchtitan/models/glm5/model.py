# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common import (
    ComplexRoPE,
    LayerNorm,
    Linear,
    RMSNorm,
)
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module


class DSAIndexerTopK(Module):
    """Compute local-query top-k indices against a global key sequence."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int
        softmax_scale: float

        def __post_init__(self) -> None:
            if self.index_topk <= 0:
                raise ValueError("index_topk must be > 0.")
            if self.softmax_scale <= 0.0:
                raise ValueError("softmax_scale must be > 0.")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.softmax_scale = config.softmax_scale

    def forward(
        self,
        q_BQNH: torch.Tensor,
        k_BKH: torch.Tensor,
        weights_BQN: torch.Tensor,
        attention_mask_BQK: torch.Tensor,
    ) -> torch.Tensor:
        if q_BQNH.ndim != 4 or k_BKH.ndim != 3:
            raise ValueError("DSA indexer q and k must have rank 4 and 3.")
        B, Q, N, H = q_BQNH.shape
        K = k_BKH.shape[1]
        if k_BKH.shape != (B, K, H):
            raise ValueError("DSA indexer k must have shape [B, K, H].")
        if weights_BQN.shape != (B, Q, N):
            raise ValueError("DSA indexer weights must have shape [B, Q, N].")
        if attention_mask_BQK.shape != (B, Q, K):
            raise ValueError(
                "DSA indexer attention_masks must have shape [B, Q, K]."
            )
        scores_BNQK = (
            torch.matmul(
                q_BQNH.float().transpose(1, 2),
                k_BKH.float().transpose(1, 2).unsqueeze(1),
            )
            * self.softmax_scale
        )
        scores_BNQK = F.relu(scores_BNQK)
        index_scores_BQK = torch.matmul(
            weights_BQN.unsqueeze(-2), scores_BNQK.transpose(1, 2)
        ).squeeze(-2)
        index_scores_BQK = index_scores_BQK + attention_mask_BQK.float()
        topk = min(self.index_topk, index_scores_BQK.shape[-1])
        return index_scores_BQK.topk(topk, dim=-1).indices.to(torch.int32)


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
        topk: DSAIndexerTopK.Config

        def __post_init__(self) -> None:
            if self.q_lora_rank <= 0:
                raise ValueError("GLM-5 DSA requires q_lora_rank > 0.")
            if self.head_dim < self.qk_rope_head_dim:
                raise ValueError("index_head_dim must be >= qk_rope_head_dim.")
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even.")
            if self.index_topk <= 0:
                raise ValueError("index_topk must be > 0.")
            if self.topk.index_topk != self.index_topk:
                raise ValueError("indexer and topk index_topk must match.")
            if self.topk.softmax_scale != self.head_dim**-0.5:
                raise ValueError("topk softmax_scale must equal head_dim**-0.5.")

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
        self.topk = config.topk.build()

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

        weights_BLN = self.weights_proj(
            hidden_states_BLD.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads**-0.5)
        if attention_mask_BLL is None:
            key_positions_11L = torch.arange(L, device=positions_BL.device)[
                None, None, :
            ]
            attention_mask_BLL = torch.zeros(
                B,
                L,
                L,
                dtype=torch.float32,
                device=positions_BL.device,
            ).masked_fill(
                key_positions_11L > positions_BL.unsqueeze(-1), float("-inf")
            )
        return self.topk(q_BLNH, k_BLH, weights_BLN, attention_mask_BLL)


class DSAInnerAttention(Module):
    """Dense reference implementation of GLM-5 DSA attention.

    Keeping score computation behind an inner-attention boundary makes the
    configured module real (rather than dead metadata) and gives distributed
    backends one well-defined kernel boundary for TP and CP support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention_dropout: float = 0.0

        def __post_init__(self) -> None:
            if not 0.0 <= self.attention_dropout < 1.0:
                raise ValueError("attention_dropout must be in [0, 1).")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        q_BQNH: torch.Tensor,
        k_BKNH: torch.Tensor,
        v_BKNV: torch.Tensor,
        attention_masks_B1QK: torch.Tensor,
        topk_indices_BQT: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        if q_BQNH.ndim != 4 or k_BKNH.ndim != 4 or v_BKNV.ndim != 4:
            raise ValueError("GLM-5 DSA q, k, and v must have rank 4.")
        B, Q, N, H = q_BQNH.shape
        K = k_BKNH.shape[1]
        if k_BKNH.shape != (B, K, N, H):
            raise ValueError("GLM-5 DSA k must have shape [B, K, N, H].")
        if v_BKNV.shape[:3] != (B, K, N):
            raise ValueError("GLM-5 DSA v must have shape [B, K, N, V].")
        if topk_indices_BQT.shape[:2] != (B, Q):
            raise ValueError("GLM-5 DSA top-k indices must have shape [B, Q, T].")
        if attention_masks_B1QK.layout != torch.strided:
            raise ValueError("GLM-5 DSA requires dense attention_masks.")
        if attention_masks_B1QK.shape != (B, 1, Q, K):
            raise ValueError(
                "attention_masks must have shape [B, 1, query_len, key_len]."
            )
        if attention_masks_B1QK.device != q_BQNH.device:
            raise ValueError("attention_masks must be on the same device as q.")
        if not attention_masks_B1QK.is_floating_point():
            raise ValueError("attention_masks must use a floating additive dtype.")

        selected_BQK = torch.zeros_like(
            attention_masks_B1QK[:, 0], dtype=torch.bool
        ).scatter(-1, topk_indices_BQT.long(), True)
        sparse_mask_B1QK = attention_masks_B1QK.masked_fill(
            ~selected_BQK.unsqueeze(1), torch.finfo(q_BQNH.dtype).min
        )
        scores_BNQK = (
            torch.matmul(
                q_BQNH.transpose(1, 2),
                k_BKNH.transpose(1, 2).transpose(-1, -2),
            )
            * scale
        )
        scores_BNQK = scores_BNQK + sparse_mask_B1QK
        probs_BNQK = F.softmax(scores_BNQK, dim=-1, dtype=torch.float32).to(
            q_BQNH.dtype
        )
        probs_BNQK = F.dropout(
            probs_BNQK, p=self.attention_dropout, training=self.training
        )
        return torch.matmul(probs_BNQK, v_BKNV.transpose(1, 2)).transpose(1, 2)


class Glm5Attention(BaseAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        q_lora_rank: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        rope: ComplexRoPE.Config
        indexer: Glm5DsaIndexer.Config
        inner_attention: DSAInnerAttention.Config

        @property
        def qk_head_dim(self) -> int:
            return self.qk_nope_head_dim + self.qk_rope_head_dim

        def __post_init__(self) -> None:
            if self.q_lora_rank <= 0:
                raise ValueError("GLM-5 MLA requires q_lora_rank > 0.")
            if self.kv_lora_rank <= 0:
                raise ValueError("GLM-5 MLA requires kv_lora_rank > 0.")
            if self.qk_nope_head_dim <= 0:
                raise ValueError("GLM-5 MLA requires qk_nope_head_dim > 0.")
            if self.qk_rope_head_dim <= 0:
                raise ValueError("GLM-5 MLA requires qk_rope_head_dim > 0.")
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("GLM-5 MLA requires an even qk_rope_head_dim.")
            if self.qk_head_dim <= 0:
                raise ValueError("GLM-5 MLA requires qk_head_dim > 0.")
            if self.v_head_dim <= 0:
                raise ValueError("GLM-5 MLA requires v_head_dim > 0.")
            if self.n_heads <= 0:
                raise ValueError("GLM-5 MLA requires n_heads > 0.")
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
        self.inner_attention = config.inner_attention.build()

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions_BL: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(attention_masks, torch.Tensor):
            raise ValueError("GLM-5 DSA requires dense attention_masks.")

        B, L, _ = x_BLD.shape
        if attention_masks.layout != torch.strided:
            raise ValueError("GLM-5 DSA requires dense attention_masks.")
        if (
            attention_masks.ndim != 4
            or attention_masks.shape[0] != B
            or attention_masks.shape[1] != 1
            or attention_masks.shape[2] != L
        ):
            raise ValueError(
                "attention_masks must have shape [B, 1, query_len, key_len]."
            )
        if attention_masks.device != x_BLD.device:
            raise ValueError("attention_masks must be on the same device as x_BLD.")
        if not attention_masks.is_floating_point():
            raise ValueError("attention_masks must use a floating additive dtype.")
        if positions_BL is None:
            positions_BL = (
                torch.arange(L, device=x_BLD.device).unsqueeze(0).expand(B, -1)
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
        q_rope_BLNR, k_rope_BL1R = self.rope(q_rope_BLNR, k_rope_BL1R, positions_BL)
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
        output_BLNV = self.inner_attention(
            q_BLNH,
            k_BLNH,
            v_BLNV,
            attention_masks,
            topk_indices_BLK,
            scale=self.softmax_scale,
        )
        return self.wo(output_BLNV.contiguous().view(B, L, -1))


class Glm5TransformerBlock(TransformerBlock):
    """GLM-5 decoder block with either a dense FFN or common MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_BLD = x_BLD + self.attention(
            self.attention_norm(x_BLD), attention_masks, positions
        )
        normalized_BLD = self.ffn_norm(x_BLD)
        ffn_output_BLD = (
            self.moe(normalized_BLD)
            if self.moe_enabled
            else self.feed_forward(normalized_BLD)
        )
        return x_BLD + ffn_output_BLD


class Glm5Model(Decoder):
    """GLM-5 decoder with dense reference DSA attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 6144
        vocab_size: int = 154880

        def update_from_config(self, *, config, **kwargs) -> None:
            # This import is deliberately local: the sharding module pulls in
            # the runtime validation, while config construction remains
            # usable standalone.
            from torchtitan.models.glm5.sharding import (
                set_glm5_sharding_config,
                validate_glm5_parallelism,
            )

            validate_glm5_parallelism(config.parallelism)
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            set_glm5_sharding_config(
                self,
                enable_sp=config.parallelism.enable_sequence_parallel,
                enable_ep=config.parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention = self.layers[0].attention
            assert isinstance(attention, Glm5Attention.Config)
            nparams, base_flops = get_moe_model_nparams_and_flops(
                self,
                model,
                attention.n_heads,
                attention.qk_head_dim + attention.v_head_dim,
                seq_len,
            )
            dsa_flops = (
                2
                * len(self.layers)
                * attention.indexer.n_heads
                * attention.indexer.head_dim
                * seq_len
            )
            return nparams, base_flops + dsa_flops

    def get_attention_masks(self, positions: torch.Tensor) -> torch.Tensor:
        positions_BL = positions
        B, L = positions_BL.shape
        # Non-first pipeline stages have tok_embeddings pruned away, but each
        # PP rank builds the mask for its own stage. Every stage receives the
        # same positions, so any float parameter dtype gives the same mask
        # dtype; fall back to the first layer's norm weight when embeddings
        # are absent. layers is a ModuleDict whose keys keep their original
        # indices after pruning, so iterate values rather than indexing [0].
        if self.tok_embeddings is not None:
            token_dtype = self.tok_embeddings.weight.dtype
        else:
            token_dtype = next(iter(self.layers.values())).attention_norm.weight.dtype
        document_ids_BL = (positions_BL == 0).cumsum(dim=1)
        sequence_indices_L = torch.arange(L, device=positions_BL.device)
        causal_BLL = (
            sequence_indices_L[None, :, None] >= sequence_indices_L[None, None, :]
        )
        same_document_BLL = document_ids_BL.unsqueeze(-1) == document_ids_BL.unsqueeze(
            -2
        )
        allowed_BLL = causal_BLL & same_document_BLL
        return torch.zeros(
            B, 1, L, L, dtype=token_dtype, device=positions_BL.device
        ).masked_fill(~allowed_BLL.unsqueeze(1), torch.finfo(token_dtype).min)

    def forward(
        self,
        tokens_BL: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L = tokens_BL.shape[:2]
        if positions is None:
            positions = torch.arange(
                tokens_BL.shape[1], device=tokens_BL.device
            ).expand(B, -1)
        if attention_masks is None:
            attention_masks = self.get_attention_masks(positions)
        return super().forward(tokens_BL, positions, attention_masks)
