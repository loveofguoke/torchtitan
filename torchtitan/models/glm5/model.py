# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear, RMSNorm
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

# Tensor dimensions used in this file:
# T/Q/K: token, query-token, and key-token dimensions.
# D/N/H/R/P/V: model, head-count, head, RoPE, pass-through, and value dimensions.


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
        q_QNH: torch.Tensor,
        k_KH: torch.Tensor,
        weights_QN: torch.Tensor,
        attention_mask_QK: torch.Tensor,
    ) -> torch.Tensor:
        if q_QNH.ndim != 3 or k_KH.ndim != 2:
            raise ValueError("DSA indexer q and k must have rank 3 and 2.")
        Q, N, H = q_QNH.shape
        K = k_KH.shape[0]
        if k_KH.shape != (K, H):
            raise ValueError("DSA indexer k must have shape [K, H].")
        if weights_QN.shape != (Q, N):
            raise ValueError("DSA indexer weights must have shape [Q, N].")
        if attention_mask_QK.shape != (Q, K):
            raise ValueError("DSA indexer attention mask must have shape [Q, K].")
        scores_NQK = torch.matmul(
            q_QNH.float().transpose(0, 1), k_KH.float().transpose(0, 1)
        ) * self.softmax_scale
        scores_NQK = F.relu(scores_NQK)
        index_scores_QK = torch.matmul(
            weights_QN.unsqueeze(1), scores_NQK.transpose(0, 1)
        ).squeeze(1)
        index_scores_QK = index_scores_QK + attention_mask_QK.float()
        topk = min(self.index_topk, index_scores_QK.shape[-1])
        return index_scores_QK.topk(topk, dim=-1).indices.to(torch.int32)


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
        # Released GLM-5 checkpoints contain a pretrained indexer. This model
        # has no auxiliary indexer objective, so LM training must keep it fixed
        # and exclude its parameters from optimizer state.
        self.requires_grad_(False)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.weights_proj.float()
        return self

    @torch.no_grad()
    def forward(
        self,
        hidden_states_TD: torch.Tensor,
        q_resid_TR: torch.Tensor,
        positions_T: torch.Tensor,
        attention_mask_TK: torch.Tensor | None,
    ) -> torch.Tensor:
        T = hidden_states_TD.shape[0]
        q_TNH = self.wq_b(q_resid_TR).view(T, self.n_heads, self.head_dim)
        q_rot_TNR, q_pass_TNP = torch.split(
            q_TNH,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_T1H = self.k_norm(self.wk(hidden_states_TD)).unsqueeze(1)
        k_rot_T1R, k_pass_T1P = torch.split(
            k_T1H,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_rot_TNR, k_rot_T1R = self.rope(q_rot_TNR, k_rot_T1R, positions_T)
        q_TNH = torch.cat((q_rot_TNR, q_pass_TNP), dim=-1)
        k_TH = torch.cat((k_rot_T1R, k_pass_T1P), dim=-1).squeeze(1)

        weights_TN = self.weights_proj(
            hidden_states_TD.to(self.weights_proj.weight.dtype)
        ).float() * (self.n_heads**-0.5)
        if attention_mask_TK is None:
            key_positions_K = torch.arange(T, device=positions_T.device)
            attention_mask_TK = torch.zeros(
                T,
                T,
                dtype=torch.float32,
                device=positions_T.device,
            ).masked_fill(key_positions_K > positions_T.unsqueeze(-1), float("-inf"))
        return self.topk(q_TNH, k_TH, weights_TN, attention_mask_TK)


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
        q_QNH: torch.Tensor,
        k_KNH: torch.Tensor,
        v_KNV: torch.Tensor,
        attention_masks_1QK: torch.Tensor,
        topk_indices_QS: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        if q_QNH.ndim != 3 or k_KNH.ndim != 3 or v_KNV.ndim != 3:
            raise ValueError("GLM-5 DSA q, k, and v must have rank 3.")
        Q, N, H = q_QNH.shape
        K = k_KNH.shape[0]
        if k_KNH.shape != (K, N, H):
            raise ValueError("GLM-5 DSA k must have shape [K, N, H].")
        if v_KNV.shape[:2] != (K, N):
            raise ValueError("GLM-5 DSA v must have shape [K, N, V].")
        if topk_indices_QS.shape[0] != Q:
            raise ValueError("GLM-5 DSA top-k indices must have shape [Q, S].")
        if attention_masks_1QK.layout != torch.strided:
            raise ValueError("GLM-5 DSA requires dense attention_masks.")
        if attention_masks_1QK.shape != (1, Q, K):
            raise ValueError("attention_masks must have shape [1, Q, K].")
        if attention_masks_1QK.device != q_QNH.device:
            raise ValueError("attention_masks must be on the same device as q.")
        if not attention_masks_1QK.is_floating_point():
            raise ValueError("attention_masks must use a floating additive dtype.")

        selected_QK = torch.zeros_like(
            attention_masks_1QK[0], dtype=torch.bool
        ).scatter(-1, topk_indices_QS.long(), True)
        sparse_mask_1QK = attention_masks_1QK.masked_fill(
            ~selected_QK.unsqueeze(0), torch.finfo(q_QNH.dtype).min
        )
        scores_NQK = torch.matmul(
            q_QNH.transpose(0, 1), k_KNH.transpose(0, 1).transpose(-1, -2)
        ) * scale
        scores_NQK = scores_NQK + sparse_mask_1QK
        probs_NQK = F.softmax(scores_NQK, dim=-1, dtype=torch.float32).to(
            q_QNH.dtype
        )
        probs_NQK = F.dropout(
            probs_NQK, p=self.attention_dropout, training=self.training
        )
        return torch.matmul(probs_NQK, v_KNV.transpose(0, 1)).transpose(0, 1)


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
                or self.indexer.qk_rope_head_dim != self.qk_rope_head_dim
            ):
                raise ValueError(
                    "indexer shared dimensions must match GLM-5 MLA dimensions."
                )

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
        x_TD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(attention_masks, torch.Tensor):
            raise ValueError("GLM-5 DSA requires dense attention_masks.")

        T = x_TD.shape[0]
        if attention_masks.layout != torch.strided:
            raise ValueError("GLM-5 DSA requires dense attention_masks.")
        if (
            attention_masks.ndim != 3
            or attention_masks.shape[0] != 1
            or attention_masks.shape[1] != T
        ):
            raise ValueError("attention_masks must have shape [1, query_len, key_len].")
        if attention_masks.device != x_TD.device:
            raise ValueError("attention_masks must be on the same device as x_TD.")
        if not attention_masks.is_floating_point():
            raise ValueError("attention_masks must use a floating additive dtype.")
        if positions_T is None:
            positions_T = torch.arange(T, device=x_TD.device)

        q_resid_TR = self.q_norm(self.wq_a(x_TD))
        q_TNH = self.wq_b(q_resid_TR).view(T, self.n_heads, self.qk_head_dim)
        q_nope_TNP, q_rope_TNR = torch.split(
            q_TNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv_TC = self.wkv_a(x_TD)
        kv_TR, k_rope_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_TR = self.kv_norm(kv_TR)
        q_rope_TNR, k_rope_T1R = self.rope(
            q_rope_TNR, k_rope_TR.unsqueeze(1), positions_T
        )
        q_TNH = torch.cat((q_nope_TNP, q_rope_TNR), dim=-1)
        kv_TNX = self.wkv_b(kv_TR).view(
            T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_TNP, v_TNV = torch.split(
            kv_TNX, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_TNH = torch.cat(
            (k_nope_TNP, k_rope_T1R.expand(-1, self.n_heads, -1)),
            dim=-1,
        )
        topk_indices_TS = self.indexer(
            x_TD,
            q_resid_TR,
            positions_T,
            attention_masks[0],
        )
        output_TNV = self.inner_attention(
            q_TNH,
            k_TNH,
            v_TNV,
            attention_masks,
            topk_indices_TS,
            scale=self.softmax_scale,
        )
        return self.wo(output_TNV.contiguous().view(T, -1))


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
        x_TD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_TD = x_TD + self.attention(
            self.attention_norm(x_TD), attention_masks, positions
        )
        normalized_TD = self.ffn_norm(x_TD)
        if self.moe_enabled:
            ffn_output_TD = self.moe(normalized_TD)
        else:
            ffn_output_TD = self.feed_forward(normalized_TD)
        return x_TD + ffn_output_TD


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
            # Generic Decoder CP uses SPMD sharding and therefore requires the
            # spmd_types backend. GLM-5 owns a model-specific partial-DTensor
            # DSA CP path, so run generic setup with CP disabled and apply the
            # GLM sharding contract below.
            generic_config = replace(
                config,
                parallelism=replace(
                    config.parallelism,
                    context_parallel_degree=1,
                ),
            )
            Decoder.Config.update_from_config(
                self,
                config=generic_config,
                **kwargs,
            )
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

    def get_attention_masks(self, positions: torch.Tensor) -> torch.Tensor | None:
        # Pipeline partitioning may create an embedding-only or output-only
        # stage. Such a stage has no attention operation and needs no mask.
        if len(self.layers) == 0:
            return None

        if positions.ndim != 1:
            raise ValueError("GLM-5 positions must have shape [T].")
        T = positions.shape[0]
        # Non-first pipeline stages have tok_embeddings pruned away, but each
        # PP stage containing attention builds its own mask. Every stage
        # receives the same positions, so any float parameter dtype gives the
        # same mask dtype; fall back to the first layer's norm weight when
        # embeddings are absent. layers is a ModuleDict whose keys keep their
        # original indices after pruning, so iterate values rather than
        # indexing [0].
        if self.tok_embeddings is not None:
            token_dtype = self.tok_embeddings.weight.dtype
        else:
            token_dtype = next(iter(self.layers.values())).attention_norm.weight.dtype
        document_ids_T = (positions == 0).cumsum(dim=0)
        sequence_indices_T = torch.arange(T, device=positions.device)
        causal_TT = sequence_indices_T[:, None] >= sequence_indices_T[None, :]
        same_document_TT = document_ids_T[:, None] == document_ids_T[None, :]
        allowed_TT = causal_TT & same_document_TT
        return torch.zeros(
            1, T, T, dtype=token_dtype, device=positions.device
        ).masked_fill(~allowed_TT.unsqueeze(0), torch.finfo(token_dtype).min)
