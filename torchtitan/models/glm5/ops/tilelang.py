# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in TileLang operators for GLM-5 DSA.

The default model uses the readable PyTorch implementations in ``model.py``.
This module implements the same component contracts with local TileLang
kernels and is imported only when explicitly requested through
``--override.imports``.
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.glm5.model import DSAIndexerTopK, SparseMLA


class _TileLangSparseMLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_QNH, kv_K1H, indices_Q1S, scale):
        from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

        q_QNH = q_QNH.contiguous()
        kv_K1H = kv_K1H.contiguous()
        indices_Q1S = indices_Q1S.contiguous()
        output_QNC, lse_QN = sparse_mla_fwd_interface(
            q_QNH,
            kv_K1H,
            indices_Q1S,
            sm_scale=scale,
        )
        ctx.scale = scale
        ctx.save_for_backward(
            q_QNH,
            kv_K1H,
            indices_Q1S,
            output_QNC,
            lse_QN,
        )
        return output_QNC

    @staticmethod
    def backward(ctx, grad_output_QNC):
        from .tilelang_sparse_mla_bwd import sparse_mla_bwd

        q_QNH, kv_K1H, indices_Q1S, output_QNC, lse_QN = ctx.saved_tensors
        grad_q_QNH, grad_kv_K1H = sparse_mla_bwd(
            q_QNH,
            kv_K1H,
            output_QNC,
            grad_output_QNC.contiguous(),
            indices_Q1S,
            lse_QN,
            sm_scale=ctx.scale,
        )
        return grad_q_QNH, grad_kv_K1H, None, None


class TileLangSparseMLA(SparseMLA):
    """Run absorbed SparseMLA with the local GLM-5 TileLang kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(SparseMLA.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.attention_dropout != 0.0:
            raise ValueError("TileLang SparseMLA does not support dropout.")

    def forward(
        self,
        q_QNH: torch.Tensor,
        kv_K1H: torch.Tensor,
        attention_masks_1QK: torch.Tensor,
        topk_indices_QS: torch.Tensor,
        *,
        scale: float,
        latent_dim: int,
    ) -> torch.Tensor:
        if q_QNH.device.type != "cuda" or kv_K1H.device.type != "cuda":
            raise RuntimeError("TileLang SparseMLA requires CUDA q and kv tensors.")
        if q_QNH.dtype != torch.bfloat16 or kv_K1H.dtype != torch.bfloat16:
            raise ValueError("TileLang SparseMLA requires BF16 q and kv tensors.")
        if latent_dim != 512 or q_QNH.shape[-1] - latent_dim != 64:
            raise ValueError(
                "TileLang SparseMLA requires 512 latent dimensions and a "
                "64-dimensional RoPE tail."
            )
        if topk_indices_QS.shape[-1] % 64 != 0:
            raise ValueError("TileLang SparseMLA requires top-k divisible by 64.")

        safe_indices_QS = topk_indices_QS.clamp_min(0).long()
        selected_mask_QS = attention_masks_1QK[0].gather(-1, safe_indices_QS)
        valid_QS = (topk_indices_QS >= 0) & (selected_mask_QS == 0)
        kernel_rows_Q = valid_QS.all(dim=-1)
        fallback_rows_Q = ~kernel_rows_Q
        output_QNC = q_QNH.new_empty(
            q_QNH.shape[0], q_QNH.shape[1], latent_dim
        )
        if fallback_rows_Q.any():
            fallback_output_QNC = super().forward(
                q_QNH[fallback_rows_Q],
                kv_K1H,
                attention_masks_1QK[:, fallback_rows_Q],
                topk_indices_QS[fallback_rows_Q],
                scale=scale,
                latent_dim=latent_dim,
            )
            output_QNC = output_QNC.index_copy(
                0,
                fallback_rows_Q.nonzero().flatten(),
                fallback_output_QNC,
            )
        if not kernel_rows_Q.any():
            return output_QNC

        kernel_output_QNC = _TileLangSparseMLAFunction.apply(
            q_QNH[kernel_rows_Q],
            kv_K1H,
            topk_indices_QS[kernel_rows_Q].unsqueeze(1),
            scale,
        )
        return output_QNC.index_copy(
            0,
            kernel_rows_Q.nonzero().flatten(),
            kernel_output_QNC,
        )


class TileLangDSAIndexerTopK(DSAIndexerTopK):
    """Compute exact GLM-5 index scores with a query-blocked TileLang kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(DSAIndexerTopK.Config):
        query_block_size: int = 8192

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.query_block_size <= 0:
                raise ValueError("query_block_size must be > 0.")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.query_block_size = config.query_block_size

    def forward(
        self,
        q_QNH: torch.Tensor,
        k_KH: torch.Tensor,
        weights_QN: torch.Tensor,
        attention_mask_QK: torch.Tensor,
    ) -> torch.Tensor:
        from .tilelang_indexer_fwd import indexer_fwd_interface

        if q_QNH.device.type != "cuda" or k_KH.device.type != "cuda":
            raise RuntimeError("TileLang DSA indexer requires CUDA q and k tensors.")
        if q_QNH.dtype != torch.bfloat16 or k_KH.dtype != torch.bfloat16:
            raise ValueError("TileLang DSA indexer requires BF16 q and k tensors.")

        allowed_QK = attention_mask_QK == 0
        if not torch.all(allowed_QK.any(dim=-1)):
            raise ValueError("each DSA query must have at least one allowed key.")
        key_positions_K = torch.arange(k_KH.shape[0], device=k_KH.device)
        starts_Q = allowed_QK.to(torch.int32).argmax(dim=-1)
        ends_Q = starts_Q + allowed_QK.sum(dim=-1)
        contiguous_QK = (key_positions_K >= starts_Q.unsqueeze(-1)) & (
            key_positions_K < ends_Q.unsqueeze(-1)
        )
        if not torch.equal(allowed_QK, contiguous_QK):
            raise ValueError(
                "TileLang DSA indexer requires one contiguous key interval "
                "per query."
            )

        topk = min(self.index_topk, k_KH.shape[0])
        indices = []
        for query_start in range(0, q_QNH.shape[0], self.query_block_size):
            query_end = min(query_start + self.query_block_size, q_QNH.shape[0])
            logits_QK = indexer_fwd_interface(
                (q_QNH[query_start:query_end] * self.softmax_scale).contiguous(),
                k_KH.contiguous(),
                weights_QN[query_start:query_end].contiguous(),
                starts_Q[query_start:query_end].to(torch.int32),
                ends_Q[query_start:query_end].to(torch.int32),
            )
            scores_QS, indices_QS = logits_QK.topk(topk, dim=-1)
            indices.append(
                indices_QS.to(torch.int32).masked_fill(
                    scores_QS == float("-inf"), -1
                )
            )
        return torch.cat(indices, dim=0)


@override(
    target=SparseMLA.Config,
    description="Use the local TileLang GLM-5 SparseMLA kernel.",
)
def tilelang_sparse_mla(cfg: SparseMLA.Config) -> TileLangSparseMLA.Config:
    return derive(cfg, TileLangSparseMLA.Config)


@override(
    target=DSAIndexerTopK.Config,
    description="Use the local TileLang GLM-5 DSA indexer kernel.",
)
def tilelang_dsa_indexer(
    cfg: DSAIndexerTopK.Config,
) -> TileLangDSAIndexerTopK.Config:
    return derive(cfg, TileLangDSAIndexerTopK.Config)

