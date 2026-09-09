# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-neutral GLM-5 model mathematics.

Read this file from the outside in:

1. ``Glm5Model`` supplies token embedding, decoder blocks, final norm, and the
   language-model head inherited from ``Decoder``.
2. ``Glm5TransformerBlock`` composes pre-norm DSA attention with either a
   dense FFN or the common TorchTitan MoE implementation.
3. ``Glm5Attention`` expands MLA's low-rank Q/KV representations, asks the
   frozen ``Glm5DsaIndexer`` which keys each query may attend to, and delegates
   score/value computation to ``Glm5FlexAttention`` in ``dsa.py``.
4. ``dsa.py`` contains the indexer and token-selected BlockMask construction.

The model uses token-first tensors because TorchTitan packs microbatches into a
single token dimension before model forward. Distributed placement is declared
separately in ``sharding.py``; the formulas below describe global logical
shapes. Distributed adaptation of this sparse path is a separate step.
"""

from dataclasses import dataclass, replace

import torch
from torch import nn

from torchtitan.models.common import ComplexRoPE, Linear, RMSNorm
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from .dsa import DSAIndexerTopK, Glm5DsaIndexer, create_dsa_causal_mask
from .dsa import Glm5FlexAttention as DSAInnerAttention

# Tensor dimensions used in this file:
# T/Q/K: token, query-token, and key-token dimensions.
# D/N/H/R/P/V: model, head-count, head, RoPE, pass-through, and value dimensions.


class Glm5Attention(BaseAttention):
    """Multi-head latent attention (MLA) gated by DSA top-k indices.

    Q follows ``D -> q_lora_rank -> N * (nope + rope)``. KV follows
    ``D -> (kv_lora_rank + rope)`` and expands only after the low-rank norm.
    The indexer consumes the same token states and returns ``[T, topk]`` key
    indices. ``inner_attention`` applies token-selected sparse attention to
    the expanded Q/K/V tensors.
    """
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
        inner_attention: Module.Config

        @property
        def qk_head_dim(self) -> int:
            return self.qk_nope_head_dim + self.qk_rope_head_dim


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
        positions: torch.Tensor | None = None,
        topk_indices_TS: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        T = x_TD.shape[0]
        if positions is None:
            positions = torch.arange(T, device=x_TD.device)

        # Query path: compress to LoRA rank R, normalize, then expand to N
        # heads. Split content (NoPE) from position-sensitive (RoPE) features.
        q_resid_TR = self.q_norm(self.wq_a(x_TD))
        q_TNH = self.wq_b(q_resid_TR).view(T, self.n_heads, self.qk_head_dim)
        q_nope_TNP, q_rope_TNR = torch.split(
            q_TNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # KV path: keep content in a compressed latent representation and keep
        # one shared rotary key. The expensive per-head expansion happens only
        # after normalization.
        compressed_kv_TC = self.wkv_a(x_TD)
        kv_TR, k_rope_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_TR = self.kv_norm(kv_TR)
        q_rope_TNR, k_rope_T1R = self.rope(
            q_rope_TNR, k_rope_TR.unsqueeze(1), positions
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
        # Selection and value computation are separate on purpose: the indexer
        # is frozen and discrete, while the attention path remains trainable.
        if topk_indices_TS is None:
            topk_indices_TS = self.indexer(
                x_TD, q_resid_TR, positions, attention_masks[0]
            )
        output_TNV = self.inner_attention(
            q_TNH,
            k_TNH,
            v_TNV,
            # local_map's input placements follow positional tensor arguments.
            attention_masks,
            topk_indices_TS,
            scale=self.softmax_scale,
        )
        output_TD = self.wo(output_TNV.contiguous().view(T, -1))
        return (output_TD, topk_indices_TS) if return_indices else output_TD


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
        topk_indices_TS: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_indices:
            attention_TD, topk_indices_TS = self.attention(
                self.attention_norm(x_TD), attention_masks, positions,
                topk_indices_TS, return_indices=True,
            )
        else:
            attention_TD = self.attention(
                self.attention_norm(x_TD), attention_masks, positions,
                topk_indices_TS,
            )
        x_TD = x_TD + attention_TD
        normalized_TD = self.ffn_norm(x_TD)
        if self.moe_enabled:
            ffn_output_TD = self.moe(normalized_TD)
        else:
            ffn_output_TD = self.feed_forward(normalized_TD)
        output_TD = x_TD + ffn_output_TD
        return (output_TD, topk_indices_TS) if return_indices else output_TD


class Glm5Model(Decoder):
    """GLM-5 decoder with FlexAttention DSA attention.

    The inherited decoder accepts token ids on the first PP stage and hidden
    states on later stages. A PP stage may own zero decoder layers, which is why
    mask construction and forward preserve embedding-only/output-only stages.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 6144
        vocab_size: int = 154880
        index_sources: tuple[int, ...] = ()

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
            # The common parameter-based estimate charges every parameter for
            # forward and backward matmuls (6 FLOPs per parameter). GLM-5's
            # pretrained indexer is frozen and runs under no_grad, so it only
            # incurs its forward cost (2 FLOPs per parameter). Keep the total
            # parameter count unchanged while removing the nonexistent
            # indexer backward work from MFU accounting.
            nparams_indexer = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if ".attention.indexer." in name
            )
            base_flops -= 4 * nparams_indexer
            # Per token, the indexer computes N dot products of width H against
            # the full sequence, followed by a weighted reduction over N heads.
            dsa_flops = (
                2
                * len(self.layers)
                * attention.indexer.n_heads
                * (attention.indexer.head_dim + 1)
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
        return create_dsa_causal_mask(positions, dtype=token_dtype)

    def __init__(self, config):
        super().__init__(config)
        self.index_sources = config.index_sources
        if self.index_sources:
            if len(self.index_sources) != len(config.layers):
                raise ValueError("index_sources must contain one source per layer")
            for layer, source in enumerate(self.index_sources):
                if not 0 <= source <= layer or self.index_sources[source] != source:
                    raise ValueError("index source must be a current/earlier producer layer")

    def forward(
        self,
        tokens: torch.Tensor,
        pipeline_indices_TS: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Preserve Decoder's positional ``model(tokens, positions)`` API. PP
        # metadata is [T, S], while GLM positions are required to be [T].
        if pipeline_indices_TS is not None and pipeline_indices_TS.ndim == 1:
            if positions is not None:
                raise ValueError("positions were provided twice")
            positions = pipeline_indices_TS
            pipeline_indices_TS = None
        if positions is None:
            positions = torch.arange(tokens.shape[0], device=tokens.device)
        if attention_masks is None:
            attention_masks = self.get_attention_masks(positions)
        if not self.index_sources:
            return super().forward(tokens, positions, attention_masks)
        hidden_TD = (
            self.tok_embeddings(tokens)
            if self.tok_embeddings is not None
            else tokens
        )
        local_layers = {int(name) for name in self.layers}
        external_sources = {
            self.index_sources[layer]
            for layer in local_layers
            if self.index_sources[layer] not in local_layers
        }
        indices_by_source: dict[int, torch.Tensor] = {}
        if external_sources:
            if pipeline_indices_TS is None:
                raise ValueError("this GLM-5 PP stage requires DSA indices")
            source = next(iter(external_sources))
            indices_by_source[source] = pipeline_indices_TS
        for name, layer in self.layers.items():
            layer_id = int(name)
            source = self.index_sources[layer_id]
            hidden_TD, indices_TS = layer(
                hidden_TD,
                attention_masks,
                positions,
                indices_by_source.get(source),
                return_indices=True,
            )
            if source == layer_id:
                indices_by_source[source] = indices_TS
        hidden_TD = self.norm(hidden_TD) if self.norm is not None else hidden_TD
        if self._skip_lm_head or self.lm_head is None:
            if local_layers:
                next_layer = max(local_layers) + 1
                if next_layer < len(self.index_sources):
                    source = self.index_sources[next_layer]
                    if source != next_layer:
                        return hidden_TD, indices_by_source[source]
            return hidden_TD
        return self.lm_head(hidden_TD)
