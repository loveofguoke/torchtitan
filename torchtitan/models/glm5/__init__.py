# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    TransformerBlock,
)
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    DSAIndexerTopK,
    DSAInnerAttention,
    Glm5Attention,
    Glm5DsaIndexer,
    Glm5Model,
    Glm5TransformerBlock,
)
from .parallelize import parallelize_glm5
from .sharding import validate_glm5_parallelism
from .state_dict_adapter import Glm5StateDictAdapter

__all__ = [
    "DSAIndexerTopK",
    "DSAInnerAttention",
    "Glm5Attention",
    "Glm5DsaIndexer",
    "Glm5Model",
    "Glm5StateDictAdapter",
    "Glm5TransformerBlock",
    "build_glm5_layers",
    "glm5_configs",
    "make_glm5_attention_config",
    "model_registry",
    "parallelize_glm5",
    "validate_glm5_parallelism",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LAYER_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}
_RMS_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def make_glm5_attention_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    attention_dropout: float,
    rope: ComplexRoPE.Config,
) -> Glm5Attention.Config:
    """Build a fully specified GLM-5 MLA plus DSA attention config."""
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    return Glm5Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        wq_a=Linear.Config(
            in_features=dim, out_features=q_lora_rank, param_init=_LINEAR_INIT
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank, eps=1e-6, param_init=_RMS_NORM_INIT
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        ),
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=kv_lora_rank, eps=1e-6, param_init=_RMS_NORM_INIT
        ),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        rope=dataclasses.replace(rope),
        indexer=Glm5DsaIndexer.Config(
            dim=dim,
            q_lora_rank=q_lora_rank,
            n_heads=index_n_heads,
            head_dim=index_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            wq_b=Linear.Config(
                in_features=q_lora_rank,
                out_features=index_n_heads * index_head_dim,
                param_init=_LINEAR_INIT,
            ),
            wk=Linear.Config(
                in_features=dim,
                out_features=index_head_dim,
                param_init=_LINEAR_INIT,
            ),
            k_norm=LayerNorm.Config(
                normalized_shape=index_head_dim,
                eps=1e-6,
                param_init=_LAYER_NORM_INIT,
            ),
            weights_proj=Linear.Config(
                in_features=dim, out_features=index_n_heads, param_init=_LINEAR_INIT
            ),
            rope=dataclasses.replace(rope),
            topk=DSAIndexerTopK.Config(
                index_topk=index_topk,
                softmax_scale=index_head_dim**-0.5,
            ),
        ),
        inner_attention=DSAInnerAttention.Config(
            attention_dropout=attention_dropout,
        ),
    )


def build_glm5_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_num_expert_groups: int,
    router_num_limited_groups: int,
    router_route_scale: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    attention_dropout: float,
    rope: ComplexRoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build GLM-5 dense and MoE block configurations for every layer."""
    if not 0 <= n_dense_layers <= n_layers:
        raise ValueError("n_dense_layers must be in [0, n_layers].")
    if router_num_expert_groups <= 0:
        raise ValueError("router_num_expert_groups must be > 0.")
    if num_experts % router_num_expert_groups != 0:
        raise ValueError("num_experts must divide evenly into router expert groups.")
    experts_per_group = num_experts // router_num_expert_groups
    if experts_per_group < 2:
        raise ValueError("experts_per_group must be >= 2.")
    if not 0 < router_num_limited_groups <= router_num_expert_groups:
        raise ValueError(
            "router_num_limited_groups must be in [1, router_num_expert_groups]."
        )
    if not 0 < router_top_k <= num_experts:
        raise ValueError("router_top_k must be in [1, num_experts].")
    if router_top_k > experts_per_group * router_num_limited_groups:
        raise ValueError("router_top_k exceeds the experts in limited groups.")

    layers: list[TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        attention = make_glm5_attention_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            attention_dropout=attention_dropout,
            rope=rope,
        )
        if layer_id < n_dense_layers:
            feed_forward = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe = None
        else:
            feed_forward = None
            moe = make_moe_config(
                num_experts=num_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    score_func="sigmoid",
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=True,
                    gate_param_init=_depth_init(layer_id),
                ),
                routed_experts=make_routed_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    param_init=_depth_experts_init(layer_id),
                    comm_backend="standard",
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
                load_balance_coeff=1e-3,
            )
        layers.append(
            Glm5TransformerBlock.Config(
                attention=attention,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_RMS_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_RMS_NORM_INIT
                ),
                feed_forward=feed_forward,
                moe=moe,
            )
        )
    return layers


def _debugmodel() -> Glm5Model.Config:
    dim = 256
    vocab_size = 2048
    rope = ComplexRoPE.Config(
        dim=32,
        max_seq_len=128,
        theta=1_000_000,
        scaling="none",
    )
    return Glm5Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_RMS_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=build_glm5_layers(
            n_layers=8,
            n_dense_layers=1,
            dim=dim,
            n_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            dense_hidden_dim=1024,
            moe_hidden_dim=256,
            num_experts=8,
            num_shared_experts=1,
            router_top_k=2,
            router_num_expert_groups=1,
            router_num_limited_groups=1,
            router_route_scale=2.5,
            index_n_heads=4,
            index_head_dim=64,
            index_topk=8,
            attention_dropout=0.0,
            rope=rope,
        ),
    )


glm5_configs = {"debugmodel": _debugmodel}


def model_registry(flavor: str = "debugmodel") -> ModelSpec:
    return ModelSpec(
        name="glm5",
        flavor=flavor,
        model=glm5_configs[flavor](),
        parallelize_fn=parallelize_glm5,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Glm5StateDictAdapter,
    )
