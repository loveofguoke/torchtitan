# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    token_id_placement,
)
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.glm5.model import Glm5Attention, Glm5DsaIndexer
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

if TYPE_CHECKING:
    from torchtitan.models.glm5.model import Glm5Model, Glm5TransformerBlock


# Routed-expert layout for the shared ``GroupedExperts`` (w1/w2/w3).
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, spmd.PerMeshAxisSpmdType] = {
    "w1_EFD": spmd.S(1),
    "w2_EDF": spmd.S(2),
    "w3_EFD": spmd.S(1),
}


def _dsa_mask_layout() -> SpmdLayout:
    """Describe dense ``[1, query, key]`` mask placement."""

    return SpmdLayout(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.R,
        },
        partition_spec=(None, (MeshAxisName.DP, MeshAxisName.CP), None),
    )


def validate_glm5_parallelism(
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims | None = None,
) -> None:
    """Reject the runtime layouts GLM-5 does not yet support.

    GLM-5 supports data parallelism (DDP/HSDP via
    ``data_parallel_replicate_degree``, and FSDP via
    ``data_parallel_shard_degree``) together with context, tensor, pipeline,
    and expert parallelism. It still rejects:

    - CP load balancing: the correctness-first DSA path requires contiguous
      sequence shards so gathered keys retain global token order.
    - The ``spmd_types`` backend. GLM-5's DSA local-map path currently uses
      DTensor-specific CP wrappers.

    ``parallel_dims`` is kept for signature parity with the runtime call site;
    once resolved, ``ParallelDims._validate`` has already enforced
    ``dp_replicate * dp_shard * cp * tp * pp == world_size``.
    """
    unsupported: list[str] = []
    cp_enabled = parallelism.context_parallel_degree > 1 or (
        parallel_dims is not None and parallel_dims.cp_enabled
    )
    if cp_enabled and parallelism.context_parallel_load_balancer is not None:
        unsupported.append(
            "CP load balancing " f"({parallelism.context_parallel_load_balancer})"
        )
    if parallelism.spmd_backend != "partial_dtensor":
        unsupported.append(f"SPMD backend ({parallelism.spmd_backend})")

    if unsupported:
        modes = ", ".join(unsupported)
        raise NotImplementedError(
            "GLM-5 supports data parallelism with CP/TP/PP/EP; "
            f"unsupported parallelism: {modes}."
        )


def set_glm5_sharding_config(
    config: "Glm5Model.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all GLM-5 sub-configs.

    Dense sub-configs (attention, norms, dense FFN) are populated
    unconditionally -- ``Module.parallelize`` filters disabled axes
    at runtime.

    MoE sub-configs (router, shared experts, routed experts) are
    populated unconditionally -- ``resolve_mesh`` filters disabled
    axes at runtime.
    """

    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_glm5_layer_sharding(layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep)


def _set_glm5_layer_sharding(
    layer_cfg: "Glm5TransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one GLM-5 transformer layer.

    Attention and norms are sharded on all blocks (MoE and non-MoE).
    Dense FFN is only sharded on non-MoE blocks; MoE FFN is routed
    through ``set_moe_sharding_config``.
    """
    attention = layer_cfg.attention
    assert isinstance(attention, Glm5Attention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )

    set_glm5_attention_sharding(attention, enable_sp=enable_sp)

    # Dense FFN (non-MoE layers only)
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )

    # MoE FFN (MoE-enabled layers only).
    if layer_cfg.moe is not None:
        set_moe_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )


def set_glm5_attention_sharding(
    attention: Glm5Attention.Config,
    *,
    enable_sp: bool,
) -> None:
    """GLM-5 MLA + DSA attention TP sharding.

    Mirrors DeepSeek-V3's MLA plan: low-rank projections and norms stay
    Replicate on TP, up-projections are Colwise (shard on the head dim),
    ``wo`` is Rowwise. The DSA indexer is kept fully Replicate on TP (see
    ``set_glm5_indexer_sharding``).
    """
    # ``attention_masks`` must arrive as a Replicate DTensor under TP: the
    # eager DSA mask construction (zeros_like + scatter, masked_fill) rejects
    # mixing a plain base tensor with a DTensor arg, so the whole mask path
    # runs on Replicate DTensors. On a single device the mesh is absent and the
    # input stays plain.
    attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "x_TD": (
                dense_sequence_parallel_placement()
                if enable_sp
                else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
            ),
            "attention_masks": _dsa_mask_layout(),
        },
        in_dst_shardings={
            "x_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            "attention_masks": _dsa_mask_layout(),
        },
    )
    attention.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )
    # Low-rank projections and norms keep Replicate weights on TP. We still
    # distribute them (Replicate DTensor) so DTensor activations flow through
    # without mixing plain Tensor + DTensor in the matmul.
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    attention.wq_a.sharding_config = replicate_weight
    attention.q_norm.sharding_config = replicate_weight
    attention.wkv_a.sharding_config = replicate_weight
    attention.kv_norm.sharding_config = replicate_weight

    attention.wq_b.sharding_config = colwise_config()
    attention.wkv_b.sharding_config = colwise_config()
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)
    set_glm5_dsa_inner_attention_sharding(attention.inner_attention)

    set_glm5_indexer_sharding(attention.indexer)


def set_glm5_dsa_inner_attention_sharding(inner_attention) -> None:
    """Run dense DSA score computation on TP-local tensors.

    Q/K/V are TP-sharded on the head dimension. The dense mask and DSA top-k
    indices are replicated on TP. Keeping this boundary in ``local_map`` avoids
    mixing DTensors with the local dense attention kernel. On the default
    backend, CP K/V gathering is installed separately by
    ``apply_glm5_cp_to_forward`` before this local-map boundary is captured.
    """
    q_layout = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    kv_src_layout = dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
    kv_dst_layout = dense_activation_placement(tp=spmd.S(1), cp=spmd.R)
    kv_grad_layout = dense_activation_placement(tp=spmd.S(1), cp=spmd.P)
    mask_layout = _dsa_mask_layout()
    topk_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    inner_attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_QNH": q_layout,
            "k_KNH": kv_src_layout,
            "v_KNV": kv_src_layout,
            "attention_masks_1QK": mask_layout,
            "topk_indices_QS": topk_layout,
        },
        in_dst_shardings={
            "q_QNH": q_layout,
            "k_KNH": kv_dst_layout,
            "v_KNV": kv_dst_layout,
            "attention_masks_1QK": mask_layout,
            "topk_indices_QS": topk_layout,
        },
        out_src_shardings=q_layout,
        local_map=LocalMapConfig(
            in_grad_placements=(
                q_layout,
                kv_grad_layout,
                kv_grad_layout,
                # Mask and top-k indices are non-differentiable metadata, but
                # they are still DTensor inputs and local_map requires their
                # placements whenever in_grad_placements is specified.
                mask_layout,
                topk_layout,
            )
        ),
    )


def set_glm5_indexer_sharding(indexer: Glm5DsaIndexer.Config) -> None:
    """DSA indexer sharding: fully Replicate on TP (correctness first).

    ``Glm5DsaIndexer.forward`` sums per-index-head scores into a
    ``[T, T]`` tensor and then runs ``topk`` on it. The declarative
    ``sharding_config`` redistributes module *outputs* after ``forward``
    returns, so sharding the index heads (Colwise) would leave ``topk``
    running on a partial (per-rank head-shard) score tensor -- incorrect.

    Keeping the whole indexer Replicated makes every TP rank compute the
    DSA indices redundantly. The indexer is small (``index_n_heads=4`` in the
    debugmodel), so the redundant cost is acceptable; a distributed
    index-head reduction with a replicated global ``topk`` is a documented
    follow-up. CP gathers the indexer key sequence in the model-specific
    forward wrapper before this TP-local top-k kernel runs.
    """
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
    )
    indexer.wq_b.sharding_config = replicate_weight
    indexer.wk.sharding_config = replicate_weight
    indexer.weights_proj.sharding_config = replicate_weight
    indexer.k_norm.sharding_config = ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
    )
    indexer.rope.sharding_config = ShardingConfig(
        state_shardings={"cache": dense_param_placement(tp=spmd.R)},
    )

    query_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    key_src_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    key_dst_layout = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    indexer.topk.sharding_config = ShardingConfig(
        in_src_shardings={
            "q_QNH": query_layout,
            "k_KH": key_src_layout,
            "weights_QN": query_layout,
            "attention_mask_QK": query_layout,
        },
        in_dst_shardings={
            "q_QNH": query_layout,
            "k_KH": key_dst_layout,
            "weights_QN": query_layout,
            "attention_mask_QK": query_layout,
        },
        out_src_shardings=query_layout,
        local_map=LocalMapConfig(in_grad_placements=None),
    )

    # Keep every indexer input Replicate on TP so the forward body operates on
    # Replicate DTensors end to end.
    indexer.sharding_config = ShardingConfig(
        in_src_shardings={
            "hidden_states_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            "q_resid_TR": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            "positions_T": token_id_placement(),
            "attention_mask_TK": query_layout,
        },
        in_dst_shardings={
            "hidden_states_TD": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            "q_resid_TR": dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
            "positions_T": token_id_placement(),
            "attention_mask_TK": query_layout,
        },
        out_src_shardings=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
        out_dst_shardings=dense_activation_placement(tp=spmd.R, cp=spmd.S(0)),
    )
