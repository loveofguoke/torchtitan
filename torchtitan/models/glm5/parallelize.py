# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime composition of GLM-5 parallel dimensions.

This file does not define model mathematics. It orders the transforms that
make the same model run on CP/TP/EP/DP and optional PP stages:

``CP forward wrapping -> TP/EP DTensor parallelize -> activation checkpoint
-> torch.compile -> composable FSDP2``.

The order is a contract. CP replaces local forward functions before
``Module.parallelize`` captures them in ``local_map``; FSDP is last so it wraps
the final parameter layout rather than parameters that later transforms replace.
"""

from functools import wraps

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._context_parallel._attention import (
    flex_cp_allgather,
)

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.models.glm5.model import Glm5Model
from torchtitan.tools.logging import logger

from .sharding import validate_glm5_parallelism

__all__ = [
    "apply_glm5_cp_to_forward",
    "parallelize_glm5",
    "validate_glm5_parallelism",
]


def _all_gather_sequence_no_grad(
    tensor: torch.Tensor,
    cp_mesh: DeviceMesh,
    *,
    sequence_dim: int,
) -> torch.Tensor:
    """Gather a sequence-sharded tensor in CP rank order without autograd.

    This helper is for non-trainable metadata or the frozen DSA index path.
    Trainable K/V uses ``flex_cp_allgather`` below so backward returns the
    correct gradient shard instead of silently treating the gather as constant.
    """

    local_tensor = tensor.contiguous()
    gathered = [torch.empty_like(local_tensor) for _ in range(cp_mesh.size())]
    dist.all_gather(gathered, local_tensor, group=cp_mesh.get_group())
    return torch.cat(gathered, dim=sequence_dim)


def apply_glm5_cp_to_forward(model: Glm5Model, cp_mesh: DeviceMesh) -> None:
    """Give local GLM DSA queries access to the complete key sequence.

    TorchTitan shards tokens, positions, and labels before model forward. This
    wrapper gathers positions and builds the local-query/global-key dense mask
    inside GLM, so no model-specific behavior leaks into the shared Trainer or
    CP input utility. The DSA indexer then gathers its key projection before
    global top-k selection, and inner attention gathers K/V before applying
    those global indices. Contiguous CP shards are required so CP rank order
    remains the original global token order.

    This wrapper must be installed before ``Module.parallelize`` so TP's
    ``local_map`` captures the wrapped local-tensor functions.
    """

    # ``flex_cp_allgather`` accepts a process-group name rather than a
    # DeviceMesh. Resolve it once and close over it for every decoder layer.
    process_group_name = dist._get_process_group_name(cp_mesh.get_group())
    original_model_forward = model.forward
    original_get_attention_masks = model.get_attention_masks

    def defer_attention_mask_to_cp_forward(positions):
        return None

    model.get_attention_masks = defer_attention_mask_to_cp_forward

    @wraps(original_model_forward)
    def cp_model_forward(
        tokens,
        positions=None,
        attention_masks=None,
        *,
        _forward=original_model_forward,
    ):
        if positions is None:
            raise ValueError("GLM-5 Context Parallel requires explicit positions.")
        # PP may leave this rank with an embedding-only or output-only stage.
        # There is no attention computation on such a stage, so neither the
        # global positions nor a dense attention mask is needed.
        if attention_masks is None and len(model.layers) > 0:
            global_positions = _all_gather_sequence_no_grad(
                positions,
                cp_mesh,
                sequence_dim=0,
            )
            global_mask = original_get_attention_masks(global_positions)
            assert global_mask is not None
            local_query_len = positions.shape[-1]
            # CP uses contiguous sequence shards. Rank r owns query interval
            # [r * Q_local, (r + 1) * Q_local) but keys span the global K axis.
            query_start = cp_mesh.get_local_rank() * local_query_len
            attention_masks = global_mask[
                :, query_start : query_start + local_query_len, :
            ]
        return _forward(tokens, positions, attention_masks)

    model.forward = cp_model_forward

    for block in model.layers.values():
        topk = block.attention.indexer.topk
        original_topk_forward = topk.forward

        @wraps(original_topk_forward)
        def cp_topk_forward(
            q_QNH,
            k_KH,
            weights_QN,
            attention_mask_QK,
            *,
            _forward=original_topk_forward,
        ):
            # The frozen indexer must rank every global key, not merely this
            # rank's local CP shard; otherwise top-k semantics change with CP.
            global_k_KH = _all_gather_sequence_no_grad(
                k_KH,
                cp_mesh,
                sequence_dim=0,
            )
            return _forward(
                q_QNH,
                global_k_KH,
                weights_QN,
                attention_mask_QK,
            )

        topk.forward = cp_topk_forward

        inner_attention = block.attention.inner_attention
        original_inner_forward = inner_attention.forward

        @wraps(original_inner_forward)
        def cp_inner_forward(
            q_QNH,
            k_KNH,
            v_KNV,
            attention_masks,
            topk_indices_QS,
            *,
            scale,
            _forward=original_inner_forward,
        ):
            # Q stays local to this CP rank. K/V are reconstructed in global
            # token order, and autograd later reduces/slices their gradients
            # back to the owner ranks through the CP-aware primitive.
            global_k_KNH, global_v_KNV = flex_cp_allgather(
                k_KNH.contiguous(),
                v_KNV.contiguous(),
                0,
                process_group_name,
            )
            return _forward(
                q_QNH,
                global_k_KNH,
                global_v_KNV,
                attention_masks,
                topk_indices_QS,
                scale=scale,
            )

        inner_attention.forward = cp_inner_forward

    logger.info("Applied GLM-5 DSA Context Parallel forward wrapping")


def parallelize_glm5(
    model: Glm5Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> Glm5Model:
    """Apply CP/TP/EP, optional model transforms, and the FSDP wrap.

    GLM-5 only runs on the ``partial_dtensor`` SPMD backend (validate_glm5_parallelism
    rejects the rest). TP/EP use declarative sharding; CP follows TorchTitan's
    partial-DTensor wrapper model and gathers global DSA keys for local queries.
    """
    validate_glm5_parallelism(parallelism, parallel_dims)
    validate_glm5_index_sharing(model)

    # Install transforms from logical-token behavior toward parameter wrapping.
    # Reordering these branches can change which callable local_map/compile sees.
    if parallel_dims.cp_enabled:
        apply_glm5_cp_to_forward(model, parallel_dims.get_mesh("cp"))

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    # Data parallelism: DDP/HSDP when dp_replicate is active (mesh [dp_replicate,
    # fsdp]), otherwise FSDP over the shard-only mesh. With EP, expert params
    # live on the efsdp mesh [dp_replicate, efsdp] (experts borrow ranks from
    # dp_shard x cp x tp); dense params stay on the dp mesh.
    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=None,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model


def validate_glm5_index_sharing(model: Glm5Model) -> None:
    """Validate the shared-index payload required by a PP model chunk."""
    if not model.index_sources:
        return
    local_layers = {int(name) for name in model.layers}
    external_sources = {
        model.index_sources[layer]
        for layer in local_layers
        if model.index_sources[layer] not in local_layers
    }
    if len(external_sources) > 1:
        raise ValueError(
            "a GLM-5 PP stage can receive indices from only one source layer"
        )
