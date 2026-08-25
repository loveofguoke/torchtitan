# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def _validate_glm5_pp_index_stage(model: Glm5Model, *, pp_enabled: bool) -> None:
    if not pp_enabled or len(model.layers) == 0:
        return
    first_layer = next(iter(model.layers.values()))
    attention = first_layer.attention
    if attention.indexer is None:
        raise NotImplementedError(
            "a GLM-5 pipeline stage cannot start with a shared-index "
            f"layer {attention.layer_id}; its source layer "
            f"{attention.index_source_layer} is on an earlier stage. "
            "Choose PP boundaries that start every stage on a full-index layer."
        )


def _all_gather_sequence_no_grad(
    tensor: torch.Tensor,
    cp_mesh: DeviceMesh,
    *,
    sequence_dim: int,
) -> torch.Tensor:
    """Gather a sequence-sharded tensor in CP rank order without autograd."""

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
    global top-k selection, and SparseMLA gathers compressed KV before applying
    those global indices. Contiguous CP shards are required so CP rank order
    remains the original global token order.

    This wrapper must be installed before ``Module.parallelize`` so TP's
    ``local_map`` captures the wrapped local-tensor functions.
    """

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
            query_start = cp_mesh.get_local_rank() * local_query_len
            attention_masks = global_mask[
                :, query_start : query_start + local_query_len, :
            ]
        return _forward(tokens, positions, attention_masks)

    model.forward = cp_model_forward

    for block in model.layers.values():
        if block.attention.indexer is not None:
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
            kv_K1H,
            attention_masks_1QK,
            topk_indices_QS,
            *,
            scale,
            latent_dim,
            _forward=original_inner_forward,
        ):
            # PyTorch's CP autograd collective is a K/V pair API. SparseMLA
            # owns one compressed KV tensor, so pass the same tensor through
            # both slots and consume one result; the unused output contributes
            # a zero gradient while the used output retains reduce-scatter
            # backward semantics.
            global_kv_K1H, _ = flex_cp_allgather(
                kv_K1H.contiguous(),
                kv_K1H.contiguous(),
                0,
                process_group_name,
            )
            return _forward(
                q_QNH,
                global_kv_K1H,
                attention_masks_1QK,
                topk_indices_QS,
                scale=scale,
                latent_dim=latent_dim,
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

    _validate_glm5_pp_index_stage(model, pp_enabled=parallel_dims.pp_enabled)

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
