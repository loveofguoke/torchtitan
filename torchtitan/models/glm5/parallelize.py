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
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
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
) -> torch.Tensor:
    """Gather a sequence-sharded tensor in CP rank order without autograd."""

    local_tensor = tensor.contiguous()
    gathered = [torch.empty_like(local_tensor) for _ in range(cp_mesh.size())]
    dist.all_gather(gathered, local_tensor, group=cp_mesh.get_group())
    return torch.cat(gathered, dim=1)


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

    process_group_name = dist._get_process_group_name(cp_mesh.get_group())
    original_model_forward = model.forward

    @wraps(original_model_forward)
    def cp_model_forward(
        tokens_BL,
        positions=None,
        attention_masks=None,
        *,
        _forward=original_model_forward,
    ):
        if positions is None:
            raise ValueError("GLM-5 Context Parallel requires explicit positions.")
        if attention_masks is None:
            global_positions = _all_gather_sequence_no_grad(positions, cp_mesh)
            global_mask = model.get_attention_masks(global_positions)
            local_query_len = positions.shape[1]
            query_start = cp_mesh.get_local_rank() * local_query_len
            attention_masks = global_mask[
                :,
                :,
                query_start : query_start + local_query_len,
                :,
            ]
        return _forward(tokens_BL, positions, attention_masks)

    model.forward = cp_model_forward

    for block in model.layers.values():
        topk = block.attention.indexer.topk
        original_topk_forward = topk.forward

        @wraps(original_topk_forward)
        def cp_topk_forward(
            q_BQNH,
            k_BKH,
            weights_BQN,
            attention_mask_BQK,
            *,
            _forward=original_topk_forward,
        ):
            global_k_BKH = _all_gather_sequence_no_grad(k_BKH, cp_mesh)
            return _forward(
                q_BQNH,
                global_k_BKH,
                weights_BQN,
                attention_mask_BQK,
            )

        topk.forward = cp_topk_forward

        inner_attention = block.attention.inner_attention
        original_inner_forward = inner_attention.forward

        @wraps(original_inner_forward)
        def cp_inner_forward(
            q_BQNH,
            k_BKNH,
            v_BKNV,
            attention_masks_B1QK,
            topk_indices_BQT,
            *,
            scale,
            _forward=original_inner_forward,
        ):
            global_k_BKNH, global_v_BKNV = flex_cp_allgather(
                k_BKNH.contiguous(),
                v_BKNV.contiguous(),
                1,
                process_group_name,
            )
            return _forward(
                q_BQNH,
                global_k_BKNH,
                global_v_BKNV,
                attention_masks_B1QK,
                topk_indices_BQT,
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

    GLM-5 only runs on the ``default`` SPMD backend (validate_glm5_parallelism
    rejects the rest). TP/EP use declarative sharding; CP follows TorchTitan's
    default-backend wrapper model and gathers global DSA keys for local queries.
    """
    validate_glm5_parallelism(parallelism, parallel_dims)

    if parallel_dims.cp_enabled:
        apply_glm5_cp_to_forward(model, parallel_dims.get_mesh("cp"))

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

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
