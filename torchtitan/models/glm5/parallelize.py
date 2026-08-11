# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

from .sharding import validate_glm5_parallelism

__all__ = ["parallelize_glm5", "validate_glm5_parallelism"]


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
    """Apply TP/EP sharding, optional model transforms, and the FSDP wrap.

    GLM-5 only runs on the ``default`` SPMD backend (validate_glm5_parallelism
    rejects the rest), so ``model.parallelize`` is always the declarative
    sharding path here. CP is blocked for GLM-5 (eager dense DSA attention has
    no inner_attention), so there is no ``apply_cp_to_forward`` step.
    """
    validate_glm5_parallelism(parallelism, parallel_dims)

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
