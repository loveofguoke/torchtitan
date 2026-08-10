# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
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
    """Apply optional model transforms and the data-parallel wrap."""
    validate_glm5_parallelism(parallelism, parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

    # Data parallelism: DDP/HSDP when dp_replicate is active (mesh [dp_replicate,
    # fsdp]), otherwise FSDP over the shard-only mesh.
    names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(names)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=None,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model
