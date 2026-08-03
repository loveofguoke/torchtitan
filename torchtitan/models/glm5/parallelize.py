# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
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
    """Apply only single-device-safe optional model transforms."""
    del training
    validate_glm5_parallelism(parallelism, parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)
    return model
