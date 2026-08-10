# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims


def validate_glm5_parallelism(
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims | None = None,
) -> None:
    """Reject every runtime layout that is not pure data parallelism.

    GLM-5's debug eager DSA path supports data parallelism (DDP/HSDP via
    ``data_parallel_replicate_degree``, and FSDP via
    ``data_parallel_shard_degree``) but rejects tensor/context/pipeline/expert
    parallelism and any non-default SPMD backend.  ``parallel_dims`` is kept
    for signature parity with the runtime call site: once resolved,
    ``ParallelDims._validate`` has already enforced
    ``dp_replicate * dp_shard * cp * tp * pp == world_size``, so a resolved
    multi-rank world is by construction a legal data-parallel layout.
    """
    unsupported: list[str] = []
    degree_names = (
        ("TP", parallelism.tensor_parallel_degree),
        ("CP", parallelism.context_parallel_degree),
        ("PP", parallelism.pipeline_parallel_degree),
        ("EP", parallelism.expert_parallel_degree),
    )
    unsupported.extend(name for name, degree in degree_names if degree > 1)

    if parallelism.spmd_backend != "default":
        unsupported.append(f"SPMD backend ({parallelism.spmd_backend})")

    if unsupported:
        modes = ", ".join(unsupported)
        raise NotImplementedError(
            "GLM-5 debugmodel supports only data-parallel layouts; "
            f"unsupported parallelism: {modes}."
        )
