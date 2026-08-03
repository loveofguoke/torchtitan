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
    """Reject every runtime layout that needs more than one rank.

    GLM-5's debug eager DSA path is intentionally single-device only.  A
    ``data_parallel_shard_degree`` of ``-1`` is unresolved until the trainer
    constructs ``ParallelDims``; it is therefore accepted only while
    ``parallel_dims`` is absent.
    """
    unsupported: list[str] = []
    degree_names = (
        ("TP", parallelism.tensor_parallel_degree),
        ("CP", parallelism.context_parallel_degree),
        ("PP", parallelism.pipeline_parallel_degree),
        ("EP", parallelism.expert_parallel_degree),
        ("DP replicate", parallelism.data_parallel_replicate_degree),
        ("DP shard", parallelism.data_parallel_shard_degree),
    )
    unsupported.extend(name for name, degree in degree_names if degree > 1)

    if parallelism.spmd_backend != "default":
        unsupported.append(f"SPMD backend ({parallelism.spmd_backend})")

    if parallel_dims is not None:
        if parallel_dims.dp_shard > 1 and "DP shard" not in unsupported:
            unsupported.append("DP shard")
        if parallel_dims.world_size != 1:
            unsupported.append(f"world_size ({parallel_dims.world_size})")

    if unsupported:
        modes = ", ".join(unsupported)
        raise NotImplementedError(
            "GLM-5 debugmodel supports only a single-device runtime; "
            f"unsupported parallelism: {modes}."
        )
