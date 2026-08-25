# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Measured-cost pipeline stage partitioning prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineUnitProfile:
    name: str
    forward_ms: float
    backward_ms: float
    activation_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pipeline unit name must be non-empty")
        if self.forward_ms < 0 or self.backward_ms < 0:
            raise ValueError("pipeline unit times must be non-negative")
        if self.activation_bytes < 0:
            raise ValueError("activation bytes must be non-negative")

    @property
    def training_ms(self) -> float:
        return self.forward_ms + self.backward_ms


@dataclass(frozen=True)
class PipelineStage:
    stage_id: int
    units: tuple[PipelineUnitProfile, ...]

    @property
    def training_ms(self) -> float:
        return sum(unit.training_ms for unit in self.units)

    @property
    def activation_bytes(self) -> int:
        return sum(unit.activation_bytes for unit in self.units)


@dataclass(frozen=True)
class PipelinePartitionPlan:
    stages: tuple[PipelineStage, ...]

    @property
    def max_stage_ms(self) -> float:
        return max(stage.training_ms for stage in self.stages)

    @property
    def min_stage_ms(self) -> float:
        return min(stage.training_ms for stage in self.stages)


def plan_contiguous_pipeline_stages(
    units: Sequence[PipelineUnitProfile],
    *,
    num_stages: int,
) -> PipelinePartitionPlan:
    """Minimize the maximum measured stage cost with contiguous partitions."""

    if not units:
        raise ValueError("units must be non-empty")
    if num_stages <= 0 or num_stages > len(units):
        raise ValueError("num_stages must be in [1, len(units)]")

    count = len(units)
    prefix = [0.0]
    for unit in units:
        prefix.append(prefix[-1] + unit.training_ms)

    infinity = float("inf")
    cost = [[infinity] * (count + 1) for _ in range(num_stages + 1)]
    split = [[-1] * (count + 1) for _ in range(num_stages + 1)]
    cost[0][0] = 0.0
    for stages in range(1, num_stages + 1):
        for end in range(stages, count + 1):
            for start in range(stages - 1, end):
                candidate = max(cost[stages - 1][start], prefix[end] - prefix[start])
                if candidate < cost[stages][end]:
                    cost[stages][end] = candidate
                    split[stages][end] = start

    boundaries = [count]
    end = count
    for stages in range(num_stages, 0, -1):
        end = split[stages][end]
        boundaries.append(end)
    boundaries.reverse()

    planned = tuple(
        PipelineStage(stage_id, tuple(units[start:end]))
        for stage_id, (start, end) in enumerate(
            zip(boundaries, boundaries[1:])
        )
    )
    return PipelinePartitionPlan(planned)


def ideal_non_interleaved_1f1b_bubble_fraction(
    *,
    num_stages: int,
    num_microbatches: int,
) -> float:
    """Return the ideal fill/drain bubble fraction for balanced 1F1B."""

    if num_stages <= 0 or num_microbatches <= 0:
        raise ValueError("num_stages and num_microbatches must be positive")
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


__all__ = [
    "PipelinePartitionPlan",
    "PipelineStage",
    "PipelineUnitProfile",
    "ideal_non_interleaved_1f1b_bubble_fraction",
    "plan_contiguous_pipeline_stages",
]
