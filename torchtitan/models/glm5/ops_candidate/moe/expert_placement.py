# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Measured expert-load placement prototype for EP experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertLoad:
    expert_id: int
    token_count: int

    def __post_init__(self) -> None:
        if self.expert_id < 0 or self.token_count < 0:
            raise ValueError("expert id and token count must be non-negative")


@dataclass(frozen=True)
class ExpertRankAssignment:
    rank: int
    expert_ids: tuple[int, ...]
    predicted_tokens: int


@dataclass(frozen=True)
class ExpertPlacementPlan:
    assignments: tuple[ExpertRankAssignment, ...]

    @property
    def imbalance_ratio(self) -> float:
        loads = tuple(item.predicted_tokens for item in self.assignments)
        mean = sum(loads) / len(loads)
        return max(loads) / mean if mean else 1.0


def plan_expert_rank_placement(
    experts: Sequence[ExpertLoad],
    *,
    num_ranks: int,
) -> ExpertPlacementPlan:
    """Greedily place measured hot experts on the least-loaded rank.

    The result is an offline what-if plan. Changing expert ownership during
    training also changes checkpoint mapping, process groups, and optimizer
    state, so this function must never be used as a dynamic runtime router.
    """

    if num_ranks <= 0:
        raise ValueError("num_ranks must be positive")
    if not experts:
        raise ValueError("experts must be non-empty")
    if num_ranks > len(experts):
        raise ValueError("num_ranks cannot exceed the number of experts")
    if len({expert.expert_id for expert in experts}) != len(experts):
        raise ValueError("expert ids must be unique")

    rank_experts: list[list[int]] = [[] for _ in range(num_ranks)]
    rank_loads = [0] * num_ranks
    for expert in sorted(
        experts,
        key=lambda item: (item.token_count, item.expert_id),
        reverse=True,
    ):
        rank = min(range(num_ranks), key=lambda item: (rank_loads[item], item))
        rank_experts[rank].append(expert.expert_id)
        rank_loads[rank] += expert.token_count

    assignments = tuple(
        ExpertRankAssignment(rank, tuple(sorted(rank_experts[rank])), rank_loads[rank])
        for rank in range(num_ranks)
    )
    return ExpertPlacementPlan(assignments)


__all__ = [
    "ExpertLoad",
    "ExpertPlacementPlan",
    "ExpertRankAssignment",
    "plan_expert_rank_placement",
]
