# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan/Megatron-inspired selective recomputation cost model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecomputeDecision(str, Enum):
    SAVE = "save"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class CandidateActivation:
    name: str
    kind: str
    output_bytes: int
    recompute_cost: float

    def __post_init__(self) -> None:
        if self.output_bytes < 0 or self.recompute_cost < 0:
            raise ValueError(
                "activation size and recompute cost must be non-negative"
            )


def plan_selective_recompute(
    activations: tuple[CandidateActivation, ...],
    *,
    save_budget_bytes: int,
) -> dict[str, RecomputeDecision]:
    """Save discrete/communication outputs, then expensive values per byte."""

    if save_budget_bytes < 0:
        raise ValueError("save_budget_bytes must be non-negative")
    if len({activation.name for activation in activations}) != len(activations):
        raise ValueError("activation names must be unique")

    decisions = {
        activation.name: RecomputeDecision.RECOMPUTE
        for activation in activations
    }
    used = 0
    mandatory = tuple(
        activation
        for activation in activations
        if activation.kind in {"discrete", "communication"}
    )
    for activation in mandatory:
        if used + activation.output_bytes > save_budget_bytes:
            raise ValueError("save budget cannot hold mandatory control-flow outputs")
        decisions[activation.name] = RecomputeDecision.SAVE
        used += activation.output_bytes

    optional = sorted(
        (
            activation
            for activation in activations
            if activation.kind not in {"discrete", "communication"}
        ),
        key=lambda activation: (
            activation.recompute_cost / max(activation.output_bytes, 1),
            activation.recompute_cost,
        ),
        reverse=True,
    )
    for activation in optional:
        if used + activation.output_bytes <= save_budget_bytes:
            decisions[activation.name] = RecomputeDecision.SAVE
            used += activation.output_bytes
    return decisions


__all__ = [
    "CandidateActivation",
    "RecomputeDecision",
    "plan_selective_recompute",
]
