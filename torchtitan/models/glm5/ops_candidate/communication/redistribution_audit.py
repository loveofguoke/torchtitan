# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Audit prototype for multi-axis Partial-to-Replicate transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshAxis:
    name: str
    size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mesh axis name must be non-empty")
        if self.size <= 0:
            raise ValueError("mesh axis size must be positive")


@dataclass(frozen=True)
class RedistributionTransition:
    tensor_name: str
    partial_axes: tuple[MeshAxis, ...]
    replicate_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        partial_names = tuple(axis.name for axis in self.partial_axes)
        if not self.tensor_name:
            raise ValueError("tensor name must be non-empty")
        if len(set(partial_names)) != len(partial_names):
            raise ValueError("partial mesh axes must be unique")
        if len(set(self.replicate_axes)) != len(self.replicate_axes):
            raise ValueError("target replicate axes must be unique")


@dataclass(frozen=True)
class RedistributionAudit:
    tensor_name: str
    current_all_reduce_count: int
    flattened_all_reduce_count: int
    removable_launches: int
    flatten_candidate: bool
    reason: str


def audit_partial_to_replicate(
    transition: RedistributionTransition,
) -> RedistributionAudit:
    """Estimate whether one flattened reduction can replace axis reductions.

    This function does not mutate a DeviceMesh or execute a collective. It
    only identifies the exact transition that needs a DTensor/SPMD trace and
    numerical proof before a flattened process group can be considered.
    """

    active_partial_axes = tuple(
        axis for axis in transition.partial_axes if axis.size > 1
    )
    active_names = {axis.name for axis in active_partial_axes}
    target_names = set(transition.replicate_axes)
    current_count = len(active_partial_axes)

    if current_count < 2:
        return RedistributionAudit(
            transition.tensor_name,
            current_count,
            current_count,
            0,
            False,
            "fewer than two non-unit partial axes",
        )
    if not active_names.issubset(target_names):
        return RedistributionAudit(
            transition.tensor_name,
            current_count,
            current_count,
            0,
            False,
            "the target does not replicate every active partial axis",
        )

    return RedistributionAudit(
        transition.tensor_name,
        current_count,
        1,
        current_count - 1,
        True,
        "all active partial axes become replicated at one dependency point",
    )


__all__ = [
    "MeshAxis",
    "RedistributionAudit",
    "RedistributionTransition",
    "audit_partial_to_replicate",
]
