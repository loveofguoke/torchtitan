# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Contiguous FSDP layer-group planning prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ShardedLayerProfile:
    name: str
    parameter_numel: int
    unsharded_bytes: int
    policy_key: str = "default"

    def __post_init__(self) -> None:
        if not self.name or not self.policy_key:
            raise ValueError("layer name and policy key must be non-empty")
        if self.parameter_numel <= 0 or self.unsharded_bytes <= 0:
            raise ValueError("layer size estimates must be positive")


@dataclass(frozen=True)
class FSDPLayerGroup:
    group_id: int
    layers: tuple[ShardedLayerProfile, ...]

    @property
    def parameter_numel(self) -> int:
        return sum(layer.parameter_numel for layer in self.layers)

    @property
    def estimated_peak_unsharded_bytes(self) -> int:
        return sum(layer.unsharded_bytes for layer in self.layers)


@dataclass(frozen=True)
class FSDPGroupingPlan:
    groups: tuple[FSDPLayerGroup, ...]
    baseline_data_collectives: int
    grouped_data_collectives: int

    @property
    def removable_data_collectives(self) -> int:
        return self.baseline_data_collectives - self.grouped_data_collectives


def plan_fsdp_layer_groups(
    layers: Sequence[ShardedLayerProfile],
    *,
    layers_per_group: int,
) -> FSDPGroupingPlan:
    """Group adjacent compatible layers without crossing policy boundaries.

    The estimate assumes one parameter AllGather and one gradient
    ReduceScatter per FSDP unit. It intentionally ignores prefetch/reuse and is
    therefore a comparison aid, not a claim about runtime collective counts.
    """

    if not layers:
        raise ValueError("layers must be non-empty")
    if layers_per_group <= 0:
        raise ValueError("layers_per_group must be positive")

    groups: list[FSDPLayerGroup] = []
    current: list[ShardedLayerProfile] = []
    for layer in layers:
        if current and (
            len(current) >= layers_per_group
            or current[-1].policy_key != layer.policy_key
        ):
            groups.append(FSDPLayerGroup(len(groups), tuple(current)))
            current = []
        current.append(layer)
    if current:
        groups.append(FSDPLayerGroup(len(groups), tuple(current)))

    return FSDPGroupingPlan(
        tuple(groups),
        baseline_data_collectives=2 * len(layers),
        grouped_data_collectives=2 * len(groups),
    )


__all__ = [
    "FSDPGroupingPlan",
    "FSDPLayerGroup",
    "ShardedLayerProfile",
    "plan_fsdp_layer_groups",
]
