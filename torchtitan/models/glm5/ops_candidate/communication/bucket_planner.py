# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Megatron-inspired parameter/gradient bucket planning prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSpan:
    name: str
    start: int
    end: int

    @property
    def numel(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CommunicationBucket:
    bucket_id: int
    start: int
    end: int
    unpadded_end: int
    parameters: tuple[ParameterSpan, ...]

    @property
    def padded_numel(self) -> int:
        return self.end - self.start


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def plan_communication_buckets(
    parameters: Sequence[tuple[str, int]],
    *,
    target_numel: int,
    alignment: int,
    reverse_registration_order: bool = True,
) -> tuple[CommunicationBucket, ...]:
    """Pack parameters into aligned buckets for ready-order communication.

    Reverse order approximates backward gradient readiness for a sequential
    transformer. A real integration must use actual module execution order and
    alias information rather than assuming the model is strictly sequential.
    """

    if target_numel <= 0 or alignment <= 0:
        raise ValueError("target_numel and alignment must be positive")
    if any(numel <= 0 for _, numel in parameters):
        raise ValueError("parameter sizes must be positive")

    ordered = (
        list(reversed(parameters))
        if reverse_registration_order
        else list(parameters)
    )
    buckets: list[CommunicationBucket] = []
    current: list[ParameterSpan] = []
    offset = 0
    bucket_start = 0

    def flush() -> None:
        nonlocal bucket_start, offset, current
        if not current:
            return
        unpadded_end = offset
        offset = _align(offset, alignment)
        buckets.append(
            CommunicationBucket(
                bucket_id=len(buckets),
                start=bucket_start,
                end=offset,
                unpadded_end=unpadded_end,
                parameters=tuple(current),
            )
        )
        bucket_start = offset
        current = []

    for name, numel in ordered:
        if current and offset - bucket_start + numel > target_numel:
            flush()
        start = offset
        offset += numel
        current.append(ParameterSpan(name, start, offset))
    flush()
    return tuple(buckets)


__all__ = [
    "CommunicationBucket",
    "ParameterSpan",
    "plan_communication_buckets",
]
