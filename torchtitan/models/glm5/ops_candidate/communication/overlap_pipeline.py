# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Standalone scheduling prototype for chunked communication overlap.

This module extracts the dispatch/compute/combine dependency pattern used by
Megatron-style MoE overlap and TorchTitan async EP experiments. It deliberately
knows nothing about GLM-5 modules, process groups, or NCCL/HCCL.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


ChunkT = TypeVar("ChunkT")
DispatchedT = TypeVar("DispatchedT")
ComputedT = TypeVar("ComputedT")
CombinedT = TypeVar("CombinedT")
OutputT = TypeVar("OutputT")


class Waitable(Protocol[DispatchedT]):
    def wait(self) -> DispatchedT: ...


@dataclass(frozen=True)
class OverlapEvent:
    chunk_id: int
    phase: str


@dataclass(frozen=True)
class OverlapResult(Generic[OutputT]):
    output: OutputT
    events: tuple[OverlapEvent, ...]


def run_chunked_overlap(
    chunks: Sequence[ChunkT],
    *,
    issue_dispatch: Callable[[int, ChunkT], Waitable[DispatchedT]],
    compute: Callable[[int, DispatchedT], ComputedT],
    issue_combine: Callable[[int, ComputedT], Waitable[CombinedT]],
    merge: Callable[[Sequence[CombinedT]], OutputT],
) -> OverlapResult[OutputT]:
    """Pipeline dispatch(i+1), compute(i), and combine(i-1).

    The callbacks own streams, process groups, buffers, and autograd. This
    function only makes wait points explicit so an integration cannot consume
    a buffer before its producer completes.
    """

    if not chunks:
        raise ValueError("chunks must be non-empty")

    events: list[OverlapEvent] = []
    dispatch = issue_dispatch(0, chunks[0])
    events.append(OverlapEvent(0, "dispatch-issued"))
    combines: list[Waitable[CombinedT]] = []

    for chunk_id in range(len(chunks)):
        next_dispatch = None
        if chunk_id + 1 < len(chunks):
            next_dispatch = issue_dispatch(chunk_id + 1, chunks[chunk_id + 1])
            events.append(OverlapEvent(chunk_id + 1, "dispatch-issued"))

        dispatched = dispatch.wait()
        events.append(OverlapEvent(chunk_id, "dispatch-ready"))
        computed = compute(chunk_id, dispatched)
        events.append(OverlapEvent(chunk_id, "compute-complete"))
        combines.append(issue_combine(chunk_id, computed))
        events.append(OverlapEvent(chunk_id, "combine-issued"))

        if next_dispatch is not None:
            dispatch = next_dispatch

    combined = []
    for chunk_id, handle in enumerate(combines):
        combined.append(handle.wait())
        events.append(OverlapEvent(chunk_id, "combine-ready"))
    return OverlapResult(merge(combined), tuple(events))


__all__ = [
    "OverlapEvent",
    "OverlapResult",
    "Waitable",
    "run_chunked_overlap",
]
