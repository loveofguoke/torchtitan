# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Bounded stream-local workspace pool inspired by TorchTitan linear fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from threading import Lock

import torch


def _stream_token(device: torch.device) -> int:
    backend = getattr(torch, device.type, None)
    if backend is None or not hasattr(backend, "current_stream"):
        return 0
    stream = backend.current_stream(device)
    for attribute in ("cuda_stream", "npu_stream", "stream_id"):
        value = getattr(stream, attribute, None)
        if value is not None:
            return int(value)
    return id(stream)


class TensorWorkspacePool:
    """Reuse exact-shape tensors on the same device stream.

    Releasing a tensor is safe for later work enqueued on the same stream: the
    allocator-visible reuse remains ordered after the previous consumer. A
    stream token is part of the key, so this class never aliases workspaces
    across streams. It is intentionally for eager experiments and is not used
    by the readable GLM-5 model or graph capture.
    """

    def __init__(self, *, max_cached_bytes: int = 512 * 1024 * 1024) -> None:
        if max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be non-negative")
        self.max_cached_bytes = max_cached_bytes
        self._cached_bytes = 0
        self._buffers: dict[tuple[object, ...], list[torch.Tensor]] = defaultdict(
            list
        )
        self._lock = Lock()

    @staticmethod
    def _key(
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[object, ...]:
        return (
            device.type,
            device.index,
            dtype,
            tuple(shape),
            _stream_token(device),
        )

    def acquire(
        self,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = self._key(shape, dtype=dtype, device=device)
        with self._lock:
            if self._buffers[key]:
                tensor = self._buffers[key].pop()
                self._cached_bytes -= tensor.numel() * tensor.element_size()
                return tensor
        return torch.empty(tuple(shape), dtype=dtype, device=device)

    def release(self, tensor: torch.Tensor) -> None:
        if tensor.requires_grad or not tensor.is_contiguous():
            return
        size = tensor.numel() * tensor.element_size()
        if size > self.max_cached_bytes:
            return
        key = self._key(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        with self._lock:
            if self._cached_bytes + size > self.max_cached_bytes:
                return
            self._buffers[key].append(tensor)
            self._cached_bytes += size

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._cached_bytes = 0

    @property
    def cached_bytes(self) -> int:
        with self._lock:
            return self._cached_bytes


__all__ = ["TensorWorkspacePool"]
