# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""vLLM-inspired paged KV block-table prototype for future GLM-5 serving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CacheBlock:
    block_id: int
    ref_count: int = 0


@dataclass
class SequenceBlocks:
    token_count: int = 0
    block_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class BlockCopy:
    source_block: int
    target_block: int
    num_tokens: int


@dataclass(frozen=True)
class AllocationPlan:
    block_ids: tuple[int, ...]
    copies: tuple[BlockCopy, ...]


class PagedKVBlockTable:
    """Map logical sequence tokens to fixed-size physical KV blocks."""

    def __init__(self, *, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self._blocks = [CacheBlock(block_id) for block_id in range(num_blocks)]
        self._free = deque(range(num_blocks))
        self._sequences: dict[str, SequenceBlocks] = {}

    def _take_free_block(self) -> int:
        block_id = self._free.popleft()
        self._blocks[block_id].ref_count = 1
        return block_id

    def allocate_tokens(
        self,
        sequence_id: str,
        num_new_tokens: int,
    ) -> AllocationPlan:
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be non-negative")
        sequence = self._sequences.setdefault(sequence_id, SequenceBlocks())
        partial_tokens = sequence.token_count % self.block_size
        needs_copy = (
            num_new_tokens > 0
            and partial_tokens > 0
            and self._blocks[sequence.block_ids[-1]].ref_count > 1
        )
        required = (
            sequence.token_count + num_new_tokens + self.block_size - 1
        ) // self.block_size
        missing = required - len(sequence.block_ids)
        if missing + int(needs_copy) > len(self._free):
            raise RuntimeError("paged KV cache is out of free blocks")

        copies: list[BlockCopy] = []
        if needs_copy:
            source_block = sequence.block_ids[-1]
            target_block = self._take_free_block()
            self._blocks[source_block].ref_count -= 1
            sequence.block_ids[-1] = target_block
            copies.append(BlockCopy(source_block, target_block, partial_tokens))

        for _ in range(missing):
            sequence.block_ids.append(self._take_free_block())
        sequence.token_count += num_new_tokens
        return AllocationPlan(tuple(sequence.block_ids), tuple(copies))

    def fork(self, source_id: str, target_id: str) -> tuple[int, ...]:
        if target_id in self._sequences:
            raise ValueError(f"target sequence already exists: {target_id}")
        source = self._sequences[source_id]
        copied = SequenceBlocks(source.token_count, list(source.block_ids))
        self._sequences[target_id] = copied
        for block_id in copied.block_ids:
            self._blocks[block_id].ref_count += 1
        return tuple(copied.block_ids)

    def free(self, sequence_id: str) -> None:
        sequence = self._sequences.pop(sequence_id)
        for block_id in sequence.block_ids:
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free.append(block_id)

    def slot_mapping(self, sequence_id: str) -> tuple[int, ...]:
        sequence = self._sequences[sequence_id]
        return tuple(
            sequence.block_ids[token // self.block_size] * self.block_size
            + token % self.block_size
            for token in range(sequence.token_count)
        )

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)


__all__ = [
    "AllocationPlan",
    "BlockCopy",
    "CacheBlock",
    "PagedKVBlockTable",
    "SequenceBlocks",
]
