# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixed-size pinned scratch pool: one [num_slots, SLOT_BYTES] uint8 tensor
(IAXL_SCRATCH_POOL_SIZE_GB), registered once as the slot pool `ScratchPool.NAME`.

Layers with different page sizes share it: a page takes cdiv(page_bytes, SLOT_BYTES)
consecutive slots, and its CPU block index is the first of them. Space is handed out
as a ring: one allocation is contiguous and never wraps, and space comes back in
allocation order, so release() of a later allocation waits for the earlier ones.
No growth: allocate() returns None once the ring is full.

allocate()/release() are not thread-safe: call them from one thread only.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import torch

from .. import torch_ext as _iqt
from ..envs import envs

logger = logging.getLogger(__name__)

SLOT_BYTES = 16 * 1024


@dataclass
class ScratchAlloc:
    start: int     # first slot
    slots: int     # slots held, including the ring tail skipped to keep this one contiguous
    per_page: int  # slots per page
    pages: int
    done: bool = False

    @property
    def indices(self) -> List[int]:
        """CPU block index of every page: its first slot."""
        return [self.start + i * self.per_page for i in range(self.pages)]


class ScratchPool:
    NAME = "scratch"

    def __init__(self, pin_memory: bool = True):
        num_slots = int(envs.IAXL_SCRATCH_POOL_SIZE_GB * 1024**3) // SLOT_BYTES
        self.pool = torch.empty((num_slots, SLOT_BYTES), dtype=torch.uint8, device="cpu", pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        _iqt.register_kv_pool_cpu_slots(self.NAME, self.pool)
        self._live: Deque[ScratchAlloc] = deque()  # allocation order
        self._head = 0  # next free slot
        self._used = 0
        self._allocate_count = 0
        self._release_count = 0
        logger.info("ScratchPool: %d slots x %d B = %.2f MB pinned",
                    num_slots, SLOT_BYTES, self.pool.nbytes / 2**20)

    def close(self):
        _iqt.unregister_kv_pool(self.NAME)

    @property
    def num_slots(self) -> int:
        return self.pool.shape[0]

    def allocate(self, pages: int, page_bytes: int) -> Optional[ScratchAlloc]:
        per_page = -(-page_bytes // SLOT_BYTES)
        need = pages * per_page
        start, slots = self._head, need
        if start + need > self.num_slots:  # skip the tail, start over at slot 0
            start, slots = 0, need + self.num_slots - self._head
        if self._used + slots > self.num_slots:
            return None
        alloc = ScratchAlloc(start, slots, per_page, pages)
        self._live.append(alloc)
        self._head = (start + need) % self.num_slots
        self._used += slots
        self._allocate_count += 1
        return alloc

    def release(self, alloc: ScratchAlloc):
        alloc.done = True
        self._release_count += 1
        live = self._live
        while live and live[0].done:
            self._used -= live.popleft().slots
        if not live:
            self._head = 0

    def available_count(self) -> int:
        return self.num_slots - self._used

    def total_bytes(self) -> int:
        return self.pool.nbytes

    def status(self) -> str:
        return (f"ScratchPool: {self.num_slots} slots x {SLOT_BYTES} B, {self.available_count()} available, "
                f"{len(self._live)} allocations live, alloc/release={self._allocate_count}/{self._release_count}, "
                f"{self.pool.nbytes / 2**20:.2f} MB")
