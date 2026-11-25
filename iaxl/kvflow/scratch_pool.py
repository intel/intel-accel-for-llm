# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixed-size pinned scratch pool: one [num_slots, SLOT_BYTES] uint8 tensor
(IAXL_SCRATCH_POOL_SIZE_GB), registered once as the slot pool `ScratchPool.NAME`.

Layers with different page sizes share it: a page takes cdiv(page_bytes, SLOT_BYTES)
consecutive slots, and its CPU block index is the first of them. The pages of one
allocation need not be adjacent, so they are cut page by page from the free ranges,
lowest address first. Released space is reusable at once, whatever order tasks finish
in; a free range shorter than a page stays unused until a neighbour is released.
No growth: allocate() returns None when the pages do not fit.

allocate()/release() are not thread-safe: call them from one thread only.
"""

import bisect
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .. import torch_ext as _iqt
from ..envs import envs

logger = logging.getLogger(__name__)

SLOT_BYTES = 16 * 1024


@dataclass
class ScratchAlloc:
    indices: List[int]  # first slot of every page: its CPU block index
    per_page: int       # slots per page


class ScratchPool:
    NAME = "scratch"

    def __init__(self, pin_memory: bool = True):
        num_slots = int(envs.IAXL_SCRATCH_POOL_SIZE_GB * 1024**3) // SLOT_BYTES
        self.pool = torch.empty((num_slots, SLOT_BYTES), dtype=torch.uint8, device="cpu", pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        _iqt.register_kv_pool_cpu_slots(self.NAME, self.pool)
        self._free: List[Tuple[int, int]] = [(0, num_slots)]  # (start, length), sorted, never adjacent
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
        if sum(length // per_page for _, length in self._free) < pages:
            return None
        indices: List[int] = []
        free: List[Tuple[int, int]] = []
        for start, length in self._free:
            take = min(length // per_page, pages - len(indices))
            indices.extend(range(start, start + take * per_page, per_page))
            if length > take * per_page:
                free.append((start + take * per_page, length - take * per_page))
        self._free = free
        self._used += pages * per_page
        self._allocate_count += 1
        return ScratchAlloc(indices, per_page)

    def release(self, alloc: ScratchAlloc):
        free = self._free
        for start in alloc.indices:
            end = start + alloc.per_page
            i = bisect.bisect(free, (start,))
            if i < len(free) and free[i][0] == end:  # merge the right neighbour
                end += free.pop(i)[1]
            if i > 0 and sum(free[i - 1]) == start:  # merge the left neighbour
                start = free[i - 1][0]
                free[i - 1] = (start, end - start)
            else:
                free.insert(i, (start, end - start))
        self._used -= len(alloc.indices) * alloc.per_page
        self._release_count += 1

    def available_count(self) -> int:
        return self.num_slots - self._used

    def total_bytes(self) -> int:
        return self.pool.nbytes

    def status(self) -> str:
        largest = max((length for _, length in self._free), default=0)
        return (f"ScratchPool: {self.num_slots} slots x {SLOT_BYTES} B, {self.available_count()} available "
                f"in {len(self._free)} ranges (largest {largest}), "
                f"alloc/release={self._allocate_count}/{self._release_count}, {self.pool.nbytes / 2**20:.2f} MB")
