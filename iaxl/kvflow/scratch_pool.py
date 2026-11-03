# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixed-size pinned scratch pool: one big [num_blocks, *block_shape] tensor
(IAXL_SCRATCH_POOL_SIZE_GB), registered once as the KV pool `ScratchPool.NAME`.
Blocks are handed out as indices into it. No growth: allocate() returns None
once the pool is exhausted.

allocate()/release() are not thread-safe: call them from one thread only.
"""

import logging
import math
from typing import List, Optional, Sequence

import torch

from .. import torch_ext as _iqt
from ..envs import envs

logger = logging.getLogger(__name__)


class ScratchPool:
    NAME = "scratch"

    def __init__(self, block_shape: Sequence[int], dtype: torch.dtype, pin_memory: bool = True):
        self.block_shape = tuple(block_shape)
        self.dtype = dtype
        self.block_bytes = math.prod(self.block_shape) * torch.tensor([], dtype=dtype).element_size()
        num_blocks = int(envs.IAXL_SCRATCH_POOL_SIZE_GB * 1024**3) // self.block_bytes
        self.pool = torch.empty((num_blocks, *self.block_shape), dtype=dtype, device="cpu", pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        _iqt.register_kv_pool_cpu(self.NAME, self.pool)
        self._free: List[int] = list(range(num_blocks))  # LIFO: pop from the end
        self._allocate_count = 0
        self._release_count = 0
        logger.info("ScratchPool: %d x %s %s = %.2f MB pinned",
                    num_blocks, self.block_shape, dtype, self.pool.nbytes / 2**20)

    def close(self):
        _iqt.unregister_kv_pool(self.NAME)

    @property
    def num_blocks(self) -> int:
        return self.pool.shape[0]

    def block(self, idx: int) -> torch.Tensor:
        return _iqt.block_tensor(self.pool, self.NAME, idx)

    def allocate(self, count: int, shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None) -> Optional[List[int]]:
        free = self._free
        if count > len(free):
            return None
        cut = len(free) - count  # not free[-count:], which is the whole list for count == 0
        idx = free[cut:]
        del free[cut:]
        self._allocate_count += count
        return idx

    def release(self, indices: List[int]):
        self._free.extend(indices)
        self._release_count += len(indices)

    def available_count(self, shape: Optional[Sequence[int]] = None, dtype: Optional[torch.dtype] = None) -> int:
        return len(self._free)

    def total_count(self, shape: Optional[Sequence[int]] = None, dtype: Optional[torch.dtype] = None) -> int:
        return self.num_blocks

    def total_bytes(self) -> int:
        return self.pool.nbytes

    def status(self) -> str:
        free, in_use = len(self._free), self._allocate_count - self._release_count
        return (f"ScratchPool: {self.num_blocks} blocks {self.block_shape} {self.dtype}, "
                f"{free} available, {in_use} in-use, alloc/release={self._allocate_count}/{self._release_count}, "
                f"{self.pool.nbytes / 2**20:.2f} MB")
