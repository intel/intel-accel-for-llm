# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixed-size pinned scratch pools, one per distinct block geometry.

Layers of a mixed-page model (MLA, indexer, mamba) need different page sizes.
A single max-page pool wastes memory: a 41.25 KB indexer page would sit in an
820 KB slot. This class keeps one pool per (page bytes, dtype) so each layer
stages through a slot sized for it.

Capacity is split by layer count: every layer gets the same number of blocks,
so a class's bytes stay proportional to ``layers * page_bytes``. A class only
serves its own geometry: two layers of the same geometry share one class, and a
dry class waits for in-flight transfers instead of taking another class's slots.

allocate()/release() are not thread-safe: call them from one thread only.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .. import torch_ext as _iqt
from ..envs import envs

logger = logging.getLogger(__name__)

PageKey = Tuple[int, torch.dtype]


class _Class:
    """One page geometry: its pinned blocks and their free list."""

    def __init__(self, name: str, class_id: int, block_bytes: int, dtype: torch.dtype,
                 shape: Sequence[int], num_blocks: int, pin_memory: bool):
        self.name = name
        self.base = class_id << 32  # global index = base | local index
        self.block_bytes = block_bytes
        self.dtype = dtype
        self.shape = tuple(shape)
        self.pool = torch.empty((num_blocks, *self.shape), dtype=dtype, device="cpu",
                                pin_memory=pin_memory)
        assert not pin_memory or self.pool.is_pinned(), "ScratchPool: pin_memory=True did not take effect"
        _iqt.register_kv_pool_cpu(self.name, self.pool)
        self._free: List[int] = list(range(num_blocks))  # LIFO: pop from the end
        self._allocate_count = 0
        self._release_count = 0

    @property
    def num_blocks(self) -> int:
        return self.pool.shape[0]

    def free(self) -> int:
        return len(self._free)

    def take(self, count: int) -> Optional[List[int]]:
        free = self._free
        if count > len(free):
            return None
        cut = len(free) - count  # not free[-count:], which is the whole list for count == 0
        idx = free[cut:]
        del free[cut:]
        self._allocate_count += count
        return idx

    def give(self, indices: List[int]) -> None:
        self._free.extend(indices)
        self._release_count += len(indices)

    def close(self) -> None:
        _iqt.unregister_kv_pool(self.name)


class ScratchPool:
    NAME = "scratch"

    def __init__(self, geometries: Sequence[Tuple[Sequence[int], torch.dtype]],
                 total_bytes: int, pin_memory: bool = True):
        groups: Dict[PageKey, List[Sequence[int]]] = {}
        for shape, dtype in geometries:
            page = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            groups.setdefault((page, dtype), []).append(shape)

        # Every layer gets the same number of blocks; the bytes then follow
        # layers * page_bytes. The largest class keeps the leftover.
        order = sorted(groups, key=lambda key: key[0])
        per_layer = max(1, total_bytes // sum(page * len(v) for (page, _), v in groups.items()))
        counts = {key: max(1, per_layer * len(v)) for key, v in groups.items()}
        while sum(counts[key] * key[0] for key in order) > total_bytes:
            biggest = max((k for k in order if counts[k] > 1),
                          key=lambda k: counts[k] * k[0], default=None)
            if biggest is None:
                break
            counts[biggest] -= 1
        used = sum(counts[key] * key[0] for key in order)
        counts[order[-1]] += (total_bytes - used) // order[-1][0]

        self._classes: List[_Class] = []
        self._by_geom: Dict[PageKey, _Class] = {}
        for index, key in enumerate(order):
            page, dtype = key
            shape = groups[key][0]
            name = self.NAME if index == len(order) - 1 else f"{self.NAME}.{page}"
            cls = _Class(name, index, page, dtype, shape, counts[key], pin_memory)
            self._classes.append(cls)
            self._by_geom[key] = cls
        for cls in self._classes:
            logger.info("ScratchPool: %s %d x %s %s = %.2f MB pinned",
                        cls.name, cls.num_blocks, cls.shape, cls.dtype,
                        cls.pool.nbytes / 2**20)

    def close(self) -> None:
        for cls in self._classes:
            cls.close()

    # ---- geometry lookup ----

    def _exact(self, shape: Sequence[int], dtype: torch.dtype) -> Optional[_Class]:
        page = math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        return self._by_geom.get((page, dtype))

    def name_for(self, shape: Sequence[int], dtype: torch.dtype) -> str:
        """Pool a layer of this geometry should stage through."""
        cls = self._exact(shape, dtype)
        if cls is None:
            raise RuntimeError(f"ScratchPool: no pool for page {math.prod(shape)} x {dtype}")
        return cls.name

    # ---- allocation ----

    def _class_of(self, idx: int) -> Tuple[_Class, int]:
        class_id, local = idx >> 32, idx & 0xFFFFFFFF
        return self._classes[class_id], local

    def name_of(self, idx: int) -> str:
        """Staging pool that owns a global block index."""
        return self._class_of(idx)[0].name

    def allocate(self, count: int, shape: Sequence[int],
                 dtype: torch.dtype) -> Optional[List[int]]:
        """Global block indices for one transfer, or None when this geometry is dry."""
        cls = self._exact(shape, dtype)
        if cls is None:
            return None
        if (blocks := cls.take(count)) is None:
            return None
        return [cls.base | i for i in blocks]

    def release(self, indices: List[int]) -> None:
        by_class: Dict[int, List[int]] = {}
        for idx in indices:
            by_class.setdefault(idx >> 32, []).append(idx & 0xFFFFFFFF)
        for class_id, local in by_class.items():
            self._classes[class_id].give(local)

    def block(self, idx: int) -> torch.Tensor:
        cls, local = self._class_of(idx)
        return _iqt.block_tensor(cls.pool, cls.name, local)

    # ---- stats ----

    def total_bytes(self) -> int:
        return sum(cls.pool.nbytes for cls in self._classes)

    def available_bytes(self) -> int:
        return sum(cls.free() * cls.block_bytes for cls in self._classes)

    def total_count(self) -> int:
        return sum(cls.num_blocks for cls in self._classes)

    def available_count(self) -> int:
        return sum(cls.free() for cls in self._classes)

    def classes(self) -> List[dict]:
        return [
            {"name": cls.name, "page_bytes": cls.block_bytes, "dtype": str(cls.dtype),
             "blocks": cls.num_blocks, "free": cls.free()}
            for cls in self._classes
        ]

    def status(self) -> str:
        parts = [
            f"{cls.name}: {cls.free()}/{cls.num_blocks} free"
            for cls in self._classes
        ]
        return "ScratchPool: " + "; ".join(parts)
