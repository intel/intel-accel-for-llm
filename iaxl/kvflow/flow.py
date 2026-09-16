# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field

from ..utils.profiler import (
    profile_scope,
    profile_cross_scope,
    profile_func,
    start_profiling,
    stop_profiling,
)
from ..envs import envs
from ..torch_ext import Record
from ..torch_ext import Context, GpuTransferDirection
from ..torch_ext import Mem, Storage
from .. import torch_ext as _iqt
from .scratch_pool import ScratchPool

logger = logging.getLogger(__name__)

debug_enabled = envs.IAXL_DEBUG
stream_sync_on_get = envs.IAXL_CACHE_STREAM_SYNC_ON_GET


def get_accelerator_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return None


@dataclass
class Task:
    ctx: Any = None
    cpu_tensors: List[torch.Tensor] = None

    label: str = None
    tensor_key: str = None
    chunk_labels: List[str] = None

    def __repr__(self):
        return f"Task(label={self.label!r}, tensor_key={self.tensor_key!r}, chunk_labels={self.chunk_labels})"


class KVFlow:
    def __init__(self, persist_dir: str, cache_size_gb: float = 8.0, rank: int = 0):

        self.enable_compression = envs.IAXL_KV_COMPRESSION
        base_dir = envs.IAXL_CACHE_DIR
        compression_subdir = "compressed" if self.enable_compression else "raw"
        self.persist_dir = os.path.join(base_dir, compression_subdir, persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.cache_size_gb = cache_size_gb
        if cache_size_gb == 0:
            self.has_only_mode = True
        else:
            self.has_only_mode = False

        if not self.has_only_mode:
            self.storage = Storage(self.persist_dir)
            logger.info("Cache size %d GB", cache_size_gb)
        else:
            self.storage = None

        self.sqlite_path = os.path.join(self.persist_dir, "chunks.db")
        self.record = Record(
            self.sqlite_path, cleanup_unpersisted=not self.has_only_mode
        )
        logger.info("C++ Record initialized at %s", self.sqlite_path)

        if self.has_only_mode:
            logger.info("Initialized in has-only mode: persist_dir=%s", persist_dir)
            return

        self.device_type = get_accelerator_device()
        if self.device_type is None:
            raise RuntimeError(
                "No accelerator available, KVFlow requires GPU support (CUDA or XPU)"
            )

        logger.info("Using accelerator: %s", self.device_type)

        self._streams_initialized = False
        self.cur_stream = None
        self.put_stream = None
        self.get_stream = None
        self.put_stream_ctx = None
        self.get_stream_ctx = None

        self.cache_size_bytes = int(cache_size_gb * 1024**3)

        self.chunk_pool = ScratchPool(cache_size_gb=cache_size_gb)

        self.mem = Mem(
            capacity_bytes=self.cache_size_bytes,
            storage=self.storage,
            record=self.record,
        )

        logger.info(
            "Initialized: persist_dir=%s, cache_size_gb=%.2f, compression=%s",
            persist_dir,
            cache_size_gb,
            self.enable_compression,
        )

        self._persist_count = 0
        self._persist_bytes = 0
        self._evict_count = 0
        self._evict_bytes = 0

    def _ensure_streams(self):
        if self._streams_initialized:
            return
        if self.device_type == "cuda":
            self.cur_stream = torch.cuda.current_stream()
            self.put_stream = torch.cuda.Stream()
            self.get_stream = torch.cuda.Stream()
            self.put_stream_ctx = lambda: torch.cuda.stream(self.put_stream)
            self.get_stream_ctx = lambda: torch.cuda.stream(self.get_stream)
        else:
            self.cur_stream = torch.xpu.current_stream()
            self.put_stream = torch.xpu.Stream()
            self.get_stream = torch.xpu.Stream()
            self.put_stream_ctx = lambda: torch.xpu.stream(self.put_stream)
            self.get_stream_ctx = lambda: torch.xpu.stream(self.get_stream)
        self._streams_initialized = True
        logger.info("CUDA/XPU streams created (lazy init, device=%s)", self.device_type)

    @profile_func(
        lambda self,
        label,
        tensors,
        chunk_dim,
        chunk_indices,
        chunk_labels,
        description="",
        skip_compression_count=0: (
            f"({description},count={len(chunk_indices)})"
        )
    )
    def put(
        self,
        label: str,
        tensors: Dict[str, torch.Tensor],
        chunk_dim: int,
        chunk_indices: List[int],
        chunk_labels: List[str],
        description: str = "",
        skip_compression_count: int = 0,
    ) -> Dict[str, Task]:

        self._ensure_streams()
        assert chunk_indices is not None and chunk_labels is not None
        assert tensors, "tensors must not be empty"
        assert chunk_indices, "chunk_indices must not be empty"
        assert len(chunk_indices) == len(chunk_labels), (
            "chunk_indices and chunk_labels must have the same length"
        )
        first_t = next(iter(tensors.values()))
        for tensor_key, tensor in tensors.items():
            assert tensor.is_cuda or tensor.is_xpu, (
                f"Tensor '{tensor_key}' must be on GPU device (CUDA or XPU)"
            )
            assert tensor.device == first_t.device, (
                "all tensors must be on the same device"
            )
            # Per-tensor chunk_shape (computed in the loop below) lets a single
            # put/get carry heterogeneous shapes/dtypes, e.g. DSv4 hybrid KV:
            # one HMA group mixes MLA ([N,64,584]) + indexer ([N,64,132]) caches.
            # ScratchPool already buckets pinned buffers by (shape, dtype).
            assert 0 <= chunk_dim < tensor.dim(), "chunk_dim is out of range"
            # The transfer copies each chunk's inner bytes at stride(chunk_dim)
            # (kv_xfer cudaMemcpy2DAsync), so FULL contiguity is not required:
            # a chunk_dim-strided view is fine (HMA tensor-sharing pads each
            # block's slot) as long as chunk_dim is the leading dim and every
            # chunk's inner data is contiguous. Fail loudly for any other layout.
            if not tensor.is_contiguous():
                assert chunk_dim == 0 and tensor.select(0, 0).is_contiguous(), (
                    f"Tensor '{tensor_key}' not transfer-compatible (inner data "
                    f"must be contiguous): shape={tuple(tensor.shape)} "
                    f"stride={tuple(tensor.stride())} chunk_dim={chunk_dim}"
                )

        results = {}
        first_tensor = True
        num_chunks = len(chunk_indices)

        for tensor_index, (tensor_key, tensor) in enumerate(tensors.items()):
            chunk_shape = list(tensor.shape)
            del chunk_shape[chunk_dim]
            chunk_shape = tuple(chunk_shape)
            cpu_tensors = self.chunk_pool.allocate(
                num_chunks, chunk_shape, tensor.dtype
            )

            ctx = Context.create(
                tensor,
                chunk_dim,
                GpuTransferDirection.D2H,
                description,
                work_stream=self.put_stream,
            )
            if first_tensor:
                ctx.xfer_wait_cur_stream(sync_cur_stream=True)
                first_tensor = False
            ctx.xfer_chunks_batch(chunk_indices, cpu_tensors)
            ctx.xfer_finish()

            compress = tensor_index >= skip_compression_count
            ctx.zip_to_mem(
                self.mem, label, tensor_key, chunk_labels, cpu_tensors, compress
            )

            results[tensor_key] = Task(
                ctx=ctx,
                cpu_tensors=cpu_tensors,
                label=label,
                tensor_key=tensor_key,
                chunk_labels=chunk_labels,
            )

        return results

    @profile_func(lambda self, put_results, *_: f"(count={len(put_results)})")
    def put_wait(
        self,
        put_results: Dict[str, Task],
        tensor_dict_keys: List[str] = None,
        wait: bool = True,
    ) -> bool:

        keys_to_wait = (
            tensor_dict_keys if tensor_dict_keys else list(put_results.keys())
        )

        if not wait:
            for key in keys_to_wait:
                assert key in put_results, f"Key {key} not in put_results"
                result = put_results[key]
                assert result.ctx is not None, f"Task ctx is None for key {key}"
                if not result.ctx.zip_is_complete():
                    return False

        for key in keys_to_wait:
            assert key in put_results, f"Key {key} not in put_results"

            result = put_results[key]
            assert result.ctx is not None, f"Task ctx is None for key {key}"
            result.ctx.zip_wait()
            result.ctx = None

            assert result.cpu_tensors is not None
            self.chunk_pool.release(result.cpu_tensors)
            result.cpu_tensors = None

        return True

    @profile_func(
        lambda self, label, tensors, chunk_dim, chunk_indices, chunk_labels, description: (
            f"({description},count={len(chunk_indices)})"
        )
    )
    def get(
        self,
        label: str,
        tensors: Dict[str, torch.Tensor],
        chunk_dim: int,
        chunk_indices: List[int],
        chunk_labels: List[str],
        description: str = "",
    ) -> Dict[str, Task]:

        self._ensure_streams()
        assert chunk_indices is not None and chunk_labels is not None
        assert tensors, "tensors must not be empty"
        assert chunk_indices, "chunk_indices must not be empty"
        assert len(chunk_indices) == len(chunk_labels), (
            "chunk_indices and chunk_labels must have the same length"
        )
        first_t = next(iter(tensors.values()))
        for tensor_key, tensor in tensors.items():
            assert tensor.is_cuda or tensor.is_xpu, (
                f"Tensor '{tensor_key}' must be on GPU device (CUDA or XPU)"
            )
            assert tensor.device == first_t.device, (
                "all tensors must be on the same device"
            )
            # Per-tensor chunk_shape (computed in the loop below) lets a single
            # put/get carry heterogeneous shapes/dtypes, e.g. DSv4 hybrid KV:
            # one HMA group mixes MLA ([N,64,584]) + indexer ([N,64,132]) caches.
            # ScratchPool already buckets pinned buffers by (shape, dtype).
            assert 0 <= chunk_dim < tensor.dim(), "chunk_dim is out of range"
            # The transfer copies each chunk's inner bytes at stride(chunk_dim)
            # (kv_xfer cudaMemcpy2DAsync), so FULL contiguity is not required:
            # a chunk_dim-strided view is fine (HMA tensor-sharing pads each
            # block's slot) as long as chunk_dim is the leading dim and every
            # chunk's inner data is contiguous. Fail loudly for any other layout.
            if not tensor.is_contiguous():
                assert chunk_dim == 0 and tensor.select(0, 0).is_contiguous(), (
                    f"Tensor '{tensor_key}' not transfer-compatible (inner data "
                    f"must be contiguous): shape={tuple(tensor.shape)} "
                    f"stride={tuple(tensor.stride())} chunk_dim={chunk_dim}"
                )

        results = {}
        num_chunks = len(chunk_indices)

        first_tensor = True
        for tensor_key, tensor in tensors.items():
            chunk_shape = list(tensor.shape)
            del chunk_shape[chunk_dim]
            chunk_shape = tuple(chunk_shape)
            cpu_tensors = self.chunk_pool.allocate(
                num_chunks, chunk_shape, tensor.dtype
            )

            ctx = Context.create(
                tensor,
                chunk_dim,
                GpuTransferDirection.H2D,
                description,
                work_stream=self.get_stream,
            )
            if first_tensor:
                if stream_sync_on_get:
                    ctx.xfer_wait_cur_stream()
                first_tensor = False
            ctx.unzip_from_mem(
                self.mem, label, tensor_key, chunk_labels, chunk_indices, cpu_tensors
            )

            results[tensor_key] = Task(
                ctx=ctx,
                cpu_tensors=cpu_tensors,
                label=label,
                tensor_key=tensor_key,
                chunk_labels=chunk_labels,
            )

        return results

    @profile_func(lambda self, get_results, *_: f"(count={len(get_results)})")
    def get_wait(
        self,
        get_results: Dict[str, Task],
        tensor_dict_keys: List[str] = None,
        wait: bool = True,
    ) -> bool:

        keys_to_wait = (
            tensor_dict_keys if tensor_dict_keys else list(get_results.keys())
        )

        for key in keys_to_wait:
            assert key in get_results, f"Key {key} not in get_results"

            result = get_results[key]

            if not wait:
                if result.ctx is None:
                    continue
                if not result.ctx.unzip_is_complete():
                    if debug_enabled:
                        logger.debug(
                            f"get_wait(wait=False): key={key} unzip NOT complete"
                        )
                    return False
                if not result.ctx.xfer_is_complete():
                    if debug_enabled:
                        logger.debug(
                            f"get_wait(wait=False): key={key} xfer NOT complete"
                        )
                    return False

            else:
                if result.ctx is None:
                    continue

                result.ctx.unzip_wait()
                result.ctx.xfer_wait()

                result.ctx = None

                if result.cpu_tensors is not None:
                    self.chunk_pool.release(result.cpu_tensors)
                    result.cpu_tensors = None

        return True

    @profile_func()
    def record_flush(self) -> bool:
        assert self.record is not None
        self.record.sync()
        return True

    @profile_func(lambda self, label, chunk_labels: f"(count={len(chunk_labels)})")
    def put_finish(self, label: str, chunk_labels: List[str]) -> None:
        assert self.record is not None
        self.record.submit(label, chunk_labels)

    @profile_func(lambda self, label, chunk_labels: f"(count={len(chunk_labels)})")
    def has(self, label: str, chunk_labels: List[str]) -> List[bool]:

        assert self.record is not None
        return self.record.has(label, chunk_labels)

    def stop(self):
        logger.info("Stopping...")
        assert self.record is not None
        self.record.sync()
        logger.info("Stopped")

    def status(self) -> dict:
        if self.has_only_mode:
            return {
                "has_only_mode": True,
                "persist_dir": self.persist_dir,
            }

        unpersisted = self.mem.unpersisted_count
        group_count = self.mem.group_count

        with self.chunk_pool._lock:
            pool_total_tensors = self.chunk_pool._total_tensors
            pool_total_bytes = self.chunk_pool._total_bytes
            pool_available = sum(len(p) for p in self.chunk_pool._pools.values())
            pool_in_use = (
                self.chunk_pool._allocate_count - self.chunk_pool._release_count
            )
            pool_shapes = []
            for (shape, dtype), tensors in self.chunk_pool._pools.items():
                pool_shapes.append(
                    {
                        "shape": list(shape),
                        "dtype": str(dtype),
                        "available": len(tensors),
                    }
                )

        return {
            "has_only_mode": False,
            "persist_dir": self.persist_dir,
            "compression": self.enable_compression,
            "cache_size_gb": self.cache_size_gb,
            "group_count": group_count,
            "cache_entries": self.mem.size,
            "current_bytes": self.mem.current_bytes,
            "capacity_bytes": self.mem.capacity_bytes,
            "usage_pct": round(self.mem.current_bytes / self.cache_size_bytes * 100, 2)
            if self.cache_size_bytes > 0
            else 0,
            "hits": self.mem.hits,
            "hits_in_storage": self.mem.hits_in_storage,
            "misses": self.mem.misses,
            "puts": self.mem.puts,
            "evictions": self.mem.evictions,
            "total_unzip_bytes": self.mem.total_unzip_bytes,
            "total_zip_bytes": self.mem.total_zip_bytes,
            "compression_ratio (unzip/zip, higher=better)": round(
                self.mem.compression_ratio, 2
            ),
            "unpersisted_count": unpersisted,
            "persisted_count": group_count - unpersisted,
            "persisted_total": self._persist_count,
            "persisted_bytes_total": self._persist_bytes,
            "evicted_total": self._evict_count,
            "evicted_bytes_total": self._evict_bytes,
            "pool_total_tensors": pool_total_tensors,
            "pool_total_bytes": pool_total_bytes,
            "pool_available": pool_available,
            "pool_in_use": pool_in_use,
            "pool_shapes": pool_shapes,
        }

    def get_persist_candidates(self, max_count: int) -> List[str]:
        if self.has_only_mode:
            return []
        entries = self.mem.get_unpersisted(max_count)

        seen: set = set()
        result: List[str] = []
        for full_label, _ in entries:
            group_key = full_label.rsplit(":", 1)[0]
            if group_key not in seen:
                seen.add(group_key)
                result.append(group_key)
        return result

    def get_evict_candidates(self, max_count: int) -> List[str]:
        if self.has_only_mode:
            return []
        entries = self.mem.get_lru_oldest(max_count)
        seen: set = set()
        result: List[str] = []
        for full_label, _ in entries:
            group_key = full_label.rsplit(":", 1)[0]
            if group_key not in seen:
                seen.add(group_key)
                result.append(group_key)
        return result

    def persist(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {
                "persisted": 0,
                "bytes_written": 0,
                "files": 0,
                "labels": [],
                "error": "has-only mode",
            }

        persisted_groups = self.mem.persist_groups(max_count)

        if not persisted_groups:
            return {"persisted": 0, "bytes_written": 0, "files": 0, "labels": []}

        groups = [entry[0] for entry in persisted_groups]
        total_bytes = sum(entry[1] for entry in persisted_groups)

        self._persist_count += len(groups)
        self._persist_bytes += total_bytes

        logger.info("Persisted %d groups (%d bytes) to disk", len(groups), total_bytes)
        return {
            "persisted": len(groups),
            "bytes_written": total_bytes,
            "labels": sorted(groups),
        }

    def evict(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {
                "evicted": 0,
                "bytes_freed": 0,
                "labels": [],
                "error": "has-only mode",
            }

        evicted_groups = self.mem.evict_groups(max_count)

        if not evicted_groups:
            return {"evicted": 0, "bytes_freed": 0, "labels": []}

        groups = [entry[0] for entry in evicted_groups]
        total_freed = sum(entry[1] for entry in evicted_groups)

        self._evict_count += len(groups)
        self._evict_bytes += total_freed

        logger.info("Evicted %d groups (%d bytes freed)", len(groups), total_freed)
        return {
            "evicted": len(groups),
            "bytes_freed": total_freed,
            "labels": sorted(groups),
        }
