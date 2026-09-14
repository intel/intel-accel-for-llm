# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""RemoteKVStore: a KVStore-compatible client backed by a remote daemon.

Drop-in replacement for :class:`iaxl.kvstore.KVStore` used by the vLLM
connector (``kvshrink_connector.py``) when remote caching is enabled. It
keeps the exact block-level API the connector relies on:

    has / put / put_wait / get / get_wait / stop

Local KV cache tensors are never compressed on the worker. Instead each
(block, layer, tensor_key) shard is transferred to the remote daemon over the
data plane; the daemon performs the compression and pool management (reusing
the same native code as the local path -- see ``server.py``). put/get are
asynchronous via a thread pool so KV transfer overlaps with GPU compute,
mirroring the Task semantics of the local KVStore.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import torch

from .client import RemoteCacheClient
from .config import RemoteCacheConfig
from .metadata import block_descriptors, tensor_keys_for_layout
from .transport import ShardRef, make_data_plane

logger = logging.getLogger(__name__)


class RemoteTask:
    """Handle returned by put()/get(), wrapping the transfer future(s)."""

    def __init__(self, future: Future):
        self.future = future

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: Optional[float] = None) -> None:
        self.future.result(timeout)


class RemoteKVStore:
    LABEL = "kv"

    def __init__(self,
                 model_name: str,
                 config: RemoteCacheConfig,
                 block_dim: Optional[int] = None,
                 kv_caches: Optional[Dict[str, torch.Tensor]] = None,
                 layer_names: Optional[List[str]] = None,
                 rank: int = 0,
                 tp_size: int = 1,
                 block_size: int = 16):
        if kv_caches is None and layer_names is None:
            raise ValueError("At least one of kv_caches or layer_names must be provided")

        self.model_name = model_name
        self.config = config
        self.block_dim = block_dim
        self.kv_caches = kv_caches
        self.rank = rank
        self.tp_size = tp_size
        self.block_size = block_size
        self.has_only_mode = kv_caches is None

        self.layer_names = list(kv_caches.keys()) if kv_caches else list(layer_names)

        role = "scheduler" if self.has_only_mode else "worker"
        self.client = RemoteCacheClient(config, model_name, tp_size, rank, role)
        self.client.connect()

        # Per-layer tensor layout (worker only).
        self._tensor_keys: Dict[str, List[str]] = {}
        dtype_name = "float16"
        shard_bytes = 0
        if kv_caches:
            first = next(iter(kv_caches.values()))
            dtype_name = str(first.dtype).split(".")[-1]
            use_mla = block_dim == 0 and (len(first.shape) <= 1 or first.shape[1] != 2)
            for name, tensor in kv_caches.items():
                self._tensor_keys[name] = tensor_keys_for_layout(
                    tensor.shape, block_dim, use_mla=use_mla,
                )
            shard_bytes = max(
                (length for _, length in block_descriptors(
                    list(first.shape), block_dim, first.element_size(), 0)),
                default=0)

        tensor_keys = self._tensor_keys.get(self.layer_names[0], []) if kv_caches else []
        self.client.create_session(
            num_layers=len(self.layer_names),
            tensor_keys=tensor_keys,
            dtype=dtype_name,
            block_size=block_size,
            shard_bytes=shard_bytes,
        )

        if not self.has_only_mode:
            self._data_plane = make_data_plane(config, self.client.session_id,
                                               staging_slots=self.client.staging_slots,
                                               tp_size=tp_size)
            self._data_plane.start()
            # One thread per concurrent layer transfer. The NIXL backend
            # allows a few rounds in flight per rank (bounded by the daemon's
            # staging budget), so these threads translate into real overlap
            # of RDMA with remote (de)compression instead of queueing on a
            # lock.
            max_workers = int(os.getenv("KVSHRINK_REMOTE_XFER_THREADS", "8"))
            self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix=f"remote-xfer-{rank}")
        else:
            self._data_plane = None
            self._pool = None

        logger.info("RemoteKVStore ready: model=%s rank=%d role=%s %s",
                    model_name, rank, role, config.describe())

    # -- shard construction ---------------------------------------------------

    def _build_shards(self, block_indices: List[int], block_hashs: List[str],
                      layer_name: str) -> List[ShardRef]:
        tensor = self.kv_caches[layer_name]
        shape = list(tensor.shape)
        elem_size = tensor.element_size()
        tks = self._tensor_keys[layer_name]
        shards: List[ShardRef] = []
        for b_idx, b_hash in zip(block_indices, block_hashs):
            ranges = block_descriptors(shape, self.block_dim, elem_size, b_idx)
            for tk, (off, length) in zip(tks, ranges):
                # Session-relative key: the daemon prepends model|tp_size|
                # tp_rank from the session. Every round carries hundreds of
                # these on the control plane, so the shorter form is worth
                # the indirection.
                key = f"{b_hash}|{layer_name}|{tk}"
                shards.append(ShardRef(key, tensor, off, length, tensor.dtype))
        return shards

    # -- KVStore-compatible API ----------------------------------------------

    def has(self, block_hashs: Optional[List[str]] = None) -> List[bool]:
        if not block_hashs:
            return []
        results = self.client.has(block_hashs)
        # Prefix-truncate: KV blocks must be contiguous from the start.
        mask = np.array(results, dtype=np.bool_)
        idx = int(np.argmin(mask))
        if not mask[idx]:
            mask[idx + 1:] = False
            results = mask.tolist()
        return results

    def put(self, block_indices: List[int], block_hashs: List[str],
            layer_names: Optional[List[str]] = None, description: str = "") -> Dict[str, RemoteTask]:
        if self.has_only_mode:
            raise RuntimeError("put() not available in has-only mode")
        if layer_names is None:
            layer_names = self.layer_names

        shards: List[ShardRef] = []
        for name in layer_names:
            shards.extend(self._build_shards(block_indices, block_hashs, name))

        # The NIC reads these KV tensors outside CUDA stream ordering, so the
        # attention kernel that fills them must complete before the transfer.
        ready = None
        if shards and shards[0].tensor.is_cuda:
            ready = torch.cuda.Event()
            ready.record()

        def _put() -> None:
            if ready is not None:
                ready.synchronize()
            self._data_plane.put(self.client.session_id, shards)

        future = self._pool.submit(_put)
        task = RemoteTask(future)

        # Publish the block only after its bytes have actually landed,
        # otherwise a concurrent has()/get() can read a half-written entry.
        if self.layer_names[-1] in layer_names:
            def _mark(fut: Future) -> None:
                if fut.exception() is None:
                    self.client.mark_ready(block_hashs)

            future.add_done_callback(_mark)

        return {name: task for name in layer_names}

    def put_wait(self, put_results: Dict[str, RemoteTask],
                 layer_names: Optional[List[str]] = None,
                 timeout: float = None, wait: bool = True) -> bool:
        if self.has_only_mode:
            raise RuntimeError("put_wait() not available in has-only mode")
        tasks = self._select_tasks(put_results, layer_names)
        if not wait:
            return all(t.done() for t in tasks)
        for t in tasks:
            t.result(timeout)
        return True

    def get(self, block_indices: List[int], block_hashs: List[str],
            layer_names: Optional[List[str]] = None, description: str = "") -> Dict[str, RemoteTask]:
        if self.has_only_mode:
            raise RuntimeError("get() not available in has-only mode")
        if layer_names is None:
            layer_names = self.layer_names

        # One future per layer so wait_for_layer_load() can wait per layer.
        results: Dict[str, RemoteTask] = {}
        for name in layer_names:
            shards = self._build_shards(block_indices, block_hashs, name)
            future = self._pool.submit(self._data_plane.get, self.client.session_id, shards)
            results[name] = RemoteTask(future)
        return results

    def get_wait(self, get_results: Dict[str, RemoteTask],
                 layer_names: Optional[List[str]] = None,
                 timeout: float = None, wait: bool = True) -> bool:
        if self.has_only_mode:
            raise RuntimeError("get_wait() not available in has-only mode")
        tasks = self._select_tasks(get_results, layer_names)
        if not wait:
            return all(t.done() for t in tasks)
        for t in tasks:
            t.result(timeout)
        return True

    @staticmethod
    def _select_tasks(results: Dict[str, RemoteTask],
                      layer_names: Optional[List[str]]) -> List[RemoteTask]:
        if layer_names is None:
            seen = set()
            tasks = []
            for t in results.values():
                if id(t) not in seen:
                    seen.add(id(t))
                    tasks.append(t)
            return tasks
        return [results[name] for name in layer_names if name in results]

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._data_plane is not None:
            self._data_plane.close()
        self.client.close()

    def status(self) -> str:
        return f"RemoteKVStore(model={self.model_name}, rank={self.rank})"
