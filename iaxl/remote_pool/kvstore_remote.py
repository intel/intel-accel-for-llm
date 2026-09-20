# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Client-side KVStore shell: same interface as `iaxl.kvstore.KVStoreLocal`, all
work forwarded to the remote_pool daemon over RPC. The daemon moves KV blocks
itself (RDMA READ for put, RDMA WRITE for get); this side only registers the
kv_caches and tracks per-job completion pushed back by the daemon."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..envs import envs
from . import rpc
from .nixl_impl import rdma_xfer
from .rpc import RpcChannel, rank_port

logger = logging.getLogger(__name__)


@dataclass
class RemoteTask:
    job_id: int
    tensor_key: str
    done: bool = False


class KVStoreRemote:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        block_dim: Optional[int] = None,
        kv_caches: Optional[Dict[str, torch.Tensor]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
        daemon_ip: Optional[str] = None,
        daemon_port: Optional[int] = None,
    ):
        if kv_caches is None and layer_names is None:
            raise ValueError("At least one of kv_caches or layer_names must be provided")
        if kv_caches is not None and block_dim is None:
            raise ValueError("block_dim is required when kv_caches is provided")
        if kv_caches is not None and layer_names is not None:
            if set(kv_caches) != set(layer_names):
                raise ValueError(
                    f"kv_caches keys {set(kv_caches)} must match layer_names {set(layer_names)}"
                )

        ip = daemon_ip or envs.IAXL_RDMA_DAEMON_IP
        port = daemon_port or envs.IAXL_RDMA_DAEMON_PORT
        if not ip:
            raise ValueError("daemon_ip (IAXL_RDMA_DAEMON_IP) is required")

        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self.rank = rank
        self.tp_size = tp_size
        self.has_only_mode = kv_caches is None
        self._pending: Dict[int, list] = {}  # job_id -> [tasks, remaining]

        if self.has_only_mode:
            name, self.peer = "client_sched", "daemon_sched"
        else:
            name, self.peer = f"client{rank}", f"daemon{rank}"
            port = rank_port(port, rank)

        self.xfer = rdma_xfer(name, local_ip=envs.IAXL_RDMA_CLIENT_IP or None)
        self.rpc = RpcChannel(self.xfer, self.peer, self._on_done)
        if kv_caches is not None:
            for t in kv_caches.values():  # rkeys ride along with our metadata
                self.xfer.register_memory(t)
        self.xfer.connect(self.peer, ip, port, timeout_s=envs.IAXL_API_TIMEOUT)
        self.rpc.handshake()

        if kv_caches is not None:
            layers = {
                n: {"base": t.data_ptr(), "shape": list(t.shape),
                    "dtype": str(t.dtype).split(".")[-1], "dev_id": t.get_device()}
                for n, t in kv_caches.items()
            }
            self.rpc.call(rpc.REGISTER_KV_CACHES, rpc.pack_json({
                "layers": layers, "block_dim": block_dim, "model_name": model_name,
                "rank": rank, "tp_size": tp_size}))
            self.layer_names = list(kv_caches.keys())
            first = next(iter(kv_caches.values()))
            self.kvcache_shape = list(first.shape)
            self.block_shape = tuple(s for i, s in enumerate(first.shape) if i != block_dim)
            self._sync = (torch.xpu.synchronize if first.is_xpu
                          else lambda: torch.cuda.current_stream().synchronize())
        else:
            self.rpc.call(rpc.REGISTER_LAYERS, rpc.pack_json({
                "layer_names": layer_names, "model_name": model_name, "tp_size": tp_size}))
            self.layer_names = list(layer_names)
            self.kvcache_shape = None
            self.block_shape = None
        self.layer_idx = {n: i for i, n in enumerate(self.layer_names)}

        logger.info("KVStoreRemote connected: peer=%s@%s:%d model=%s rank=%d has_only=%s layers=%d",
                    self.peer, ip, port, model_name, rank, self.has_only_mode, len(self.layer_names))

    # -- data path -------------------------------------------------------------
    def _xfer(self, method, block_indices, block_hashs, layer_names, description, what, label=None):
        if self.has_only_mode:
            raise RuntimeError(f"{what}() not available in has-only mode (kv_caches not provided)")
        names = layer_names or self.layer_names
        payload = rpc.pack_blocks(block_indices, block_hashs, [self.layer_idx[n] for n in names],
                                  description, label or "")
        job_id = int.from_bytes(self.rpc.call(method, payload), "little")
        tasks = {n: RemoteTask(job_id, n) for n in names}
        self._pending[job_id] = [tasks, len(tasks)]
        return tasks

    def put(self, block_indices, block_hashs, layer_names=None, description="",
            label=None) -> Dict[str, RemoteTask]:
        if not self.has_only_mode:
            self._sync()  # attention kernels must have written kv_caches before the daemon READs
        return self._xfer(rpc.PUT, block_indices, block_hashs, layer_names, description, "put", label=label)

    def get(self, block_indices, block_hashs, layer_names=None, description="",
            label=None) -> Dict[str, RemoteTask]:
        return self._xfer(rpc.GET, block_indices, block_hashs, layer_names, description, "get", label=label)

    def _wait(self, results: Dict[str, RemoteTask], layer_names, wait: bool, what: str) -> bool:
        if self.has_only_mode:
            raise RuntimeError(f"{what}() not available in has-only mode (kv_caches not provided)")
        tasks = [results[n] for n in layer_names] if layer_names else list(results.values())
        self.rpc.drain()
        while not all(t.done for t in tasks):
            if not wait:
                return False
            self.rpc.drain()
        return True

    def put_wait(self, put_results, layer_names=None, wait=True) -> bool:
        return self._wait(put_results, layer_names, wait, "put_wait")

    def get_wait(self, get_results, layer_names=None, wait=True) -> bool:
        return self._wait(get_results, layer_names, wait, "get_wait")

    def _on_done(self, job_id: int, layer_idx: int):
        entry = self._pending.get(job_id)
        if entry is None:
            logger.warning("done for unknown job %d layer %d", job_id, layer_idx)
            return
        entry[0][self.layer_names[layer_idx]].done = True
        entry[1] -= 1
        if entry[1] == 0:
            del self._pending[job_id]

    # -- control path ------------------------------------------------------------
    def has(self, block_hashs: Optional[List[str]] = None,
            label: Optional[str] = None,
            truncate: bool = True) -> List[bool]:
        """Presence per block hash; ``label`` selects the namespace and
        ``truncate`` the prefix semantics, both applied daemon-side."""
        resp = self.rpc.call(rpc.HAS, rpc.pack_has(block_hashs or [], label or "", truncate))
        return [bool(b) for b in resp]

    def _json(self, method, *args):
        return rpc.unpack_json(self.rpc.call(method, rpc.pack_json(list(args))))

    def stop(self):
        """Stops the daemon-side KVStore; the serving daemon process exits afterwards."""
        self.rpc.call(rpc.STOP)
        self.xfer.disconnect(self.peer)

    def status(self) -> dict:
        return self._json(rpc.STATUS)

    def metrics(self, params: Optional[dict] = None) -> dict:
        return self._json(rpc.METRICS, params)

    def persist(self, max_count: int) -> dict:
        return self._json(rpc.PERSIST, max_count)

    def evict(self, max_count: int) -> dict:
        return self._json(rpc.EVICT, max_count)

    def get_persist_candidates(self, max_count: int) -> List[str]:
        return self._json(rpc.PERSIST_CANDIDATES, max_count)

    def get_evict_candidates(self, max_count: int) -> List[str]:
        return self._json(rpc.EVICT_CANDIDATES, max_count)
