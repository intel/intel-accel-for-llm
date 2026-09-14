# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""NIXL (RDMA/GDR) data-plane backend.

This is the production transport. The vLLM worker registers its GPU KV cache
tensors with a local NIXL agent once, then drives zero-copy RDMA transfers
against staging slots that the remote daemon advertises over the control
plane. The control plane only carries metadata and put-commit/get-begin
triggers (see ``iaxl.remote.protocol`` for why it stays on TCP).

Protocol (client-driven):

  PUT  begin  -> daemon allocates + returns remote staging descriptors
       WRITE  -> client RDMA-writes local KV shards into remote staging
       commit -> daemon reads staging, compresses (native QAT/IAA/CPU zip
                 pipeline) into the native ``Mem`` pool, frees the slots
  GET  begin  -> daemon decompresses into staging, returns descriptors
       READ   -> client RDMA-reads remote staging into local KV shards
       done   -> daemon frees slots

Both sides use :class:`NixlEndpoint`. The daemon staging pool lives in
:class:`NixlStagingPool`. ``import`` succeeds without nixl installed; the
agent is created lazily so only transport=nixl users need the dependency.

Concurrency model: rounds are *not* serialized per rank. Each round only
needs exclusive access to the NIXL agent while it posts descriptors, so the
agent lock lives inside :class:`NixlEndpoint` and is dropped while a round
waits on the control-plane RPC or polls a transfer. What bounds the number of
concurrent rounds is :class:`_ShardBudget`, sized from the staging capacity
the daemon advertises at session creation, so a rank can keep several rounds
in flight (one layer's RDMA READ overlaps the next layer's daemon-side
decompression) without ever oversubscribing the daemon's staging pool.

Multiple RDMA links: ``KVSHRINK_REMOTE_NIXL_DEVICE`` (client) / ``NIXL_DEVICE``
(daemon) may each be a comma-separated device list (e.g.
``"mlx5_0:1,mlx5_1:1"``); it is forwarded verbatim to ``UCX_NET_DEVICES``,
and UCX itself stripes a single NIXL session's transfers across all listed
devices ("multi-rail"). No per-device application logic is needed here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch

from .. import protocol
from .base import DataPlane, ShardRef, dtype_to_str, tensor_byte_view

logger = logging.getLogger(__name__)

# One put()/get() call from a single request can carry far more shards
# (blocks x tensor_keys) than the daemon's staging pool has slots for -- e.g.
# an 8k token prompt is ~500 blocks. Split into rounds no larger than this so
# a big request degrades to more round trips instead of "staging pool
# exhausted".
_MAX_SHARDS_PER_ROUND = int(os.environ.get("KVSHRINK_REMOTE_NIXL_MAX_SHARDS_PER_ROUND", "128"))

# Override for the per-rank in-flight shard budget. 0 = derive it from the
# rounds-in-flight target below, clamped to the daemon's advertised capacity.
_MAX_INFLIGHT_SHARDS = int(os.environ.get("KVSHRINK_REMOTE_NIXL_MAX_INFLIGHT_SHARDS", "0"))

# Rounds a rank may keep in flight. The point of >1 is overlap: while one
# round waits on the daemon's decompression, another one's RDMA is on the
# wire. Going much higher does not add throughput -- the daemon is already
# saturated -- and costs GIL time on the vLLM worker, so the default is
# deliberately small.
_ROUNDS_IN_FLIGHT = int(os.environ.get("KVSHRINK_REMOTE_NIXL_ROUNDS_IN_FLIGHT", "2"))

# Threads that issue the rounds of a *single* put()/get() call. RemoteKVStore
# already drives one call per layer concurrently, which is normally enough to
# keep the daemon busy, so the default is to walk a call's rounds inline;
# raise this only when a request has few layers but very many blocks.
_ROUND_THREADS = int(os.environ.get("KVSHRINK_REMOTE_NIXL_ROUND_THREADS", "1"))

# Back-off between transfer progress polls once the initial spin is exhausted.
_POLL_INTERVAL_SEC = float(os.environ.get("KVSHRINK_REMOTE_NIXL_POLL_US", "50")) / 1e6


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class _ShardBudget:
    """Counting semaphore admitting a whole round's worth of staging slots.

    Admission is by shard count rather than round count so the limit tracks
    what the daemon actually runs out of (staging slots): many small rounds
    run concurrently, one huge round runs alone.
    """

    def __init__(self, capacity: int):
        self._capacity = max(1, capacity)
        self._available = self._capacity
        self._cv = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    def acquire(self, n: int, timeout: float) -> int:
        n = min(max(1, n), self._capacity)
        with self._cv:
            if not self._cv.wait_for(lambda: self._available >= n, timeout=timeout):
                raise TimeoutError(
                    f"remote staging budget exhausted: needed {n} of {self._capacity} "
                    f"shards, {self._available} free after {timeout:.0f}s")
            self._available -= n
        return n

    def release(self, n: int) -> None:
        with self._cv:
            self._available = min(self._capacity, self._available + n)
            self._cv.notify_all()


def _load_nixl():
    try:
        from nixl._api import nixl_agent, nixl_agent_config  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "transport=nixl requires the 'nixl' package. Install NIXL and its "
            "Python bindings, or use transport=tcp for single-host testing."
        ) from exc
    return nixl_agent, nixl_agent_config


def _agent_config(nixl_agent_config):
    """Build an agent config, requesting NIXL's own thread synchronization.

    By default ``nixl_agent_config`` selects ``NIXL_THREAD_SYNC_NONE``, which
    is why the agent then has to be serialized behind a Python lock -- and
    that lock also serializes the progress polling of every in-flight
    transfer. Asking for the reader/writer sync mode moves that locking into
    C++ (outside the GIL) and at a far finer granularity. Returns ``(config,
    native_sync)``; older bindings without ``sync_mode`` fall back to the
    Python lock.
    """
    mode = os.environ.get("KVSHRINK_REMOTE_NIXL_SYNC_MODE", "rw").strip().lower()
    if mode != "none":
        try:
            from nixl._api import nixl_thread_sync_t  # type: ignore

            sync = (nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT if mode == "strict"
                    else nixl_thread_sync_t.NIXL_THREAD_SYNC_RW)
            return nixl_agent_config(backends=["UCX"], sync_mode=sync), True
        except Exception as exc:  # noqa: BLE001
            logger.warning("NIXL thread sync mode unavailable (%s); "
                           "falling back to a process-level agent lock", exc)
    return nixl_agent_config(backends=["UCX"]), False


def _mem_type(tensor: torch.Tensor) -> str:
    return "VRAM" if tensor.is_cuda else "DRAM"


def _dev_id(tensor: torch.Tensor) -> int:
    return tensor.get_device() if tensor.is_cuda else 0


class NixlEndpoint:
    """Thin wrapper over a NIXL agent for registration and transfers.

    When the agent is created with NIXL's own thread synchronization the
    wrapper adds no locking of its own; otherwise it falls back to one
    process-level lock held only for the (short) posting operations --
    crucially never across the wait loop, so many transfers can be in flight
    at once.
    """

    def __init__(self, name: str):
        nixl_agent, nixl_agent_config = _load_nixl()
        config, native_sync = _agent_config(nixl_agent_config)
        self.agent = nixl_agent(name, config)
        self._remote_names: Dict[str, str] = {}
        self._lock = nullcontext() if native_sync else threading.Lock()

    def metadata(self) -> bytes:
        with self._lock:
            return self.agent.get_agent_metadata()

    def add_remote(self, key: str, meta: bytes) -> str:
        with self._lock:
            name = self.agent.add_remote_agent(meta)
        self._remote_names[key] = name
        return name

    def remote_name(self, key: str) -> str:
        return self._remote_names[key]

    def register(self, triples: List[Tuple[int, int, int]], mem_type: str):
        # NIXL register_memory expects (ptr, size, dev_id, "") tuples.
        addrs = [(ptr, size, dev, "") for (ptr, size, dev) in triples]
        with self._lock:
            descs = self.agent.register_memory(addrs, mem_type)
        if not descs:
            raise RuntimeError("NIXL memory registration failed")
        return descs

    def post(self, op: str, local_triples, local_mem: str,
             remote_triples, remote_mem: str, remote_name: str, notif: bytes) -> object:
        """Build both descriptor lists and post the transfer in one critical section."""
        with self._lock:
            local = self.agent.get_xfer_descs(local_triples, local_mem)
            remote = self.agent.get_xfer_descs(remote_triples, remote_mem)
            handle = self.agent.initialize_xfer(op, local, remote, remote_name, notif)
            if not handle:
                raise RuntimeError("NIXL failed to create transfer handle")
            state = self.agent.transfer(handle)
        if state == "ERR":
            self.release(handle)
            raise RuntimeError("NIXL failed to post transfer")
        return handle

    def wait(self, handle, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        spins = 0
        while True:
            with self._lock:
                state = self.agent.check_xfer_state(handle)
            if state == "DONE":
                return
            if state == "ERR":
                raise RuntimeError("NIXL transfer failed")
            if time.monotonic() > deadline:
                raise TimeoutError("NIXL transfer timed out")
            # check_xfer_state is a real syscall (ibverbs CQ poll), not a
            # cheap memory read, and the agent has its own progress thread --
            # so a hot spin from every in-flight round would just burn GIL
            # time that the posting threads need. Spin a little for latency,
            # then back off.
            spins += 1
            time.sleep(0.0 if spins < 16 else _POLL_INTERVAL_SEC)

    def release(self, handle) -> None:
        """Free a completed transfer handle (otherwise NIXL leaks it per round)."""
        try:
            with self._lock:
                self.agent.release_xfer_handle(handle)
        except Exception:  # noqa: BLE001 - best effort cleanup
            logger.debug("release_xfer_handle failed", exc_info=True)

    def close(self) -> None:
        try:
            self.agent = None
        except Exception:  # noqa: BLE001
            pass


class NixlDataPlane(DataPlane):
    """Client-side NIXL data plane used by RemoteKVStore workers."""

    name = "nixl"

    def __init__(self, config, session_id: str, staging_slots: int = 0, tp_size: int = 1):
        self._config = config
        self._session_id = session_id
        self._timeout = config.request_timeout_sec
        self._endpoint: Optional[NixlEndpoint] = None
        self._remote_key = "daemon"
        self._registered: Dict[int, object] = {}
        self._register_lock = threading.Lock()
        # How many staging slots this rank may hold at once. The daemon's
        # pool is shared by every rank *that talks to that daemon instance*
        # (with a multi-process daemon and one instance per rank, tp_size
        # here is effectively 1 -- see the design doc), so each takes an
        # equal share of it; within that share a rank keeps a few rounds in
        # flight so the RDMA of one round overlaps the daemon-side
        # (de)compression of the next.
        share = (staging_slots // max(1, tp_size)) if staging_slots else _MAX_SHARDS_PER_ROUND
        budget = _MAX_INFLIGHT_SHARDS or min(share, _ROUNDS_IN_FLIGHT * _MAX_SHARDS_PER_ROUND)
        self._budget = _ShardBudget(budget)
        self._round_shards = max(1, min(_MAX_SHARDS_PER_ROUND, self._budget.capacity))
        logger.info("NIXL data plane: rounds<=%d shards, in-flight budget=%d shards "
                    "(daemon staging_slots=%d, tp_size=%d)",
                    self._round_shards, self._budget.capacity, staging_slots, tp_size)
        # RemoteKVStore drives put/get for many layers concurrently from a
        # thread pool. A daemon-side "begin" can legitimately block (waiting
        # for staging slots freed by *another* thread's "commit"), so all
        # rounds sharing one socket behind a lock would deadlock: the blocked
        # begin holds the lock for its whole wait, starving the very commit
        # that would free the slots it is waiting for. Give every thread its
        # own control connection instead.
        self._tls = threading.local()
        self._all_socks: List[object] = []
        self._socks_lock = threading.Lock()
        self._round_pool = (ThreadPoolExecutor(
            max_workers=_ROUND_THREADS, thread_name_prefix="nixl-round")
            if _ROUND_THREADS > 1 else None)

    def _sock_for_thread(self):
        sock = getattr(self._tls, "sock", None)
        if sock is None:
            import socket

            sock = socket.create_connection(
                (self._config.daemon_host, self._config.daemon_port), timeout=self._timeout
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._tls.sock = sock
            with self._socks_lock:
                self._all_socks.append(sock)
        return sock

    def _rpc(self, header: dict) -> dict:
        sock = self._sock_for_thread()
        protocol.send_message(sock, header)
        resp, _ = protocol.recv_message(sock)
        return resp

    def _notify(self, header: dict) -> None:
        """Fire-and-forget control message; the daemon sends no reply."""
        header = dict(header, noreply=True)
        protocol.send_message(self._sock_for_thread(), header)

    def start(self) -> None:
        self._endpoint = NixlEndpoint(f"kvshrink-{self._session_id}")
        # Exchange agent metadata with the daemon over the control plane.
        resp = self._rpc(
            {
                "type": protocol.MSG_NIXL_HANDSHAKE,
                "session_id": self._session_id,
                "agent_metadata": self._endpoint.metadata().hex(),
            }
        )
        if resp.get("status") != "ok":
            raise RuntimeError(f"nixl handshake failed: {resp.get('error')}")
        self._endpoint.add_remote(self._remote_key, bytes.fromhex(resp["agent_metadata"]))

    def _ensure_registered(self, tensor: torch.Tensor) -> None:
        tid = tensor.data_ptr()
        if tid in self._registered:
            return
        with self._register_lock:
            if tid not in self._registered:
                nbytes = tensor.numel() * tensor.element_size()
                triple = (tensor.data_ptr(), nbytes, _dev_id(tensor))
                self._registered[tid] = self._endpoint.register([triple], _mem_type(tensor))

    def _local_triples(self, shards: List[ShardRef]) -> List[Tuple[int, int, int]]:
        triples = []
        for s in shards:
            self._ensure_registered(s.tensor)
            base = s.tensor.data_ptr()
            triples.append((base + s.offset, s.length, _dev_id(s.tensor)))
        return triples

    def _rdma(self, op: str, shards: List[ShardRef], resp: dict, notif: bytes) -> None:
        base = resp["staging_base"]
        stride = resp["staging_stride"]
        dev = resp["staging_dev"]
        remote_triples = [(base + sid * stride, s.length, dev)
                          for sid, s in zip(resp["slots"], shards)]
        handle = self._endpoint.post(
            op, self._local_triples(shards), _mem_type(shards[0].tensor),
            remote_triples, resp["staging_mem"],
            self._endpoint.remote_name(self._remote_key), notif)
        try:
            self._endpoint.wait(handle, self._timeout)
        finally:
            self._endpoint.release(handle)

    def _run_rounds(self, fn, session_id: str, shards: List[ShardRef]) -> None:
        batches = list(_chunked(shards, self._round_shards))
        if self._round_pool is None or len(batches) <= 1:
            for batch in batches:
                fn(session_id, batch)
            return
        futures = [self._round_pool.submit(fn, session_id, b) for b in batches]
        first_error = None
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - drain all before reporting
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _round_spec(shards: List[ShardRef]) -> dict:
        """Factored shard list: keys plus one dtype/size when they are uniform.

        All shards of a round come from the same layer, so dtype and length
        are normally identical; sending them once keeps the control-plane
        JSON (which the vLLM worker pays for under the GIL) proportional to
        the key text only.
        """
        lengths = [s.length for s in shards]
        first = lengths[0]
        return {
            "keys": [s.key for s in shards],
            "dtype": dtype_to_str(shards[0].dtype),
            "nbytes": first if all(n == first for n in lengths) else lengths,
        }

    def put(self, session_id: str, shards: List[ShardRef]) -> None:
        self._run_rounds(self._put_round, session_id, shards)

    def _put_round(self, session_id: str, shards: List[ShardRef]) -> None:
        if not shards:
            return
        held = self._budget.acquire(len(shards), self._timeout)
        try:
            resp = self._rpc(
                {"type": protocol.MSG_PUT, "mode": "nixl", "phase": "begin",
                 "session_id": session_id, **self._round_spec(shards)},
            )
            if resp.get("status") != "ok":
                raise RuntimeError(f"remote put begin failed: {resp.get('error')}")

            self._rdma("WRITE", shards, resp, f"put:{session_id}".encode())

            # Commit must be synchronous: the caller marks the block ready as
            # soon as put() returns, so the shards have to be in the pool
            # first.
            done = self._rpc(
                {"type": protocol.MSG_PUT, "mode": "nixl", "phase": "commit",
                 "session_id": session_id, "token": resp["token"]},
            )
            if done.get("status") != "ok":
                raise RuntimeError(f"remote put commit failed: {done.get('error')}")
        finally:
            self._budget.release(held)

    def get(self, session_id: str, shards: List[ShardRef]) -> None:
        self._run_rounds(self._get_round, session_id, shards)

    def _get_round(self, session_id: str, shards: List[ShardRef]) -> None:
        if not shards:
            return
        held = self._budget.acquire(len(shards), self._timeout)
        try:
            resp = self._rpc(
                {"type": protocol.MSG_GET, "mode": "nixl", "phase": "begin",
                 "session_id": session_id, **self._round_spec(shards)},
            )
            if resp.get("status") != "ok":
                raise RuntimeError(f"remote get begin failed: {resp.get('error')}")

            self._rdma("READ", shards, resp, f"get:{session_id}".encode())

            # The KV bytes have already landed, so waiting for the daemon to
            # ack the slot release would only add a round trip to the load
            # path that attention is blocked on. Fire and forget; the
            # daemon's stale-token reaper covers the case where this message
            # is lost.
            self._notify(
                {"type": protocol.MSG_GET, "mode": "nixl", "phase": "done",
                 "session_id": session_id, "token": resp["token"]},
            )
        finally:
            self._budget.release(held)

    def close(self) -> None:
        try:
            if self._round_pool is not None:
                self._round_pool.shutdown(wait=True)
            with self._socks_lock:
                for sock in self._all_socks:
                    try:
                        sock.close()
                    except OSError:
                        pass
                self._all_socks.clear()
        finally:
            if self._endpoint is not None:
                self._endpoint.close()


class NixlStagingPool:
    """Daemon-side registered staging arena with a simple free list.

    One contiguous buffer (VRAM if the daemon has a GPU, else DRAM -- the
    normal case, since the compress/decompress path this daemon uses is
    CPU-only, see ``server.py``) is registered with NIXL exactly once and
    carved into fixed-size slots. Keeping it to a single registration is what
    makes a large slot count affordable: with per-slot registration, going
    from a few hundred to a few thousand in-flight shards would cost
    thousands of ibverbs MR registrations at startup. ``alloc`` returns
    per-shard (addr, len, dev) triples that the client RDMA-writes/reads; the
    daemon then reads/fills those regions.
    """

    _ALIGN = 4096

    def __init__(self, endpoint: NixlEndpoint, slot_bytes: int, num_slots: int, device: str,
                alloc_timeout_sec: float = 60.0):
        self._endpoint = endpoint
        self._slot_bytes = (slot_bytes + self._ALIGN - 1) // self._ALIGN * self._ALIGN
        self._device = device
        self._num_slots = num_slots
        self._alloc_timeout_sec = alloc_timeout_sec
        self._mem = "VRAM" if device != "cpu" else "DRAM"

        # Over-allocate by one alignment unit so slot 0 (and therefore every
        # slot) starts page-aligned, which both the NIC and the compression
        # kernels prefer, and so uint8 slices can be reinterpreted as
        # fp16/bf16.
        total = self._slot_bytes * num_slots
        self._buffer = torch.empty(total + self._ALIGN, dtype=torch.uint8, device=device)
        raw = self._buffer.data_ptr()
        self._pad = (-raw) % self._ALIGN
        self._base = raw + self._pad
        self._dev = self._buffer.get_device() if self._buffer.is_cuda else 0
        self._endpoint.register([(self._base, total, self._dev)], self._mem)

        # Multiple client sessions/ranks request slots concurrently (one
        # thread per connection in server.py), so the free list needs its
        # own lock; the condition also lets alloc() wait for a concurrent
        # put/get to finish and release slots instead of failing outright.
        self._cv = threading.Condition()
        self._free: List[int] = list(range(num_slots))
        logger.info("NIXL staging arena: %d slots x %d KiB = %.2f GiB (%s)",
                    num_slots, self._slot_bytes // 1024, total / 1024 ** 3, self._mem)

    @property
    def mem_type(self) -> str:
        return self._mem

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    @property
    def base_addr(self) -> int:
        return self._base

    @property
    def dev_id(self) -> int:
        return self._dev

    def alloc(self, n: int) -> List[int]:
        if n > self._num_slots:
            raise RuntimeError(
                f"staging pool too small: request needs {n} slots but only "
                f"{self._num_slots} exist; raise STAGING_SLOTS (and/or "
                f"STAGING_SLOT_MB if shards exceed the slot size)"
            )
        with self._cv:
            ok = self._cv.wait_for(lambda: len(self._free) >= n, timeout=self._alloc_timeout_sec)
            if not ok:
                raise RuntimeError(
                    f"staging pool exhausted: needed {n} slots, "
                    f"{len(self._free)}/{self._num_slots} free after "
                    f"{self._alloc_timeout_sec}s wait"
                )
            return [self._free.pop() for _ in range(n)]

    def free(self, slot_ids: List[int]) -> None:
        if not slot_ids:
            return
        with self._cv:
            self._free.extend(slot_ids)
            self._cv.notify_all()

    def triple(self, slot_id: int, length: int) -> Tuple[int, int, int]:
        return (self._base + slot_id * self._slot_bytes, length, self._dev)

    def view(self, slot_id: int, length: int) -> torch.Tensor:
        off = self._pad + slot_id * self._slot_bytes
        return self._buffer[off:off + length]
