# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Remote cache daemon: control-plane server + native pool/compression.

Long-running service that vLLM workers connect to. Responsibilities:

  - Negotiate capability and create per-(model, tp_size, tp_rank) sessions.
  - Track block readiness for scheduler ``has()`` queries.
  - Receive KV shards (TCP inline, or NIXL RDMA into staging) and hand them
    to the *same native code the local, GPU-attached KVStore uses*:
    ``iaxl.torch_ext.zip_compress_to_mem`` / ``zip_decompress_from_mem`` (the
    shared QAT/IAA/CPU zip task pool) and ``iaxl.torch_ext.Mem`` / ``Storage``
    / ``Record`` (grouped DDR pool with LRU + SQLite/disk persistence). This
    is what makes the remote daemon reuse iaxl's pool management, compression
    and chunk-persistence logic rather than reimplementing it -- see the
    design doc's "reuse scope" section.

Each distinct (model_name, tp_size, tp_rank) gets its own :class:`RemoteCacheGroup`
(its own ``Mem``/``Storage``/``Record`` triple and on-disk directory), mirroring
exactly the per-rank ``{model}_rank{rank}`` layout ``iaxl.kvflow.KVFlow`` uses
locally (with ``tp_size`` folded into the directory name to disambiguate
multiple TP configurations of the same model sharing one daemon).

One thread per client connection. A whole round of shards is handed to the
zip pipeline as a single batch so QAT/IAA can drive all of their instances in
parallel; on a DRAM staging pool the (de)compression reads/writes straight
into/out of the staging slots, so there is no extra host copy on either path.
"""

from __future__ import annotations

import collections
import logging
import os
import socket
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import torch

from . import protocol
from .metadata import full_chunk_label
from .transport.base import str_to_dtype

logger = logging.getLogger(__name__)

_STAGING_ALIGN = 4096


def _env_flag(name: str) -> bool:
    return os.environ.get(f"KVSHRINK_REMOTE_{name}", "").strip().lower() in ("1", "true", "yes", "on")


def _elem_size(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


class _CodecWorker:
    """Single-worker priority queue for the native zip pipeline.

    The daemon has one connection thread per (client, rank), so PUT and GET
    RPCs land on the codec concurrently. The native ``kv_zip_*`` batch
    entry points index QAT/IAA/CPU slots by OMP thread id and are only safe
    when at most one call is in flight per process (see the ``_run_codec``
    docstring in :class:`RemoteCacheDaemon` for the exact failure mode).
    This queue enforces that invariant while still letting GET (critical
    path) preempt any pending PUT (background save), matching the intent of
    the C++ ``omp_queue`` TaskQueue used by the local, GPU-attached KVStore
    (iaxl/csrc/torch_ext/context.h::omp_queue).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        # Two FIFO queues; GET is drained before PUT.
        self._get_q: "collections.deque[_CodecJob]" = collections.deque()
        self._put_q: "collections.deque[_CodecJob]" = collections.deque()
        self._stop = False
        self._thread = threading.Thread(
            target=self._worker_loop, name="iaxl-remote-codec", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, op: str, fn, args) -> Tuple[object, float]:
        """Enqueue ``fn(*args)`` and block until it finishes.

        Returns ``(result, busy_sec)`` where ``busy_sec`` is the actual time
        spent in ``fn`` (excludes queue wait), matching what ``_run_codec``
        previously reported as "codec busy" time.
        """
        job = _CodecJob(fn, args)
        with self._cv:
            if self._stop:
                raise RuntimeError("codec worker is shut down")
            (self._get_q if op == "get" else self._put_q).append(job)
            self._cv.notify()
        job.done.wait()
        if job.exc is not None:
            raise job.exc
        return job.result, job.busy_sec

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._get_q and not self._put_q:
                    self._cv.wait()
                if self._stop and not self._get_q and not self._put_q:
                    return
                job = self._get_q.popleft() if self._get_q else self._put_q.popleft()
            t0 = time.monotonic()
            try:
                job.result = job.fn(*job.args)
            except BaseException as exc:  # propagate to caller thread
                job.exc = exc
            finally:
                job.busy_sec = time.monotonic() - t0
                job.done.set()


class _CodecJob:
    __slots__ = ("fn", "args", "done", "result", "exc", "busy_sec")

    def __init__(self, fn, args) -> None:
        self.fn = fn
        self.args = args
        self.done = threading.Event()
        self.result = None
        self.exc: Optional[BaseException] = None
        self.busy_sec = 0.0


class RoundStats:
    """Aggregate daemon-side timings, logged periodically.

    The remote (de)compression sits directly on the vLLM prefill critical
    path, so knowing how much of a round is codec time versus staging wait is
    what tells an operator whether to add QAT instances, staging slots, or
    neither.
    """

    def __init__(self, interval_sec: float):
        self.interval = interval_sec
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        self._n = {"get": 0, "put": 0}
        self._shards = {"get": 0, "put": 0}
        self._bytes = {"get": 0, "put": 0}
        self._codec = {"get": 0.0, "put": 0.0}
        self._alloc = {"get": 0.0, "put": 0.0}
        self._queue = {"get": 0.0, "put": 0.0}

    def record(self, op: str, shards: int, nbytes: int, codec_sec: float,
              alloc_sec: float, queue_sec: float = 0.0) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            self._n[op] += 1
            self._shards[op] += shards
            self._bytes[op] += nbytes
            self._codec[op] += codec_sec
            self._alloc[op] += alloc_sec
            self._queue[op] += queue_sec
            now = time.monotonic()
            if now - self._last < self.interval:
                return
            elapsed = now - self._last
            self._last = now
            parts = []
            for k in ("get", "put"):
                if not self._n[k]:
                    continue
                parts.append(
                    "%s: %d rounds / %d shards / %.2f GiB, codec %.1fs busy "
                    "(%.0f MiB/s/thread), codec queue %.1fs, staging wait %.2fs" % (
                        k.upper(), self._n[k], self._shards[k], self._bytes[k] / 1024 ** 3,
                        self._codec[k],
                        self._bytes[k] / 1024 ** 2 / max(self._codec[k], 1e-9),
                        self._queue[k], self._alloc[k]))
            self._reset()
        if parts:
            logger.info("last %.0fs -- %s", elapsed, "; ".join(parts))


class RemoteCacheGroup:
    """One native (Storage, Record, Mem) triple per (model, tp_size, tp_rank).

    Mirrors ``iaxl.kvflow.KVFlow.__init__``'s own on-disk layout
    (``chunks.db`` + ``chunks/`` under a persist directory) but skips the
    GPU-bound streams/ScratchPool/Context machinery that class also sets up,
    since this daemon never transfers to/from a GPU (see the design doc:
    ``zip_compress_to_mem``/``zip_decompress_from_mem`` operate on
    CPU-resident staging tensors only).
    """

    def __init__(self, base_dir: str, model_name: str, tp_size: int, tp_rank: int,
                pool_bytes: int, cleanup_unpersisted: bool = True):
        from ..torch_ext import Mem, Record, Storage

        safe_model = model_name.replace("/", "_").replace(":", "_")
        self.persist_dir = os.path.join(base_dir, f"{safe_model}_tp{tp_size}_rank{tp_rank}")
        os.makedirs(self.persist_dir, exist_ok=True)
        self.storage = Storage(self.persist_dir)
        self.record = Record(os.path.join(self.persist_dir, "chunks.db"), cleanup_unpersisted)
        self.mem = Mem(capacity_bytes=pool_bytes, storage=self.storage, record=self.record)
        logger.info("remote cache group ready: model=%s tp=%d rank=%d persist_dir=%s pool=%.2f GiB",
                    model_name, tp_size, tp_rank, self.persist_dir, pool_bytes / 1024 ** 3)


class SessionState:
    def __init__(self, session_id: str, header: dict, group: RemoteCacheGroup):
        self.session_id = session_id
        self.model_name = header["model_name"]
        self.tp_size = header["tp_size"]
        self.tp_rank = header["tp_rank"]
        self.num_layers = header["num_layers"]
        self.tensor_keys = header.get("tensor_keys", [])
        self.dtype = header.get("dtype", "float16")
        self.role = header.get("role", "worker")
        self.group = group


class RemoteCacheDaemon:
    def __init__(self, host: str, port: int, pool_bytes: int, cache_dir: str,
                 compress: bool = True,
                 nixl_host: str = "", nixl_port: int = 0, staging_slot_bytes: int = 0,
                 staging_slots: int = 0, device: str = "cpu"):
        self.host = host
        self.port = port
        self.pool_bytes = pool_bytes
        self.cache_dir = cache_dir
        self.compress = compress
        self.device = device

        self._groups: Dict[Tuple[str, int, int], RemoteCacheGroup] = {}
        self._groups_lock = threading.Lock()

        self._sessions: Dict[str, SessionState] = {}
        # Readiness: (model, tp_size, tp_rank, block_hash) present.
        self._ready: set[Tuple[str, int, int, str]] = set()
        self._state_lock = threading.Lock()

        self._nixl_host = nixl_host
        self._nixl_port = nixl_port
        self._staging_slot_bytes = staging_slot_bytes
        self._staging_slots = staging_slots
        # Slots are what limits how many shards can be in flight, and
        # operators size them generously (MiB) while a KV shard is tens of
        # KiB. Keep the configured *total* staging bytes but re-cut them once
        # we learn the real shard size from a worker, so the same memory buys
        # 1-2 orders of magnitude more concurrency. Disable with
        # STAGING_FIXED_SLOT_SIZE=1.
        self._staging_total_bytes = staging_slot_bytes * staging_slots
        self._staging_autosize = bool(staging_slots) and not _env_flag("STAGING_FIXED_SLOT_SIZE")
        self._staging_max_slots = int(os.environ.get("KVSHRINK_REMOTE_MAX_STAGING_SLOTS", "4096"))
        self._stats = RoundStats(float(os.environ.get("KVSHRINK_REMOTE_STATS_INTERVAL_SEC", "30")))
        # The native zip pipeline (kv_zip_compress_batch /
        # kv_zip_decompress_batch) launches an OpenMP team that indexes
        # QAT/IAA/CPU slots by *thread id* (base + k in
        # iaxl/csrc/kv_zip/kv_zip.cpp:zip_pipeline). Those slot arrays are
        # process-global, so two concurrent calls -- e.g. one GET and one
        # PUT, or two GETs -- have OMP threads with the same t clobbering
        # each other's inputs in submit_slot(memcpy(sl->in, ...)) and reading
        # back the wrong output in wait_op, which manifests as
        #   kv_zip: decompressed size does not match tensor byte size
        #   kv_zip: zip wait failed
        # and aborts the daemon. The local, GPU-attached KVStore avoids this
        # by dispatching every zip_compress_to_mem/zip_decompress_from_mem
        # through the single-worker omp_queue TaskQueue in
        # iaxl/csrc/torch_ext/context.h. Mirror that here with one
        # priority-aware serial worker: GET runs ahead of any pending PUT
        # (critical path vs. background save), but only one codec call is
        # ever in flight per daemon process. Native OMP still parallelises
        # each call across all IAXL_OMP_THREAD_NUM workers, so throughput is
        # not lost -- only unsafe outer concurrency is.
        self._codec_worker = _CodecWorker()
        self._codec_worker.start()
        self._nixl_endpoint = None
        self._staging = None
        # Worker handshakes arrive concurrently (one connection/thread per
        # rank); without this lock two of them race in _ensure_nixl and each
        # build a separate NIXL agent + staging pool.
        self._nixl_lock = threading.Lock()
        # token -> {"entries": [(slot_id, spec), ...], "ts": monotonic-alloc-time}.
        self._nixl_tokens: Dict[str, dict] = {}
        self._nixl_token_ttl_sec = 300.0
        self._token_lock = threading.Lock()

        self._server_sock: Optional[socket.socket] = None
        self._stop = threading.Event()

    # -- native pool group management ------------------------------------------

    def _group_for(self, model_name: str, tp_size: int, tp_rank: int) -> RemoteCacheGroup:
        key = (model_name, tp_size, tp_rank)
        group = self._groups.get(key)
        if group is not None:
            return group
        with self._groups_lock:
            group = self._groups.get(key)
            if group is None:
                group = RemoteCacheGroup(self.cache_dir, model_name, tp_size, tp_rank, self.pool_bytes)
                self._groups[key] = group
            return group

    # -- lifecycle ------------------------------------------------------------

    def serve_forever(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(128)
        logger.info("Remote cache daemon listening on %s:%d (compress=%s, device=%s, cache_dir=%s)",
                    self.host, self.port, self.compress, self.device, self.cache_dir)
        threading.Thread(target=self._reap_stale_nixl_tokens, daemon=True).start()
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = self._server_sock.accept()
                except OSError:
                    break
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=self._handle_conn, args=(conn, addr),
                                 daemon=True).start()
        finally:
            self._server_sock.close()

    def _reap_stale_nixl_tokens(self) -> None:
        while not self._stop.wait(self._nixl_token_ttl_sec / 6):
            now = time.monotonic()
            with self._token_lock:
                stale = [t for t, v in self._nixl_tokens.items()
                         if now - v["ts"] > self._nixl_token_ttl_sec]
                entries = [(t, self._nixl_tokens.pop(t)) for t in stale]
            for token, entry in entries:
                if self._staging is None:
                    continue
                slot_ids = [sid for sid, _ in entry["entries"]]
                self._staging.free(slot_ids)
                logger.warning(
                    "reaped %d staging slot(s) from abandoned nixl token %s "
                    "(no commit/done for over %.0fs; client likely died mid-transfer)",
                    len(slot_ids), token, self._nixl_token_ttl_sec)

    def shutdown(self) -> None:
        self._stop.set()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        self._codec_worker.shutdown()

    # -- connection loop ------------------------------------------------------

    def _handle_conn(self, conn: socket.socket, addr) -> None:
        logger.debug("client connected: %s", addr)
        header: dict = {}
        try:
            while True:
                header, blob = protocol.recv_message(conn)
                mtype = header.get("type")
                if mtype == protocol.MSG_STOP:
                    break
                handler = self._dispatch.get(mtype)
                if handler is None:
                    protocol.send_message(conn, {"status": "error", "error": f"unknown type {mtype}"})
                    continue
                handler(self, conn, header, blob)
        except protocol.ProtocolError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("error handling client %s", addr)
            if not header.get("noreply"):
                try:
                    protocol.send_message(conn, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                except OSError:
                    pass
        finally:
            conn.close()

    # -- shard key parsing ------------------------------------------------------

    def _shard_specs(self, header: dict) -> List[dict]:
        """Expand a round's compact shard list into ``[{block_hash, layer_id,
        tensor_key, dtype, nbytes}, ...]``.

        The wire key is ``block_hash|layer_id|tensor_key`` (session-relative
        -- no model/tp/rank prefix is needed any more since the session
        already maps 1:1 to a :class:`RemoteCacheGroup`).
        """
        keys = header.get("keys")
        if keys is None:
            pairs = [(s["key"], s["dtype"], s["nbytes"]) for s in header["shards"]]
        else:
            dtype = header["dtype"]
            nbytes = header["nbytes"]
            if isinstance(nbytes, list):
                pairs = [(k, dtype, n) for k, n in zip(keys, nbytes)]
            else:
                pairs = [(k, dtype, nbytes) for k in keys]
        specs = []
        for key, dtype, nbytes in pairs:
            block_hash, layer_id, tensor_key = key.split("|", 2)
            specs.append({
                "block_hash": block_hash, "layer_id": layer_id, "tensor_key": tensor_key,
                "dtype": dtype, "nbytes": nbytes,
            })
        return specs

    @staticmethod
    def _full_keys(specs: List[dict]) -> List[str]:
        return [full_chunk_label(s["block_hash"], s["layer_id"], s["tensor_key"]) for s in specs]

    # -- handlers -------------------------------------------------------------

    def _h_capability(self, conn, header, blob):
        protocol.send_message(conn, {
            "status": "ok",
            "protocol_version": protocol.PROTOCOL_VERSION,
            "daemon_id": "iaxl-remote-cache",
            "features": {"nixl": bool(self._staging_slots), "compress": self.compress,
                         "admin": True},
        })

    def _h_session_create(self, conn, header, blob):
        session_id = uuid.uuid4().hex[:12]
        group = self._group_for(header["model_name"], header["tp_size"], header["tp_rank"])
        with self._state_lock:
            self._sessions[session_id] = SessionState(session_id, header, group)
        self._autosize_staging(int(header.get("shard_bytes", 0) or 0))
        logger.info("session %s: model=%s tp=%d rank=%d role=%s",
                    session_id, header["model_name"], header["tp_size"],
                    header["tp_rank"], header.get("role"))
        protocol.send_message(conn, {
            "status": "ok",
            "session_id": session_id,
            "staging_slots": self._staging_slots,
            "staging_slot_bytes": self._staging_slot_bytes,
        })

    def _autosize_staging(self, shard_bytes: int) -> None:
        """Re-cut the staging arena into slots that match the real shard size."""
        with self._nixl_lock:
            if (not self._staging_autosize or self._nixl_endpoint is not None
                    or shard_bytes <= 0):
                return
            self._staging_autosize = False
            slot = max(_STAGING_ALIGN,
                       (shard_bytes + _STAGING_ALIGN - 1) // _STAGING_ALIGN * _STAGING_ALIGN)
            if slot >= self._staging_slot_bytes:
                return
            slots = min(self._staging_max_slots, self._staging_total_bytes // slot)
            if slots <= self._staging_slots:
                return
            logger.info("staging autosize: shard=%d B -> %d slots x %d KiB "
                        "(was %d x %d KiB, same %.2f GiB budget)",
                        shard_bytes, slots, slot // 1024, self._staging_slots,
                        self._staging_slot_bytes // 1024,
                        self._staging_total_bytes / 1024 ** 3)
            self._staging_slot_bytes = slot
            self._staging_slots = slots

    def _h_mark_ready(self, conn, header, blob):
        model = header["model_name"]
        tp = header["tp_size"]
        rank = header["tp_rank"]
        with self._state_lock:
            for bh in header["block_hashes"]:
                self._ready.add((model, tp, rank, bh))
        protocol.send_message(conn, {"status": "ok"})

    def _h_has(self, conn, header, blob):
        model = header["model_name"]
        tp = header["tp_size"]
        # Scheduler proxy: a block is present if rank 0 has recorded it.
        with self._state_lock:
            exists = [(model, tp, 0, bh) in self._ready for bh in header["block_hashes"]]
        protocol.send_message(conn, {"status": "ok", "exists": exists})

    def _run_codec(self, op, fn, *args):
        """Call the native zip entry point through the serial codec worker.

        ``op`` picks the priority so a background save never queues in front
        of a critical-path load. Returns ``(result, busy_sec, queue_sec)``.
        """
        t0 = time.monotonic()
        result, busy_sec = self._codec_worker.submit(op, fn, args)
        queue_sec = max(0.0, (time.monotonic() - t0) - busy_sec)
        return result, busy_sec, queue_sec

    def _h_put(self, conn, header, blob):
        if header.get("mode") == "nixl":
            return self._h_put_nixl(conn, header, blob)
        from ..torch_ext import zip_compress_to_mem

        session = self._sessions[header["session_id"]]
        specs = self._shard_specs(header)
        cursor = 0
        tensors: List[torch.Tensor] = []
        nbytes_total = 0
        for spec in specs:
            nbytes = spec["nbytes"]
            raw = blob[cursor:cursor + nbytes]
            cursor += nbytes
            dtype = str_to_dtype(spec["dtype"])
            numel = nbytes // _elem_size(dtype)
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).view(numel)
            tensors.append(tensor)
            nbytes_total += nbytes
        full_keys = self._full_keys(specs)
        # One batch so QAT/IAA use every instance; the tensors are private
        # copies of the received bytes, so in-place preprocessing is safe.
        _, busy_sec, queue_sec = self._run_codec(
            "put", zip_compress_to_mem, session.group.mem, full_keys, tensors, self.compress)
        self._stats.record("put", len(specs), nbytes_total, busy_sec, 0.0, queue_sec)
        protocol.send_message(conn, {"status": "ok"})

    def _h_get(self, conn, header, blob):
        if header.get("mode") == "nixl":
            return self._h_get_nixl(conn, header, blob)
        from ..torch_ext import zip_decompress_from_mem

        session = self._sessions[header["session_id"]]
        specs = self._shard_specs(header)
        full_keys = self._full_keys(specs)
        outs: List[torch.Tensor] = []
        lengths = []
        for spec in specs:
            dtype = str_to_dtype(spec["dtype"])
            numel = spec["nbytes"] // _elem_size(dtype)
            outs.append(torch.empty(numel, dtype=dtype))
            lengths.append(spec["nbytes"])

        # zip_decompress_from_mem aborts the whole process on a missing key
        # (native IAXL_CHECK semantics -- see torch_ext.cpp), so filter out
        # cache misses (e.g. evicted between has() and get()) before calling
        # it, exactly like the local KVStore's own retry/guard does for a
        # different reason (see Context::unzip_from_mem).
        present = session.group.mem.has(full_keys)
        decompress_keys = [k for k, ok in zip(full_keys, present) if ok]
        decompress_outs = [o for o, ok in zip(outs, present) if ok]
        if decompress_keys:
            _, busy_sec, queue_sec = self._run_codec(
                "get", zip_decompress_from_mem, session.group.mem, decompress_keys, decompress_outs)
        else:
            busy_sec = queue_sec = 0.0
        missing = [k for k, ok in zip(full_keys, present) if not ok]
        for o, ok in zip(outs, present):
            if not ok:
                o.zero_()
        self._stats.record("get", len(specs), sum(lengths), busy_sec, 0.0, queue_sec)
        if missing:
            logger.warning("GET cache miss for %d shard(s): %s", len(missing), missing[:3])
        payload = b"".join(_tensor_to_bytes(t) for t in outs)
        protocol.send_message(conn, {"status": "ok", "lengths": lengths}, payload)

    # -- NIXL handlers (production path; requires nixl on both hosts) ----------

    def _ensure_nixl(self):
        if self._nixl_endpoint is None:
            from .transport.nixl_backend import NixlEndpoint, NixlStagingPool

            with self._nixl_lock:
                if self._nixl_endpoint is None:
                    endpoint = NixlEndpoint("iaxl-remote-daemon")
                    self._staging = NixlStagingPool(
                        endpoint, self._staging_slot_bytes,
                        self._staging_slots, self.device)
                    self._nixl_endpoint = endpoint

    def _h_nixl_handshake(self, conn, header, blob):
        self._ensure_nixl()
        client_meta = bytes.fromhex(header["agent_metadata"])
        self._nixl_endpoint.add_remote(header["session_id"], client_meta)
        protocol.send_message(conn, {
            "status": "ok",
            "agent_metadata": self._nixl_endpoint.metadata().hex(),
        })

    def _begin_round(self, specs) -> Tuple[List[int], str, float]:
        """Reserve one staging slot per shard and register a cleanup token."""
        t0 = time.monotonic()
        slots = self._staging.alloc(len(specs))
        alloc_sec = time.monotonic() - t0
        token = uuid.uuid4().hex[:12]
        with self._token_lock:
            self._nixl_tokens[token] = {"entries": list(zip(slots, specs)),
                                        "ts": time.monotonic(), "alloc_sec": alloc_sec}
        return slots, token, alloc_sec

    def _pop_round(self, token) -> dict:
        with self._token_lock:
            return self._nixl_tokens.pop(token, {"entries": [], "alloc_sec": 0.0})

    def _staging_reply(self, token: str, slots: List[int], specs: List[dict]) -> dict:
        return {
            "status": "ok", "token": token,
            "slots": slots,
            "staging_base": self._staging.base_addr,
            "staging_stride": self._staging.slot_bytes,
            "staging_dev": self._staging.dev_id,
            "staging_mem": self._staging.mem_type,
        }

    def _h_put_nixl(self, conn, header, blob):
        phase = header["phase"]
        if phase == "begin":
            specs = self._shard_specs(header)
            slots, token, _ = self._begin_round(specs)
            protocol.send_message(conn, self._staging_reply(token, slots, specs))
        else:  # commit: read staging, compress, store, free
            from ..torch_ext import zip_compress_to_mem

            session = self._sessions[header["session_id"]]
            record = self._pop_round(header["token"])
            entries = record["entries"]
            tensors: List[torch.Tensor] = []
            nbytes = 0
            for slot_id, spec in entries:
                dtype = str_to_dtype(spec["dtype"])
                numel = spec["nbytes"] // _elem_size(dtype)
                view = self._staging.view(slot_id, spec["nbytes"])
                if view.is_cuda:
                    view = view.cpu()
                tensors.append(view.view(dtype)[:numel])
                nbytes += spec["nbytes"]
            full_keys = self._full_keys([s for _, s in entries])
            _, busy_sec, queue_sec = self._run_codec(
                "put", zip_compress_to_mem, session.group.mem, full_keys, tensors, self.compress)
            self._staging.free([sid for sid, _ in entries])
            self._stats.record("put", len(entries), nbytes, busy_sec, record["alloc_sec"], queue_sec)
            protocol.send_message(conn, {"status": "ok"})

    def _h_get_nixl(self, conn, header, blob):
        phase = header["phase"]
        if phase == "begin":
            from ..torch_ext import zip_decompress_from_mem

            session = self._sessions[header["session_id"]]
            specs = self._shard_specs(header)
            slots, token, alloc_sec = self._begin_round(specs)
            full_keys = self._full_keys(specs)
            present = session.group.mem.has(full_keys)

            decompress_keys = []
            decompress_outs = []
            gpu_copies: List[Tuple[torch.Tensor, torch.Tensor]] = []
            missing = []
            for slot_id, spec, key, ok in zip(slots, specs, full_keys, present):
                dst = self._staging.view(slot_id, spec["nbytes"])
                if not ok:
                    missing.append(key)
                    dst.zero_()
                    continue
                dtype = str_to_dtype(spec["dtype"])
                numel = spec["nbytes"] // _elem_size(dtype)
                if dst.is_cuda:
                    # (De)compression is CPU-side; stage through a host buffer.
                    out = torch.empty(numel, dtype=dtype)
                    gpu_copies.append((dst, out))
                else:
                    # DRAM staging: decompress straight into the RDMA-visible
                    # slot, so GET costs one pass over the data instead of
                    # decompress-to-temp plus a copy.
                    out = dst.view(dtype)[:numel]
                decompress_keys.append(key)
                decompress_outs.append(out)
            if decompress_keys:
                _, busy_sec, queue_sec = self._run_codec(
                    "get", zip_decompress_from_mem, session.group.mem,
                    decompress_keys, decompress_outs)
            else:
                busy_sec = queue_sec = 0.0
            for dst, src in gpu_copies:
                dst.copy_(src.reshape(-1).view(torch.uint8))
            self._stats.record("get", len(specs), sum(s["nbytes"] for s in specs),
                               busy_sec, alloc_sec, queue_sec)
            if missing:
                logger.warning("GET cache miss for %d/%d shard(s): %s",
                               len(missing), len(specs), missing[:3])
            protocol.send_message(conn, self._staging_reply(token, slots, specs))
        else:  # done: free slots
            entries = self._pop_round(header["token"])["entries"]
            self._staging.free([sid for sid, _ in entries])
            if not header.get("noreply"):
                protocol.send_message(conn, {"status": "ok"})

    # -- admin / CLI handlers -------------------------------------------------
    # Session-less; the CLI (iaxl.remote.admin) connects, negotiates
    # capability, then issues one of these directly. Persist/evict apply per
    # RemoteCacheGroup, so a missing (model_name, tp_size, tp_rank) filter
    # means "every group".

    def _select_groups(self, header):
        model = header.get("model_name")
        tp = header.get("tp_size")
        rank = header.get("tp_rank")
        with self._groups_lock:
            items = list(self._groups.items())
        selected = []
        for key, group in items:
            m, t, r = key
            if model is not None and m != model:
                continue
            if tp is not None and int(tp) != t:
                continue
            if rank is not None and int(rank) != r:
                continue
            selected.append((key, group))
        return selected

    def _group_snapshot(self, key, group):
        m, tp, rank = key
        mem = group.mem
        cap = int(mem.capacity_bytes)
        cur = int(mem.current_bytes)
        return {
            "model_name": m, "tp_size": tp, "tp_rank": rank,
            "persist_dir": group.persist_dir,
            "cache_entries": int(mem.size),
            "group_count": int(mem.group_count),
            "current_bytes": cur,
            "capacity_bytes": cap,
            "usage_pct": round(cur / cap * 100, 2) if cap > 0 else 0.0,
            "hits": int(mem.hits),
            "hits_in_storage": int(mem.hits_in_storage),
            "misses": int(mem.misses),
            "puts": int(mem.puts),
            "evictions": int(mem.evictions),
            "total_unzip_bytes": int(mem.total_unzip_bytes),
            "total_zip_bytes": int(mem.total_zip_bytes),
            "compression_ratio": round(float(mem.compression_ratio), 2),
            "unpersisted_count": int(mem.unpersisted_count),
        }

    def _h_status(self, conn, header, blob):
        selected = self._select_groups(header)
        groups = [self._group_snapshot(k, g) for k, g in selected]
        with self._state_lock:
            sessions = [
                {"session_id": s.session_id, "model_name": s.model_name,
                 "tp_size": s.tp_size, "tp_rank": s.tp_rank, "role": s.role}
                for s in self._sessions.values()
            ]
            num_ready = len(self._ready)
        protocol.send_message(conn, {
            "status": "ok",
            "daemon": {
                "host": self.host,
                "port": self.port,
                "compress": self.compress,
                "device": self.device,
                "cache_dir": self.cache_dir,
                "pool_bytes": self.pool_bytes,
                "staging_slots": self._staging_slots,
                "staging_slot_bytes": self._staging_slot_bytes,
                "nixl_ready": self._nixl_endpoint is not None,
                "num_sessions": len(sessions),
                "num_ready_blocks": num_ready,
                "num_groups": len(groups),
            },
            "groups": groups,
            "sessions": sessions,
        })

    @staticmethod
    def _group_keys_from_entries(entries):
        """entries is a list of (full_label, bytes) from Mem.get_unpersisted /
        Mem.get_lru_oldest; dedup down to per-group keys the way flow.py does."""
        seen: set = set()
        out = []
        for full_label, _ in entries:
            group_key = full_label.rsplit(":", 1)[0]
            if group_key not in seen:
                seen.add(group_key)
                out.append(group_key)
        return out

    def _h_persist(self, conn, header, blob):
        count = max(0, int(header.get("count", 10)))
        selected = self._select_groups(header)
        per_group = []
        total_groups = 0
        total_bytes = 0
        for key, group in selected:
            entries = group.mem.persist_groups(count)
            labels = sorted(e[0] for e in entries)
            nbytes = sum(int(e[1]) for e in entries)
            total_groups += len(entries)
            total_bytes += nbytes
            m, tp, rank = key
            per_group.append({
                "model_name": m, "tp_size": tp, "tp_rank": rank,
                "persisted": len(entries),
                "bytes_written": nbytes,
                "labels": labels,
            })
        if not selected:
            logger.warning("persist: no matching group for filter %s", {
                k: header.get(k) for k in ("model_name", "tp_size", "tp_rank")})
        protocol.send_message(conn, {
            "status": "ok",
            "persisted": total_groups,
            "bytes_written": total_bytes,
            "groups": per_group,
        })

    def _h_evict(self, conn, header, blob):
        count = max(0, int(header.get("count", 10)))
        selected = self._select_groups(header)
        per_group = []
        total_groups = 0
        total_bytes = 0
        for key, group in selected:
            entries = group.mem.evict_groups(count)
            labels = sorted(e[0] for e in entries)
            nbytes = sum(int(e[1]) for e in entries)
            total_groups += len(entries)
            total_bytes += nbytes
            m, tp, rank = key
            per_group.append({
                "model_name": m, "tp_size": tp, "tp_rank": rank,
                "evicted": len(entries),
                "bytes_freed": nbytes,
                "labels": labels,
            })
        if not selected:
            logger.warning("evict: no matching group for filter %s", {
                k: header.get(k) for k in ("model_name", "tp_size", "tp_rank")})
        protocol.send_message(conn, {
            "status": "ok",
            "evicted": total_groups,
            "bytes_freed": total_bytes,
            "groups": per_group,
        })

    def _h_persist_candidates(self, conn, header, blob):
        count = max(0, int(header.get("count", 10)))
        selected = self._select_groups(header)
        per_group = []
        for key, group in selected:
            entries = group.mem.get_unpersisted(count)
            m, tp, rank = key
            per_group.append({
                "model_name": m, "tp_size": tp, "tp_rank": rank,
                "candidates": self._group_keys_from_entries(entries),
            })
        protocol.send_message(conn, {"status": "ok", "groups": per_group})

    def _h_evict_candidates(self, conn, header, blob):
        count = max(0, int(header.get("count", 10)))
        selected = self._select_groups(header)
        per_group = []
        for key, group in selected:
            entries = group.mem.get_lru_oldest(count)
            m, tp, rank = key
            per_group.append({
                "model_name": m, "tp_size": tp, "tp_rank": rank,
                "candidates": self._group_keys_from_entries(entries),
            })
        protocol.send_message(conn, {"status": "ok", "groups": per_group})

    def _h_metrics(self, conn, header, blob):
        from ..torch_ext import metrics_set_enabled, metrics_reset, metrics_read

        if "enable" in header:
            metrics_set_enabled(bool(header.get("enable")))
        if header.get("reset"):
            metrics_reset()
        protocol.send_message(conn, {"status": "ok", "metrics": metrics_read()})

    _dispatch = {
        protocol.MSG_CAPABILITY: _h_capability,
        protocol.MSG_SESSION_CREATE: _h_session_create,
        protocol.MSG_HAS: _h_has,
        protocol.MSG_PUT: _h_put,
        protocol.MSG_GET: _h_get,
        protocol.MSG_MARK_READY: _h_mark_ready,
        protocol.MSG_NIXL_HANDSHAKE: _h_nixl_handshake,
        protocol.MSG_STATUS: _h_status,
        protocol.MSG_PERSIST: _h_persist,
        protocol.MSG_EVICT: _h_evict,
        protocol.MSG_METRICS: _h_metrics,
        protocol.MSG_PERSIST_CANDIDATES: _h_persist_candidates,
        protocol.MSG_EVICT_CANDIDATES: _h_evict_candidates,
    }
