"""RPC between KVStoreRemote (client) and the daemon's KVStoreLocal, over one NIXL agent.

Transport (§3.8 of DESIGN.md): every side owns a registered control buffer.
Client requests RDMA-WRITE their payload into the daemon's buffer and carry the
header in the notification attached to that write, so the header arrives after
the data has landed. Responses and `done` pushes are plain notifications.

  request  header  "<cBIQ"  kind (b"Q": payload in daemon buf | b"I": inline), method, seq, len
  response header  "<cIB"   b"R", seq, ok            + payload
  push     header  "<cIH"   b"D", job_id, layer_idx
"""

import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set

import numpy as np
import torch

from ..utils import cuda_available

logger = logging.getLogger(__name__)

CTRL_BYTES = 4 << 20

HELLO, REGISTER_KV_CACHES, REGISTER_LAYERS, PUT, GET, HAS, STATUS, METRICS, PERSIST, EVICT, \
    PERSIST_CANDIDATES, EVICT_CANDIDATES, STOP, UNREGISTER = range(14)

REQ_FMT, RESP_FMT, PUSH_FMT = "<cBIQ", "<cIB", "<cIH"
REQ_SIZE, RESP_SIZE, PUSH_SIZE = (struct.calcsize(f) for f in (REQ_FMT, RESP_FMT, PUSH_FMT))


def rank_port(port: int, rank: int) -> int:
    return port + 1 + rank


# -- codec ---------------------------------------------------------------------

def pack_hashes(hashes: List[str]) -> bytes:
    """Fixed-width ASCII records (width = longest hash, NUL padded)."""
    if not hashes:
        return struct.pack("<IH", 0, 0)
    h = max(map(len, hashes))
    return struct.pack("<IH", len(hashes), h) + np.array(hashes, dtype=f"S{h}").tobytes()


def unpack_hashes(buf: bytes, off: int = 0):
    n, h = struct.unpack_from("<IH", buf, off)
    off += 6
    if n == 0:
        return [], off
    hashes = np.frombuffer(buf, f"S{h}", count=n, offset=off).astype(str).tolist()
    return hashes, off + n * h


def pack_blocks(block_indices, block_hashs, layer_idx, description: str, label: str = "") -> bytes:
    desc = description.encode()
    lab = label.encode()
    return b"".join((
        struct.pack("<IHHH", len(block_indices), len(layer_idx), len(desc), len(lab)),
        np.asarray(block_indices, dtype=np.int32).tobytes(),
        np.asarray(layer_idx, dtype=np.int16).tobytes(),
        pack_hashes(block_hashs),
        desc,
        lab,
    ))


def unpack_blocks(buf: bytes):
    n, nl, nd, nlab = struct.unpack_from("<IHHH", buf)
    off = 10
    indices = np.frombuffer(buf, np.int32, count=n, offset=off).tolist()
    off += 4 * n
    layer_idx = np.frombuffer(buf, np.int16, count=nl, offset=off).tolist()
    off += 2 * nl
    hashes, off = unpack_hashes(buf, off)
    desc = buf[off:off + nd].decode()
    off += nd
    return indices, hashes, layer_idx, desc, buf[off:off + nlab].decode()


def pack_has(block_hashs: List[str], label: str = "", truncate: bool = True) -> bytes:
    lab = label.encode()
    return b"".join((
        struct.pack("<BH", len(lab), 1 if truncate else 0),
        lab,
        pack_hashes(block_hashs),
    ))


def unpack_has(buf: bytes):
    nlab, truncate = struct.unpack_from("<BH", buf)
    off = 3
    label = buf[off:off + nlab].decode()
    off += nlab
    hashes, off = unpack_hashes(buf, off)
    return hashes, label, bool(truncate)


def pack_json(obj) -> bytes:
    return json.dumps(obj).encode()


def unpack_json(buf: bytes):
    return json.loads(buf) if buf else None


def _ctrl_buffer(pin: bool) -> torch.Tensor:
    return torch.empty(CTRL_BYTES, dtype=torch.uint8, device="cpu", pin_memory=pin)


def _json_call(name: str):
    """Handler forwarding JSON-encoded positional args to `KVStore.<name>`."""
    def handler(self, peer, payload):
        args = unpack_json(payload) or []
        return pack_json(getattr(self.kvstore, name)(*args))
    return handler


# -- client --------------------------------------------------------------------

class RpcChannel:
    """Synchronous, single-outstanding client channel. `on_push(job_id, layer_idx)`
    receives `done` pushes that arrive while waiting for responses or in drain()."""

    def __init__(self, xfer, peer: str, on_push: Callable[[int, int], None]):
        self.xfer, self.peer, self.on_push = xfer, peer, on_push
        self.buf = _ctrl_buffer(pin=True)
        self.buf_np = self.buf.numpy()
        self.buf_ptr = self.buf.data_ptr()
        xfer.register_memory(self.buf)
        self.remote_base = 0
        self.remote_bytes = 0
        self._seq = 0
        self._resp = None

    def handshake(self):
        info = unpack_json(self.call(HELLO))
        self.remote_base, self.remote_bytes = info["base"], info["bytes"]

    def call(self, method: int, payload: bytes = b"") -> bytes:
        self._seq = seq = (self._seq + 1) & 0xFFFFFFFF
        n = len(payload)
        if 0 < n <= self.remote_bytes:
            self.buf_np[:n] = np.frombuffer(payload, np.uint8)
            self.xfer.write(self.peer, self.buf_ptr, self.remote_base, n,
                            struct.pack(REQ_FMT, b"Q", method, seq, n))
        else:
            self.xfer.send_notif(self.peer, struct.pack(REQ_FMT, b"I", method, seq, n) + payload)
        while self._resp is None:
            self.drain()
        rseq, ok, body = self._resp
        self._resp = None
        assert rseq == seq, f"rpc: response seq {rseq} != {seq}"
        if not ok:
            raise RuntimeError(f"rpc method {method} failed on {self.peer}: {body.decode(errors='replace')}")
        return body

    def drain(self):
        for _, msg in self.xfer.iter_notifs():
            kind = msg[:1]
            if kind == b"D":
                _, job_id, layer_idx = struct.unpack_from(PUSH_FMT, msg)
                self.on_push(job_id, layer_idx)
            elif kind == b"R":
                _, seq, ok = struct.unpack_from(RESP_FMT, msg)
                self._resp = (seq, ok, msg[RESP_SIZE:])
            else:
                logger.warning("rpc: unexpected notif kind %r", kind)


# -- daemon --------------------------------------------------------------------

@dataclass
class Job:
    peer: str
    tasks: dict
    is_put: bool
    not_done: Set[str] = field(default_factory=set)


class KVStoreService:
    """Daemon-side handlers. Owns the KVStore (created on register_*) and the
    registered control buffer clients write their request payloads into."""

    def __init__(self, xfer, role: str, rank: int = 0, tp_size: int = 1):
        self.xfer, self.role, self.rank, self.tp_size = xfer, role, rank, tp_size
        self.ctrl = _ctrl_buffer(pin=cuda_available())
        self.ctrl_np = self.ctrl.numpy()
        xfer.register_memory(self.ctrl)
        self.kvstore = None
        self.layer_names: List[str] = []
        self.layer_idx: Dict[str, int] = {}
        self.remote_bases: List[int] = []
        self.jobs: Dict[int, Job] = {}
        self._next_job = 0

    # -- dispatch ------------------------------------------------------------
    def dispatch(self, peer: str, method: int, payload: bytes) -> bytes:
        return self._HANDLERS[method](self, peer, payload)

    def _hello(self, peer, payload):
        self.xfer.wait_peer(peer)  # we need the client's metadata to notify it
        return pack_json({"base": self.ctrl.data_ptr(), "bytes": CTRL_BYTES})

    def _check_topology(self, req):
        if req["tp_size"] != self.tp_size:
            raise ValueError(f"client tp_size {req['tp_size']} != daemon tp_size {self.tp_size}")
        if req.get("rank", 0) != self.rank:
            raise ValueError(f"client rank {req.get('rank')} != daemon rank {self.rank}")

    def _register_kv_caches(self, peer, payload):
        from iaxl import torch_ext
        from iaxl.kvstore import KVStoreLocal

        from .remote_tensor import RemoteTensor

        req = unpack_json(payload)
        self._check_topology(req)
        if self.kvstore is not None:
            raise RuntimeError("kv_caches already registered")
        block_dim = req["block_dim"]
        kv_caches = {}
        for name, t in req["layers"].items():
            dtype = getattr(torch, t["dtype"])
            shape = tuple(t["shape"])
            rt = RemoteTensor(peer, t["base"], shape, dtype, t["dev_id"])
            torch_ext.rdma_register_remote(peer, rt.base, list(shape), rt.element_size(), rt.dev_id, block_dim)
            self.remote_bases.append(rt.base)
            kv_caches[name] = rt
        self.kvstore = KVStoreLocal(model_name=req["model_name"], block_dim=block_dim, kv_caches=kv_caches,
                                    rank=self.rank, tp_size=self.tp_size)
        self._set_layers(self.kvstore.layer_names)
        return pack_json({"layer_names": self.layer_names})

    def _register_layers(self, peer, payload):
        from iaxl.kvstore import KVStoreLocal

        req = unpack_json(payload)
        self._check_topology(req)
        if self.kvstore is None:
            self.kvstore = KVStoreLocal(model_name=req["model_name"], layer_names=req["layer_names"],
                                        tp_size=self.tp_size)
        self._set_layers(self.kvstore.layer_names)
        return b""

    def _set_layers(self, names):
        self.layer_names = list(names)
        self.layer_idx = {n: i for i, n in enumerate(self.layer_names)}

    def _xfer(self, peer, payload, is_put):
        indices, hashes, layer_idx, desc, label = unpack_blocks(payload)
        names = [self.layer_names[i] for i in layer_idx] or None
        fn = self.kvstore.put if is_put else self.kvstore.get
        tasks = fn(indices, hashes, names, desc, label=label or None)
        self._next_job = job_id = (self._next_job + 1) & 0xFFFFFFFF
        self.jobs[job_id] = Job(peer, tasks, is_put, set(tasks.keys()))
        return struct.pack("<I", job_id)

    def _put(self, peer, payload):
        return self._xfer(peer, payload, True)

    def _get(self, peer, payload):
        return self._xfer(peer, payload, False)

    def _has(self, peer, payload):
        hashes, label, truncate = unpack_has(payload)
        flags = self.kvstore.has(hashes, label=label or None, truncate=truncate)
        return np.asarray(flags, dtype=np.uint8).tobytes()

    def _stop(self, peer, payload):
        if self.kvstore is not None:
            self.kvstore.stop()
        self._release_remote()
        return b""

    def _release_remote(self):
        if self.remote_bases:
            from iaxl import torch_ext

            for base in self.remote_bases:
                torch_ext.rdma_unregister_remote(base)
            self.remote_bases.clear()

    def _unregister(self, peer, payload):
        self._release_remote()
        return b""

    _HANDLERS = {
        HELLO: _hello,
        REGISTER_KV_CACHES: _register_kv_caches,
        REGISTER_LAYERS: _register_layers,
        PUT: _put,
        GET: _get,
        HAS: _has,
        STATUS: _json_call("status"),
        METRICS: _json_call("metrics"),
        PERSIST: _json_call("persist"),
        EVICT: _json_call("evict"),
        PERSIST_CANDIDATES: _json_call("get_persist_candidates"),
        EVICT_CANDIDATES: _json_call("get_evict_candidates"),
        STOP: _stop,
        UNREGISTER: _unregister,
    }

    # -- completion polling --------------------------------------------------
    def poll_done(self):
        """Yield (job_id, layer_idx, peer) for every layer that completed since the
        last call; completed layers release their pool blocks immediately."""
        kv = self.kvstore
        for job_id in list(self.jobs):
            job = self.jobs[job_id]
            wait = kv.put_wait if job.is_put else kv.get_wait
            for name in list(job.not_done):
                if wait(job.tasks, [name], wait=False):
                    if job.tasks[name].ctx is not None:  # get_wait(wait=False) only checks
                        wait(job.tasks, [name], wait=True)
                    job.not_done.discard(name)
                    yield job_id, self.layer_idx[name], job.peer
            if not job.not_done:
                del self.jobs[job_id]


def serve(xfer, service: KVStoreService, idle_spin: int = 2000, idle_sleep: float = 50e-6):
    """Single-threaded daemon loop: dispatch requests, poll completions, push `done`."""
    idle = 0
    while True:
        busy = False
        for peer, msg in xfer.iter_notifs():
            busy = True
            kind, method, seq, n = struct.unpack_from(REQ_FMT, msg)
            payload = service.ctrl_np[:n].tobytes() if kind == b"Q" else msg[REQ_SIZE:REQ_SIZE + n]
            try:
                ok, body = True, service.dispatch(peer, method, payload)
            except Exception as e:  # report to the caller, keep serving
                logger.exception("rpc: method %d from %s failed", method, peer)
                ok, body = False, str(e).encode()
            xfer.send_notif(peer, struct.pack(RESP_FMT, b"R", seq, ok) + body)
            if method == STOP:
                return
        for job_id, layer_idx, peer in service.poll_done():
            busy = True
            xfer.send_notif(peer, struct.pack(PUSH_FMT, b"D", job_id, layer_idx))
        if busy or service.jobs:
            idle = 0
        elif idle < idle_spin:
            idle += 1
        else:
            time.sleep(idle_sleep)
