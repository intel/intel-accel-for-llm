# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""TCP data-plane backend (portable, single-host, no RDMA).

Shard bytes travel inline over the same length-prefixed framing as the
control plane. Each worker thread gets its own socket to the daemon so the
``ThreadPoolExecutor`` in :class:`~iaxl.remote.remote_kvstore.RemoteKVStore`
can issue concurrent put/get without serializing on one connection.

PUT sends raw (uncompressed) shard bytes; the daemon compresses (via the
native QAT/IAA/CPU zip pipeline) and stores into the native ``Mem`` pool. GET
requests shards; the daemon decompresses and returns raw bytes, which are
scattered back into the local KV tensors in place.
"""

from __future__ import annotations

import socket
import threading
from typing import List

import torch

from .. import protocol
from .base import DataPlane, ShardRef, dtype_to_str, tensor_byte_view


class TcpDataPlane(DataPlane):
    name = "tcp"

    def __init__(self, config, session_id: str):
        self._host = config.daemon_host
        self._port = config.daemon_port
        self._timeout = config.request_timeout_sec
        self._session_id = session_id
        self._local = threading.local()

    def _sock(self) -> socket.socket:
        sock = getattr(self._local, "sock", None)
        if sock is None:
            sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._local.sock = sock
        return sock

    def put(self, session_id: str, shards: List[ShardRef]) -> None:
        if not shards:
            return
        specs = []
        parts = []
        for s in shards:
            view = tensor_byte_view(s.tensor)
            blob = bytes(view[s.offset:s.offset + s.length].cpu().numpy().tobytes())
            specs.append({"key": s.key, "dtype": dtype_to_str(s.dtype), "nbytes": len(blob)})
            parts.append(blob)
        header = {"type": protocol.MSG_PUT, "session_id": session_id, "shards": specs}
        sock = self._sock()
        protocol.send_message(sock, header, b"".join(parts))
        resp, _ = protocol.recv_message(sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"remote put failed: {resp.get('error')}")

    def get(self, session_id: str, shards: List[ShardRef]) -> None:
        if not shards:
            return
        specs = [
            {"key": s.key, "dtype": dtype_to_str(s.dtype), "nbytes": s.length}
            for s in shards
        ]
        header = {"type": protocol.MSG_GET, "session_id": session_id, "shards": specs}
        sock = self._sock()
        protocol.send_message(sock, header)
        resp, blob = protocol.recv_message(sock)
        if resp.get("status") != "ok":
            raise RuntimeError(f"remote get failed: {resp.get('error')}")
        # Scatter returned bytes back into the local KV tensors in place.
        cursor = 0
        for s, out_len in zip(shards, resp["lengths"]):
            chunk = blob[cursor:cursor + out_len]
            cursor += out_len
            view = tensor_byte_view(s.tensor)
            src = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
            view[s.offset:s.offset + s.length].copy_(src.to(view.device))

    def close(self) -> None:
        sock = getattr(self._local, "sock", None)
        if sock is not None:
            try:
                sock.close()
            finally:
                self._local.sock = None
