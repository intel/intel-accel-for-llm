# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Control-plane client used by the vLLM connector.

Carries metadata only: capability negotiation, session creation, block
existence (``has``) queries, and block-readiness records. Bulk KV bytes
travel on the data plane (see :mod:`iaxl.remote.transport`). One instance is
created per connector role (scheduler or worker).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import List, Optional

from . import protocol
from .config import RemoteCacheConfig

logger = logging.getLogger(__name__)


class RemoteCacheClient:
    def __init__(self, config: RemoteCacheConfig, model_name: str, tp_size: int,
                 tp_rank: int, role: str):
        self.config = config
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.role = role  # "scheduler" or "worker"
        self.session_id: Optional[str] = None
        # Daemon-advertised NIXL staging capacity, learned at session creation.
        self.staging_slots: int = 0
        self.staging_slot_bytes: int = 0
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    # -- connection -----------------------------------------------------------

    def connect(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_sec
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                self._sock = socket.create_connection(
                    (self.config.daemon_host, self.config.daemon_port),
                    timeout=self.config.request_timeout_sec,
                )
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._capability()
                logger.info("Connected to remote cache daemon %s:%d (%s rank %d)",
                            self.config.daemon_host, self.config.daemon_port,
                            self.role, self.tp_rank)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(self.config.connect_retry_sec)
        msg = (f"cannot reach remote cache daemon at "
               f"{self.config.daemon_host}:{self.config.daemon_port}: {last_err}")
        if self.config.fail_if_unreachable:
            raise RuntimeError(msg)
        logger.warning("%s; continuing without remote cache", msg)

    def _rpc(self, header: dict, blob: bytes = b"") -> tuple[dict, bytes]:
        with self._lock:
            protocol.send_message(self._sock, header, blob or None)
            return protocol.recv_message(self._sock)

    def _capability(self) -> None:
        resp, _ = self._rpc({
            "type": protocol.MSG_CAPABILITY,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "client_type": "iaxl-kvshrink",
            "required_features": [],
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"capability negotiation failed: {resp.get('error')}")
        if resp.get("protocol_version") != protocol.PROTOCOL_VERSION:
            raise RuntimeError("daemon protocol version mismatch")

    # -- session --------------------------------------------------------------

    def create_session(self, num_layers: int, tensor_keys: List[str],
                       dtype: str, block_size: int, shard_bytes: int = 0) -> str:
        resp, _ = self._rpc({
            "type": protocol.MSG_SESSION_CREATE,
            "role": self.role,
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "num_layers": num_layers,
            "tensor_keys": tensor_keys,
            "dtype": dtype,
            "block_size": block_size,
            # Lets the daemon cut its staging arena into shard-sized slots.
            "shard_bytes": shard_bytes,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"session create failed: {resp.get('error')}")
        self.session_id = resp["session_id"]
        self.staging_slots = int(resp.get("staging_slots", 0) or 0)
        self.staging_slot_bytes = int(resp.get("staging_slot_bytes", 0) or 0)
        return self.session_id

    # -- runtime --------------------------------------------------------------

    def has(self, block_hashes: List[str]) -> List[bool]:
        if not block_hashes:
            return []
        resp, _ = self._rpc({
            "type": protocol.MSG_HAS,
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "block_hashes": block_hashes,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"has query failed: {resp.get('error')}")
        return resp["exists"]

    def mark_ready(self, block_hashes: List[str]) -> None:
        """Record that this rank has stored every shard of these blocks."""
        if not block_hashes:
            return
        resp, _ = self._rpc({
            "type": protocol.MSG_MARK_READY,
            "session_id": self.session_id,
            "model_name": self.model_name,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "block_hashes": block_hashes,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"mark_ready failed: {resp.get('error')}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                protocol.send_message(self._sock, {"type": protocol.MSG_STOP})
            except Exception:  # noqa: BLE001
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None
