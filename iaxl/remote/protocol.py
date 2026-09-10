# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Control-plane wire protocol.

The control plane carries metadata: capability negotiation, session creation,
existence checks, readiness records and (for NIXL) begin/commit/done tokens.
It never carries bulk KV bytes when the NIXL data plane is used -- those bytes
travel over RDMA. The TCP fallback data plane reuses this same framing to
move bytes inline, for single-host development and CI.

Framing: a 4-byte big-endian unsigned length prefix followed by a UTF-8 JSON
object. A message may optionally carry a trailing binary blob whose length is
declared in the JSON header field ``_blob_len``; this is used only by the TCP
data plane.

Why keep the control plane on TCP instead of RDMA SEND (see the design doc,
section "control-plane transport"): control frames are small and infrequent
relative to the data volume moved by NIXL WRITE/READ, so the few
microseconds a loopback/local RoCE TCP round trip costs are negligible next
to the actual RDMA transfer + QAT (de)compress time of a round. Moving them
onto RDMA SEND would add a second reliable-message channel (its own
completion queue, its own agent bootstrap) for no measurable end-to-end win.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, Optional, Tuple

PROTOCOL_VERSION = 1

# Control message types.
MSG_HEALTH = "health"
MSG_CAPABILITY = "capability"
MSG_SESSION_CREATE = "session_create"
MSG_HAS = "has"
MSG_PUT = "put"
MSG_GET = "get"
MSG_MARK_READY = "mark_ready"
MSG_NIXL_HANDSHAKE = "nixl_handshake"
MSG_STOP = "stop"

# Admin / CLI messages (session-less, safe to omit for capability check).
# Added in an additive way -- old clients that never send these still see
# PROTOCOL_VERSION=1 and continue to work unchanged.
MSG_STATUS = "status"
MSG_PERSIST = "persist"
MSG_EVICT = "evict"
MSG_METRICS = "metrics"
MSG_PERSIST_CANDIDATES = "persist_candidates"
MSG_EVICT_CANDIDATES = "evict_candidates"

_LEN = struct.Struct(">I")


class ProtocolError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise on premature EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError(f"connection closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, header: Dict[str, Any], blob: Optional[bytes] = None) -> None:
    """Send one framed message with an optional trailing binary blob."""
    header = dict(header)
    if blob is not None:
        header["_blob_len"] = len(blob)
    payload = json.dumps(header).encode("utf-8")
    sock.sendall(_LEN.pack(len(payload)))
    sock.sendall(payload)
    if blob:
        sock.sendall(blob)


def recv_message(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    """Receive one framed message. Returns (header, blob)."""
    (length,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    payload = _recv_exact(sock, length)
    header = json.loads(payload.decode("utf-8"))
    blob_len = int(header.get("_blob_len", 0))
    blob = _recv_exact(sock, blob_len) if blob_len else b""
    return header, blob
