# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Remote cache configuration parsing.

Settings can come from three places, in decreasing priority:

  1. Environment variables (prefix ``KVSHRINK_REMOTE_``).
  2. The ``kv_connector_extra_config`` dict inside vLLM's
     ``--kv-transfer-config``.
  3. Built-in defaults.

The connector calls :meth:`RemoteCacheConfig.from_vllm` with the parsed
``kv_connector_extra_config`` dict, the worker's ``rank`` and ``tp_size``; the
daemon builds its own config from CLI flags (see ``daemon.py``).

Per-rank daemon addressing (multi-daemon-process deployments)
---------------------------------------------------------------
To avoid every GPU worker rank talking to the *same* single daemon process
(many-to-one contention -- see the design doc's "why multi-process daemon"
section), ``KVSHRINK_REMOTE_DAEMON_ADDR`` and ``KVSHRINK_REMOTE_NIXL_ADDR``
may each be a single ``host:port`` (all ranks share one daemon) **or** a
``|``-separated list of ``host:port`` entries, one per rank, exactly like the
existing ``KVSHRINK_QAT_DEVICES``/``KVSHRINK_DSA_DEVICES`` convention in
``kvshrink_connector._bind_intel_accel``. Rank ``r`` connects only to entry
``r``. The scheduler role always resolves to entry 0, matching rank 0's
"has()" being used as the deployment-level hit proxy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Transport backends for the data plane (bulk KV bytes).
TRANSPORT_NIXL = "nixl"
TRANSPORT_TCP = "tcp"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(f"KVSHRINK_REMOTE_{name}", default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _split_hostport(addr: Optional[str], default_port: int) -> tuple[str, int]:
    """Parse ``host:port`` (port optional) into a tuple."""
    if not addr:
        return "", default_port
    if ":" in addr:
        host, _, port = addr.rpartition(":")
        return host, int(port)
    return addr, default_port


def _resolve_per_rank(value: Optional[str], rank: int) -> Optional[str]:
    """Pick rank ``rank``'s entry out of a ``|``-separated list, if any.

    A single (non-``|``) value applies to every rank unchanged.
    """
    if not value or "|" not in value:
        return value
    parts = [p.strip() for p in value.split("|")]
    if rank >= len(parts):
        raise ValueError(
            f"expected at least {rank + 1} '|'-separated entries for rank {rank}, "
            f"got {len(parts)}: {value!r}"
        )
    return parts[rank]


@dataclass
class RemoteCacheConfig:
    """Client-side (vLLM worker/scheduler) remote-cache settings."""

    enabled: bool = False

    # Control plane (metadata / has / put-commit / get-begin RPC).
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 19000

    # Data plane (bulk KV transfer).
    transport: str = TRANSPORT_NIXL
    nixl_host: str = "127.0.0.1"
    nixl_port: int = 19100
    # NIXL device string, e.g. "mlx5_0:1", forwarded to UCX_NET_DEVICES.
    # May be a comma-separated list ("mlx5_0:1,mlx5_1:1"): UCX itself performs
    # multi-rail striping across the listed devices, so multiple RDMA ports
    # can already be used simultaneously for one session -- this is preserved
    # unchanged from the previous implementation; see the design doc.
    nixl_device: str = ""

    # Connection behaviour.
    connect_timeout_sec: float = 30.0
    connect_retry_sec: float = 2.0
    request_timeout_sec: float = 120.0
    session_reconnect: bool = True

    # Fail fast if the daemon cannot be reached at init.
    fail_if_unreachable: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_vllm(
        cls,
        extra_config: Optional[Dict[str, Any]],
        rank: int = 0,
        tp_size: int = 1,
    ) -> "RemoteCacheConfig":
        """Build a config from vLLM ``kv_connector_extra_config`` + env overrides.

        Environment variables always override the dict so an operator can
        flip remote caching on/off, or repoint a rank at a different daemon
        instance, without editing the serve command.
        """
        extra_config = dict(extra_config or {})

        enabled = _as_bool(
            _env("CACHE_ENABLE", extra_config.get("remote_cache_enable")),
            default=False,
        )

        daemon_addr_raw = _env("DAEMON_ADDR", extra_config.get("remote_daemon_addr"))
        daemon_addr = _resolve_per_rank(daemon_addr_raw, rank)
        daemon_host, daemon_port = _split_hostport(daemon_addr, 19000)

        nixl_addr_raw = _env("NIXL_ADDR", extra_config.get("remote_nixl_addr"))
        nixl_addr = _resolve_per_rank(nixl_addr_raw, rank)
        nixl_host, nixl_port = _split_hostport(nixl_addr, 19100)
        # Fall back to the daemon host if only the daemon address was provided.
        if not nixl_host:
            nixl_host = daemon_host

        transport = (_env("TRANSPORT", extra_config.get("transport")) or TRANSPORT_NIXL).lower()

        return cls(
            enabled=enabled,
            daemon_host=daemon_host or "127.0.0.1",
            daemon_port=daemon_port,
            transport=transport,
            nixl_host=nixl_host or "127.0.0.1",
            nixl_port=nixl_port,
            nixl_device=_env("NIXL_DEVICE", extra_config.get("nixl_device")) or "",
            connect_timeout_sec=float(
                _env("CONNECT_TIMEOUT_SEC", extra_config.get("connect_timeout_sec", 30.0))
            ),
            connect_retry_sec=float(
                _env("CONNECT_RETRY_SEC", extra_config.get("connect_retry_sec", 2.0))
            ),
            request_timeout_sec=float(
                _env("REQUEST_TIMEOUT_SEC", extra_config.get("request_timeout_sec", 120.0))
            ),
            session_reconnect=_as_bool(
                _env("SESSION_RECONNECT", extra_config.get("session_reconnect")), default=True
            ),
            fail_if_unreachable=_as_bool(
                _env("FAIL_IF_UNREACHABLE", extra_config.get("fail_if_unreachable")),
                default=True,
            ),
            extra=extra_config,
        )

    def describe(self) -> str:
        return (
            f"RemoteCacheConfig(enabled={self.enabled}, "
            f"daemon={self.daemon_host}:{self.daemon_port}, "
            f"transport={self.transport}, "
            f"nixl={self.nixl_host}:{self.nixl_port}, device={self.nixl_device!r})"
        )
