# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Admin CLI for the remote NIXL KV cache daemon.

Mirrors the local KVStore's ``/v1/cache/*`` HTTP endpoints (see
``iaxl/kvstore/kvstore.py``) but talks the daemon's native TCP framing
directly, so no session is created and the running vLLM workers are not
disturbed.

Usage:

    python -m iaxl.remote.admin --daemon HOST:PORT status
    python -m iaxl.remote.admin --daemon HOST:PORT persist --count 32
    python -m iaxl.remote.admin --daemon HOST:PORT evict --count 32 \
        --model Qwen2.5-32B-Instruct --tp-size 4 --tp-rank 0
    python -m iaxl.remote.admin --daemon HOST:PORT metrics --reset
    python -m iaxl.remote.admin --daemon HOST:PORT candidates --which evict --count 5

For a multi-process daemon (``run-daemon-multi.sh``) each instance listens
on its own control port; pass the address of the instance you want to poke
(usually one at a time) or use ``tools/remote_daemon/remote-cli.sh`` which
loops over ``KVSHRINK_REMOTE_DAEMON_ADDR``.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any, Dict, Optional

from . import protocol


class AdminClient:
    """Session-less RPC client for the admin messages defined in protocol.py."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def __enter__(self) -> "AdminClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> Dict[str, Any]:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        resp = self._rpc({
            "type": protocol.MSG_CAPABILITY,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "client_type": "iaxl-remote-admin",
            "required_features": [],
        })
        if resp.get("status") != "ok":
            raise RuntimeError(f"capability failed: {resp.get('error')}")
        if resp.get("protocol_version") != protocol.PROTOCOL_VERSION:
            raise RuntimeError(
                f"protocol version mismatch: daemon={resp.get('protocol_version')} "
                f"client={protocol.PROTOCOL_VERSION}"
            )
        if not resp.get("features", {}).get("admin"):
            raise RuntimeError(
                "daemon does not advertise the 'admin' feature: it was built before "
                "the CLI admin RPCs were added; rebuild/redeploy the daemon."
            )
        return resp

    def close(self) -> None:
        if self._sock is not None:
            try:
                protocol.send_message(self._sock, {"type": protocol.MSG_STOP})
            except OSError:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _rpc(self, header: Dict[str, Any]) -> Dict[str, Any]:
        protocol.send_message(self._sock, header)
        resp, _ = protocol.recv_message(self._sock)
        if resp.get("status") == "error":
            raise RuntimeError(f"daemon error: {resp.get('error')}")
        return resp

    # -- typed wrappers ------------------------------------------------------

    def status(self, **filt) -> Dict[str, Any]:
        return self._rpc({"type": protocol.MSG_STATUS, **_group_filter(filt)})

    def persist(self, count: int, **filt) -> Dict[str, Any]:
        return self._rpc({"type": protocol.MSG_PERSIST, "count": count, **_group_filter(filt)})

    def evict(self, count: int, **filt) -> Dict[str, Any]:
        return self._rpc({"type": protocol.MSG_EVICT, "count": count, **_group_filter(filt)})

    def metrics(self, enable: Optional[bool] = None, reset: bool = False) -> Dict[str, Any]:
        header: Dict[str, Any] = {"type": protocol.MSG_METRICS}
        if enable is not None:
            header["enable"] = bool(enable)
        if reset:
            header["reset"] = True
        return self._rpc(header)

    def persist_candidates(self, count: int, **filt) -> Dict[str, Any]:
        return self._rpc({"type": protocol.MSG_PERSIST_CANDIDATES, "count": count,
                          **_group_filter(filt)})

    def evict_candidates(self, count: int, **filt) -> Dict[str, Any]:
        return self._rpc({"type": protocol.MSG_EVICT_CANDIDATES, "count": count,
                          **_group_filter(filt)})


def _group_filter(filt: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unset fields so the server treats them as 'match every group'."""
    out: Dict[str, Any] = {}
    if filt.get("model_name"):
        out["model_name"] = filt["model_name"]
    if filt.get("tp_size") is not None:
        out["tp_size"] = int(filt["tp_size"])
    if filt.get("tp_rank") is not None:
        out["tp_rank"] = int(filt["tp_rank"])
    return out


# -- CLI ---------------------------------------------------------------------


def _parse_daemon(addr: str) -> tuple[str, int]:
    if ":" not in addr:
        raise argparse.ArgumentTypeError(
            f"--daemon must be host:port, got {addr!r}")
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _add_group_filter(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", "--model-name", dest="model_name",
                    help="Filter to this model only (default: every group).")
    sp.add_argument("--tp-size", type=int, default=None,
                    help="Filter to this tensor-parallel size only.")
    sp.add_argument("--tp-rank", type=int, default=None,
                    help="Filter to this TP rank only.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iaxl.remote.admin",
        description="Admin CLI for the remote NIXL KV cache daemon "
                    "(persist / evict / status / metrics).")
    p.add_argument("--daemon", required=True,
                   help="Daemon control-plane address, HOST:PORT.")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-RPC socket timeout in seconds.")
    p.add_argument("--json", action="store_true",
                   help="Print raw JSON reply instead of a compact summary.")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("status", help="Show daemon + per-group stats.")
    _add_group_filter(sp)

    sp = sub.add_parser("persist", help="Persist oldest N unpersisted groups to disk.")
    sp.add_argument("--count", type=int, default=10)
    _add_group_filter(sp)

    sp = sub.add_parser("evict", help="Evict N oldest LRU groups from the DDR pool.")
    sp.add_argument("--count", type=int, default=10)
    _add_group_filter(sp)

    sp = sub.add_parser("candidates", help="List persist/evict candidates.")
    sp.add_argument("--which", choices=("persist", "evict"), required=True)
    sp.add_argument("--count", type=int, default=10)
    _add_group_filter(sp)

    sp = sub.add_parser("metrics", help="Read (and optionally toggle/reset) codec metrics.")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--enable", dest="enable", action="store_const", const=True)
    g.add_argument("--disable", dest="enable", action="store_const", const=False)
    sp.set_defaults(enable=None)
    sp.add_argument("--reset", action="store_true")

    return p


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def _print_status(resp: Dict[str, Any]) -> None:
    d = resp["daemon"]
    print(f"daemon: {d['host']}:{d['port']}  compress={d['compress']}  "
          f"nixl_ready={d['nixl_ready']}  staging={d['staging_slots']} x "
          f"{_fmt_bytes(d['staging_slot_bytes'])}  "
          f"sessions={d['num_sessions']}  ready_blocks={d['num_ready_blocks']}  "
          f"groups={d['num_groups']}  cache_dir={d['cache_dir']}")
    for g in resp["groups"]:
        print(f"  group model={g['model_name']} tp={g['tp_size']} rank={g['tp_rank']} "
              f"entries={g['cache_entries']} groups={g['group_count']} "
              f"usage={_fmt_bytes(g['current_bytes'])}/{_fmt_bytes(g['capacity_bytes'])} "
              f"({g['usage_pct']}%)  "
              f"hits={g['hits']}(st={g['hits_in_storage']}) miss={g['misses']} "
              f"puts={g['puts']} evicts={g['evictions']} "
              f"unpersisted={g['unpersisted_count']} "
              f"ratio={g['compression_ratio']}")


def _print_persist_evict(resp: Dict[str, Any], op: str) -> None:
    total_key = "persisted" if op == "persist" else "evicted"
    bytes_key = "bytes_written" if op == "persist" else "bytes_freed"
    print(f"{op}: total_groups={resp[total_key]} total_bytes={_fmt_bytes(resp[bytes_key])}")
    for g in resp["groups"]:
        print(f"  model={g['model_name']} tp={g['tp_size']} rank={g['tp_rank']} "
              f"{total_key}={g[total_key]} bytes={_fmt_bytes(g[bytes_key])}")
        for lbl in g["labels"][:10]:
            print(f"    - {lbl}")
        if len(g["labels"]) > 10:
            print(f"    ... {len(g['labels']) - 10} more")


def _print_candidates(resp: Dict[str, Any], which: str) -> None:
    for g in resp["groups"]:
        print(f"{which} candidates for model={g['model_name']} tp={g['tp_size']} "
              f"rank={g['tp_rank']}: {len(g['candidates'])}")
        for lbl in g["candidates"]:
            print(f"  - {lbl}")


def _print_metrics(resp: Dict[str, Any]) -> None:
    m = resp["metrics"]
    print(f"metrics enabled={m.get('enabled')} "
          f"compress: {_fmt_bytes(m.get('compress_bytes', 0))} / "
          f"{m.get('compress_ns', 0) / 1e9:.3f}s  "
          f"({m.get('compress_gbps', 0):.2f} GB/s)  "
          f"decompress: {_fmt_bytes(m.get('decompress_bytes', 0))} / "
          f"{m.get('decompress_ns', 0) / 1e9:.3f}s  "
          f"({m.get('decompress_gbps', 0):.2f} GB/s)")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    host, port = _parse_daemon(args.daemon)
    filt = dict(model_name=getattr(args, "model_name", None),
                tp_size=getattr(args, "tp_size", None),
                tp_rank=getattr(args, "tp_rank", None))

    with AdminClient(host, port, timeout=args.timeout) as cli:
        if args.cmd == "status":
            resp = cli.status(**filt)
        elif args.cmd == "persist":
            resp = cli.persist(args.count, **filt)
        elif args.cmd == "evict":
            resp = cli.evict(args.count, **filt)
        elif args.cmd == "candidates":
            resp = (cli.persist_candidates(args.count, **filt)
                    if args.which == "persist"
                    else cli.evict_candidates(args.count, **filt))
        elif args.cmd == "metrics":
            resp = cli.metrics(enable=args.enable, reset=args.reset)
        else:
            raise SystemExit(f"unknown command: {args.cmd}")

    if args.json:
        print(json.dumps(resp, indent=2, sort_keys=True))
        return 0

    if args.cmd == "status":
        _print_status(resp)
    elif args.cmd in ("persist", "evict"):
        _print_persist_evict(resp, args.cmd)
    elif args.cmd == "candidates":
        _print_candidates(resp, args.which)
    elif args.cmd == "metrics":
        _print_metrics(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
