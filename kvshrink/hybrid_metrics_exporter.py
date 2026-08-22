# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVShrink standalone metrics exporter.

Exposes the 12 ``kvshrink_*`` metrics on a DEDICATED HTTP port, following
the project's own observability pattern (KVCacheClip kvclip.mgmt: a
threaded HTTP server inside the worker process, independent of vLLM's
prometheus stack). vLLM's /metrics endpoint aggregates a private
``CollectorRegistry`` from multiprocess mmap files only, so metrics
registered on prometheus_client's DEFAULT REGISTRY never surface there
(verified against v0.23.0: the EngineCore process had no
PROMETHEUS_MULTIPROC_DIR set and /metrics returned zero kvshrink
series).

Data source is the in-process store in ``metrics.py`` (never modified --
this module only reads via ``metric_names()``/``get_value()``), so the
exporter works with or without prometheus_client.

Endpoints:
  GET /metrics  Prometheus exposition text (counters with ``_total``
                alias, histograms as _bucket/_sum/_count, gauges plain)
  GET /health   {"status": "ok"}

Port: ``KVSHRINK_METRICS_PORT`` (default 18801), offset by rank for
TP>1 worker processes. Every function is fail-open: an exporter failure
must NEVER affect the inference path.
"""
from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import hybrid_metrics as _kvshrink_metrics

logger = None  # set lazily below; never used on the inference path

_SERVER = None
_LOCK = threading.Lock()
_DEFAULT_PORT = 18801


def _log(msg: str) -> None:
    """INFO log under the vllm.* namespace, acquiring the logger lazily
    on first use so importing this module has no logging side effects.
    Never invoked on the inference path."""
    global logger
    if logger is None:
        import logging
        # log under the vllm.* namespace: vLLM only configures the
        # "vllm" logger (handler+level); a kvclip.* logger would drop
        # INFO evidence lines that the GPU probes grep for.
        logger = logging.getLogger("vllm." + __name__)
    logger.info("kvshrink metrics exporter: %s", msg)


# labelnames are ignored here; the in-process store keys values by
# (name, labels) so we emit one series per distinct label tuple.
def _series() -> list[tuple[str, str, dict, float]]:
    """[(exposition_name, kind, labels, value)] for every stored metric.

    Metrics that were never recorded still appear with value 0 so the
    full required set is always observable.

    Exposition rules:
    - counters: logical names ending in ``_total`` keep that exact name
      (no double suffix); other counters gain ``_total``.
    - histograms: emit ``_bucket{le=\"+Inf\"}``, ``_sum`` and ``_count``
      (count is tracked in the in-process store by metrics.py)."""
    out = []
    try:
        type_of = {name: mtype for name, mtype, _d, _l
                   in _kvshrink_metrics._METRIC_DEFS}
        with _kvshrink_metrics._lock:
            items = list(_kvshrink_metrics._store.items())
        seen = set()
        for (name, labels_key), value in items:
            if name.endswith(_kvshrink_metrics._COUNT_SUFFIX):
                continue
            mtype = type_of.get(name, "counter")
            labels = dict(labels_key) if labels_key else {}
            if mtype == "histogram":
                count = _kvshrink_metrics._store.get(
                    (name + _kvshrink_metrics._COUNT_SUFFIX, labels_key), 0.0)
                out.append((f"{name}_bucket", "bucket",
                            dict(labels, le="+Inf"), float(count)))
                out.append((f"{name}_sum", "sum", labels, float(value)))
                out.append((f"{name}_count", "count", labels, float(count)))
            else:
                expo = (name if name.endswith("_total")
                        else f"{name}_total") if mtype == "counter" else name
                out.append((expo, mtype, labels, float(value)))
            seen.add(name)
        for name, mtype, _d, _l in _kvshrink_metrics._METRIC_DEFS:
            if name in seen:
                continue
            if mtype == "histogram":
                out.append((f"{name}_bucket", "bucket",
                            {"le": "+Inf"}, 0.0))
                out.append((f"{name}_sum", "sum", {}, 0.0))
                out.append((f"{name}_count", "count", {}, 0.0))
            else:
                expo = (name if name.endswith("_total")
                        else f"{name}_total") if mtype == "counter" else name
                out.append((expo, mtype, {}, 0.0))
    except Exception:  # pragma: no cover - fail-open by design
        return []
    return out


class _Handler(BaseHTTPRequestHandler):
    """Serves /health and the /metrics exposition text; every failure
    path is downgraded to a 500 response and never propagates."""

    def do_GET(self):  # noqa: N802 - stdlib signature
        """Serve /metrics (prometheus exposition text) or /health JSON.
        Fail-open by design: any exception is turned into a 500 so a
        broken exporter can never crash the worker."""
        try:
            if self.path.split("?", 1)[0] == "/health":
                body = b'{"status": "ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                lines = []
                for expo, kind, labels, value in _series():
                    if labels:
                        lstr = ",".join(
                            f'{k}="{v}"' for k, v in labels.items())
                        lines.append(f"{expo}{{{lstr}}} {value:g}")
                    else:
                        lines.append(f"{expo} {value:g}")
                body = ("\n".join(lines) + "\n").encode()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:  # pragma: no cover - fail-open by design
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def log_message(self, *args):  # silence default stderr logging
        """Silence the stdlib per-request stderr logging (noise on the
        inference path)."""
        pass


def start_metrics_server(rank: int = 0) -> object:
    """Start the exporter on ``KVSHRINK_METRICS_PORT`` + rank (fail-open).

    Returns the server object, or None when disabled/start fails. Idempotent.
    """
    global _SERVER
    try:
        with _LOCK:
            if _SERVER is not None:
                return _SERVER
            port_env = os.getenv("KVSHRINK_METRICS_PORT")
            if port_env is not None and port_env.strip() == "0":
                return None
            port = int(port_env or _DEFAULT_PORT) + int(rank)
            _SERVER = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
            _SERVER.daemon_threads = True
            t = threading.Thread(target=_SERVER.serve_forever, daemon=True)
            t.start()
            _log(f"listening on :{port} (rank {rank})")
            return _SERVER
    except Exception:  # pragma: no cover - fail-open by design
        try:
            if _SERVER is not None:
                _SERVER.server_close()
        except Exception:
            pass
        _SERVER = None
        _log("failed to start; exporter disabled")
        return None


def stop_metrics_server() -> None:
    """Stop the exporter (idempotent, fail-open)."""
    global _SERVER
    try:
        with _LOCK:
            if _SERVER is None:
                return
            try:
                _SERVER.shutdown()
            except Exception:
                pass
            try:
                _SERVER.server_close()
            except Exception:
                pass
            _SERVER = None
    except Exception:  # pragma: no cover - fail-open by design
        pass
