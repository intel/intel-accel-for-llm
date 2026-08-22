# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVShrink metrics -- self-contained, fail-open.

Exposes the 9 metrics plus 3 lifecycle gauges through a
thread-safe in-process store that the standalone exporter
(hybrid_metrics_exporter.py) reads on its dedicated HTTP port. There is no
prometheus_client default-registry path: vLLM's /metrics endpoint
aggregates a private CollectorRegistry from multiprocess mmap files only,
so metrics registered on the DEFAULT REGISTRY never surface there
(verified on vLLM v0.21+). The in-process store is the single source
of truth.

Every public function is fail-open: any exception inside metrics code is
swallowed so metrics can NEVER affect the inference path
("metrics 不可用时绝不影响推理路径").

Registered metrics (9 + 3 lifecycle gauges):

  kvshrink_lookup_boundary{group,kind,result}    Counter   policy lookups
  kvshrink_external_hit_tokens                   Counter   external hit tokens
  kvshrink_state_snapshot_boundary               Counter   mamba snapshots restored
  kvshrink_transfer_bytes{direction,group,rank}  Counter   D2H/H2D bytes
  kvshrink_job_latency_seconds{kind}             Histogram store/load/flush latency
  kvshrink_manifest_incomplete_total             Counter   load-side manifest gaps
  kvshrink_checksum_failure_total                Counter   load checksum failures
  kvshrink_deferred_blocks                       Gauge     deferred-free blocks (sync model: 0)
  kvshrink_pinned_pool_bytes{page_size}          Gauge     in-flight pinned staging bytes
  kvshrink_pending_store_jobs                    Gauge     pending async store jobs
  kvshrink_inflight_boundaries                   Gauge     in-flight boundary writers
  kvshrink_cursor_rollbacks                      Gauge     save-cursor rollbacks
"""
from __future__ import annotations

import threading


# name -> (type, doc, labelnames)
_METRIC_DEFS = [
    ("kvshrink_lookup_boundary", "counter",
     "KVShrink external boundary lookups by group/kind/result",
     ("group", "kind", "result")),
    ("kvshrink_external_hit_tokens", "counter",
     "Total external cache hit tokens", ()),
    ("kvshrink_state_snapshot_boundary", "counter",
     "Mamba state snapshots restored from the external tier", ()),
    ("kvshrink_transfer_bytes", "counter",
     "Bytes transferred by direction/group/rank",
     ("direction", "group", "rank")),
    ("kvshrink_job_latency_seconds", "histogram",
     "Transfer job latency by kind (store/load/flush)", ("kind",)),
    ("kvshrink_manifest_incomplete_total", "counter",
     "Load-side manifest completeness failures", ()),
    ("kvshrink_checksum_failure_total", "counter",
     "Load checksum verification failures", ()),
    ("kvshrink_deferred_blocks", "gauge",
     "Blocks deferred from free by async saves (sync model: always 0)", ()),
    ("kvshrink_pinned_pool_bytes", "gauge",
     "In-flight pinned staging pool bytes by page size", ("page_size",)),
    ("kvshrink_pending_store_jobs", "gauge",
     "Pending async store jobs", ()),
    ("kvshrink_inflight_boundaries", "gauge",
     "In-flight boundary write cohorts", ()),
    ("kvshrink_cursor_rollbacks", "gauge",
     "Save-cursor rollbacks (preemption/resume)", ()),
]

REGISTERED = frozenset(name for name, _t, _d, _l in _METRIC_DEFS)

_lock = threading.Lock()
_store = {}  # (name, labels_key) -> value (counter: sum; gauge: last; hist: sum)


def _labels_key(labels) -> tuple:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


_COUNT_SUFFIX = "\x00count"  # internal key suffix for histogram counts


def _record(name: str, labels, delta: float, mode: str) -> None:
    """Update the in-process store (the standalone exporter reads it).

    ``mode``: 'inc' accumulates, 'set' overwrites (gauge), 'observe'
    accumulates the histogram sum AND its sample count (stored under
    ``name + _COUNT_SUFFIX`` so the standalone exporter can emit a
    well-formed _sum/_count family without prometheus internals). Any
    failure is swallowed."""
    try:
        lk = _labels_key(labels)
        with _lock:
            old = _store.get((name, lk), 0.0)
            _store[(name, lk)] = (old + delta) if mode in ("inc", "observe") \
                else delta
            if mode == "observe":
                ck = (name + _COUNT_SUFFIX, lk)
                _store[ck] = _store.get(ck, 0.0) + 1.0
    except Exception:  # pragma: no cover - fail-open
        pass


def inc(name: str, labels=None, value: float = 1.0) -> None:
    """Increment a counter metric (fail-open)."""
    _record(name, labels, float(value), "inc")


def set_gauge(name: str, labels=None, value: float = 0.0) -> None:
    """Set a gauge metric (fail-open)."""
    _record(name, labels, float(value), "set")


def observe(name: str, labels=None, value: float = 0.0) -> None:
    """Observe a histogram metric (fail-open); accumulates the _sum."""
    _record(name, labels, float(value), "observe")


def get_value(name: str, labels=None) -> float:
    """Read the in-process value for a metric+labels (tests / debug)."""
    try:
        with _lock:
            return _store.get((name, _labels_key(labels)), 0.0)
    except Exception:
        return 0.0


def metric_names() -> set[str]:
    """The base names of every registered metric."""
    return set(REGISTERED)


def clear_store() -> None:
    """Reset the in-process store (tests only)."""
    with _lock:
        _store.clear()
