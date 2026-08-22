# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hybrid boundary backend: the iaxl.kvstore.HybridStore adapter.

The connector/scheduler talk ONLY to this adapter; the durable store is
the iaxl HybridStore behind it (KVFlow chunk engine, schema-4
manifests, atomic group commit, checksum harvest, flock single writer).
Key mapping: CacheKey (per-page, layer_name != "") maps onto
CacheBoundary (per-group); the adapter owns the translation and never
lets callers build HybridStore private objects.

Directory layout: page payloads live in the KVFlow chunk store
(``$IAXL_CACHE_DIR/compressed|raw/<namespace>_rank<r>/``); commit
manifests live under the manifest root (default
``$IAXL_CACHE_DIR/kv4-manifests``, override with KVSHRINK_PERSIST_DIR),
namespaced per boundary via CacheBoundary.path_prefix.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# log under the vllm.* namespace: vLLM only configures the
# "vllm" logger (handler+level); an unconfigured logger would drop
# INFO evidence lines that the GPU probes grep for.
logger = logging.getLogger("vllm." + __name__)


def _default_persist_dir() -> str:
    """Default commit-manifest root: $IAXL_CACHE_DIR/kv4-manifests."""
    from iaxl.envs import envs
    return os.path.join(envs.IAXL_CACHE_DIR, "kv4-manifests")


def _cache_boundary_from_key(key) -> object:
    """CacheKey -> iaxl CacheBoundary."""
    from iaxl.kvstore.hybrid import CacheBoundary
    return CacheBoundary(
        namespace=key.namespace, tp_size=key.tp_size, rank=key.rank,
        block_hash=key.block_hash, group_idx=key.group_idx)


class KVShrinkHybridBackendAdapter:
    """Thin adapter over iaxl.kvstore.HybridStore (chunk tier).

    All HybridStore fail-closed semantics (atomic group commit,
    checksum harvest, manifest-as-commit-point, flock single writer)
    are preserved: this adapter only translates keys and delegates.

    Codec policy: lossless-only. Hybrid pages are opaque int8 canonical
    views; GDN state pages are lossless by construction (the engine's
    DEFLATE compression is lossless; the lossy transforms must stay
    off: IAXL_KV_LOSSY_TRUNC=0).
    """

    def __init__(self, persist_dir: Optional[str] = None,
                 writer_rank: Optional[int] = None,
                 role: str = "scheduler"):
        """Record role, writer lease and manifest root; the real
        HybridStore is built lazily in ``register_layout``. Layout facts
        kept here feed the lazy per-rank presence backends."""
        self._persist_dir = persist_dir or _default_persist_dir()
        self._writer_rank = writer_rank
        self._role = role
        self._backend = None
        # Layout facts kept for lazy per-rank backends (TP cross-rank
        # presence checks, see lookup_boundary).
        self._groups = None
        self._layer_infos = None
        self._namespace = ""
        self._tp_size = 1
        self._own_rank = 0
        self._rank_backends: dict = {}

    def _make_chunk_flow(self, layout):
        """KVFlow engine for the chunk tier. Worker: full engine (GPU
        streams are created lazily in the worker process). Scheduler:
        has-only mode (cache_size_gb=0) — Record/SQLite only, for
        presence checks.
        """
        from iaxl.kvflow import KVFlow
        if self._role == "worker":
            from iaxl.envs import envs
            cache_size_gb = envs.IAXL_DDR_POOL_SIZE_GB
            if cache_size_gb is None:
                from iaxl.kvstore.kvstore import _get_default_cache_size_gb
                cache_size_gb = _get_default_cache_size_gb()
        else:
            cache_size_gb = 0.0  # has-only: scheduler never stages pages
        flow = KVFlow(
            persist_dir=f"{layout.namespace}_rank{layout.rank}",
            cache_size_gb=cache_size_gb,
            rank=layout.rank)
        # Hybrid correctness: the engine's get_stream must wait for the
        # compute stream's pending work (preprocess_mamba prev->curr
        # copy) before any H2D restore; host call order proves nothing
        # about GPU order. Per-instance flag: the pure-attention default
        # stays off and other in-process engines are unaffected.
        flow.stream_sync_on_get = True
        return flow

    def register_layout(self, groups, layer_infos, namespace, tp_size, rank):
        """Install the layout facts and build this rank's HybridStore
        over a role-sized KVFlow engine, returning the CacheLayout.
        Must be called once before any chunk-tier op; the retained
        groups/layer_infos feed the lazy per-rank backends."""
        from iaxl.kvstore.hybrid import CacheLayout, HybridStore
        if isinstance(layer_infos, dict):
            layer_infos = list(layer_infos.values())
        self._groups = groups
        self._layer_infos = layer_infos
        self._namespace = namespace
        self._tp_size = tp_size
        self._own_rank = rank
        layout = CacheLayout.from_group_infos(
            namespace=namespace, tp_size=tp_size, rank=rank,
            group_infos=groups, layer_infos=layer_infos)
        engine = self._make_chunk_flow(layout)
        self._backend = HybridStore(
            layout=layout, persist_dir=self._persist_dir,
            writer_rank=self._writer_rank, flow=engine)
        return layout

    def _backend_for_rank(self, rank: int):
        """Read-only HybridStore view onto another TP rank's shard
        (scheduler role only). Lazily built; the engine is has-only
        (Record presence only, no GPU/DDR pool) and takes no writer
        flock, so it can never mutate that rank's data."""
        be = self._rank_backends.get(rank)
        if be is None:
            from iaxl.kvstore.hybrid import CacheLayout, HybridStore
            layout = CacheLayout.from_group_infos(
                namespace=self._namespace, tp_size=self._tp_size,
                rank=rank, group_infos=self._groups,
                layer_infos=self._layer_infos)
            engine = self._make_chunk_flow(layout)
            be = HybridStore(
                layout=layout, persist_dir=self._persist_dir,
                writer_rank=None, flow=engine)
            self._rank_backends[rank] = be
        return be

    # ------------------------------------------------------------------
    # Chunk-tier protocol (thin forwards)
    # ------------------------------------------------------------------

    def submit_group_stores(self, group_idx, layer_views, chunk_indices,
                            chunk_labels):
        """Thin forward: enqueue GPU->store chunk writes for one group;
        returns tasks to hand to ``wait_group_stores``."""
        return self._backend.submit_layer_puts(
            group_idx, layer_views, chunk_indices, chunk_labels)

    def wait_group_stores(self, tasks):
        """Thin forward: block until the submitted chunk writes land."""
        return self._backend.wait_layer_puts(tasks)

    def submit_group_loads(self, group_idx, layer_views, chunk_indices,
                           chunk_labels):
        """Thin forward: enqueue store->GPU chunk reads for one group;
        returns tasks to hand to ``wait_layer_loads``."""
        return self._backend.submit_layer_gets(
            group_idx, layer_views, chunk_indices, chunk_labels)

    def wait_layer_loads(self, layer_tasks):
        """Thin forward: block until the submitted chunk reads complete
        (loads are synchronous before forward)."""
        self._backend.wait_layer_gets(layer_tasks)

    def commit_chunks(self, key, expected_layers, checksums, chunk_labels,
                      expected_boundary_tokens=None) -> bool:
        """Thin forward: commit one boundary's manifest (atomic group
        commit of the listed layers + checksums). Maps the page CacheKey
        onto the iaxl CacheBoundary; the caller treats False/raise as a
        MISS (fail closed)."""
        boundary = _cache_boundary_from_key(key)
        return self._backend.commit_boundary_chunks(
            boundary, list(expected_layers), checksums,
            chunk_labels=list(chunk_labels),
            boundary_tokens=expected_boundary_tokens)

    def persist_engine(self, max_count: int) -> dict:
        """Thin forward: flush up to ``max_count`` staged records to the
        durable store; returns a stats dict."""
        return self._backend.persist(max_count)

    def evict_over_watermark(self, high_water: float = 0.8,
                             low_water: float = 0.6) -> dict:
        """Thin forward: once usage exceeds ``high_water``, evict until
        it drops below ``low_water``; returns a stats dict."""
        return self._backend.evict_over_watermark(high_water, low_water)

    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        """Presence check for one boundary on this rank plus, under
        TP>1, every peer rank's read-only shard backend. HIT only when
        ALL ranks report committed; any miss, anomaly or exception ->
        MISS (fail closed: a wrong hit silently corrupts output, a
        wrong miss costs one recompute). A TP partial commit heals on
        the request's own re-save (commit is idempotent per rank)."""
        from dataclasses import replace as _dc_replace
        from .hybrid_policy import LookupStatus
        try:
            boundary = _cache_boundary_from_key(key)
            if not self._backend.is_committed(
                    boundary, expected_layers=expected_layers,
                    expected_boundary_tokens=expected_boundary_tokens):
                return LookupStatus.MISS
            # TP partial-commit guard: under TP>1 each rank commits its
            # own shard independently -- there is no cross-rank
            # transaction. A boundary present on this rank but missing
            # on any other rank must be treated as MISS. The hit path
            # then heals it with no background repair: the request
            # recomputes and its own save re-commits every rank's
            # shard (commit is idempotent per rank).
            for r in range(self._tp_size):
                if r == self._own_rank:
                    continue
                rb = _dc_replace(boundary, rank=r)
                if not self._backend_for_rank(r).is_committed(
                        rb, expected_layers=expected_layers,
                        expected_boundary_tokens=expected_boundary_tokens):
                    logger.info(
                        "boundary %s g%d present on rank %d but missing "
                        "on rank %d; MISS (hit-path heal: recompute + "
                        "re-save will re-commit all ranks)",
                        boundary.hash_str[:12], boundary.group_idx,
                        self._own_rank, r)
                    return LookupStatus.MISS
            return LookupStatus.HIT
        except Exception:  # pragma: no cover - fail-closed to MISS
            logger.exception("hybrid lookup error; treating as MISS")
            return LookupStatus.MISS

    def orphan_stats(self) -> dict:
        """Chunk-store orphan stats, or {} before a backend exists."""
        return self._backend.orphan_stats() if self._backend else {}

    def close(self) -> None:
        """Release the backend and every per-rank view; failures are
        swallowed so shutdown never raises (fail-open)."""
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:  # pragma: no cover - fail-open
                pass
            self._backend = None
        for be in self._rank_backends.values():
            try:
                be.close()
            except Exception:  # pragma: no cover - fail-open
                pass
        self._rank_backends.clear()


def create_boundary_backend(persist_dir: Optional[str] = None,
                            writer_rank: Optional[int] = None,
                            role: str = "scheduler"):
    """Build the boundary backend (iaxl HybridStore, chunk tier).
    Scheduler side is read-only: no writer flock."""
    writer = writer_rank if role == "worker" else None
    return KVShrinkHybridBackendAdapter(persist_dir=persist_dir,
                                        writer_rank=writer, role=role)
