# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Worker-side execution engine for the hybrid (GDN) connector path.

Owns everything the worker role does with a hybrid
KVShrinkConnectorMetadata: canonical page views, load submission,
piggybacked GDN waits, pipelined attention save, and the post-forward
save commit. The connector facade (kvshrink_connector.py) only
dispatches.

Load pipelining without any vLLM patch
--------------------------------------
vLLM calls ``wait_for_layer_load`` at every ATTENTION layer's entry
(piecewise cudagraph is forced) but never at GDN layers. So:

- ``start_load``: submit ALL loads (attention pages + GDN snapshots)
  to the engine (async unzip+H2D on the engine's get_stream), then
  host-block ONLY on the LEADING GDN segment -- the GDN layers that
  execute before the first attention layer and therefore have no
  attention hook to ride on.
- ``wait_layer_load(attn_i)``: wait attention layer i's pages AND the
  GDN segment between attn_i and the next attention layer (those GDN
  layers execute after attn_i, so waiting at attn_i's entry is in
  time). Their transfers overlapped the preceding layers' compute --
  this IS the layer pipeline.

GDN snapshots are written into the CURR state slot: v0.23.0's GDN
execution metadata is pinned to the CURR block for both chunked-prefill
and decode, and preprocess_mamba's prev->curr copy runs before
start_load_kv, so a CURR write during forward is always safe and a PREV
write would be dead work.

Save path
---------
Attention groups save PIPELINED: ``save_kv_layer`` submits each layer's
async D2H+zip at that layer's exit (the layer's pages for this step are
final then). GDN groups save in ``wait_save`` (their state is final
only post-forward). Checksum harvest, schema-4 manifest commit, persist
drain and watermark eviction all happen in ``wait_save``.

Fail-stop contract: any load/save anomaly raises (EngineCore fatal).
Silently dropping a save would lose a boundary permanently (the
scheduler already advanced its incremental indices); entering forward
with unrestored pages would emit wrong tokens (the core already skipped
recompute).
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Optional

from .hybrid_metadata import CacheKey
from .hybrid_policy import LookupStatus

# log under the vllm.* namespace: vLLM only configures the "vllm"
# logger (handler+level); an unconfigured logger would drop INFO
# evidence lines that the GPU probes grep for.
logger = logging.getLogger("vllm." + __name__)

# Guarded import + no-op fallbacks so a metrics failure can NEVER
# affect the inference path.
try:
    from .hybrid_metrics import (  # noqa: E402
        inc as _metric_inc,
        observe as _metric_observe,
    )
except Exception:  # pragma: no cover - fail-open by design
    def _metric_inc(*a, **k):
        """Metrics no-op fallback (fail-open): a broken metrics stack
        must never break serving."""
        pass

    def _metric_observe(*a, **k):
        """Metrics no-op fallback (fail-open): a broken metrics stack
        must never break serving."""
        pass


def _now() -> float:
    """Monotonic clock for step-latency accounting: immune to NTP
    steps, so measured durations are never negative."""
    import time as _t
    return _t.monotonic()


class HybridWorker:
    """Worker-role executor for the hybrid path (see module docstring)."""

    def __init__(self, groups, layer_infos, num_blocks, backend,
                 canonicalizer, rank: int, tp_size: int):
        self._groups = groups
        self._layer_infos = layer_infos
        self._num_blocks = num_blocks
        self._backend = backend
        self._canon = canonicalizer
        self.rank = rank
        self.tp_size = tp_size

        self._kv_caches_ref = None
        # Per-step load tasks: layer_name -> list of per-call engine
        # task dicts. Populated by start_load, popped by the piggyback
        # waits. A leftover at step end means a wait never ran ->
        # fail-stop (residue check in wait_save).
        self._load_tasks: dict[str, list] = {}
        # Pipelined attention saves: layer_name -> (group_idx, tasks).
        self._step_attn_saves: dict = {}
        # Sticky LOAD poison (allocation-after-HIT failures must
        # fail-stop every later worker hook, never degrade to recompute).
        self._load_poison: Optional[BaseException] = None

        # attention layer_name -> group idx (mamba layers map out).
        self._attn_layer_group = {
            ln: g.group_idx for g in groups if g.kind != "mamba"
            for ln in g.layer_names}
        # Piggyback map (built in register): attention layer name ->
        # tuple of GDN layer names that execute after it and before the
        # next attention layer. Plus the leading GDN segment.
        self._piggyback_map: dict[str, tuple[str, ...]] = {}
        self._leading_gdn: tuple[str, ...] = ()
        """Wire up the worker-role pieces: the group/layer layout, the
        boundary backend and the canonical page-view builder for this
        rank's block pool, plus this rank's TP identity (the worker
        persists and loads its OWN shard).

        Initializes the per-step task bookkeeping (load tasks, stashed
        attention saves) and the sticky load-poison latch -- the
        worker is the EXECUTE side; it owns the writer lease, while
        the scheduler only plans against a read-only backend."""

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, kv_caches, execution_order: list[str]) -> None:
        """Bind canonical page views + build the piggyback map.

        ``execution_order``: all cached layer names in model execution
        order (the connector derives it from static_forward_context or
        the layer-index naming convention, fail-closed).

        Why a map at all: vLLM's per-layer load hook is attached to
        attention operators only, so a GDN layer is never told "your
        state is ready". Each GDN layer is therefore waited for by the
        attention layer that runs BEFORE it. That is early enough --
        the hook returns before those GDN layers execute -- while still
        letting their transfers overlap the compute of every layer up
        to that hook. GDN layers that run before the first attention
        layer have no such host, so they are collected separately and
        host-blocked in start_load (see ``_leading_gdn``).

        The map must be exhaustive: a GDN layer nobody waits for would
        enter forward with unrestored state, which is silent output
        corruption. Anything unaccounted for refuses to start, and the
        end-of-step check in wait_for_save catches a hook that never
        fired at runtime.
        """
        self._kv_caches_ref = kv_caches
        self._canon.register(kv_caches)

        mamba_layers = {ln for g in self._groups if g.kind == "mamba"
                        for ln in g.layer_names}
        attn_order = [ln for ln in execution_order
                      if ln in self._attn_layer_group]
        if not attn_order:
            raise RuntimeError(
                "kvshrink hybrid: no attention layers found in the "
                "execution order; cannot schedule piggybacked GDN loads")
        seen = set()
        segments: dict[str, list[str]] = {ln: [] for ln in attn_order}
        leading: list[str] = []
        # Walk the model in execution order carrying the most recent
        # attention layer: every GDN layer met afterwards is waited by
        # that one. Before the first attention layer `current` is None
        # and the layers land in `leading` instead.
        current: Optional[str] = None  # waiting for first attention
        for ln in execution_order:
            if ln in self._attn_layer_group:
                current = ln
                seen.add(ln)
            elif ln in mamba_layers:
                (segments[current] if current is not None
                 else leading).append(ln)
        # Exhaustiveness check: every GDN layer must have an owner. A
        # layer missing from execution_order would silently never be
        # waited for, so this refuses to start rather than risk it.
        unknown = mamba_layers - set(leading) - {
            ln for seg in segments.values() for ln in seg}
        if unknown:
            raise RuntimeError(
                f"kvshrink hybrid: GDN layers {sorted(unknown)} missing "
                "from the execution order; refusing to start")
        self._piggyback_map = {k: tuple(v) for k, v in segments.items()}
        self._leading_gdn = tuple(leading)
        logger.info(
            "kvshrink hybrid worker registered: %d layers, %d attention "
            "hook points, leading GDN segment=%d layers (namespace tp=%d "
            "rank=%d)",
            len(self._layer_infos), len(attn_order), len(leading),
            self.tp_size, self.rank)

    # ------------------------------------------------------------------
    # poison (sticky fail-stop)
    # ------------------------------------------------------------------
    def raise_load_poison(self) -> None:
        """Re-raise the sticky load failure so no later hook proceeds
        (fail-closed).

        Poison is a latched, sticky failure: once a load fails, the
        affected layers must never be treated as restored, because
        forward reading unrestored GDN state emits wrong tokens with
        no error. Retrying or ignoring would silently degrade."""
        if self._load_poison is not None:
            raise self._load_poison

    def _poison_load(self, error: BaseException) -> None:
        """Latch a load failure as sticky (first error wins). The
        failure is re-raised by every later worker hook, so a
        partially-loaded step can never enter forward and silently
        emit wrong output."""
        if self._load_poison is None:
            self._load_poison = error
        logger.error("kvshrink load poison: %s", error)

    # ------------------------------------------------------------------
    # metrics helpers (fail-open)
    # ------------------------------------------------------------------
    def _emit_transfer_bytes(self, direction: str, group_idx: int,
                             nbytes: int) -> None:
        """Count bytes moved per direction and group. Metrics are
        observability only and fail-open: a metrics problem must
        never break serving, so the call is always guarded."""
        try:
            _metric_inc(
                "kvshrink_transfer_bytes",
                {"direction": direction, "group": f"g{group_idx}",
                 "rank": str(self.rank)},
                value=float(nbytes))
        except Exception:  # pragma: no cover - fail-open
            pass

    def _emit_job_latency(self, kind: str, seconds: float) -> None:
        """Record one per-step latency observation (load/store), with
        the same fail-open guard as transfer bytes: a metrics failure
        is swallowed, never surfaced."""
        try:
            _metric_observe("kvshrink_job_latency_seconds",
                            {"kind": kind}, value=float(seconds))
        except Exception:  # pragma: no cover - fail-open
            pass

    def _worker_key(self, key: CacheKey) -> CacheKey:
        """Remap a scheduler-built key (rank 0) to this worker's own
        rank: each TP rank persists and loads its OWN shard under its
        own rank path. Without this, TP>1 workers overwrite each
        other's pages under the shared rank-0 key."""
        if key.rank == self.rank:
            return key
        return replace(key, rank=self.rank)

    def _layer_views(self, layer_name: str):
        """Canonical page views over the raw KV tensors of one layer:
        part key -> (num_blocks, page_bytes) GPU view. The chunk
        engine moves rows of these views, indexed by GPU block id."""
        parts, _chunk_dim = self._canon.page_view_parts(layer_name)
        return parts

    # ------------------------------------------------------------------
    # load path
    # ------------------------------------------------------------------
    def start_load(self, metadata) -> int:
        """Submit ALL of this step's loads (attention + GDN), then
        host-block ONLY on the leading GDN segment. Everything else is
        waited by the piggyback hooks during forward. Returns the number
        of (layer, block) pages submitted."""
        self.raise_load_poison()
        if self._load_tasks:
            # A previous step's submit aborted midway and its residue
            # was never drained -- refuse to silently drop in-flight
            # engine tasks (fail-stop).
            err = RuntimeError(
                "kvshrink chunk load: stale step residue "
                f"{sorted(self._load_tasks)}: the previous step's load "
                "was never drained (hook path aborted?)")
            self._poison_load(err)
            raise err
        self._load_tasks = {}
        self._step_attn_saves = {}
        npages = 0
        _t0 = _now()
        try:
            for req_meta in getattr(metadata, "requests", []):
                for op in getattr(req_meta, "group_ops", []):
                    npages += self._submit_op_load(req_meta, op)
        except BaseException as e:
            # Submit-stage failures (pool budget, engine errors) must
            # poison like wait-stage failures: a partially submitted
            # step can never enter forward.
            self._poison_load(e)
            raise
        # The leading GDN segment executes before the first attention
        # layer: no attention hook can cover it, so barrier-wait here.
        leading = [td for ln in self._leading_gdn
                   for td in self._load_tasks.pop(ln, [])]
        if leading:
            self._wait_tasks(leading)
        if npages:
            logger.info(
                "start_load_kv: %d pages loaded [chunk tier piggyback] "
                "elapsed_ms=%.3f (rank %d/%d)", npages,
                (_now() - _t0) * 1e3, self.rank, self.tp_size)
        self._emit_job_latency("load", _now() - _t0)
        return npages

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages + the
        trailing GDN segment riding on this layer."""
        self.raise_load_poison()
        tds = self._load_tasks.pop(layer_name, [])
        for gdn_ln in self._piggyback_map.get(layer_name, ()):
            tds += self._load_tasks.pop(gdn_ln, [])
        if tds:
            self._wait_tasks(tds)

    def loads_drained_check(self) -> None:
        """Fail-stop if any submitted load was never waited (a hook
        never ran -> forward just read unrestored state)."""
        if self._load_tasks:
            err = RuntimeError(
                "kvshrink load left unrestored layers "
                f"{sorted(self._load_tasks)}: the piggyback wait never "
                "ran for them")
            self._poison_load(err)
            raise err

    def _submit_op_load(self, req_meta, op) -> int:
        """Submit one GroupTransferMeta to the engine (async get).

        Returns the number of (layer, block) pages covered. Fail-stop:
        a mamba boundary whose tokens no longer match the scheduler's
        HIT (TOCTOU) poisons the load path and raises BEFORE any
        transfer for this op is submitted.
        """
        group = self._groups[op.group_idx]
        if not op.keys:
            return 0
        if group.kind == "mamba":
            # TOCTOU gate: the committed boundary must still match the
            # scheduler's snapshot point.
            first = self._worker_key(op.keys[0])
            manifest_key = replace(first, layer_name="")
            if self._backend.lookup_boundary(
                    manifest_key, group.layer_names,
                    expected_boundary_tokens=op.snapshot_boundary_tokens
                    ) != LookupStatus.HIT:
                err = RuntimeError(
                    "kvshrink mamba load TOCTOU: boundary changed after "
                    f"HIT req={req_meta.req_id} boundary="
                    f"{op.snapshot_boundary_tokens}; refusing to enter "
                    "forward with unrestored state")
                self._poison_load(err)
                raise err
        by_layer: dict[str, list] = {}
        for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
            by_layer.setdefault(key.layer_name, []).append(
                (gpu_block_id, key.hash_str))
        if not by_layer:
            return 0
        # Every layer of the group addresses the same chunk sequence
        # (scheduler invariant: keys expand per block x layer).
        any_layer = next(iter(by_layer))
        entries = by_layer[any_layer]
        for layer_name, ent in by_layer.items():
            if ent != entries:
                err = RuntimeError(
                    "kvshrink chunk load: inconsistent op expansion "
                    f"req={req_meta.req_id} group={op.group_idx} "
                    f"layer={layer_name}")
                self._poison_load(err)
                raise err
        views = {ln: self._layer_views(ln) for ln in by_layer}
        # Split into calls with unique chunk labels (one engine call
        # maps chunk_labels 1:1 to chunk_indices).
        calls: list[tuple[list, list]] = []
        slot_of: dict[str, int] = {}
        for gpu_block_id, h in entries:
            c = slot_of.get(h)
            if c is None:
                slot_of[h] = len(calls)
                calls.append(([], []))
                c = slot_of[h]
            calls[c][0].append(gpu_block_id)
            calls[c][1].append(h)
        npages = 0
        for indices, labels in calls:
            tasks = self._backend.submit_group_loads(
                op.group_idx, views, indices, labels)
            for layer_name, td in tasks.items():
                self._load_tasks.setdefault(layer_name, []).append(td)
                npages += len(indices)
        if npages:
            self._emit_transfer_bytes(
                "load", op.group_idx,
                npages * group.page_size_bytes)
        return npages

    def _wait_tasks(self, task_dicts) -> None:
        """Host-block until these engine tasks landed (fail-stop)."""
        try:
            for td in task_dicts:
                self._backend.wait_layer_loads(td)
        except BaseException as e:
            self._poison_load(e)
            raise

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def save_enabled(self) -> bool:
        """Reflects the KVSHRINK_SAVE switch: ON by default; "0"
        disables production saving and KVSHRINK_DEBUG_AUTOSAVE=1
        force-enables it."""
        return (os.getenv("KVSHRINK_SAVE", "1") != "0"
                or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")

    def _gather_save_candidates(self, metadata) -> dict:
        """Batch-level boundary candidates with cross-request dedup.
        Returns boundary_key -> {"group_idx", "pages": {layer_name:
        (key, gpu_block_id)}, "boundary_tokens"}."""
        candidates: dict[tuple, dict] = {}
        for req_meta in getattr(metadata, "save_requests", []):
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    key = self._worker_key(key)
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = {"group_idx": op.group_idx,
                                "pages": {},
                                "boundary_tokens": None}
                        candidates[key.boundary_key] = cand
                    cand["pages"][key.layer_name] = (key, gpu_block_id)
                    if op.snapshot_boundary_tokens is not None:
                        cand["boundary_tokens"] = \
                            op.snapshot_boundary_tokens
        return candidates

    def _submit_group_layers_save(self, g_idx, layer_names, entries):
        """Submit ONE async engine put covering ``layer_names`` for the
        blocks in ``entries`` (list of (gpu_block_id, chunk_label), same
        order for every layer -- scheduler invariant). Async D2H+zip on
        the engine's put_stream, self-gated on the compute stream so it
        reads final values. Returns the engine tasks dict."""
        chunk_indices = [gpu for gpu, _ in entries]
        chunk_labels = [h for _, h in entries]
        if len(set(chunk_labels)) != len(chunk_labels):
            raise RuntimeError(
                "kvshrink chunk save: duplicate chunk labels in one "
                f"engine call group={g_idx} (candidates dedup broken)")
        views = {ln: self._layer_views(ln) for ln in layer_names}
        tasks, _expanded = self._backend.submit_group_stores(
            g_idx, views, chunk_indices, chunk_labels)
        return tasks

    def save_kv_layer(self, layer_name: str, metadata) -> None:
        """Pipelined attention save. vLLM calls this on exit of EVERY
        attention layer during forward (kv_transfer_utils decorator).

        An attention layer's page for this step's tokens is final the
        moment that layer returns, so this layer's D2H+zip can overlap
        the remaining layers' compute instead of adding to the
        post-forward critical path. GDN groups are NOT covered here:
        their layers never call this hook and their state is only final
        after forward -- they save in wait_save.

        This method only SUBMITS. Checksum harvest, manifest commit,
        persist and eviction all stay in wait_save. Partial-boundary
        candidates are skipped here exactly as wait_save skips them, so
        the stashed per-layer entries stay aligned with the committable
        boundary list. KVSHRINK_SAVE_PIPELINED=0 disables this path
        (everything then submits in wait_save).
        """
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not self.save_enabled():
            return
        g_idx = self._attn_layer_group.get(layer_name)
        if g_idx is None:
            return  # not an attention layer we serve (fast path)
        expected = sorted(self._groups[g_idx].layer_names)
        entries = []
        for _bkey, cand in self._gather_save_candidates(metadata).items():
            if cand["group_idx"] != g_idx:
                continue
            if sorted(cand["pages"]) != expected:
                continue  # partial boundary: skipped at commit time too
            if layer_name not in cand["pages"]:
                continue
            key, gpu_block_id = cand["pages"][layer_name]
            entries.append((gpu_block_id, key.hash_str))
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, [layer_name],
                                               entries)
        self._step_attn_saves[layer_name] = (g_idx, tasks)

    def wait_save(self, metadata) -> tuple[int, int]:
        """Post-forward save: GDN groups submit here; attention groups
        collect their pipelined tasks; then wait, harvest checksums,
        commit schema-4 manifests, drain persist, evict over watermark.
        Fail-stop on any anomaly (the scheduler already advanced its
        incremental indices). Returns (pages, boundaries)."""
        if self._kv_caches_ref is None:
            return 0, 0
        _t0 = _now()
        candidates = self._gather_save_candidates(metadata)
        pipelined = os.getenv("KVSHRINK_SAVE_PIPELINED", "1") != "0"
        # group the complete candidates for one engine put per group
        per_group: dict[int, dict] = {}
        for bkey, cand in candidates.items():
            namespace, tp_size, rank, blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand["pages"]) != expected:
                # A partial boundary is skipped (the scheduler's
                # incremental cursor has advanced, so it ages out as a
                # MISS, never wrong data). Log unconditionally: this is
                # the one save-side anomaly that does not fail-stop.
                logger.warning(
                    "chunk_save skip commit g%d h=%s: expected %d "
                    "layers, stored %d (%s)", g_idx, blk_hash,
                    len(expected), len(cand["pages"]),
                    set(expected) ^ set(cand["pages"]))
                continue
            gslot = per_group.setdefault(g_idx, {"bnds": [],
                                                 "layers": {}})
            gslot["bnds"].append((bkey, cand))
            for layer_name in expected:
                key, gpu_block_id = cand["pages"][layer_name]
                gslot["layers"].setdefault(layer_name, []).append(
                    (gpu_block_id, key.hash_str))
        npages = 0
        nbound = 0
        for g_idx, gslot in per_group.items():
            layers = gslot["layers"]
            any_layer = next(iter(layers))
            entries = layers[any_layer]
            for layer_name, ent in layers.items():
                if ent != entries:
                    raise RuntimeError(
                        "kvshrink chunk save: inconsistent op expansion "
                        f"group={g_idx} layer={layer_name}")
            if self._groups[g_idx].kind != "mamba" and pipelined:
                # Attention: save_kv_layer submitted each layer during
                # forward; wait + harvest checksums here. A layer the
                # decorator never fired for is submitted now -- the
                # plan must be fully covered either way.
                cks = {}
                for layer_name in layers:
                    stashed = self._step_attn_saves.pop(layer_name, None)
                    tasks = (stashed[1] if stashed is not None
                             else self._submit_group_layers_save(
                                 g_idx, [layer_name], entries))
                    cks.update(self._backend.wait_group_stores(tasks))
            else:
                tasks = self._submit_group_layers_save(
                    g_idx, list(layers), entries)
                cks = self._backend.wait_group_stores(tasks)
            chunk_indices = [gpu for gpu, _ in entries]
            chunk_labels = [h for _, h in entries]
            npages += len(chunk_indices) * len(layers)
            self._emit_transfer_bytes(
                "save", g_idx,
                len(chunk_indices) * len(layers)
                * self._groups[g_idx].page_size_bytes)
            expected = sorted(layers)
            for i, (bkey, cand) in enumerate(gslot["bnds"]):
                namespace, tp_size, rank, blk_hash, _ = bkey
                manifest_key = CacheKey(
                    namespace=namespace, tp_size=tp_size, rank=rank,
                    block_hash=blk_hash, group_idx=g_idx, layer_name="")
                layer_ck = {ln: cks[ln][i] for ln in layers}
                if self._backend.commit_chunks(
                        manifest_key, expected, layer_ck,
                        [chunk_labels[i]],
                        expected_boundary_tokens=cand["boundary_tokens"]):
                    nbound += 1
        if self._step_attn_saves:
            # Layers whose submit was never consumed by a group above
            # (plan changed mid-step): wait them so pinned staging is
            # released, then drop. Their data is identical to what the
            # group path stored (same labels), so nothing is lost.
            logger.warning(
                "chunk_save: %d stashed layer saves unconsumed (%s); "
                "draining", len(self._step_attn_saves),
                sorted(self._step_attn_saves))
            for _ln, (_g, tasks) in self._step_attn_saves.items():
                self._backend.wait_group_stores(tasks)
            self._step_attn_saves.clear()
        if per_group:
            # durability: flush the freshly-zipped chunk groups to the
            # Storage tier (a boundary whose chunks never persisted is
            # cleaned from the Record at startup -> fail-closed MISS).
            # Drain in batches: a step whose new groups exceed one batch
            # must not publish manifests for chunks still memory-only.
            while True:
                res = self._backend.persist_engine(4096)
                if res.get("persisted", 0) < 4096:
                    break
            # memory: keep the engine's Mem under its watermark. Safe
            # here only because the drain above left nothing unpersisted
            # (evicting an unpersisted group would drop its Record
            # entry -> permanent MISS).
            ev = self._backend.evict_over_watermark()
            if ev.get("evicted", 0) and os.getenv("KVSHRINK_DEBUG_LOG"):
                logger.info(
                    "evict_over_watermark: evicted=%d bytes_freed=%d "
                    "(memory copies only, persisted data intact)",
                    ev["evicted"], ev.get("bytes_freed", 0))
        if npages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            logger.info(
                "chunk_save: %d pages stored, %d boundaries committed "
                "elapsed_ms=%.3f (rank %d/%d)", npages, nbound,
                (_now() - _t0) * 1e3, self.rank, self.tp_size)
        self._emit_job_latency("store", _now() - _t0)
        return npages, nbound

    # ------------------------------------------------------------------
    # debug dump
    # ------------------------------------------------------------------
    def debug_dump_state(self) -> None:
        """KVSHRINK_DEBUG_DUMP=1: log sha256 of the first layer page of
        every mamba group at gpu blocks 0..9, so cold-vs-hot GPU states
        can be compared byte-exactly."""
        if not os.getenv("KVSHRINK_DEBUG_DUMP") \
                or self._kv_caches_ref is None:
            return
        import hashlib
        for group in self._groups:
            if group.kind != "mamba":
                continue
            ln = group.layer_names[0]
            for blk in range(10):
                page = self._canon.get_page(ln, blk)
                h = hashlib.sha256(
                    page.cpu().numpy().tobytes()).hexdigest()
                logger.info("DUMP g%d block=%d sha=%s",
                            group.group_idx, blk, h[:16])

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Flush and release the backend (Record sync, writer lease),
        then re-raise the first sticky load poison so cleanup never
        masks it."""
        errors = []
        try:
            self._backend.close()
        except BaseException as e:  # pragma: no cover - collect
            errors.append(e)
        if self._load_poison is not None:
            errors.append(self._load_poison)
        if errors:
            raise errors[0]
