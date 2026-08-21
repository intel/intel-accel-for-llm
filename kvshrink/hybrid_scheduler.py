# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-side request state for the hybrid connector.

For each request we:
1. run the hit policy (find_longest_cache_hit) against the backend,
2. after vLLM allocates blocks, record per-group block tables,
3. build load metadata: for attention groups, every hit block in the
   prefix; for mamba groups, the single state snapshot block at the
   restore boundary (GDN loads piggyback on the preceding attention layer's
   wait_for_layer_load; the leading GDN segment waits at
   start_load_kv),
4. build incremental save metadata and track resume/cursor
   rollback lifecycle.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .hybrid_metadata import (
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta, RequestGroupState,
    RequestState, make_boundary_key,
)
from .hybrid_policy import HybridHitPolicy, LookupResult, LookupStatus

# Log under the vllm.* namespace: vLLM's init_logger attaches NO
# handler and relies on propagation to the configured "vllm" parent
# logger, so a bare __name__ logger silently drops every record in the
# EngineCore process.
logger = logging.getLogger("vllm." + __name__)

# Guarded import + no-op fallbacks so a
# metrics failure can NEVER affect the inference path.
try:
    from .hybrid_metrics import (  # noqa: E402
        inc as _metric_inc,
        set_gauge as _metric_set_gauge,
    )
except Exception:  # pragma: no cover - fail-open by design
    def _metric_inc(*a, **k):
        """Metrics no-op fallback (fail-open): a broken metrics stack
        must never break serving."""
        pass

    def _metric_set_gauge(*a, **k):
        """Metrics no-op fallback (fail-open): a broken metrics stack
        must never break serving."""
        pass


class HybridRequestScheduler:
    """Scheduler-side request state machine: hit detection + load/save
    plan builder.

    This object owns no vLLM interface methods itself; the connector
    facade (connector.py) holds the vLLM hooks and delegates. The vLLM
    trigger map below says WHEN each entry point fires and WHAT it is
    for; per-function docstrings describe HOW.

    One scheduling pass looks like this:

    1. NEW request arrives -> the core asks the connector
       ``get_num_new_matched_tokens(request, num_computed_tokens)``
       -> :meth:`get_num_new_matched_tokens`.
       Purpose: how many tokens beyond the local prefix-cache hit can be
       treated as already computed thanks to the external store.
       We run the hit policy over the request's block hashes (Record-
       gated, always synchronous), remember the authoritative restore
       point as ``snapshot_boundary``, and return the external token
       count. The core then skips recomputing those tokens.
    2. Block allocation succeeded (same pass) -> the core calls
       ``connector.update_state_after_alloc(request, blocks,
       num_external_computed_tokens)`` -> :meth:`update_state_after_alloc`.
       Purpose: tell us where the GPU blocks landed and how many
       external tokens the core accepted (i.e. will skip recompute for).
       We snapshot per-group block_ids and set ``pending_load_tokens`` --
       the external tokens the worker MUST restore before forward.
    3. End of the pass: the core calls
       ``connector.build_connector_meta(scheduler_output)``. The facade
       asks this object for per-request plans and ships them to the
       worker inside the connector metadata. Four kinds of work:

       a) LOAD plan for NEW requests -> :meth:`build_load_meta`.
          Restores pages up to the snapshot boundary recorded in step 1
          (never re-looks-up: after step 2 the progress counters already
          include external tokens, a fresh lookup would be polluted).
       b) LOAD plan for PREEMPTION-RESUMED requests ->
          :meth:`build_resumed_load_meta`. v1 carries resumed requests
          in ``scheduled_cached_reqs.resumed_req_ids``, NOT in
          ``scheduled_new_reqs``, so they need their own loop --
          missing them would yield garbage output after preemption. Guard: if the
          core accepted external tokens for the request but we find
          zero restorable pages, raise instead of entering forward with
          unrestored KV while the core skips recompute.
       c) SAVE plan for EVERY request scheduled this pass ->
          :meth:`build_save_meta`. Incremental: only blocks/boundaries
          not previously emitted are saved. The worker executes it
          AFTER forward, when the GPU pages hold state up to
          computed+scheduled tokens.
       d) Bookkeeping for RUNNING cached requests ->
          :meth:`on_cached_request`, done before (c). Sync the
          authoritative progress and block tables from upstream. On
          resume (or any progress regression) roll the save cursor
          back, so boundaries emitted before a preemption but never
          provably persisted get re-emitted (safe: overwrite is
          idempotent; skipping them would lose data).
    4. Request teardown -> the core calls
       ``connector.request_finished(request)`` ->
       :meth:`on_request_finished`, which drops the RequestState.

    How per-group block tables (RequestGroupState.block_ids) stay
    current
    ---------------------------------------------------------------
    block_ids is our copy of vLLM's block table for the request, one
    list per KV cache group. A block id is an index into that group's
    GPU block pool (not a raw address; the worker multiplies it by the
    page layout to locate the data).

    vLLM allocates new blocks inside ``kv_cache_manager.allocate_slots``
    during every scheduling pass, whenever a request crosses a block
    boundary (decode: a new block every block_size tokens; chunked
    prefill: at each boundary crossing). Those new block ids reach us
    through TWO channels, depending on which scheduling loop the
    request is in:

    - Requests scheduled from the WAITING queue (new and
      preemption-resumed): immediately after allocate_slots succeeds,
      the core calls ``connector.update_state_after_alloc`` with the
      request's FULL current block table
      (``kv_cache_manager.get_blocks(request_id)``). We replace our
      copy wholesale.
    - RUNNING requests: the core's running loop calls allocate_slots
      but does NOT notify the connector. Instead the newly allocated
      blocks travel inside the SchedulerOutput as
      ``scheduled_cached_reqs.new_block_ids`` (a parallel array to
      ``req_ids``). :meth:`on_cached_request` appends them to our copy
      (or replaces it for resumed requests, where upstream sends the
      full table again).

    Ordering inside build_connector_meta matters:
    :meth:`on_cached_request` (table sync) runs BEFORE
    :meth:`build_save_meta`, so the save plan already sees blocks
    allocated in the SAME pass. The plan is executed by the worker
    after forward (wait_for_save), at which point those blocks
    actually contain the computed KV data.

    One scheduling pass, top to bottom
    ---------------------------------------------------------------
    ::

      vLLM core (scheduler process)        this object
      ===================================  ==============================
      new/resumed request:
        get_num_new_matched_tokens(req)  -> hit lookup; pin
                                            snapshot_boundary
        allocate_slots(req)              (core allocates GPU blocks)
        update_state_after_alloc(req,    -> replace block_ids copies;
          blocks, num_external_tokens)       set pending_load_tokens
      running requests:
        allocate_slots(req)              (no connector callback; new
                                          blocks ride new_block_ids)
      build_connector_meta(sched_output)
        per new request                -> build_load_meta
        per resumed request            -> build_resumed_load_meta
                                          (fail-closed guard)
        per running request            -> on_cached_request FIRST
                                          (sync progress + tables),
                                          then build_save_meta
      ================= pickle: KVShrinkConnectorMetadata ==============
      worker process (per GPU rank)
        start_load_kv                  execute LOAD plans (BEFORE fwd)
        forward                        GPU computes this pass's tokens
        wait_for_save                  execute SAVE plans (AFTER fwd)

    Structure and data flow
    ---------------------------------------------------------------
    ::

      scheduler process                  worker process
      ============================       ============================
      HybridRequestScheduler             connector (worker role)
        RequestState (per request)         start_load_kv
          block_hashes ............        wait_for_save
            content hash per block             |
          snapshot_boundary ......             v
            restore point pinned           BoundaryBackend
            at lookup                          |
          pending_load_tokens ....             v
            external tokens the            TensorZip chunk engine
            worker must restore                |
          groups[g]:                           v
            block_ids ............         disk / host memory
              copy of vLLM's block           (content-addressed
              table, group g's pool           chunk store)
            next_stored_chunk_idx
              save cursor (rolled back
              on resume/regression)

    Save addressing: one logical block, two addresses
    ---------------------------------------------------------------
    ::

      token stream     | block 0 | block 1 | ... | block i |
                       (block_size tokens each)

      block_hashes[i] --> store key     (content-addressed: which
                                         chunks the page is written to
                                         / found under)
      block_ids[i]    --> GPU pool blk  (position-addressed: which
                                         physical block the worker
                                         reads the page from)

      A save op pairs them: keys[k] <-> gpu_block_ids[k].

    Everything else in this file is internal plumbing for the above
    (key builders, hash recompute, metrics glue).
    """

    def __init__(
        self,
        groups: list[GroupInfo],
        backend,
        hash_block_size: int,
        namespace: str,
        tp_size: int,
        rank: int,
        prefix_caching_hash_algo: str = "sha256",
    ):
        self._groups = groups
        self._backend = backend
        self._hash_block_size = hash_block_size
        self._namespace = namespace
        self._tp_size = tp_size
        self._rank = rank
        # Engine-configured block-hash algorithm
        # (cache_config.prefix_caching_hash_algo). Only used by the
        # defensive _hashes_from_prompt path, which MUST reproduce
        # vLLM's own hashes byte for byte.
        self._prefix_caching_hash_algo = prefix_caching_hash_algo
        self._req_states: dict[str, RequestState] = {}
        self.cursor_rollbacks = 0
        """Record the per-group layout, hit-policy backend and TP
        identity, and own the per-request RequestState table plus the
        resume/cursor-rollback counter that spans a request's whole
        scheduling lifecycle.

        This is the DECISION side of the hybrid path: it only plans
        (hit lookup, load/save ReqMeta) against a read-only backend;
        the worker executes transfers and owns the page views and this
        rank's writer lease."""

    def lifecycle_stats(self) -> dict:
        """Expose lifecycle counters for tests and operators: the
        number of live RequestStates must drop to 0 once every request
        has finished (or state leaked), alongside the cumulative cursor
        rollbacks caused by preemption/resume."""
        return {
            "request_states": len(self._req_states),
            "cursor_rollbacks": self.cursor_rollbacks,
        }

    # ------------------------------------------------------------------
    def on_new_request(
        self, req_id: str, block_hashes: list[int],
        num_computed_tokens: int,
    ) -> None:
        """Register a fresh RequestState. Internal: called by us from
        get_num_new_matched_tokens / build_load_meta /
        update_state_after_alloc when a request first becomes visible
        (vLLM has no dedicated "new request" connector hook)."""
        self._req_states[req_id] = RequestState(
            request=req_id,
            block_hashes=list(block_hashes),
            num_locally_computed_tokens=num_computed_tokens,
            groups=tuple(
                RequestGroupState() for _ in self._groups),
        )

    def on_request_finished(self, req_id: str) -> None:
        """vLLM trigger: core frees the request ->
        connector.request_finished -> here. Drop the RequestState;
        committed boundaries are content-addressed and stay."""
        self._req_states.pop(req_id, None)

    def on_cached_request(
        self, req_id: str, new_block_ids, resumed: bool,
        num_computed_tokens: Optional[int],
    ) -> None:
        """vLLM trigger: every scheduling pass, for each running
        (cached) request, via connector.build_connector_meta.

        Track a scheduled CACHED request: sync the authoritative
        num_computed from upstream and extend block tables with newly
        allocated blocks so incremental save targets the right slots.
        ``resumed`` (preemption resume) REPLACES tables per upstream
        CachedRequestData semantics.

        Cursor rollback: the
        incremental save cursor means "proven not to need re-emission in
        THIS request lifecycle", NOT "metadata was once constructed".
        On resume (or any authoritative progress regression, even with a
        missing resumed flag -- fail-closed) every group's cursor rolls
        back to floor(N / block_size): boundaries emitted before a
        preemption but never provably persisted are re-emitted.
        Re-emission is safe (idempotent overwrite + checksum re-commit);
        NOT rolling back permanently skips un-persisted
        boundaries."""
        state = self._req_states.get(req_id)
        if state is None:
            return
        old_progress = max(state.num_locally_computed_tokens,
                           state.last_known_progress)
        regression = (num_computed_tokens is not None
                      and num_computed_tokens < old_progress)
        if num_computed_tokens is not None:
            state.num_locally_computed_tokens = num_computed_tokens
            state.last_known_progress = num_computed_tokens
        if resumed or regression:
            # fail-closed: a missing progress on resume is treated as
            # N=0 (roll everything back) rather than skipping the
            # rollback.
            safe_n = num_computed_tokens or 0
            for g_idx, group in enumerate(self._groups):
                gstate = state.groups[g_idx]
                safe = safe_n // group.block_size
                if gstate.next_stored_chunk_idx > safe:
                    self.cursor_rollbacks += 1
                    _metric_set_gauge(
                        "kvshrink_cursor_rollbacks",
                        value=float(self.cursor_rollbacks))
                    if os.getenv("KVSHRINK_DEBUG_LOG"):
                        logger.info(
                            "cursor rollback req=%s g%d: %d -> %d "
                            "(progress %d -> %s, resumed=%s)",
                            req_id, g_idx, gstate.next_stored_chunk_idx,
                            safe, old_progress, num_computed_tokens,
                            resumed)
                    gstate.next_stored_chunk_idx = safe
        if new_block_ids:
            for g_idx in range(min(len(self._groups),
                                   len(new_block_ids))):
                ids = new_block_ids[g_idx]
                if resumed:
                    # upstream semantics: for resumed requests
                    # new_block_ids IS the table (replace), per group --
                    # including an EMPTY list, which clears stale blocks
                    state.groups[g_idx].block_ids = list(ids) if ids \
                        else []
                elif ids:
                    state.groups[g_idx].block_ids.extend(ids)

    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        """External lookup; returns (hit_tokens, has_async_load).

        vLLM trigger: the core calls connector.get_num_new_matched_tokens
        while scheduling a NEW request, BEFORE block allocation, to ask
        how many tokens the external store can vouch for. Our answer is
        added to num_computed_tokens by the core, so it must be backed
        by restorable pages.

        Always synchronous: the chunk-tier lookup is Record-gated and
        never defers (no PENDING/RETRY states exist anymore).
        """
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        self.on_new_request(
            request.request_id, list(request.block_hashes),
            num_computed_tokens)
        policy = HybridHitPolicy(
            self._groups, self._backend, self._hash_block_size,
            num_computed_tokens, self._namespace, self._tp_size, self._rank)
        result, trace = policy.find_longest_cache_hit(
            list(request.block_hashes), request.num_tokens)
        if result.status == LookupStatus.HIT:
            # The policy HIT already gated on live chunk presence
            # (engine Record), so the boundary is complete by
            # construction; only record the snapshot point.
            state = self._req_states.get(request.request_id)
            if state is not None:
                if state.block_hashes:
                    state.snapshot_boundary = result.boundary_tokens
                else:
                    result = LookupResult(LookupStatus.MISS, 0)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "policy result req=%s status=%s boundary=%d hashes=%d "
                "trace=%s",
                request.request_id, result.status.value,
                result.boundary_tokens, len(request.block_hashes), trace)
        # Metrics recorded on the FINAL completeness result (not the
        # raw policy trace).
        try:
            _metric_inc(
                "kvshrink_lookup_boundary",
                {"group": "all", "kind": "completeness",
                 "result": result.status.value})
        except Exception:  # pragma: no cover - fail-open
            pass
        external = result.boundary_tokens - num_computed_tokens
        if external < 0:
            external = 0
        if result.status == LookupStatus.HIT:
            _metric_inc("kvshrink_external_hit_tokens", value=float(external))
        logger.debug(
            "req=%s external_hit=%d boundary=%d trace=%s",
            request.request_id, external, result.boundary_tokens, trace)
        return external, False

    # ------------------------------------------------------------------
    def update_state_after_alloc(
        self, request, blocks, num_external_tokens: int
    ) -> None:
        """Record the allocated block tables per group (after alloc).

        vLLM trigger: the core calls connector.update_state_after_alloc
        right AFTER successful block allocation (same pass as the
        lookup), passing the num_external_tokens it accepted. Only on
        this path is pending_load_tokens set -- the alloc-failure path
        never calls us, so no pending load obligation can leak.

        Design note -- why this hook only RECORDS facts and defers plan
        building to build_connector_meta (unlike the legacy
        pure-attention connector, which computes the load range inline
        here):

        1. The restore point is fixed at lookup time, not by arithmetic.
           GDN state can only be restored at segment boundaries, so the
           hit policy pins ``snapshot_boundary`` during
           get_num_new_matched_tokens. Dividing
           num_external_tokens by block_size here would not necessarily
           land on a legal boundary; the plan must follow the recorded
           boundary, never a fresh computation.
        2. Multiple KV groups keep separate block tables. The attention
           group and the GDN group have independent block pools; the
           per-group ids recorded here are later assembled by different
           rules (attention per block, mamba = the last non-null
           snapshot slot). The legacy connector can hardcode
           ``get_block_ids()[0]``; we cannot.
        3. New and preemption-resumed requests share one plan builder.
           Resumed requests arrive via resumed_req_ids, not the new
           request path; assembling plans at alloc time would fork the
           logic. Recording facts here and building plans from the
           recorded state in build_connector_meta keeps both entries on
           the same code path (_build_load_meta_from_state).

        ``blocks`` is the result of kv_cache_manager.get_blocks(request_id):
        a tuple of per-group block sequences (KVCacheBlock objects).
        """
        req_id = request.request_id
        state = self._req_states.get(req_id)
        if state is None:
            self.on_new_request(
                req_id, list(request.block_hashes), 0)
            state = self._req_states[req_id]
        state.num_locally_computed_tokens = (
            state.num_locally_computed_tokens + num_external_tokens)
        state.pending_load_tokens = num_external_tokens
        if hasattr(blocks, "get_block_ids"):
            all_block_ids = blocks.get_block_ids()
        else:
            all_block_ids = tuple(
                [b.block_id for b in group_blocks] if group_blocks else []
                for group_blocks in blocks)
        for g_idx, group in enumerate(self._groups):
            if g_idx >= len(all_block_ids):
                continue
            ids = list(all_block_ids[g_idx])
            state.groups[g_idx].block_ids = ids
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "update_state req=%s per-group block_ids: %s hashes=%d",
                req_id, [[b for b in g.block_ids] for g in state.groups],
                len(state.block_hashes))

    # ------------------------------------------------------------------
    def build_load_meta(self, new_req, scheduled_tokens: int = 0) -> ReqMeta:
        """Build the LOAD ReqMeta for a NewRequestData entry.

        vLLM trigger: connector.build_connector_meta iterates
        ``scheduler_output.scheduled_new_reqs`` at the end of the pass.

        attention groups: all prefix blocks whose boundary hash is HIT;
        mamba groups: the single state snapshot block at the restore
        boundary (written into the CURR state block; see
           _build_load_meta_from_state).
        """
        req_id = new_req.req_id
        state = self._req_states.get(req_id)
        if state is None:
            self.on_new_request(
                req_id, list(new_req.prompt_token_ids or []), 0)
            state = self._req_states[req_id]
            state.block_hashes = self._hashes_from_prompt(
                new_req.prompt_token_ids or [])
            state.num_locally_computed_tokens = new_req.num_computed_tokens
            state.groups = tuple(
                RequestGroupState() for _ in self._groups)
            # populate block tables from NewRequestData
            for g_idx in range(len(self._groups)):
                if g_idx < len(new_req.block_ids):
                    state.groups[g_idx].block_ids = list(
                        new_req.block_ids[g_idx])
        return self._build_load_meta_from_state(
            req_id, state, scheduled_tokens,
            num_tokens=getattr(new_req, "num_tokens", "?"))

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> Optional[ReqMeta]:
        """Build the LOAD ReqMeta for a PREEMPTION-RESUMED request.

        vLLM v1 carries resumed requests in
        ``scheduled_cached_reqs.resumed_req_ids`` (NOT
        ``scheduled_new_reqs``), so build_connector_meta must ask for
        their load meta explicitly. State (block tables, snapshot
        boundary, pending_load_tokens) was already refreshed this
        scheduling pass by
        get_num_new_matched_tokens + update_state_after_alloc.

        Fail-closed: if the core accepted external tokens
        (pending_load_tokens > 0) the meta MUST carry restorable pages;
        an empty load with pending external tokens means the forward
        would read
        unrestored KV while num_computed_tokens skips recompute. Raise
        instead of silently emitting wrong tokens.
        Returns None when the request was never seen by the connector
        (no external tokens could have been accepted).
        """
        state = self._req_states.get(req_id)
        if state is None:
            return None
        meta = self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)
        if state.pending_load_tokens > 0:
            npages = sum(len(op.keys) for op in meta.group_ops)
            if npages == 0:
                raise RuntimeError(
                    "kvshrink resumed request has accepted external "
                    "tokens but no restorable pages (req="
                    f"{req_id} pending={state.pending_load_tokens} "
                    f"boundary={state.snapshot_boundary} "
                    f"sched={scheduled_tokens}): refusing to enter "
                    "forward with unrestored state")
        return meta

    def _build_load_meta_from_state(
        self, req_id: str, state, scheduled_tokens: int,
        num_tokens="?",
    ) -> ReqMeta:
        """The snapshot_boundary recorded by get_num_new_matched_tokens
        is the AUTHORITATIVE restore boundary for this alloc/load. NEVER recompute here: after
        update_state_after_alloc the locally-computed counter already
        includes external tokens, and a fresh lookup would be polluted.
        Missing/expired boundary must
        FAIL CLOSED (boundary 0), not guess."""
        boundary = state.snapshot_boundary
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "TAIL req=%s snapshot_boundary=%d computed_before_fwd=%d "
                "external=%d num_tokens=%s",
                req_id, state.snapshot_boundary,
                state.num_locally_computed_tokens,
                state.snapshot_boundary - state.num_locally_computed_tokens,
                num_tokens)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            ids = state.groups[g_idx].block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            if group.kind == "attention":
                gran = group.block_size
                num_hash = boundary // gran
                for i in range(num_hash):
                    if i >= len(state.block_hashes):
                        break
                    blk_hash = state.block_hashes[i]
                    key = self._boundary_key(group, blk_hash)
                    if self._backend.lookup_boundary(
                            key, group.layer_names) != LookupStatus.HIT:
                        break
                    # v0.21 hashes are per complete block: hash i == block i
                    if i < len(ids):
                        # one page key + gpu block per layer (full expansion)
                        for layer_name in group.layer_names:
                            keys.append(self._page_key(key, layer_name))
                            gpu_ids.append(ids[i])
            elif group.kind == "mamba":
                # Load the snapshot into the CURR state block ONLY
                # (v0.23.0 semantics, verified upstream): the align-mode
                # block table pins the GDN execution metadata to column 0
                # = the block holding this step's last scheduled token,
                # for BOTH the chunked-prefill and decode paths -- there
                # is no prev/curr distinction at execution time.
                # preprocess_mamba's prev -> curr copy runs BEFORE
                # start_load_kv (execute_model order), and our H2D write
                # lands during forward (piggybacked on the preceding
                # attention layer's wait_for_layer_load), i.e. after the
                # copy and before the GDN layer executes, so CURR is the
                # one correct target. Writing PREV would be dead work:
                # the kernel never reads it that step.
                if state.block_hashes and boundary > 0:
                    # hash index of the snapshot AT boundary:
                    # hash[i] covers [i*bs, (i+1)*bs) -> snapshot at
                    # boundary lives at hash[boundary//bs - 1]
                    idx = boundary // group.block_size - 1
                    if 0 <= idx < len(state.block_hashes):
                        blk_hash = state.block_hashes[idx]
                        key = self._boundary_key(group, blk_hash)
                        if self._backend.lookup_boundary(
                                key, group.layer_names,
                                boundary) == LookupStatus.HIT:
                            bs = group.block_size
                            # CURR running-state index for this step
                            # (upstream align-mode formula):
                            # (num_computed + num_scheduled - 1) // bs
                            # with num_computed == boundary here.
                            #
                            # Why exactly one slot, and why this one:
                            # in align mode the kernels do not scan the
                            # table, they gather a single column --
                            # mamba_get_block_table_tensor computes
                            # start = (seq_lens - 1) // block_size and
                            # mamba_attn then takes column 0 of the
                            # gathered result. So this index is the only
                            # location forward will ever read for this
                            # step. The previous generation wrote both a
                            # prev and a curr slot because the timing of
                            # vLLM's prev->curr copy was unclear; the
                            # v0.23 source settles it, making the second
                            # write dead weight. The flip side is that
                            # there is no fallback slot, so an invalid
                            # index below must fail-stop rather than
                            # degrade.
                            curr_idx = (boundary + scheduled_tokens -
                                        1) // bs

                            def _slot_ok(t):
                                return 0 <= t < len(ids) and ids[t] != 0

                            # Fail-closed contract: an external HIT has already
                            # committed num_computed_tokens=boundary via
                            # get_num_new_matched_tokens; silently skipping a
                            # required slot
                            # would let forward read unrestored state and
                            # emit wrong tokens. Fail-stop (EngineCore
                            # fatal, same semantics as the TOCTOU gate)
                            # instead of producing a partial mamba load.
                            if scheduled_tokens <= 0:
                                raise RuntimeError(
                                    "kvshrink mamba external HIT with "
                                    "scheduled_tokens=0 "
                                    f"(req={req_id} boundary={boundary}): "
                                    "production hits must schedule >= 1 "
                                    "token; refusing to build load meta")
                            if not _slot_ok(curr_idx):
                                raise RuntimeError(
                                    "kvshrink mamba load curr slot "
                                    f"invalid (req={req_id} "
                                    f"boundary={boundary} "
                                    f"sched={scheduled_tokens} "
                                    f"table_idx={curr_idx} "
                                    f"table={ids}): refusing to enter "
                                    "forward with unrestored state")
                            targets = {curr_idx}
                            for table_idx in sorted(targets):
                                gpu_block = ids[table_idx]
                                for layer_name in group.layer_names:
                                    keys.append(self._page_key(
                                        key, layer_name))
                                    gpu_ids.append(gpu_block)
            if group.kind == "mamba" and keys:
                # One state snapshot restored at a boundary.
                # The SCHEDULER is the single
                # authoritative counter -- the worker-side increment in
                # connector.py is removed so a TP>1 restore is never
                # multi-source added.
                _metric_inc("kvshrink_state_snapshot_boundary", value=1.0)
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids),
                snapshot_boundary_tokens=boundary if group.kind == "mamba"
                else None))
        return ReqMeta(
            req_id=req_id,
            external_hit_tokens=boundary - state.num_locally_computed_tokens,
            group_ops=tuple(group_ops),
        )

    def build_save_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Production save: INCREMENTAL per-group page persistence.

        vLLM trigger: connector.build_connector_meta asks for a save
        plan for EVERY request scheduled this pass (new + cached); the
        worker executes it after forward (wait_for_save), so the GPU
        pages then hold state up to ``computed + scheduled`` tokens;
        boundaries are computed against THAT progress.

        The save path for NEWLY COMPUTED KV, end to end
        (pass N, request advances by S scheduled tokens)
        ---------------------------------------------------------------
        ::

          [scheduler, pass N]
            progress P = num_locally_computed_tokens + S   (predictive:
                                  the plan is built BEFORE forward but
                                  describes the state AFTER forward)
            per group, emit ops for work not previously emitted:
              attention: blocks [next_stored_chunk_idx, P//block_size)
                         -- every newly COMPLETED block, per layer
              mamba:     snapshot of the running state block, ONLY if
                         P lands exactly on a block boundary
            next_stored_chunk_idx advances at EMIT time (the worker
            save is fail-stop, so indices cannot silently diverge)
                    |
                    |  ReqMeta pickled inside connector metadata
                    v
          [worker, pass N]
            forward          -> KV for the S new tokens is now in the
                                GPU blocks (block_ids recorded earlier)
            wait_for_save    -> per (layer, block) in the plan:
                                read GPU block -> compress -> stage
                                chunks under the content-hash keys
            commit           -> atomic schema-4 manifest write; ONLY
                                now does the boundary become visible
                                (is_committed -> HIT) to future lookups
                    |
                    v
          [any later pass / any later request]
            get_num_new_matched_tokens can now HIT these pages:
            same content hash -> same keys -> restore instead of
            recompute.

        Each group tracks ``next_stored_chunk_idx``; a step emits save
        ops only for blocks/boundaries not previously emitted.

        - attention: every completed block hash in
          [next_stored, progress//gran) is saved (per-block pages are
          valid as soon as the block completes).
        - mamba: the running state block (last NON-NULL table slot) is
          saved only when progress lands EXACTLY on a block boundary --
          a partial tail is not a valid restore point, and snapshots are
          only addressable by boundary hashes.
        """
        state = self._req_states.get(req_id)
        if state is None:
            return ReqMeta(req_id=req_id)
        progress = state.num_locally_computed_tokens + scheduled_tokens
        state.last_known_progress = max(state.last_known_progress,
                                        progress)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            gstate = state.groups[g_idx]
            ids = gstate.block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            snapshot_boundary: Optional[int] = None
            if group.kind == "attention":
                num_hash = min(progress // group.block_size, len(ids),
                               len(state.block_hashes))
                start = gstate.next_stored_chunk_idx
                for i in range(start, num_hash):
                    blk_hash = state.block_hashes[i]
                    for layer_name in group.layer_names:
                        keys.append(self._page_key(
                            self._boundary_key(group, blk_hash),
                            layer_name))
                        gpu_ids.append(ids[i])
                if num_hash > start:
                    gstate.next_stored_chunk_idx = num_hash
            elif group.kind == "mamba":
                # Save the running state block: the last NON-NULL block in
                # the group's table. Block tables vary by token count:
                # single-element [X] (545-token req), null-prefixed
                # [0,0,X], or [null, X] -- block 0 is the reserved null
                # block. Do NOT assume len(ids) > 1.
                if state.block_hashes:
                    block_pos = None
                    for pos in range(len(ids) - 1, -1, -1):
                        if ids[pos] != 0:  # 0 = null block placeholder
                            block_pos = pos
                            break
                    if block_pos is None:
                        if os.getenv("KVSHRINK_DEBUG_LOG"):
                            logger.info(
                                "save mamba g%d: no non-null block in "
                                "ids=%s", g_idx, ids)
                    elif (progress > 0
                          and progress % group.block_size == 0):
                        idx = progress // group.block_size - 1
                        if (idx >= gstate.next_stored_chunk_idx
                                and idx < len(state.block_hashes)):
                            blk_hash = state.block_hashes[idx]
                            snapshot_boundary = progress
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    self._boundary_key(group, blk_hash),
                                    layer_name))
                                gpu_ids.append(ids[block_pos])
                            gstate.next_stored_chunk_idx = idx + 1
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids),
                snapshot_boundary_tokens=snapshot_boundary))
        return ReqMeta(
            req_id=req_id,
            external_hit_tokens=0,
            group_ops=tuple(group_ops),
        )

    def _boundary_key(self, group: GroupInfo, block_hash) -> CacheKey:
        """Content-addressed boundary key for one group (the
        layer_name="" identity). Carries namespace, tp_size, rank,
        group and block hash, so under TP each rank addresses its own
        shard."""
        return make_boundary_key(self._namespace, self._tp_size,
                                 self._rank, group.group_idx, block_hash)

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same
        namespace/tp/rank/hash/group as the boundary, plus the layer
        name. This is the exact page address the worker must move."""
        return CacheKey(
            namespace=boundary_key.namespace,
            tp_size=boundary_key.tp_size,
            rank=boundary_key.rank,
            block_hash=boundary_key.block_hash,
            group_idx=boundary_key.group_idx,
            layer_name=layer_name)

    def _hashes_from_prompt(self, token_ids: list[int]) -> list:
        """Recompute block hashes when RequestState was never created.

        Reproduces vLLM v0.23's ``request_block_hasher`` exactly: the
        same ``hash_block_tokens`` chain over FULL blocks only, with the
        engine-configured hash function and the process-global
        ``NONE_HASH`` as the first parent (hashes are sha256 BYTES in
        v0.23, not v0.21's ints). A divergent hash here would not be
        wrong data -- keys simply would not match -- but it would make
        every such request an unconditional MISS, so it is worth
        deriving from vLLM's own primitives rather than reimplementing.

        Defensive path only: the production flow always creates the
        RequestState in get_num_new_matched_tokens with the request's
        own ``block_hashes``.
        """
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import hash_block_tokens

        hash_fn = get_hash_fn_by_name(self._prefix_caching_hash_algo)
        bs = self._hash_block_size
        hashes: list = []
        parent = None
        for i in range(0, len(token_ids), bs):
            tokens = token_ids[i:i + bs]
            if len(tokens) < bs:
                break  # only complete blocks are hashed
            # extra_keys=None: the defensive path never carries MM/LoRA
            # keys, matching generate_block_hash_extra_keys for plain
            # text requests.
            parent = hash_block_tokens(hash_fn, parent, tokens, None)
            hashes.append(parent)
        return hashes
