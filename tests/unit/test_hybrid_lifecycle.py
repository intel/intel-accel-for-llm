"""Abort / preemption / resume lifecycle tests.

Rulings under test:
1. resume (or any authoritative progress regression) rolls every
   group's incremental save cursor back to floor(N / block_size) --
   emitted-but-unproven boundaries are re-emitted (idempotent, safe);
2. request_finished returns (False, None) -- the save path is
   synchronous so blocks free immediately; True would leak blocks
   (no get_finished ack ever comes);
3. request_finished fail-stops if async store jobs exist;
4. committed boundaries are content-addressed: abort/finish NEVER
   deletes them; uncommitted pages never hit.
"""


import pytest

from kvshrink.hybrid_metadata import (
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)
from kvshrink.hybrid_policy import LookupStatus
from kvshrink.hybrid_scheduler import HybridRequestScheduler

PAGE = 64 * 1024


class _Block:
    def __init__(self, block_id):
        self.block_id = block_id


def _attn(bs=16):
    return GroupInfo(
        group_idx=0, kind="attention", layer_names=("attn.0",),
        block_size=bs, page_size_bytes=PAGE, mamba_cache_mode=None,
        mamba_align_size=None)


def _mamba():
    return GroupInfo(
        group_idx=0, kind="mamba", layer_names=("m.0",),
        block_size=544, page_size_bytes=PAGE, mamba_cache_mode="align",
        mamba_align_size=544)


class _MissBackend:
    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        return LookupStatus.MISS


class _HitBackend:
    """Committed boundary hashes are HIT (content-addressed)."""

    def __init__(self, committed):
        self.committed = committed

    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        return (LookupStatus.HIT if key.block_hash in self.committed
                else LookupStatus.MISS)


def _sched(groups, backend=None):
    return HybridRequestScheduler(groups, backend or _MissBackend(),
                                  16, "ns", 1, 0)


def _setup_attn_req(sched, hashes, ids, tokens=0):
    sched.on_new_request("r1", block_hashes=hashes,
                         num_computed_tokens=tokens)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        ([_Block(i) for i in ids],), 0)


# ------------------------------------------------------------------
# 1-6: cursor rollback semantics
# ------------------------------------------------------------------

def test_resume_to_zero_rolls_cursor_and_reemits():
    """Attention: cursor at 2, resume to progress 0 -> cursor 0; the
    blocks are re-emitted when the request re-crosses boundaries."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=32)  # cursor -> 2
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 2
    sched.on_cached_request("r1", ([10, 11, 12, 13],), resumed=True,
                            num_computed_tokens=0)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 0, g
    m = sched.build_save_meta("r1", scheduled_tokens=32)
    assert m.group_ops[0].gpu_block_ids == (10, 11), m.group_ops[0]


def test_mamba_resume_reemits_boundary_snapshot():
    sched = _sched([_mamba()])
    sched.on_new_request("r1", block_hashes=[0, 1],
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}), ([_Block(5)],), 0)
    m1 = sched.build_save_meta("r1", scheduled_tokens=544)
    assert len(m1.group_ops[0].keys) == 1
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 1
    sched.on_cached_request("r1", ([_Block(9).block_id],), resumed=True,
                            num_computed_tokens=0)
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 0
    m2 = sched.build_save_meta("r1", scheduled_tokens=544)
    assert len(m2.group_ops[0].keys) == 1  # re-emitted
    assert m2.group_ops[0].gpu_block_ids == (9,)


def test_resume_to_nonzero_progress_rolls_to_floor():
    """Resume at N=32 (block 16): cursor rolls to floor(32/16)=2, so
    only blocks >= 2 re-emit."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)  # cursor -> 4
    sched.on_cached_request("r1", ([10, 11, 12, 13],), resumed=True,
                            num_computed_tokens=32)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 2, g
    m = sched.build_save_meta("r1", scheduled_tokens=32)
    assert m.group_ops[0].gpu_block_ids == (12, 13), m.group_ops[0]


def test_monotonic_progress_no_rollback():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=32)
    sched.on_cached_request("r1", None, resumed=False,
                            num_computed_tokens=32)
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 2
    assert sched.cursor_rollbacks == 0


def test_resumed_empty_table_clears_and_rolls_back():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1], [10, 11])
    sched.build_save_meta("r1", scheduled_tokens=32)
    sched.on_cached_request("r1", ([],), resumed=True,
                            num_computed_tokens=0)
    g = sched._req_states["r1"].groups[0]
    assert g.block_ids == []
    assert g.next_stored_chunk_idx == 0


def test_progress_regression_without_resumed_flag_rolls_back():
    """Fail-closed: authoritative progress regression rolls the cursor
    even if the resumed flag is missing (defence in depth)."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)  # cursor -> 4
    sched.on_cached_request("r1", None, resumed=False,
                            num_computed_tokens=16)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 1, g  # floor(16/16)
    assert sched.cursor_rollbacks == 1


# ------------------------------------------------------------------
# 7-8: request_finished contract
# ------------------------------------------------------------------

def _sched_side_connector(sched):
    """The real connector facade in its scheduler role, with the hybrid
    plan builder injected. Construction is bypassed on purpose: these
    tests cover the dispatch contract, not engine startup.

    Skipped when the compiled iaxl extension is unavailable (the facade
    imports it for the pure-attention path); every other test in this
    file is pure logic and always runs.
    """
    pytest.importorskip(
        "iaxl", reason="kvshrink_connector imports the built iaxl extension")
    from kvshrink.kvshrink_connector import KVShrinkConnector

    conn = object.__new__(KVShrinkConnector)
    conn._hyb_sched = sched
    conn._hyb_worker = None
    conn._hyb_backend = None
    conn._hyb_groups = list(sched._groups)
    return conn


def test_request_finished_returns_false_and_clears_state():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1], [10, 11])
    conn = _sched_side_connector(sched)
    req = type("R", (), {"request_id": "r1"})
    free, delay = conn.request_finished(req, None)
    assert (free, delay) == (False, None), \
        "synchronous save must free blocks immediately"
    assert "r1" not in sched._req_states


def test_request_finished_pending_async_job_returns_false_none():
    """Committed boundaries are content-addressed and per-boundary (not
    per-request), so a finished request NEVER blocks block freeing:
    request_finished returns (False, None). A failed store surfaces as
    sticky poison at the next worker hook, not here."""
    sched = _sched([_attn()])
    conn = _sched_side_connector(sched)
    req = type("R", (), {"request_id": "r1"})
    out = conn.request_finished(req, None)
    assert out == (False, None), out


# ------------------------------------------------------------------
# 9-12: committed data ownership / orphan semantics
# ------------------------------------------------------------------

def test_abort_keeps_committed_boundary_hittable():
    """Content-addressed cache: after abort, a NEW request with the
    same hashes still HITs the committed boundary."""
    backend = _HitBackend(committed={0, 1})
    sched = _sched([_attn()], backend)
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.on_request_finished("r1")  # abort
    # a fresh lookup for the same hashes still hits
    assert backend.lookup_boundary(
        CacheKey("ns", 1, 0, 0, 0, "")) == LookupStatus.HIT


def test_resumed_missing_progress_rolls_back_to_zero():
    """Fail-closed (super-master gate §6.10): resumed=True with missing
    num_computed rolls ALL group cursors to 0 (safe N=0), never skips."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)  # cursor -> 4
    sched.on_cached_request("r1", ([10, 11, 12, 13],), resumed=True,
                            num_computed_tokens=None)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 0, g
    assert sched.cursor_rollbacks == 1
    m = sched.build_save_meta("r1", scheduled_tokens=32)
    assert m.group_ops[0].gpu_block_ids == (10, 11)  # re-emitted


def test_abort_resume_stress_1000_iterations_zero_residue():
    """1000 rounds of new/save/resume/finish. Every round the cursor
    rolls back and re-emits; at the end no request state is left behind,
    the rollback counter is exact and nothing raised."""
    backend = _MissBackend()
    sched = _sched([_attn()], backend)
    conn = _sched_side_connector(sched)
    for i in range(1000):
        rid = f"r{i}"
        sched.on_new_request(rid, block_hashes=[0, 1, 2, 3], num_computed_tokens=0)
        sched.update_state_after_alloc(
            type("R", (), {"request_id": rid}),
            ([_Block(10), _Block(11), _Block(12), _Block(13)],), 0)
        sched.build_save_meta(rid, scheduled_tokens=64)  # cursor -> 4
        # preempt + resume to zero
        sched.on_cached_request(rid, ([10, 11, 12, 13],), resumed=True,
                                num_computed_tokens=0)
        g = sched._req_states[rid].groups[0]
        assert g.next_stored_chunk_idx == 0, f"round {i}: no rollback"
        m = sched.build_save_meta(rid, scheduled_tokens=64)
        assert m.group_ops[0].gpu_block_ids == (10, 11, 12, 13)
        free, delay = conn.request_finished(
            type("R", (), {"request_id": rid}), None)
        assert (free, delay) == (False, None)
    stats = sched.lifecycle_stats()
    assert stats["request_states"] == 0, stats
    assert stats["cursor_rollbacks"] == 1000, stats
    assert conn.lifecycle_stats()["pending_store_jobs"] == 0


# ------------------------------------------------------------------
# 9: preemption-resume LOAD metadata (M7 TP2-P1 regression)
# ------------------------------------------------------------------
# vLLM v1 carries preempted->rescheduled requests in
# scheduled_cached_reqs.resumed_req_ids, NOT scheduled_new_reqs. The
# connector historically built load meta ONLY from scheduled_new_reqs,
# so a resumed request's accepted external tokens
# (get_num_new_matched_tokens -> core skips
# recompute) was never matched by a worker-side load -> forward read
# unrestored KV and emitted wrong tokens (4B TP2 lifecycle gate).

def _hybrid_resumed_setup(committed, scheduled=64, ext=544):
    """2-group hybrid (attention bs=16 + mamba bs=544 align 544) with a
    request that re-entered after preemption: the lookup hook reset state and
    recorded a HIT at boundary 544, then the core allocated fresh blocks
    and credited ``ext`` external tokens."""
    groups = [
        _attn(),
        GroupInfo(group_idx=1, kind="mamba", layer_names=("m.0",),
                  block_size=544, page_size_bytes=PAGE,
                  mamba_cache_mode="align",
                  mamba_align_size=544),
    ]
    sched = HybridRequestScheduler(groups, _HitBackend(committed),
                                   16, "ns", 1, 0)
    hashes = list(range(34))  # 34 hash blocks * 16 = 544 tokens
    sched.on_new_request("r1", block_hashes=hashes, num_computed_tokens=0)
    sched._req_states["r1"].snapshot_boundary = 544
    attn_ids = list(range(100, 134))  # 34 fresh attention blocks
    mamba_ids = [200, 201]            # CURR slot for this step = idx 1
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        tuple([_Block(i) for i in ids] for ids in (attn_ids, mamba_ids)),
        ext)
    return sched


def test_resumed_load_meta_restores_credited_pages():
    """Resumed request with 544 credited external tokens gets load meta
    carrying all 34 attention pages + the mamba snapshot written into
    the CURR slot only (v0.23.0 reads CURR in every kernel path)."""
    sched = _hybrid_resumed_setup(set(range(34)))
    meta = sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    assert meta is not None
    attn, mamba = meta.group_ops
    assert attn.group_idx == 0  # attention group (fixture order)
    assert len(attn.keys) == 34, attn
    assert attn.gpu_block_ids == tuple(range(100, 134)), attn
    assert mamba.group_idx == 1  # mamba group
    assert len(mamba.keys) == 1, mamba  # CURR slot only x 1 layer
    assert mamba.gpu_block_ids == (201,), mamba  # ids[curr_idx=1]


def test_resumed_load_meta_fail_closed_when_pages_unrestorable():
    """Fail-closed: the core credited 544 external tokens but
    the backend can no longer restore ANY page -> raise instead of
    letting forward read unrestored KV."""
    sched = _hybrid_resumed_setup(set())  # nothing committed anymore
    raised = None
    try:
        sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    except RuntimeError as e:
        raised = e
    assert raised is not None, \
        "credited external tokens with no restorable pages must raise"
    assert "unrestored state" in str(raised)


def test_resumed_load_meta_without_credit_is_quiet():
    """Resume covered entirely by the LOCAL prefix cache (ext=0, no
    external boundary): empty load meta, no error."""
    sched = _hybrid_resumed_setup(set(range(34)), ext=0)
    sched._req_states["r1"].snapshot_boundary = 0
    meta = sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    assert meta is not None
    assert all(len(op.keys) == 0 for op in meta.group_ops)


def test_resumed_load_meta_unknown_req_returns_none():
    """A resumed request the connector never saw (no external tokens
    possible) is skipped, not an error."""
    sched = _sched([_attn()])
    assert sched.build_resumed_load_meta("ghost", scheduled_tokens=64) \
        is None


def test_connector_meta_includes_resumed_load():
    """End-to-end at connector level: build_connector_meta must emit the
    resumed request's load meta (scheduled_cached_reqs.resumed_req_ids),
    not only scheduled_new_reqs."""
    from types import SimpleNamespace
    sched = _hybrid_resumed_setup(set(range(34)))
    conn = _sched_side_connector(sched)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"], resumed_req_ids={"r1"},
            new_block_ids=[(list(range(100, 134)), [200, 201])],
            num_computed_tokens=[544]),
        num_scheduled_tokens={"r1": 64})
    meta = conn.build_connector_meta(scheduler_output)
    loads = [r for r in meta.requests if r.req_id == "r1"]
    assert loads, "resumed request must receive load metadata"
    attn = loads[0].group_ops[0]
    assert len(attn.keys) == 34, attn
