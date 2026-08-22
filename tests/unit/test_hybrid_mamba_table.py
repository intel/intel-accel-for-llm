"""Mamba block-table shape handling in build_save_meta/build_load_meta.

Regression for super-master gate (devlog §5.10): block tables vary by
token count -- single-element [X] (545-token req), null-prefixed
[0,0,X], or [null, X]. Block 0 is the reserved null block (M0 verified).
Old code assumed len(ids) > 1 and silently skipped mamba snapshots for
single-element tables; the fix picks the last NON-NULL block.

These tests import the real HybridRequestScheduler (needs vllm env,
so they run in the container test runner).
"""

from __future__ import annotations

from kvshrink.hybrid_metadata import GroupInfo
from kvshrink.hybrid_scheduler import HybridRequestScheduler
from kvshrink.hybrid_policy import LookupStatus


class _Backend:
    """BoundaryBackend stub: hash value N is HIT iff N in committed.
    Pages live in `pages` {(hash, layer): bytes}; sizes default 1024.
    Checksums computed on demand.

    committed_pairs (optional): set of (group_idx, hash) for group-aware
    manifests. Real backends commit a MAMBA group's manifest ONLY at the
    final progress boundary (debug_save stores hash[progress//bs-1]), so
    mixed-group tests must use pairs that reflect that -- a bare
    `committed` set would pretend mamba snapshots exist at every block.
    """

    PAGE_SIZE = 1024

    def __init__(self, committed, pages=None, committed_pairs=None):
        self.committed = committed
        self.committed_pairs = committed_pairs
        self.pages = dict(pages) if pages else {}

    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        if self.committed_pairs is not None:
            hit = (key.group_idx, key.block_hash) in self.committed_pairs
        else:
            hit = key.block_hash in self.committed
        if not hit:
            return LookupStatus.MISS
        # Model the chunk tier's Record-gated LIVE presence: a committed
        # manifest whose pages are missing/corrupt is a MISS at lookup
        # (the real engine checks _chunks_present per chunk). Tests that
        # never populate page bookkeeping are presence-vacuous.
        if self.pages or getattr(self, "_manifest_checksums", None):
            import hashlib
            mc = getattr(self, "_manifest_checksums", {}) or {}
            for ln in (expected_layers or ()):
                data = self.pages.get((key.block_hash, ln))
                if data is None:
                    return LookupStatus.MISS
                base = mc.get((key.block_hash, ln))
                if base is not None and hashlib.sha256(
                        data).hexdigest() != base:
                    return LookupStatus.MISS
        return LookupStatus.HIT


    def _all_pages_present(self, groups, hashes):
        """Populate pages for all (hash, layer) pairs of the groups."""
        for g in groups:
            for i in hashes:
                for ln in g.layer_names:
                    self.pages.setdefault((i, ln), b"x" * self.PAGE_SIZE)
        return self


def _group(g_idx, kind, block_size, align=None):
    return GroupInfo(
        group_idx=g_idx, kind=kind,
        layer_names=(f"l{g_idx}.0", f"l{g_idx}.1"),
        block_size=block_size, page_size_bytes=1024,
        mamba_cache_mode="align" if kind == "mamba" else None,
        mamba_align_size=align)


def _hybrid_pairs(attn_hashes, mamba_hashes, attn_g=0, mamba_g=1):
    """Group-aware committed pairs for a 2-group hybrid model. The mamba
    group commits ONLY the final progress boundary (debug_save stores a
    single snapshot), so mamba_hashes is normally just [final_hash]."""
    return ({(attn_g, h) for h in attn_hashes}
            | {(mamba_g, h) for h in mamba_hashes})


class _Block:
    def __init__(self, block_id):
        self.block_id = block_id


def _make(groups, committed, block_ids_per_group):
    sched = HybridRequestScheduler(groups, _Backend(committed), 16,
                                   "ns", 1, 0)
    sched.on_new_request("r1", block_hashes=[0, 1],
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),  # request-like
        tuple([_Block(i) for i in ids] for ids in block_ids_per_group), 0)
    return sched


def test_save_meta_single_element_table():
    """545-token request: mamba table [X] (no null prefix). Snapshot at
    the 544 boundary must be saved (regression: old len(ids)>1 skip)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5]])  # ids=[5] only
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    op = meta.group_ops[0]
    # progress = 0 + 544 = 544 -> boundary -> idx=0 -> hash0 committed
    assert len(op.keys) == 2, op  # 2 layers
    assert op.gpu_block_ids == (5, 5), op.gpu_block_ids


def test_save_meta_null_prefixed_table():
    """Null-prefixed [0,0,X]: last non-null block is used."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 2
    assert op.gpu_block_ids == (7, 7), op.gpu_block_ids


def test_save_meta_skips_partial_tail():
    """progress 1088+472 = 1560 (not a boundary) -> no mamba snapshot."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0, 1}, [[0, 0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=1560)
    op = meta.group_ops[0]
    assert len(op.keys) == 0, op  # partial tail never saved


def test_save_meta_multi_block_boundary():
    """progress 1088 = 2 complete blocks -> hash idx=1 (hash1)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=1088)
    op = meta.group_ops[0]
    assert len(op.keys) == 2
    assert op.gpu_block_ids == (7, 7)


def test_load_meta_curr_slot_is_last_scheduled_block():
    """The snapshot lands in the block the GDN kernel actually reads.

    v0.23.0 mamba_get_block_table_tensor (align mode) gathers
    block_table[(seq_len - 1) // block_size] and the kernel uses column
    0 of that gather, where seq_len = computed + scheduled. Restoring
    boundary 544 with 544 scheduled tokens therefore targets table index
    (544 + 544 - 1) // 544 = 1 -> block 6."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5, 6]])
    sched._req_states["r1"].snapshot_boundary = 544
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1088,
                       "block_ids": ([5, 6],)}),
        scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op  # CURR slot x 2 layers
    assert op.gpu_block_ids == (6, 6), op.gpu_block_ids


def test_load_meta_chunk_tail_table_index():
    """Real shape [0,1,6], boundary 1088, sched 472 (1560-token prompt):
    (1088 + 472 - 1) // 544 = 2 -> block 6, the only slot the kernel
    reads this step."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 6]])
    sched._req_states["r1"].snapshot_boundary = 1088
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1560,
                       "block_ids": ([0, 1, 6],)}),
        scheduled_tokens=472)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op
    assert op.gpu_block_ids == (6, 6), op.gpu_block_ids


def test_load_meta_null_prefixed_table():
    """Null-prefixed table [0, 7, 9]: boundary 1088 + 544 scheduled ->
    index 2 -> block 9. Leading nulls are reserved placeholders, never
    load targets."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 7, 9]])
    sched._req_states["r1"].snapshot_boundary = 1088
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1632,
                       "block_ids": ([0, 7, 9],)}),
        scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op
    assert op.gpu_block_ids == (9, 9), op.gpu_block_ids


def test_load_meta_decode_tail():
    """Decode tail (sched == 1, e.g. a 1089-token prompt with boundary
    1088): (1088 + 1 - 1) // 544 = 2 -> block 6."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 6]])
    sched._req_states["r1"].snapshot_boundary = 1088
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1089,
                       "block_ids": ([0, 1, 6],)}),
        scheduled_tokens=1)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op  # CURR slot x 2 layers
    assert op.gpu_block_ids == (6, 6), op.gpu_block_ids


def test_load_meta_curr_null_fail_stop():
    """Chunk path (sched >= 2) with a null CURR slot -> FAIL-STOP.

    There is no second slot to fall back on: the kernel reads exactly
    the gathered column, so a null there means the state cannot be
    restored at all."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 0]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1090,
                           "block_ids": ([0, 1, 0],)}),
            scheduled_tokens=2)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "null curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_curr_null_decode_fail_stop():
    """Decode tail (sched == 1) with a null CURR slot -> FAIL-STOP:
    never enter forward with unrestored state."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 0]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1089,
                           "block_ids": ([0, 1, 0],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "decode tail with null curr must raise"
    assert "unrestored state" in str(raised)


def test_load_meta_curr_out_of_range_fail_stop():
    """Chunk path (sched >= 2): the gathered index is beyond the table
    -> FAIL-STOP (the block the kernel will read was never allocated)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1090,
                           "block_ids": ([0, 1],)}),
            scheduled_tokens=2)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "out-of-range curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_curr_out_of_range_decode_fail_stop():
    """Decode tail (sched == 1): gathered index beyond table -> FAIL-STOP."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1089,
                           "block_ids": ([0, 1],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "decode tail with out-of-range curr" \
        " must raise"
    assert "unrestored state" in str(raised)


def test_load_meta_table_idx_null_fail_closed():
    """Gathered table index resolves to a null block -> FAIL-STOP:
    get_num_new_matched_tokens already credited the external boundary,
    so proceeding would enter forward with unrestored state."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0, 6]])
    sched._req_states["r1"].snapshot_boundary = 544  # idx (544+1-1)//544 = 1
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 545,
                           "block_ids": ([0, 0, 6],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "null curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_table_idx_out_of_range_fail_closed():
    """Gathered table index beyond the table length -> FAIL-STOP."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = HybridRequestScheduler(groups, _Backend({2}), 16, "ns", 1, 0)
    sched.on_new_request("r1", block_hashes=[0, 1, 2],
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        ((_Block(5),),), 0)
    sched._req_states["r1"].snapshot_boundary = 1632  # idx 1631//544 = 2
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1633,
                           "block_ids": ([5],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "out-of-range curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_fail_closed_without_boundary():
    """No snapshot_boundary recorded -> fail closed (0 keys), never guess
    by recomputing."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5]])
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 545,
                       "block_ids": ([5],)}))
    op = meta.group_ops[0]
    assert len(op.keys) == 0, op  # boundary=0 -> idx=-1 -> no load


def test_save_meta_all_null_table_skipped():
    """All-null table [0,0] -> no mamba save keys."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0]])
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 0, op


def test_load_meta_all_null_table_fail_closed():
    """All-null table with boundary > 0 -> FAIL-STOP (curr slot null)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0]])
    sched._req_states["r1"].snapshot_boundary = 544
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 545,
                           "block_ids": ([0, 0],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "all-null table must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_hit_sched_zero_fail_stop():
    """External HIT with scheduled_tokens=0 -> FAIL-STOP (super-master
    gate §5.33: production hits must schedule >= 1 token; sched=0 is a
    test-only path that must never drive a real mamba load)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5, 9]])
    sched._req_states["r1"].snapshot_boundary = 544
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 545,
                           "block_ids": ([5, 9],)}))
    except RuntimeError as e:
        raised = e
    assert raised is not None, "sched=0 external HIT must raise"
    assert "scheduled_tokens=0" in str(raised)


def test_gnnmt_completeness_intact_boundary_unchanged():
    """Lookup: all pages present -> boundary unchanged (1088)."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    backend = _Backend(set(), committed_pairs=_hybrid_pairs([0, 1], [1])
                       )._all_pages_present(groups, [0, 1])
    sched = HybridRequestScheduler(groups, backend, 16, "ns", 1, 0)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1], "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 1088, ext
    assert sched._req_states["r1"].snapshot_boundary == 1088


def test_gnnmt_mamba_align_granularity():
    """mamba_align_size > block_size (32 vs 16): final boundary 96 with
    a missing attention hash2 page -> mamba snapshot only at 96 -> full
    MISS (no intermediate boundary usable)."""
    groups = [
        _group(0, "attention", 16),
        _group(1, "mamba", 16, align=32),
    ]
    backend = _Backend(
        set(),
        committed_pairs=(_hybrid_pairs([0, 1, 2, 3, 4, 5], [5]))
    )._all_pages_present(groups, [0, 1, 2, 3, 4, 5])
    del backend.pages[(2, "l0.0")]
    del backend.pages[(2, "l1.0")]
    sched = HybridRequestScheduler(groups, backend, 16, "ns", 1, 0)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1, 2, 3, 4, 5],
        "num_tokens": 96})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 0, ext  # mamba snapshot unusable below 96 -> MISS


def test_gnnmt_mamba_partial_recovery_with_earlier_snapshot():
    """If an EARLIER mamba snapshot is committed and intact (future M3
    multi-snapshot save), partial recovery is legal: hash1 attention
    page missing -> 1088 unusable, but 544 has a complete mamba
    snapshot -> restore 544 (super-master gate, devlog §5.21)."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    backend = _Backend(
        set(), committed_pairs=_hybrid_pairs([0, 1], [0, 1])
    )._all_pages_present(groups, [0, 1])
    del backend.pages[(1, "l0.1")]  # hash1 attention page missing
    sched = HybridRequestScheduler(groups, backend, 16, "ns", 1, 0)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1], "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 544, ext  # earlier intact mamba snapshot -> recover


def test_gnnmt_hetero_attention_shrinks_by_lcm():
    """Two attention groups bs=16/32 (LCM=32), no mamba. Policy hits 96
    (all manifests committed) but g1's hash2 page is missing -> the
    completeness check walks LCM-aligned candidates: 96 incomplete ->
    64 complete (g0 needs hash0..3, g1 needs hash0..1) -> external 64.
    Guards the align_down start + per-group manifest validation
    (super-master gate, devlog §5.21)."""
    groups = [_group(0, "attention", 16), _group(1, "attention", 32)]
    pairs = {(0, i) for i in range(6)} | {(1, i) for i in range(3)}
    backend = _Backend(set(), committed_pairs=pairs)
    import hashlib
    backend._manifest_checksums = {}
    for i in range(6):  # g0 pages hash0..5
        for ln in ("l0.0", "l0.1"):
            data = b"x" * _Backend.PAGE_SIZE
            backend.pages[(i, ln)] = data
            backend._manifest_checksums[(i, ln)] = hashlib.sha256(
                data).hexdigest()
    for i in range(2):  # g1 pages hash0..1; hash2 deliberately absent
        for ln in ("l1.0", "l1.1"):
            data = b"x" * _Backend.PAGE_SIZE
            backend.pages[(i, ln)] = data
            backend._manifest_checksums[(i, ln)] = hashlib.sha256(
                data).hexdigest()
    sched = HybridRequestScheduler(groups, backend, 16, "ns", 1, 0)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1, 2, 3, 4, 5],
        "num_tokens": 96})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 64, ext


def test_partial_recovery_load_meta_targets_earlier_snapshot():
    """Scheduler-level chain: hash1 attention page missing + mamba hash0
    snapshot committed -> lookup recovers 544 -> build_load_meta targets
    the mamba hash0 pages in the slot the kernel reads this step
    ((544 + 544 - 1) // 544 = 1), attention loads only hash0's pages."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    backend = _Backend(
        set(), committed_pairs=_hybrid_pairs([0, 1], [0, 1])
    )._all_pages_present(groups, [0, 1])
    del backend.pages[(1, "l0.1")]  # 1088 attention incomplete
    sched = HybridRequestScheduler(groups, backend, 16, "ns", 1, 0)
    req = type("R", (), {"request_id": "r1", "block_hashes": [0, 1],
                         "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 544, ext
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        tuple([_Block(i) for i in ids] for ids in ([3, 4], [5, 8])), 544)
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1088,
                       "block_ids": ([3, 4], [5, 8])}),
        scheduled_tokens=544)
    attn_op = meta.group_ops[0]
    mamba_op = meta.group_ops[1]
    # attention: boundary 544 -> 1 hash (hash0) x 2 layers, gpu block 3
    assert len(attn_op.keys) == 2, attn_op
    assert all(k.block_hash == 0 for k in attn_op.keys)
    assert set(attn_op.gpu_block_ids) == {3}
    # mamba: snapshot at hash0 -> curr table_idx 1 -> block 8
    assert len(mamba_op.keys) == 2, mamba_op
    assert all(k.block_hash == 0 for k in mamba_op.keys)
    assert set(mamba_op.gpu_block_ids) == {8}
    assert mamba_op.snapshot_boundary_tokens == 544


def test_gnnmt_real_528_544_hetero_attention():
    """Real 528/544 attention pair: LCM = 17952, so the only legal common
    boundaries are 0 and 17952. A missing page at the top boundary
    degrades to full MISS (no intermediate boundary exists); proves the
    align_down start + LCM step on the production block sizes
    (super-master gate, devlog §5.23)."""
    import hashlib
    groups = [_group(0, "attention", 528), _group(1, "attention", 544)]
    n0, n1 = 17952 // 528, 17952 // 544  # 34, 33
    pairs = {(0, i) for i in range(n0)} | {(1, i) for i in range(n1)}
    backend = _Backend(set(), committed_pairs=pairs)
    backend._manifest_checksums = {}
    for g_idx, n in ((0, n0), (1, n1)):
        for i in range(n):
            for ln in (f"l{g_idx}.0", f"l{g_idx}.1"):
                data = b"x" * _Backend.PAGE_SIZE
                backend.pages[(i, ln)] = data
                backend._manifest_checksums[(i, ln)] = hashlib.sha256(
                    data).hexdigest()
    # one layer page of g1's LAST hash (the 17952-boundary block) missing
    del backend.pages[(n1 - 1, "l1.0")]
    sched = HybridRequestScheduler(groups, backend, 16, "ns", 1, 0)
    req = type("R", (), {"request_id": "r1",
                         "block_hashes": list(range(max(n0, n1))),
                         "num_tokens": 17952})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 0, ext


# ----------------------------------------------------------------------
