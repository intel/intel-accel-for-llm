# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Piggybacked GDN loading (the replacement for the old layer-hook patch).

vLLM calls ``wait_for_layer_load`` only at ATTENTION layers, never at
GDN/mamba layers. Instead of patching vLLM, HybridWorker rides the
attention hooks:

- ``start_load`` submits every transfer, then host-blocks ONLY on the
  LEADING GDN segment (GDN layers that execute before the first
  attention layer, so no hook can cover them);
- ``wait_layer_load(attn_i)`` waits attn_i's own pages plus the GDN
  segment that executes AFTER attn_i and before the next attention
  layer -- those layers run after this hook returns, so waiting here is
  in time and their transfers overlapped the preceding compute.

Every GDN layer must be covered exactly once, and anything left
un-waited at the end of a step is a fail-stop: it would mean the
forward read unrestored state.

Pure logic: fake backend and canonicalizer, no GPU, no disk, no model.
"""

from __future__ import annotations

import pytest

from kvshrink.hybrid_metadata import (
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)
from kvshrink.hybrid_policy import LookupStatus
from kvshrink.hybrid_worker import HybridWorker

PAGE = 4096


def _group(g_idx, kind, layers, block_size=16):
    return GroupInfo(
        group_idx=g_idx, kind=kind, layer_names=tuple(layers),
        block_size=block_size, page_size_bytes=PAGE,
        mamba_cache_mode="align" if kind == "mamba" else None,
        mamba_align_size=block_size if kind == "mamba" else None)


class _FakeBackend:
    """Records submits and the ORDER in which tasks are waited."""

    def __init__(self, committed=True):
        self.submitted = []          # layer names, in submit order
        self.waited = []             # layer names, in wait order
        self.committed = committed

    def submit_group_loads(self, g_idx, views, indices, labels):
        tasks = {}
        for ln in views:
            self.submitted.append(ln)
            tasks[ln] = {"layer": ln}
        return tasks

    def wait_layer_loads(self, task):
        self.waited.append(task["layer"])

    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        return LookupStatus.HIT if self.committed else LookupStatus.MISS


class _FakeCanon:
    def register(self, kv_caches):
        pass

    def page_view_parts(self, layer_name):
        return {"page": layer_name}, 0


# Execution order: a leading GDN layer, then attention, more GDN, and a
# final attention layer with nothing after it.
ORDER = ["m0", "a1", "m2", "m3", "a4"]
ATTN = ["a1", "a4"]
GDN = ["m0", "m2", "m3"]


def _worker(backend=None, order=ORDER, gdn=None):
    """Worker whose groups match ``order`` unless ``gdn`` overrides the
    mamba membership (used to test an unplaceable GDN layer)."""
    attn = [ln for ln in order if ln in ATTN]
    groups = [_group(0, "attention", attn),
              _group(1, "mamba", gdn if gdn is not None
                     else [ln for ln in order if ln in GDN])]
    layer_infos = {ln: None for ln in order}
    w = HybridWorker(groups, layer_infos, 64, backend or _FakeBackend(),
                     _FakeCanon(), rank=0, tp_size=1)
    w.register({ln: None for ln in order}, order)
    return w


def _load_meta(layers, group_idx, boundary=None):
    """One load op covering ``layers`` for a single block."""
    keys = tuple(CacheKey(namespace="ns", tp_size=1, rank=0,
                          block_hash=7, group_idx=group_idx,
                          layer_name=ln) for ln in layers)
    op = GroupTransferMeta(group_idx=group_idx, keys=keys,
                           gpu_block_ids=(5,) * len(layers),
                           snapshot_boundary_tokens=boundary)
    return type("M", (), {"requests": [ReqMeta(req_id="r1",
                                               group_ops=(op,))]})


# ------------------------------------------------------------------
# map construction
# ------------------------------------------------------------------

def test_leading_segment_and_trailing_segments():
    w = _worker()
    assert w._leading_gdn == ("m0",)
    assert w._piggyback_map == {"a1": ("m2", "m3"), "a4": ()}


def test_every_gdn_layer_is_covered_exactly_once():
    w = _worker()
    covered = list(w._leading_gdn) + [ln for seg in
                                      w._piggyback_map.values() for ln in seg]
    assert sorted(covered) == sorted(GDN)
    assert len(covered) == len(set(covered))


def test_no_leading_segment_when_attention_runs_first():
    w = _worker(order=["a1", "m2", "m3", "a4"])
    assert w._leading_gdn == ()
    assert w._piggyback_map == {"a1": ("m2", "m3"), "a4": ()}


def test_gdn_layer_missing_from_execution_order_fails_closed():
    """A GDN layer we never place could never be waited for."""
    with pytest.raises(RuntimeError, match="missing from the execution order"):
        _worker(order=["m0", "a1", "m2", "a4"], gdn=GDN)  # m3 unplaced


def test_model_without_attention_layers_fails_closed():
    """With no attention hook there is nothing to ride on."""
    groups = [_group(0, "attention", []), _group(1, "mamba", GDN)]
    w = HybridWorker(groups, {ln: None for ln in GDN}, 64, _FakeBackend(),
                     _FakeCanon(), rank=0, tp_size=1)
    with pytest.raises(RuntimeError, match="no attention layers"):
        w.register({ln: None for ln in GDN}, GDN)


# ------------------------------------------------------------------
# load scheduling
# ------------------------------------------------------------------

def test_start_load_waits_only_the_leading_segment():
    be = _FakeBackend()
    w = _worker(be)
    w.start_load(_load_meta(GDN, 1, boundary=16))
    assert sorted(be.submitted) == sorted(GDN), be.submitted
    # only the leading GDN layer is host-blocked before forward
    assert be.waited == ["m0"], be.waited
    # the rest stay pending for their piggyback hooks
    assert sorted(w._load_tasks) == ["m2", "m3"]


def test_attention_hook_waits_its_own_pages_and_trailing_gdn():
    be = _FakeBackend()
    w = _worker(be)
    meta = _load_meta(ATTN, 0)
    meta.requests.append(_load_meta(GDN, 1, boundary=16).requests[0])
    w.start_load(meta)
    assert be.waited == ["m0"]

    w.wait_layer_load("a1")
    # a1's own page plus the GDN segment that runs before a4
    assert sorted(be.waited[1:]) == ["a1", "m2", "m3"], be.waited

    w.wait_layer_load("a4")
    assert be.waited[-1] == "a4"
    w.loads_drained_check()  # nothing left un-waited


def test_unwaited_layer_at_step_end_fails_stop():
    be = _FakeBackend()
    w = _worker(be)
    w.start_load(_load_meta(GDN, 1, boundary=16))
    # a1's hook never fired -> m2/m3 were never restored
    with pytest.raises(RuntimeError, match="never ran"):
        w.loads_drained_check()


def test_stale_residue_from_previous_step_fails_stop():
    be = _FakeBackend()
    w = _worker(be)
    w._load_tasks = {"m2": [{"layer": "m2"}]}
    with pytest.raises(RuntimeError, match="stale step residue"):
        w.start_load(_load_meta(GDN, 1, boundary=16))


def test_load_poison_is_sticky_across_hooks():
    """A failed load must fail every later hook of the step, never
    degrade into a silent recompute."""
    be = _FakeBackend()

    def _boom(task):
        raise RuntimeError("h2d failed")

    be.wait_layer_loads = _boom
    w = _worker(be)
    with pytest.raises(RuntimeError, match="h2d failed"):
        w.start_load(_load_meta(GDN, 1, boundary=16))
    for call in (lambda: w.wait_layer_load("a1"),
                 lambda: w.raise_load_poison(),
                 lambda: w.start_load(_load_meta(GDN, 1, boundary=16))):
        with pytest.raises(RuntimeError, match="h2d failed"):
            call()


def test_mamba_toctou_change_fails_stop():
    """The committed boundary must still match the scheduler's HIT when
    the worker executes; otherwise the state we would restore is not the
    state the core credited."""
    be = _FakeBackend(committed=False)
    w = _worker(be)
    with pytest.raises(RuntimeError, match="TOCTOU"):
        w.start_load(_load_meta(GDN, 1, boundary=16))
    assert be.submitted == [], "no transfer may be submitted after TOCTOU"


def test_attention_load_needs_no_boundary_check():
    """Attention pages are per-block and content-addressed: they carry no
    snapshot boundary and are submitted without the mamba TOCTOU gate."""
    be = _FakeBackend(committed=False)  # would fail a boundary check
    w = _worker(be)
    w.start_load(_load_meta(ATTN, 0))
    assert sorted(be.submitted) == sorted(ATTN)
