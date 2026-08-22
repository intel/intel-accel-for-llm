"""Pipelined attention save via save_kv_layer.

vLLM calls save_kv_layer on exit of every attention layer during
forward. HybridWorker submits that layer's async put immediately
(overlapping the remaining layers' compute); wait_save then only
waits/harvests/commits. GDN groups always save in wait_save (their
state is final only post-forward). These tests use a fake backend and
fake canonicalizer -- no GPU, no disk, no model.
"""

import os
from types import SimpleNamespace

from kvshrink.hybrid_worker import HybridWorker
from kvshrink.hybrid_metadata import (
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)


def _group(g_idx, kind, layers):
    return GroupInfo(group_idx=g_idx, kind=kind,
                     layer_names=tuple(layers), block_size=16,
                     page_size_bytes=1024,
                     mamba_cache_mode=None, mamba_align_size=None)


def _key(layer_name, blk_hash=777, g_idx=0):
    return CacheKey(namespace="ns", tp_size=1, rank=0,
                    block_hash=blk_hash, group_idx=g_idx,
                    layer_name=layer_name)


class _FakeBackend:
    """Records submit/wait/commit calls; checksums are fabricated."""

    def __init__(self):
        self.submits = []   # (g_idx, sorted(layers), labels)
        self.waits = 0
        self.commits = []   # (key, expected_layers)
        self.persisted = 0
        self.evicted = 0

    def submit_group_stores(self, g_idx, views, indices, labels):
        self.submits.append((g_idx, sorted(views), list(labels)))
        return {ln: {"layer": ln, "labels": list(labels)}
                for ln in views}, None

    def wait_group_stores(self, tasks):
        self.waits += 1
        return {ln: [f"ck-{h}" for h in td["labels"]]
                for ln, td in tasks.items()}

    def commit_chunks(self, key, expected_layers, checksums,
                      chunk_labels, expected_boundary_tokens=None):
        self.commits.append((key, sorted(expected_layers),
                             dict(checksums)))
        return True

    def persist_engine(self, max_count):
        self.persisted += 1
        return {"persisted": 0}

    def evict_over_watermark(self, *a, **k):
        self.evicted += 1
        return {}


class _FakeCanon:
    def page_view_parts(self, layer_name):
        return [layer_name], 0


def _save_meta():
    """One attention boundary (2 layers) + one mamba boundary."""
    attn_ops = GroupTransferMeta(
        group_idx=0,
        keys=tuple(_key(ln) for ln in ("a0", "a1")),
        gpu_block_ids=(10, 10))
    mamba_ops = GroupTransferMeta(
        group_idx=1,
        keys=(_key("m0", blk_hash=888, g_idx=1),),
        gpu_block_ids=(20,),
        snapshot_boundary_tokens=544)
    return SimpleNamespace(save_requests=[ReqMeta(
        req_id="r1", group_ops=(attn_ops, mamba_ops))])


def _worker():
    groups = [_group(0, "attention", ["a0", "a1"]),
              _group(1, "mamba", ["m0"])]
    w = HybridWorker(groups, {"a0": None, "a1": None, "m0": None},
                     num_blocks=64, backend=_FakeBackend(),
                     canonicalizer=_FakeCanon(), rank=0, tp_size=1)
    w._kv_caches_ref = object()  # truthy: kv caches registered
    return w


def _env_off(monkeypatch_env=None):
    os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)
    os.environ.pop("KVSHRINK_SAVE", None)
    os.environ.pop("KVSHRINK_DEBUG_AUTOSAVE", None)


def test_pipelined_attention_submits_during_forward():
    _env_off()
    c = _worker()
    # forward: vLLM calls save_kv_layer on exit of each attention layer
    c.save_kv_layer("a0", _save_meta())
    c.save_kv_layer("a1", _save_meta())
    submits_during_fwd = list(c._backend.submits)
    assert len(submits_during_fwd) == 2
    assert submits_during_fwd[0][1] == ["a0"]  # one layer per call
    assert submits_during_fwd[1][1] == ["a1"]

    c.wait_save(_save_meta())
    # attention layers were NOT re-submitted; mamba submitted at wait
    submit_layers = [sorted(v) for _g, v, _l in c._backend.submits]
    assert ["a0", "a1"] not in submit_layers  # no bulk re-submit
    assert ["m0"] in submit_layers
    # both boundaries committed with checksums from all their layers
    by_group = {k.group_idx: (exp, ck) for k, exp, ck in
                c._backend.commits}
    assert sorted(by_group[0][0]) == ["a0", "a1"]
    assert by_group[0][1]["a0"] == "ck-777"
    assert by_group[0][1]["a1"] == "ck-777"
    assert by_group[1][0] == ["m0"]
    assert c._backend.persisted >= 1 and c._backend.evicted == 1


def test_fallback_when_hook_never_fired():
    """Older vLLM / decorator missing: attention submits at wait time,
    commits still correct (idempotent full coverage)."""
    _env_off()
    c = _worker()
    c.wait_save(_save_meta())  # no save_kv_layer calls beforehand
    submit_layers = [sorted(v) for _g, v, _l in c._backend.submits]
    assert ["a0"] in submit_layers and ["a1"] in submit_layers
    assert ["m0"] in submit_layers
    assert len(c._backend.commits) == 2


def test_pipelined_disabled_by_env():
    _env_off()
    os.environ["KVSHRINK_SAVE_PIPELINED"] = "0"
    try:
        c = _worker()
        c.save_kv_layer("a0", _save_meta())
        assert c._backend.submits == []  # nothing during forward
        c.wait_save(_save_meta())
        assert len(c._backend.commits) == 2
    finally:
        os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)


def test_save_kv_layer_ignores_mamba_and_unknown_layers():
    _env_off()
    c = _worker()
    c.save_kv_layer("m0", _save_meta())       # mamba layer: never served
    c.save_kv_layer("no.such.layer", _save_meta())
    assert c._backend.submits == []
    c.wait_save(_save_meta())
    assert len(c._backend.commits) == 2
