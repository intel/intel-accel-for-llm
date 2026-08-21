"""TP partial-commit guard in boundary_backend.lookup_boundary.

Under TP>1 each rank commits its own shard independently (no cross-rank
transaction). A boundary visible on the scheduler's own rank but missing
on any other rank must look up as MISS, so the request recomputes and
its save re-commits every rank's shard (hit-path heal). These tests
exercise the guard with stubbed per-rank backends -- no GPU, no disk.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from kvshrink import hybrid_backend
from kvshrink.hybrid_backend import KVShrinkHybridBackendAdapter
from kvshrink.hybrid_metadata import CacheKey
from kvshrink.hybrid_policy import LookupStatus


def _key():
    return CacheKey(namespace="ns", tp_size=2, rank=0,
                    block_hash=12345, group_idx=0, layer_name="")


@dataclass(frozen=True)
class _StubBoundary:
    """Stands in for iaxl's CacheBoundary so the guard is testable
    without the compiled extension (the adapter only needs a
    dataclass it can ``replace(rank=...)`` and log)."""
    namespace: str
    tp_size: int
    rank: int
    block_hash: object
    group_idx: int

    @property
    def hash_str(self) -> str:
        return str(self.block_hash)


@pytest.fixture(autouse=True)
def _stub_cache_boundary(monkeypatch):
    monkeypatch.setattr(
        hybrid_backend, "_cache_boundary_from_key",
        lambda key: _StubBoundary(
            namespace=key.namespace, tp_size=key.tp_size, rank=key.rank,
            block_hash=key.block_hash, group_idx=key.group_idx))


def _adapter(rank0_hit=True, rank1_hit=True, tp_size=2):
    """Adapter with stubbed backends; never touches __init__ I/O."""
    a = object.__new__(KVShrinkHybridBackendAdapter)
    a._tp_size = tp_size
    a._own_rank = 0
    a._backend = SimpleNamespace(
        is_committed=lambda *a_, **k: rank0_hit)
    a._rank_backends = {1: SimpleNamespace(
        is_committed=lambda *a_, **k: rank1_hit)}
    return a


def test_all_ranks_present_hit():
    a = _adapter(rank0_hit=True, rank1_hit=True)
    assert a.lookup_boundary(_key()) == LookupStatus.HIT


def test_other_rank_missing_is_miss():
    """Partial commit: rank 0 committed, rank 1 did not -> MISS."""
    a = _adapter(rank0_hit=True, rank1_hit=False)
    assert a.lookup_boundary(_key()) == LookupStatus.MISS


def test_own_rank_missing_is_miss():
    a = _adapter(rank0_hit=False, rank1_hit=True)
    assert a.lookup_boundary(_key()) == LookupStatus.MISS


def test_single_rank_skips_cross_rank_check():
    """tp_size=1: no cross-rank validation, own rank decides."""
    a = _adapter(rank0_hit=True, rank1_hit=False, tp_size=1)
    assert a.lookup_boundary(_key()) == LookupStatus.HIT


def test_backend_error_fails_closed_to_miss():
    def _boom(*a_, **k):
        raise RuntimeError("record unavailable")
    a = _adapter()
    a._backend = SimpleNamespace(is_committed=_boom)
    assert a.lookup_boundary(_key()) == LookupStatus.MISS
