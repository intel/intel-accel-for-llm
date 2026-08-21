"""Cache hit policy for hybrid (Full Attention + GDN) models.

Implements the hit-detection algorithm, verified against vLLM v0.21.0's
HybridKVCacheCoordinator semantics:

- Attention groups: left-to-right prefix scan; the prefix must exist
  contiguously (downward-closed).
- Mamba/GDN groups: right-to-left scan for the NEAREST committed snapshot;
  earlier snapshots need not exist. Candidates are aligned down to
  mamba_align_size and the final boundary recomputes exactly 1 token.
- Multiple groups converge via fixed-point iteration (full attention
  first, then mamba groups).
- Backend lookups return HIT / MISS (chunk tier is Record-gated,
  request, never allocates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hybrid_metadata import CacheKey, GroupInfo, make_boundary_key


class LookupStatus(Enum):
    """Two-valued lookup verdict: HIT restores from external cache,
    MISS recomputes (the fail-closed default)."""
    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class LookupResult:
    """Frozen policy output: the verdict and the number of boundary
    tokens restorable from the external cache."""
    status: LookupStatus
    boundary_tokens: int = 0  # tokens that can be restored from external cache


def align_down(tokens: int, align: int) -> int:
    """Largest multiple of ``align`` not exceeding ``tokens``. Snaps
    mamba candidates to align boundaries: GDN state is only addressable
    on aligned positions."""
    return (tokens // align) * align


class HybridHitPolicy:
    """Fixed-point multi-group hit detection (pure function, testable)."""

    def __init__(
        self,
        groups: list[GroupInfo],
        backend,
        hash_block_size: int,
        num_computed_tokens: int,
        namespace: str,
        tp_size: int,
        rank: int,
    ):
        """Configure the policy for one request: groups, backend, the
        request's computed tokens and its namespace/tp/rank identity.
        Orders groups attention-first (tighter initial bound) and takes
        the global mamba alignment as the minimum across mamba groups."""
        self._groups = groups
        self._backend = backend
        self._hash_block_size = hash_block_size
        self._num_computed = num_computed_tokens
        self._namespace = namespace
        self._tp_size = tp_size
        self._rank = rank
        # full attention first (tighter initial bound)
        self._ordered = sorted(
            groups, key=lambda g: 0 if g.kind == "attention" else 1)
        self._mamba_align = None
        for g in groups:
            if g.kind == "mamba":
                a = g.mamba_align_size
                self._mamba_align = a if self._mamba_align is None \
                    else min(self._mamba_align, a)

    # ------------------------------------------------------------------
    def _boundary_key(self, group: GroupInfo, block_hash: int) -> CacheKey:
        """Boundary-manifest CacheKey (layer_name="") for one group at
        one block hash; namespace/tp/rank/group are part of the identity
        so the same hash can never alias across groups or ranks."""
        return make_boundary_key(self._namespace, self._tp_size,
                                 self._rank, group.group_idx, block_hash)

    @staticmethod
    def _hash_granularity(group: GroupInfo) -> int:
        """vLLM computes request.block_hashes at BLOCK size granularity
        (one hash per complete block). The connector must use the same
        granularity; hash_block_size (16) is unrelated to block hashes.
        """
        return group.block_size

    def _lookup_attention(self, group: GroupInfo, block_hashes,
                          candidate: int) -> tuple[LookupStatus, int]:
        """Left-to-right prefix scan. Downward closed."""
        gran = self._hash_granularity(group)
        num_hash_blocks = candidate // gran
        hit = 0
        for i in range(num_hash_blocks):
            if i >= len(block_hashes):
                break
            key = self._boundary_key(group, block_hashes[i])
            st = self._backend.lookup_boundary(key, group.layer_names)
            if st != LookupStatus.HIT:
                break
            hit = (i + 1) * gran
        return LookupStatus.HIT, hit

    def _lookup_mamba(self, group: GroupInfo, block_hashes,
                      candidate: int) -> tuple[LookupStatus, int]:
        """Right-to-left scan for the nearest committed snapshot.

        The candidate is first aligned down to mamba_align_size; then we
        walk hashes backwards from that boundary to find the closest
        committed snapshot (no requirement of contiguous prefixes).
        mamba_cache_mode == "none" never hits.
        """
        if group.mamba_cache_mode == "none":
            return LookupStatus.MISS, 0
        align = group.mamba_align_size or group.block_size
        aligned = align_down(candidate, align)
        gran = self._hash_granularity(group)
        # Scan right-to-left for the nearest committed snapshot AT or
        # BELOW the aligned candidate. hash[i] covers tokens
        # [i*gran, (i+1)*gran); the snapshot AT `aligned` tokens lives at
        # hash[aligned//gran - 1]. Mirroring upstream MambaManager
        # (single_type_kv_cache_manager.py find_longest_cache_hit), a
        # snapshot boundary must ALSO be a multiple of alignment_tokens:
        #   if (i+1)*gran % align != 0: continue
        # Without this, a non-aligned snapshot could shrink the candidate
        # below an align boundary and destabilize the fixed-point loop.
        max_idx = aligned // gran - 1 if aligned >= gran else -1
        for i in range(max_idx, -1, -1):
            if i >= len(block_hashes):
                continue
            key = self._boundary_key(group, block_hashes[i])
            st = self._backend.lookup_boundary(key, group.layer_names,
                                               (i + 1) * gran)
            if st != LookupStatus.HIT:
                continue
            if (i + 1) * gran % align != 0:
                continue  # snapshot not on an alignment boundary
            return LookupStatus.HIT, (i + 1) * gran
        return LookupStatus.MISS, 0

    # ------------------------------------------------------------------
    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> tuple[LookupResult, dict]:
        """Fixed-point convergence over all groups.

        Returns (LookupResult, trace) where trace records per-group
        decisions for logging / differential testing.
        """
        candidate = max_length
        if self._mamba_align is not None:
            # the last prompt token is always recomputed (logprobs + state)
            candidate = min(candidate - 1,
                            align_down(candidate - 1, self._mamba_align))
        trace = {"iterations": [], "final": candidate}

        while True:
            changed = False
            iteration = {}
            for group in self._ordered:
                kind = group.kind
                # Every attention group is looked up against its own
                # manifest: trimming a
                # later attention group without validating its manifest
                # would let a boundary hit on group A while group B's
                # pages were never committed.
                st, hit = (
                    self._lookup_attention(group, block_hashes, candidate)
                    if kind == "attention"
                    else self._lookup_mamba(group, block_hashes, candidate))
                iteration[group.group_idx] = {
                    "kind": kind, "status": st.value, "hit": hit}
                if hit < candidate:
                    candidate = hit
                    changed = True
                if candidate <= self._num_computed:
                    trace["iterations"].append(iteration)
                    trace["final"] = 0
                    return (LookupResult(LookupStatus.MISS, 0), trace)
            trace["iterations"].append(iteration)
            if not changed:
                break

        external = candidate - self._num_computed
        trace["final"] = candidate
        trace["external"] = external
        if external <= 0:
            return (LookupResult(LookupStatus.MISS, 0), trace)
        return (LookupResult(LookupStatus.HIT, candidate), trace)
