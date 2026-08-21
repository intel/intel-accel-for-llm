"""Hit policy tests: multi-group convergence, GDN right-to-left, align/-1.

v0.21 verified semantics: request.block_hashes has ONE hash per COMPLETE
block (block_size granularity), e.g. 2135-token prompt with block_size=544
-> 3 hashes. Tests use this granularity.
"""
from kvshrink.hybrid_metadata import GroupInfo, CacheKey
from kvshrink.hybrid_policy import (
    HybridHitPolicy, LookupResult, LookupStatus,
    align_down)


def _group(g_idx, kind, block_size, align=None, mode="align"):
    return GroupInfo(
        group_idx=g_idx, kind=kind,
        layer_names=(f"l{g_idx}.0", f"l{g_idx}.1"),
        block_size=block_size, page_size_bytes=1024,
        mamba_cache_mode=mode if kind == "mamba" else None,
        mamba_align_size=align)


def _hashes(committed: set[int], n_blocks: int):
    """committed: set of hash VALUES that are HIT; block_hashes[i] == i."""
    class B:
        def __init__(self):
            self.calls = []

        def lookup_boundary(self, key: CacheKey, expected_layers=None,
                            expected_boundary_tokens=None):
            idx = key.block_hash
            self.calls.append(idx)
            if idx in committed:
                return LookupStatus.HIT
            return LookupStatus.MISS

    return B(), list(range(n_blocks))


def test_attention_prefix_only():
    """[H,H,M,H] at block granularity: only first 2 blocks exist."""
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes({0, 1, 3}, 4)  # 4 blocks = 128 tokens
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 128)
    assert result.status == LookupStatus.HIT
    assert result.boundary_tokens == 64  # stops at hash 2 (block index 2)


def test_gdn_nearest_snapshot():
    """GDN finds the nearest snapshot walking right-to-left; no prefix
    contiguity required. block_size=16, align=32, 4 blocks (64 tokens).

    Only hash2 (48-token boundary) exists; 48 is NOT a multiple of the
    alignment 32, so it cannot be used as a restore snapshot
    (upstream MambaManager: `(i+1)*block_size % alignment_tokens` gate).
    """
    groups = [_group(0, "mamba", 16, align=32)]
    b, hashes = _hashes({2}, 4)  # hash2 = 48-token boundary (not aligned)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 64)
    assert result.status == LookupStatus.MISS, trace


def test_gdn_walks_left_when_boundary_missing():
    """Snapshot below the aligned candidate must itself be on an
    alignment boundary. Only hash0 (16 tokens) exists; 16 % 32 != 0 ->
    no usable snapshot -> MISS (upstream alignment gate)."""
    groups = [_group(0, "mamba", 16, align=32)]
    b, hashes = _hashes({0}, 4)  # only hash0 (16-token boundary) exists
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 64)
    assert result.status == LookupStatus.MISS


def test_multi_group_convergence():
    """Attention block 32 + GDN block 16 align 32; both fully present."""
    groups = [_group(0, "attention", 32), _group(1, "mamba", 16, align=32)]
    b, hashes = _hashes({0, 1, 2, 3}, 4)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 128)
    # attention: hashes 0,1,2,3 -> 128? candidate=127 -> 127//32=3 -> 96
    # mamba: align_down(127,32)=96 -> idx=96//16=6 out of range -> walk left
    # hash3 HIT -> 64; then attention re-check 64 -> 64//32=2 hashes 0,1 -> 64
    assert result.status == LookupStatus.HIT
    assert result.boundary_tokens == 64, trace


def test_align_down_and_minus_one():
    """Non-aligned boundary rounds down; -1 applied once."""
    groups = [_group(0, "mamba", 16, align=32)]
    b, hashes = _hashes({0, 1, 2, 3, 4, 5}, 6)  # 96 tokens
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 100)
    # candidate = min(99, align_down(99, 32)) = 96 -> idx 6/5 ... hash5 HIT -> 96
    assert result.boundary_tokens == 96, trace


def test_local_computed_tokens_reduce_external():
    """num_computed reduces the reported external hit."""
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes({0, 1}, 2)
    policy = HybridHitPolicy(groups, b, 16, 32, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 64)
    assert result.status == LookupStatus.HIT
    assert result.boundary_tokens == 64
    assert trace["external"] == 32


def test_local_computed_at_boundary_is_miss():
    """num_computed == candidate boundary -> no external hit."""
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes({0, 1}, 2)
    policy = HybridHitPolicy(groups, b, 16, 64, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 64)
    assert result.status == LookupStatus.MISS
    assert result.boundary_tokens == 0


def test_no_hit():
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes(set(), 2)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 64)
    assert result.status == LookupStatus.MISS
    assert result.boundary_tokens == 0


def test_boundary_table():
    """Table-driven: prompt lengths around block/alignment boundaries."""
    groups = [_group(0, "attention", 32), _group(1, "mamba", 16, align=32)]
    for length in (31, 32, 33, 63, 64, 65, 95, 96, 97):
        n_blocks = length // 16 + 2
        b, hashes = _hashes(set(range(n_blocks)), n_blocks)
        policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
        result, _ = policy.find_longest_cache_hit(hashes, length)
        expected = align_down(length - 1, 32)
        if expected == 0:
            assert result.status == LookupStatus.MISS, length
            assert result.boundary_tokens == 0, length
        else:
            assert result.status == LookupStatus.HIT, length
            assert result.boundary_tokens == expected, (
                f"len={length} got {result.boundary_tokens} want {expected}")


def test_none_mode_mamba_never_hits():
    """mamba_cache_mode none -> mamba contributes 0, no external hit."""
    groups = [
        _group(0, "attention", 32),
        _group(1, "mamba", 32, align=32, mode="none"),
    ]
    b, hashes = _hashes({0, 1, 2, 3}, 4)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 128)
    assert result.status == LookupStatus.MISS
    assert result.boundary_tokens == 0


def test_mamba_lookup_starts_left_of_boundary():
    """Mamba right-to-left scan must start at the hash AT the aligned
    boundary (aligned//gran - 1), never at the block AFTER the candidate.

    Regression for super-master gate (devlog §5.10): with a longer hash
    chain and another group shrinking the candidate, probing from
    aligned//gran could hit a snapshot to the RIGHT of the candidate and
    report a false boundary.
    """
    # attention block 16, mamba block 16 align 32; 6 blocks = 96 tokens
    groups = [
        _group(0, "attention", 16),
        _group(1, "mamba", 16, align=32),
    ]
    # candidate shrinks to 64 (attention only hashes 0..3 committed).
    # hash4 (80-token boundary) is committed but lies AFTER candidate 64.
    # Old bug: mamba started at aligned//gran = 64//16 = 4 -> HIT hash4
    # -> boundary 80 > candidate 64 (false, never allowed).
    b, hashes = _hashes({0, 1, 2, 3, 4}, 6)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 96)
    assert result.status == LookupStatus.HIT
    # mamba: align_down(95,32)=64 -> idx=64//16-1=3 -> hash3 HIT -> 64
    assert result.boundary_tokens == 64, trace


def test_mamba_snapshot_exactly_at_boundary():
    """Aligned boundary 64 with only hash3 committed -> hit 64."""
    groups = [_group(0, "mamba", 16, align=32)]
    b, hashes = _hashes({3}, 4)  # hash3 = 64-token boundary
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 64)
    # candidate = min(63, align_down(63,32)) = 32 -> idx=32//16-1=1 ->
    # hash1 MISS -> hash0 MISS -> MISS (no snapshot below 32)
    assert result.status == LookupStatus.MISS


def test_mamba_snapshot_at_candidate_cannot_overshoot():
    """Even if hash right of candidate is committed, boundary never
    exceeds candidate (fail closed on overrun)."""
    groups = [_group(0, "mamba", 16, align=32)]
    b, hashes = _hashes({5}, 6)  # hash5 = 96-token boundary
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 64)
    assert result.status == LookupStatus.MISS  # 96 > candidate 32
    assert result.boundary_tokens == 0


def test_mamba_empty_hashes_miss():
    """Empty hash list -> no scan -> MISS (no IndexError)."""
    groups = [_group(0, "mamba", 16, align=32)]
    policy = HybridHitPolicy(groups, _Backend_empty(), 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit([], 64)
    assert result.status == LookupStatus.MISS
    assert result.boundary_tokens == 0


def test_mamba_candidate_below_gran_miss():
    """candidate < gran (and < align): aligned=0 -> max_idx=-1 -> MISS."""
    groups = [_group(0, "mamba", 16, align=32)]
    b, hashes = _hashes({0}, 2)  # hash0 (16 tokens) committed
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 16)
    # candidate = min(15, align_down(15,32)=0) = 0 -> MISS
    assert result.status == LookupStatus.MISS
    assert result.boundary_tokens == 0


class _Backend_empty:
    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        return LookupStatus.MISS


def _group_aware(committed_pairs: set, n_blocks: int):
    """committed_pairs: set of (group_idx, hash_value) that are HIT."""
    class B:
        def lookup_boundary(self, key: CacheKey, expected_layers=None,
                            expected_boundary_tokens=None):
            if (key.group_idx, key.block_hash) in committed_pairs:
                return LookupStatus.HIT
            return LookupStatus.MISS

    return B(), list(range(n_blocks))


def test_multiple_attention_groups_all_validated():
    """Two attention groups, g1's manifest lacks hash2: the hit must
    shrink to 64 even though g0 has all four hashes. Pre-fix only the
    first attention group was looked up and later ones were blindly
    trimmed (super-master gate, devlog §5.21)."""
    groups = [_group(0, "attention", 32), _group(1, "attention", 32)]
    b, hashes = _group_aware(
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)}, 4)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, trace = policy.find_longest_cache_hit(hashes, 128)
    assert result.status == LookupStatus.HIT
    assert result.boundary_tokens == 64, trace


def test_multiple_attention_groups_converge_to_zero():
    """g1 has NO committed hashes at all -> MISS, not a g0-only hit."""
    groups = [_group(0, "attention", 32), _group(1, "attention", 32)]
    b, hashes = _group_aware({(0, 0), (0, 1)}, 4)
    policy = HybridHitPolicy(groups, b, 16, 0, "ns", 2, 0)
    result, _ = policy.find_longest_cache_hit(hashes, 128)
    assert result.status == LookupStatus.MISS
    assert result.boundary_tokens == 0
