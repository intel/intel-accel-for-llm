"""Table-driven tests for vLLM's mamba block-aligned prefill split.

``Scheduler._mamba_block_aligned_split`` decides how many tokens a
prefill chunk may schedule so a GDN/mamba state snapshot always lands
on a block boundary. It is called with BOTH locally prefix-cached and
externally restored token counts, so an external connector hit must
flow through exactly the same math as a local hit -- otherwise a
restored request could be scheduled off-boundary and never produce a
cacheable snapshot.

v0.21/0.22 refused external tokens here (an assert that a patch had to
remove); v0.23.0 takes ``num_external_computed_tokens`` as a first-class
parameter, so these tests call the REAL upstream method and pin the
contract we depend on:

  P1. Isomorphism: for a fixed total computed count,
      (local=N, external=0) == (local=0, external=N) == any (l, e)
      split with l + e == N.
  P2. Boundary behavior: computed start at n-1 / n / n+1 (block edges)
      and num_new_tokens landing before / exactly on / crossing / past
      last_cache_position.
  P3. Budget interaction: num_new_tokens budget < / == / > block_size.
  P4. Resumed-request semantics: num_computed_tokens >= the prefill
      horizon skips the split branch entirely.

No GPU, no model, no machine specifics: only the scheduler method.
"""

from __future__ import annotations

import inspect
import itertools

import pytest

from vllm.v1.core.sched.scheduler import Scheduler


def test_external_tokens_are_a_first_class_parameter():
    """Regression guard replacing the old source patch: upstream must
    keep accepting external-connector tokens here. If a future vLLM
    re-introduces a restriction, every hybrid external hit would be
    mis-scheduled, so fail loudly at test time instead."""
    sig = inspect.signature(Scheduler._mamba_block_aligned_split)
    assert "num_external_computed_tokens" in sig.parameters
    src = inspect.getsource(Scheduler._mamba_block_aligned_split)
    assert "not verified" not in src, (
        "upstream re-added an external-connector restriction to "
        "_mamba_block_aligned_split; review the hybrid load path")


class _Req:
    def __init__(self, num_tokens, num_prompt_tokens, num_computed_tokens):
        self.num_tokens = num_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_output_tokens = max(num_tokens - num_prompt_tokens, 0)


class _Sched:
    def __init__(self, block_size, use_eagle=False):
        self.cache_config = type("C", (), {"block_size": block_size})()
        self.use_eagle = use_eagle


def _split(req, num_new_tokens, local, external, sched):
    """Call the REAL upstream method with a stub ``self`` carrying only
    the two attributes it reads (block size + eagle flag)."""
    return Scheduler._mamba_block_aligned_split(
        sched, req, num_new_tokens, local, external)


def _both(req, num_new_tokens, local, external, sched):
    """Result of the upstream method (the contract under test)."""
    return _split(req, num_new_tokens, local, external, sched)


def test_p1_isomorphism_all_splits():
    """P1: all (local, external) splits with equal sums give equal results."""
    for block_size in (528, 544):
        sched = _Sched(block_size)
        for num_tokens in (1088, 1560, 2135, 4096):
            for total_computed in (0, block_size, 2 * block_size):
                for budget in (block_size // 2, block_size,
                               2 * block_size, 2048):
                    req = _Req(num_tokens, num_tokens, 0)
                    results = []
                    splits = [
                        (total_computed, 0), (0, total_computed),
                        (total_computed // 2,
                         total_computed - total_computed // 2),
                    ]
                    if total_computed > 0:
                        splits.append((max(total_computed - 1, 0), 1))
                    for local, external in splits:
                        out = _both(req, budget, local, external, sched)
                        results.append(out)
                    assert all(r == results[0] for r in results), (
                        f"bs={block_size} nt={num_tokens} "
                        f"computed={total_computed} budget={budget}: "
                        f"{results}")


def test_p1_isomorphism_local_external_swap():
    """P1 minimal: local=N == external=N."""
    for block_size in (528, 544):
        sched = _Sched(block_size)
        for num_tokens in (1560, 2135):
            for N in (0, block_size // 2, block_size, block_size * 2):
                for budget in (100, block_size, 2048):
                    req = _Req(num_tokens, num_tokens, 0)
                    a = _both(req, budget, N, 0, sched)
                    b = _both(req, budget, 0, N, sched)
                    assert a == b, f"bs={block_size} nt={num_tokens} N={N}"


def test_p2_boundary_edges():
    """P2: computed start at n-1 / n / n+1 relative to block edge."""
    for block_size in (528, 544):
        sched = _Sched(block_size)
        num_tokens = 3 * block_size + 37  # 3 full blocks + tail
        for edge in range(block_size - 1, block_size + 2):  # n-1, n, n+1
            for budget in (block_size, 2048):
                req = _Req(num_tokens, num_tokens, 0)
                for local in (0, edge):
                    for external in (edge, 0):
                        if local + external != edge:
                            continue
                        out = _both(req, budget, local, external, sched)
                        assert 0 <= out <= budget, out


def test_p2_landing_before_exact_cross_past():
    """P2: num_new_tokens lands before / exactly on / crosses / past
    last_cache_position."""
    block_size = 544
    sched = _Sched(block_size)
    num_tokens = 3 * block_size  # 1632
    last = num_tokens - num_tokens % block_size  # 1632
    cases = [
        # (computed, budget, expected behavior)
        (0, block_size, "before"),        # after=544 < 1632 -> align
        (0, 2 * block_size, "before"),    # after=1088 < 1632 -> align
        (block_size, block_size, "before"),  # after=1088 < 1632
        (block_size, 2 * block_size, "exact"),  # after=1632 == last
        (block_size + 1, block_size, "cross"),  # after=1089? no: 544+1+544=1089 < 1632
    ]
    # recompute expectations precisely
    expectations = {}
    for computed in (0, block_size, block_size + 1):
        for budget in (block_size, 2 * block_size, 3 * block_size):
            req = _Req(num_tokens, num_tokens, 0)
            out = _both(req, budget, computed, 0, sched)
            after = computed + budget
            if after < last:
                exp = budget // block_size * block_size
            elif computed < last < after:
                exp = last - computed
            else:
                exp = budget
            assert out == exp, (
                f"computed={computed} budget={budget}: got {out} exp {exp}")


def test_p2_last_cache_position_crossing():
    """P2: crossing last_cache_position forces caching the last chunk."""
    block_size = 544
    sched = _Sched(block_size)
    num_tokens = 2 * block_size + 100  # 1188
    last = 2 * block_size  # 1088
    req = _Req(num_tokens, num_tokens, 0)
    # computed=544 (local 544 + ext 0): budget 600 crosses 1088
    out = _both(req, 600, 544, 0, sched)
    assert out == last - 544 == 544, out
    # same with swapped split
    out2 = _both(req, 600, 0, 544, sched)
    assert out2 == out, out2


def test_p3_budget_interaction():
    """P3: budget < / == / > block_size all keep alignment rules."""
    block_size = 544
    sched = _Sched(block_size)
    num_tokens = 4 * block_size
    for computed in (0, block_size):
        for budget in (block_size // 2, block_size, block_size + 1,
                       2 * block_size - 1, 2 * block_size):
            req = _Req(num_tokens, num_tokens, 0)
            out = _both(req, budget, computed, 0, sched)
            after = computed + budget
            last = 4 * block_size
            if after < last:
                exp = budget // block_size * block_size
            elif computed < last < after:
                exp = last - computed
            else:
                exp = budget
            assert out == exp, (computed, budget, out, exp)


def test_p4_resumed_request_skips_split():
    """P4: resumed requests (computed >= prompt) bypass the branch."""
    block_size = 544
    sched = _Sched(block_size)
    num_prompt = 1560
    # resumed: num_computed_tokens >= num_prompt_tokens
    for computed in (num_prompt, num_prompt + 8):
        req = _Req(num_tokens=num_prompt + 8,
                   num_prompt_tokens=num_prompt,
                   num_computed_tokens=computed)
        for local, external in ((0, 0), (64, 0), (0, 64), (64, 64)):
            for budget in (8, 100, block_size):
                out = _both(req, budget, local, external, sched)
                assert out == budget, (computed, local, external, budget, out)


def test_p4_non_resumed_uses_max():
    """P4: branch condition uses max(prompt, num_tokens-1)."""
    block_size = 544
    sched = _Sched(block_size)
    # num_tokens == num_prompt (prefill, no output yet): max == num_prompt
    req = _Req(num_tokens=1560, num_prompt_tokens=1560, num_computed_tokens=0)
    out = _both(req, 700, 0, 544, sched)
    # computed=544 < max(1560, 1559) -> branch active
    last = 1560 - 1560 % 544  # 1088
    assert out == last - 544 == 544, out
