#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""iaxl unit tests. Run inside the dev container from the repo root:

    python3 tests/iaxl_unittest.py [-v]

The zip backends are selected with the usual IAXL_*_ZIP_ENABLE variables; the CPU backend is on
unless disabled explicitly. Round trips are checked bit-exactly, so lossy truncation is off.
"""

import os
import unittest

os.environ.setdefault("IAXL_CPU_ZIP_ENABLE", "1")
os.environ["IAXL_KV_LOSSY_TRUNC"] = "0"

import torch  # noqa: E402
from iaxl import torch_ext  # noqa: E402


class TestKvZip(unittest.TestCase):
    """Sample usage of compress_tensors / decompress_tensors."""

    def test_roundtrip(self):
        # Inputs: contiguous CPU tensors of one dtype and size.
        tensors = [torch.randn(2, 16, 128, dtype=torch.bfloat16) for _ in range(8)]
        # Segmented compression (IAXL_ZIP_SEGMENT_SIZE_KB) uses the inputs as scratch, so keep
        # copies to compare against.
        expected = [t.clone() for t in tensors]

        # Compress: one uint8 tensor (the block) per input.
        blocks = torch_ext.compress_tensors(tensors)
        self.assertEqual(len(blocks), len(tensors))
        for b in blocks:
            self.assertEqual(b.dtype, torch.uint8)

        # Decompress into preallocated tensors of the original shape and dtype.
        outs = [torch.empty_like(t) for t in tensors]
        torch_ext.decompress_tensors(blocks, outs)
        for e, o in zip(expected, outs):
            self.assertTrue(torch.equal(e, o))

    def test_compressible(self):
        tensors = [torch.zeros(64 * 1024, dtype=torch.bfloat16) for _ in range(4)]
        blocks = torch_ext.compress_tensors(tensors)
        for t, b in zip(tensors, blocks):
            # IAXL_ZIP_EXPONENT_ONLY keeps half of every segment raw.
            self.assertLess(b.numel(), t.numel() * t.element_size() * 0.6)


class TestScratchPool(unittest.TestCase):
    """The scratch pool cuts pages from free slot ranges, lowest address first."""

    def setUp(self):
        from iaxl.envs import envs
        from iaxl.kvflow.scratch_pool import SLOT_BYTES, ScratchPool

        self.slot = SLOT_BYTES
        saved, envs.IAXL_SCRATCH_POOL_SIZE_GB = envs.IAXL_SCRATCH_POOL_SIZE_GB, 10 * SLOT_BYTES / 1024**3
        try:
            self.pool = ScratchPool(pin_memory=False)
        finally:
            envs.IAXL_SCRATCH_POOL_SIZE_GB = saved
        self.addCleanup(self.pool.close)

    def test_pages_take_whole_slots(self):
        a = self.pool.allocate(2, self.slot + 1)
        self.assertEqual(a.indices, [0, 2])
        b = self.pool.allocate(3, 100)
        self.assertEqual(b.indices, [4, 5, 6])
        self.assertEqual(self.pool.available_count(), 3)

    def test_release_in_any_order_and_reuse_holes(self):
        a = self.pool.allocate(2, self.slot)  # slots 0-1
        b = self.pool.allocate(3, self.slot)  # slots 2-4
        c = self.pool.allocate(2, self.slot)  # slots 5-6
        self.pool.release(a)  # free: 0-1 and 7-9, before b and c come back
        d = self.pool.allocate(3, self.slot)  # one allocation spread over both holes
        self.assertEqual(d.indices, [0, 1, 7])
        e = self.pool.allocate(1, 2 * self.slot)
        self.assertEqual(e.indices, [8])
        self.assertIsNone(self.pool.allocate(1, self.slot))  # full
        for x in (c, e, b, d):
            self.pool.release(x)
        self.assertEqual(self.pool._free, [(0, 10)])  # every neighbour merged back

    def test_failure_changes_nothing(self):
        self.pool.allocate(4, self.slot)
        before = list(self.pool._free)
        self.assertIsNone(self.pool.allocate(4, 2 * self.slot))  # 6 free slots hold 3 such pages
        self.assertEqual(self.pool._free, before)

    def test_random_mixed_pages_never_overlap(self):
        import random

        rng = random.Random(0)
        live = []
        for _ in range(2000):
            if live and rng.random() < 0.5:
                self.pool.release(live.pop(rng.randrange(len(live))))
            else:
                a = self.pool.allocate(rng.randint(1, 3), rng.choice([1, 2, 3]) * self.slot)
                if a is not None:
                    live.append(a)
            held = sorted((i, i + a.per_page) for a in live for i in a.indices)
            spans = sorted(held + [(s, s + n) for s, n in self.pool._free])
            self.assertTrue(all(x[1] <= y[0] for x, y in zip(spans, spans[1:])))
            self.assertEqual(sum(e - s for s, e in spans), 10)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestKVFlowMixedPages(unittest.TestCase):
    """Layers of different page sizes share the scratch pool through put and get."""

    def test_roundtrip(self):
        import tempfile

        from iaxl.envs import envs
        from iaxl.kvflow.flow import KVFlow

        num_blocks = 16
        # 20,000 B pages (2 slots, not slot-aligned) and 2,048 B pages (1 slot, mostly empty).
        caches = {
            "big": torch.randn(num_blocks, 10000, dtype=torch.bfloat16, device="cuda"),
            "small": torch.randn(num_blocks, 1024, dtype=torch.bfloat16, device="cuda"),
        }
        expected = {k: v.clone() for k, v in caches.items()}
        with tempfile.TemporaryDirectory() as d:
            saved = envs.IAXL_CACHE_DIR
            envs.IAXL_CACHE_DIR = d
            try:
                flow = KVFlow("mixed", cache_size_gb=1, kv_caches=caches, block_dim=0)
            finally:
                envs.IAXL_CACHE_DIR = saved
            try:
                chunks = [1, 3, 4, 9]
                labels = [f"c{i}" for i in chunks]
                flow.put_wait(flow.put("req", list(caches), chunks, labels))
                for t in caches.values():
                    t.zero_()
                flow.get_wait(flow.get("req", list(caches), chunks, labels))
                torch.cuda.synchronize()
                for k, t in caches.items():
                    self.assertTrue(torch.equal(t[chunks], expected[k][chunks]), k)
                    others = [i for i in range(num_blocks) if i not in chunks]
                    self.assertEqual(t[others].count_nonzero().item(), 0, k)
            finally:
                flow.close()


if __name__ == "__main__":
    unittest.main()
