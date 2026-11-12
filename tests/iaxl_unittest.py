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


if __name__ == "__main__":
    unittest.main()
