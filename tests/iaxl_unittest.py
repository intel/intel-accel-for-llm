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


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTransferAfter(unittest.TestCase):
    """Sample usage of xfer_after(): a save of GPU blocks followed by a load into the same
    blocks. H2D and D2H contexts each have their own thread and stream, so the two directions
    overlap unless the load is made to follow the save. Runs with whichever copy engine
    IAXL_DSA_GD_ENABLE / IAXL_KV_XFER_FULL_DUPLEX select."""

    def test_load_after_save(self):
        num_blocks, block_bytes = 256, 64 * 1024
        gpu = torch.zeros((num_blocks, block_bytes), dtype=torch.uint8, device="cuda")
        saved = torch.zeros((num_blocks, block_bytes), dtype=torch.uint8, pin_memory=True)
        loaded = torch.full((num_blocks, block_bytes), 255, dtype=torch.uint8).pin_memory()
        # Every tensor is registered once as a pool of blocks along dim 0.
        torch_ext.register_kv_pool_gpu("test.gpu", gpu, 0)
        torch_ext.register_kv_pool_cpu("test.saved", saved)
        torch_ext.register_kv_pool_cpu("test.loaded", loaded)
        save, load = (
            torch_ext.Context.create("test.gpu", cpu_pool, direction, name, torch.cuda.Stream())
            for cpu_pool, direction, name in (
                ("test.saved", torch_ext.GpuTransferDirection.D2H, "save"),
                ("test.loaded", torch_ext.GpuTransferDirection.H2D, "load"),
            )
        )
        indices = torch.arange(num_blocks)  # block indices: an int64 tensor or a list
        try:
            for value in range(1, 6):
                gpu.fill_(value)
                for context in (save, load):
                    # The transfers start once the fill, pending on the current stream, is done.
                    context.xfer_wait_stream(torch.cuda.current_stream())
                save.xfer_chunks_batch(indices, indices)
                load.xfer_after(save)  # the loads from here on follow the saves so far
                load.xfer_chunks_batch(indices.tolist(), indices.tolist())
                load.xfer_finish()
                load.xfer_wait()
                self.assertTrue((saved == value).all())  # saved before the load overwrote it
                self.assertTrue((gpu == 255).all())
        finally:
            # A context must be waited for before it is destroyed, and destroyed before its
            # pools are unregistered.
            for context in (save, load):
                context.xfer_finish()
                context.xfer_wait()
            del save, load
            for name in ("test.gpu", "test.saved", "test.loaded"):
                torch_ext.unregister_kv_pool(name)


if __name__ == "__main__":
    unittest.main()
