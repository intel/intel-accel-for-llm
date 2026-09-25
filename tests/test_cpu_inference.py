import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ZIP_BACKEND = os.environ.get("IAXL_TEST_ZIP_BACKEND", "cpu")
if ZIP_BACKEND not in ("cpu", "qat"):
    raise ValueError("IAXL_TEST_ZIP_BACKEND must be cpu or qat")

os.environ.update(
    IAXL_QAT_ZIP_ENABLE=str(int(ZIP_BACKEND == "qat")),
    IAXL_CPU_ZIP_ENABLE=str(int(ZIP_BACKEND == "cpu")),
    IAXL_IAA_ZIP_ENABLE="0",
    IAXL_QAT_INSTANCE_NUM="2",
    IAXL_CPU_ZIP_THREADS="2",
    OMP_NUM_THREADS="8",
    IAXL_KV_COMPRESSION="1",
    IAXL_KV_LOSSY_TRUNC="0",
    IAXL_KV_DATA_SHUFFLE="1",
    IAXL_PROFILE_MODE="disabled",
)

import torch

from iaxl import KVStore, torch_ext
from iaxl.envs import envs
from iaxl.kvflow import KVFlow, get_accelerator_device
from iaxl.kvflow.scratch_pool import ScratchPool


class CPUInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_ext.device_type != "cpu":
            raise RuntimeError("Build IAXL with DEVICE=cpu before running CPU tests")

    def setUp(self):
        cache_dir = self.enterContext(tempfile.TemporaryDirectory(prefix="iaxl-cpu-"))
        self.enterContext(patch.object(envs, "IAXL_CACHE_DIR", cache_dir))
        self.flow = KVFlow("roundtrip", cache_size_gb=0.01)
        self.addCleanup(self.flow.stop)

    def test_cpu_build_does_not_access_accelerator_apis(self):
        self.assertEqual(torch.get_num_threads(), 8)
        self.assertEqual(
            Path(self.flow.persist_dir),
            Path(envs.IAXL_CACHE_DIR) / "cpu" / "compressed" / "roundtrip",
        )
        with (
            patch.object(torch.cuda, "is_available", side_effect=AssertionError),
            patch.object(torch.xpu, "is_available", side_effect=AssertionError),
            patch.object(torch.cuda, "Stream", side_effect=AssertionError),
            patch.object(torch.xpu, "Stream", side_effect=AssertionError),
        ):
            self.assertEqual(get_accelerator_device(), "cpu")
            self.flow._ensure_streams()
        self.assertIsNone(self.flow.put_stream)
        self.assertIsNone(self.flow.get_stream)

    def test_direct_codec_bypasses_scratch_pool(self):
        self.assertTrue(self.flow.direct_codec)
        tensor = (torch.arange(2 * 6 * 2 * 32 * 64) % 19).to(torch.bfloat16)
        tensor = tensor.reshape(2, 6, 2, 32, 64)
        original = tensor.clone()
        labels = ["direct4", "direct0", "direct2"]
        indices = [4, 0, 2]
        tasks = self.flow.put("kv", {"layer0": tensor}, 1, indices, labels)
        self.assertTrue(all(task.cpu_tensors is None for task in tasks.values()))
        self.flow.put_wait(tasks)
        self.assertTrue(torch.equal(tensor, original), "PUT must not modify the KV tensor")
        tensor.fill_(-1)
        tasks = self.flow.get("kv", {"layer0": tensor}, 1, indices, labels)
        self.assertTrue(all(task.cpu_tensors is None for task in tasks.values()))
        self.flow.get_wait(tasks)
        for index in range(tensor.shape[1]):
            expected = original.select(1, index) if index in indices else torch.full_like(
                original.select(1, index), -1)
            self.assertTrue(torch.equal(tensor.select(1, index), expected), index)
        status = self.flow.status()
        self.assertEqual(status["pool_in_use"], 0)
        self.assertIsNone(self.flow.chunk_pool, "scratch pool must not be created")

    def test_direct_codec_rejects_out_of_range_chunks(self):
        tensor = torch.zeros((2, 4, 16), dtype=torch.bfloat16)
        for direction, method in (
            (torch_ext.GpuTransferDirection.D2H, "zip_to_mem_direct"),
            (torch_ext.GpuTransferDirection.H2D, "unzip_from_mem_direct"),
        ):
            with self.subTest(method=method):
                context = torch_ext.Context.create(tensor, 1, direction)
                with self.assertRaisesRegex(RuntimeError, "out of range"):
                    getattr(context, method)(self.flow.mem, "kv", "layer0", ["a"], [4])
                with self.assertRaisesRegex(RuntimeError, "must match"):
                    getattr(context, method)(self.flow.mem, "kv", "layer0", ["a", "b"], [0])

    NATIVE_THREAD_PROBE = textwrap.dedent(
        """
        import json, os, sys, tempfile
        import torch
        from iaxl.envs import envs
        from iaxl.kvflow import KVFlow

        with tempfile.TemporaryDirectory() as directory:
            envs.IAXL_CACHE_DIR = directory
            flow = KVFlow("probe", cache_size_gb=0.01)
            tensor = (torch.arange(2 * 8 * 2 * 32 * 64) % 13).to(torch.bfloat16)
            tensor = tensor.reshape(2, 8, 2, 32, 64)
            original = tensor.clone()
            labels = [f"b{i}" for i in range(8)]
            for name, skip in (("zip", 0), ("raw", 1)):
                flow.put_wait(flow.put("kv", {name: tensor}, 1, list(range(8)), labels,
                                       skip_compression_count=skip))
                tensor.fill_(0)
                flow.get_wait(flow.get("kv", {name: tensor}, 1, list(range(8)), labels))
                assert torch.equal(tensor, original), name
            masks = {}
            for tid in os.listdir("/proc/self/task"):
                with open(f"/proc/self/task/{tid}/status") as status:
                    fields = dict(line.split(":", 1) for line in status if ":" in line)
                masks.setdefault(fields["Name"].strip(), set()).add(
                    fields["Cpus_allowed_list"].strip())
            flow.stop()
            print("PROBE " + json.dumps({k: sorted(v) for k, v in masks.items()}))
        """
    )

    def _run_native_probe(self, **extra_env):
        environment = {**os.environ, **extra_env}
        completed = subprocess.run(
            [sys.executable, "-u", "-c", self.NATIVE_THREAD_PROBE],
            env=environment, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        line = next(l for l in completed.stdout.splitlines() if l.startswith("PROBE "))
        return json.loads(line[len("PROBE "):]), completed.stdout + completed.stderr

    def test_native_threads_honour_cpu_affinity(self):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < 4:
            self.skipTest("need at least four CPUs to pin a disjoint set")
        pinned = f"{allowed[-2]}-{allowed[-1]}" if allowed[-1] == allowed[-2] + 1 else (
            f"{allowed[-2]},{allowed[-1]}")
        masks, _ = self._run_native_probe(IAXL_CPU_AFFINITY=pinned, IAXL_CPU_ZIP_THREADS="2")
        for name in ("D2H", "H2D", "OMP-Main"):
            self.assertEqual(masks[name], [pinned], f"{name} thread not pinned: {masks}")
        self.assertNotEqual(masks["python"], [pinned], "the Python threads must keep their mask")

    def test_dsa_memcpy_falls_back_without_hardware(self):
        _, output = self._run_native_probe(IAXL_DSA_MEMCPY_ENABLE="1",
                                           IAXL_DSA_MEMCPY_MIN_BYTES="0",
                                           IAXL_DSA_WQS="wq-iaxl-does-not-exist")
        self.assertIn("dsa_memcpy=ON", output)
        self.assertEqual(output.count("falling back to CPU memcpy"), 1, output)

    def test_dsa_memcpy_skips_small_batches(self):
        _, output = self._run_native_probe(IAXL_DSA_MEMCPY_ENABLE="1",
                                           IAXL_DSA_WQS="wq-iaxl-does-not-exist")
        self.assertIn("dsa_memcpy_min_bytes=1048576", output)
        # Device codecs may queue per-block staging copies on DSA regardless of the batch floor.
        if ZIP_BACKEND == "cpu":
            self.assertNotIn("falling back to CPU memcpy", output)

    DSA_CODEC_PROBE = textwrap.dedent(
        """
        import tempfile
        import torch
        from iaxl.envs import envs
        from iaxl.kvflow import KVFlow

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            envs.IAXL_CACHE_DIR = directory
            flow = KVFlow("dsa-probe", cache_size_gb=0.05)
            # Blocks on axis 1 (two strided segments each) and on axis 0 (one contiguous run).
            for name, shape, dim in (("kv5d", (2, 12, 4, 32, 64), 1), ("kv4d", (12, 8, 32, 256), 0)):
                tensor = (torch.randn(shape) * 0.05).to(torch.bfloat16)
                original = tensor.clone()
                labels = [f"{name}{i}" for i in range(0, 12, 2)]
                indices = list(range(0, 12, 2))
                flow.put_wait(flow.put("kv", {name: tensor}, dim, indices, labels))
                assert torch.equal(tensor, original), "PUT modified the KV tensor"
                tensor.fill_(0)
                flow.get_wait(flow.get("kv", {name: tensor}, dim, indices, labels))
                for index in range(12):
                    got, want = tensor.select(dim, index), original.select(dim, index)
                    assert torch.equal(got, want if index in indices else torch.zeros_like(want)), (
                        name, index)
            flow.stop()
        print("PROBE-OK")
        """
    )

    @unittest.skipUnless(ZIP_BACKEND == "qat" and os.path.exists("/dev/dsa/wq0.0"),
                         "requires QAT hardware and DSA work queue wq0.0")
    def test_qat_codec_copies_run_on_dsa(self):
        for shuffle in ("0", "1"):
            with self.subTest(shuffle=shuffle):
                completed = subprocess.run(
                    [sys.executable, "-u", "-c", self.DSA_CODEC_PROBE],
                    env={**os.environ, "IAXL_DSA_MEMCPY_ENABLE": "1", "IAXL_DSA_WQS": "wq0.0",
                         "IAXL_KV_DATA_SHUFFLE": shuffle},
                    capture_output=True, text=True, timeout=300,
                    cwd=str(Path(__file__).resolve().parents[1]),
                )
                output = completed.stdout + completed.stderr
                self.assertEqual(completed.returncode, 0, output)
                self.assertIn("PROBE-OK", output)
                self.assertIn("codec staging copies: using Intel DSA", output)
                self.assertNotIn("falling back to CPU memcpy", output)

    @unittest.skipUnless(ZIP_BACKEND == "qat", "requires QAT hardware")
    def test_qat_single_poller_drives_all_instances(self):
        _, output = self._run_native_probe(IAXL_QAT_POLL_THREADS="1")
        self.assertIn("qat_instances=2 qat_pollers=1", output)
        self.assertIn("omp_threads=1", output)

    SHUFFLE_PERSIST_PROBE = textwrap.dedent(
        """
        import sys
        import torch
        from iaxl.envs import envs
        from iaxl.kvflow import KVFlow

        mode, directory = sys.argv[1], sys.argv[2]
        envs.IAXL_CACHE_DIR = directory
        flow = KVFlow("shuffle-probe", cache_size_gb=0.01)
        tensor = (torch.arange(2 * 6 * 2 * 32 * 64) % 23).to(torch.bfloat16)
        tensor = tensor.reshape(2, 6, 2, 32, 64)
        original = tensor.clone()
        labels = [f"s{i}" for i in range(6)]
        if mode == "put":
            flow.put_wait(flow.put("kv", {"layer0": tensor}, 1, list(range(6)), labels))
            flow.put_finish("kv", labels)
            flow.record_flush()
            assert flow.persist(10)["persisted"] == 6
        else:
            assert flow.has("kv", labels) == [True] * 6
            tensor.fill_(-1)
            flow.get_wait(flow.get("kv", {"layer0": tensor}, 1, list(range(6)), labels))
            assert flow.mem.hits_in_storage > 0
            assert torch.equal(tensor, original), "persisted block decoded with wrong shuffle"
        flow.stop()
        print("PROBE-OK")
        """
    )

    def test_persisted_blocks_decode_regardless_of_shuffle_setting(self):
        for put_shuffle, get_shuffle in (("1", "0"), ("0", "1")):
            with self.subTest(put=put_shuffle, get=get_shuffle):
                with tempfile.TemporaryDirectory(prefix="iaxl-shuffle-") as directory:
                    for mode, shuffle in (("put", put_shuffle), ("get", get_shuffle)):
                        completed = subprocess.run(
                            [sys.executable, "-u", "-c", self.SHUFFLE_PERSIST_PROBE, mode,
                             directory],
                            env={**os.environ, "IAXL_KV_DATA_SHUFFLE": shuffle},
                            capture_output=True, text=True, timeout=300,
                            cwd=str(Path(__file__).resolve().parents[1]),
                        )
                        self.assertEqual(completed.returncode, 0,
                                         f"{mode} shuffle={shuffle}\n"
                                         + completed.stdout + completed.stderr)
                        self.assertIn("PROBE-OK", completed.stdout)

    def _run_persist_probe(self, mode, directory, **extra_env):
        return subprocess.run(
            [sys.executable, "-u", "-c", self.SHUFFLE_PERSIST_PROBE, mode, directory],
            env={**os.environ, **extra_env}, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

    def test_persisted_cache_refuses_foreign_format_version(self):
        import sqlite3

        with tempfile.TemporaryDirectory(prefix="iaxl-fmt-") as directory:
            completed = self._run_persist_probe("put", directory)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            db_path = next(Path(directory).rglob("chunks.db"))
            with sqlite3.connect(db_path) as db:
                (version,) = db.execute(
                    "SELECT value FROM meta WHERE key = 'format_version'").fetchone()
                self.assertEqual(version, 1)
                self.assertGreater(db.execute("SELECT count(*) FROM chunks").fetchone()[0], 0)

            completed = self._run_persist_probe("get", directory)
            self.assertEqual(completed.returncode, 0, "same version must reload\n"
                             + completed.stdout + completed.stderr)

            for tamper, label in (("UPDATE meta SET value = 0 WHERE key = 'format_version'",
                                   "version 0"),
                                  ("DROP TABLE meta", "pre-version cache")):
                with self.subTest(tamper=label):
                    with sqlite3.connect(db_path) as db:
                        db.execute(tamper)
                    completed = self._run_persist_probe("get", directory)
                    self.assertNotEqual(completed.returncode, 0, label)
                    self.assertIn("has format version", completed.stderr, label)
                    self.assertIn("this build writes version 1", completed.stderr, label)
                    with sqlite3.connect(db_path) as db:
                        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, "
                                   "value INTEGER)")
                        db.execute("INSERT OR REPLACE INTO meta VALUES ('format_version', 1)")

            completed = self._run_persist_probe("get", directory)
            self.assertEqual(completed.returncode, 0, "restored version must reload\n"
                             + completed.stdout + completed.stderr)

            with sqlite3.connect(db_path) as db:
                db.execute("DROP TABLE meta")
            completed = self._run_persist_probe("get", directory, IAXL_KV_DATA_SHUFFLE="0")
            self.assertEqual(completed.returncode, 0, "unshuffled pre-version cache must load\n"
                             + completed.stdout + completed.stderr)
            self.assertIn("adopting pre-version cache", completed.stderr)
            with sqlite3.connect(db_path) as db:
                (version,) = db.execute(
                    "SELECT value FROM meta WHERE key = 'format_version'").fetchone()
                self.assertEqual(version, 1)

    def test_unpinned_scratch_is_reused(self):
        with patch.object(envs, "IAXL_SCRATCH_POOL_SIZE_GB", 1e-4):
            pool = ScratchPool((2, 16, 64), torch.bfloat16, pin_memory=False)
        tensors = pool.allocate(3, (2, 16, 64), torch.bfloat16)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in tensors))
        self.assertTrue(all(not tensor.is_pinned() for tensor in tensors))
        addresses = {tensor.data_ptr() for tensor in tensors}
        pool.release(tensors)
        reused = pool.allocate(3, (2, 16, 64), torch.bfloat16)
        self.assertEqual({tensor.data_ptr() for tensor in reused}, addresses)
        pool.release(reused)
        self.assertEqual(pool._allocate_count, pool._release_count)

    def test_native_cpu_transfer(self):
        tensor = torch.arange(2 * 7 * 64, dtype=torch.float32).reshape(2, 7, 64)
        original = tensor.clone()
        indices = [5, 1, 3]
        scratch = [torch.empty((2, 64)) for _ in indices]
        context = torch_ext.Context.create(
            tensor, 1, torch_ext.GpuTransferDirection.D2H
        )
        context.xfer_wait_cur_stream(sync_cur_stream=True)
        context.xfer_chunks_batch(indices, scratch)
        context.xfer_finish()
        context.xfer_wait()
        self.assertTrue(context.xfer_is_complete())
        for index, chunk in zip(indices, scratch):
            self.assertTrue(torch.equal(chunk, original.select(1, index)))

        tensor.fill_(-1)
        context = torch_ext.Context.create(
            tensor, 1, torch_ext.GpuTransferDirection.H2D
        )
        for index, chunk in zip(indices, scratch):
            context.xfer_chunk(chunk, index)
        context.xfer_finish()
        context.xfer_wait()
        for index in range(tensor.shape[1]):
            if index in indices:
                self.assertTrue(torch.equal(tensor.select(1, index), original.select(1, index)))
            else:
                self.assertTrue(torch.all(tensor.select(1, index) == -1))

    def test_native_context_rejects_incompatible_tensors_and_streams(self):
        tensor = torch.empty((2, 7, 64))
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch_ext.Context.create(tensor.transpose(0, 1), 0)
        for dimension in (-1, tensor.ndim):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(RuntimeError, "dimension"):
                    torch_ext.Context.create(tensor, dimension)
        with self.assertRaisesRegex(RuntimeError, "built for cpu"):
            torch_ext.Context.create(torch.empty((2, 7, 64), device="meta"), 0)
        with self.assertRaisesRegex(ValueError, "accelerator streams"):
            torch_ext.Context.create(tensor, 0, work_stream=object())

    def test_raw_and_compressed_round_trips(self):
        # (3, 5, 16, 64) on dim 1 gives three segments, so the shuffle half-way point splits one.
        layouts = (((7, 2, 16, 64), 0), ((2, 7, 16, 64), 1), ((2, 3, 7, 64), 2),
                   ((3, 5, 16, 64), 1))
        source_indices = [4, 1, 3]
        target_indices = [2, 4, 0]

        for shape, dimension in layouts:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for skip_count in (0, 1, 2):
                    with self.subTest(shape=shape, dtype=dtype, skip_count=skip_count):
                        prefix = f"{dimension}-{dtype}-{skip_count}"
                        labels = [f"{prefix}-{index}" for index in source_indices]
                        values = torch.arange(torch.Size(shape).numel()).reshape(shape) % 17
                        tensors = {
                            "layer0": values.to(dtype),
                            "layer1": (values + 32).to(dtype),
                        }
                        originals = {name: tensor.clone() for name, tensor in tensors.items()}
                        before_raw = self.flow.mem.total_unzip_bytes
                        before_stored = self.flow.mem.total_zip_bytes
                        tasks = self.flow.put(
                            "kv", tensors, dimension, source_indices, labels,
                            skip_compression_count=skip_count,
                        )
                        if not self.flow.put_wait(tasks, wait=False):
                            self.flow.put_wait(tasks)
                        self.assertTrue(all(task.ctx is None for task in tasks.values()))
                        self.flow.put_finish("kv", labels)
                        self.flow.record_flush()
                        self.assertEqual(self.flow.has("kv", labels), [True] * len(labels))

                        raw_bytes = self.flow.mem.total_unzip_bytes - before_raw
                        stored_bytes = self.flow.mem.total_zip_bytes - before_stored
                        self.assertGreater(raw_bytes, 0)
                        if skip_count == len(tensors):
                            self.assertEqual(stored_bytes, raw_bytes + 8 * len(labels) * len(tensors))
                        else:
                            self.assertLess(stored_bytes, raw_bytes)

                        expected = {}
                        for name, tensor in tensors.items():
                            self.assertTrue(torch.equal(tensor, originals[name]))
                            tensor.fill_(-1)
                            expected[name] = tensor.clone()
                            for source, target in zip(source_indices, target_indices):
                                expected[name].select(dimension, target).copy_(
                                    originals[name].select(dimension, source)
                                )

                        tasks = self.flow.get("kv", tensors, dimension, target_indices, labels)
                        self.assertIsInstance(self.flow.get_wait(tasks, wait=False), bool)
                        self.flow.get_wait(tasks)
                        for name, tensor in tensors.items():
                            self.assertTrue(torch.equal(tensor, expected[name]), name)
                        self.assertTrue(all(task.ctx is None for task in tasks.values()))
                        self.assertEqual(self.flow.status()["pool_in_use"], 0)

    def test_persist_evict_and_reload(self):
        tensor = (torch.arange(2 * 7 * 2 * 32 * 64) % 17).to(torch.bfloat16)
        tensor = tensor.reshape(2, 7, 2, 32, 64)
        original = tensor.clone()
        labels = ["persist5", "persist1", "persist3"]
        indices = [5, 1, 3]
        tasks = self.flow.put("kv", {"layer0": tensor}, 1, indices, labels)
        self.flow.put_wait(tasks)
        self.flow.put_finish("kv", labels)
        self.flow.record_flush()
        persisted = self.flow.persist(10)
        self.assertGreater(persisted["persisted"], 0)
        self.assertGreater(persisted["bytes_written"], 0)
        self.assertEqual(self.flow.evict(10)["evicted"], persisted["persisted"])
        self.assertEqual(self.flow.mem.size, 0)
        self.flow.record_flush()
        self.assertEqual(self.flow.has("kv", labels), [True] * len(labels))

        tensor.fill_(-1)
        tasks = self.flow.get("kv", {"layer0": tensor}, 1, indices, labels)
        self.flow.get_wait(tasks)
        self.assertGreater(self.flow.mem.hits_in_storage, 0)
        for index in indices:
            self.assertTrue(torch.equal(tensor.select(1, index), original.select(1, index)))
        self.assertEqual(self.flow.status()["pool_in_use"], 0)

    def test_kvstore_layerwise_save_and_restore(self):
        values = torch.arange(2 * 7 * 2 * 32 * 64).reshape(2, 7, 2, 32, 64) % 17
        tensors = {"layer0": values.to(torch.bfloat16), "layer1": (values + 32).to(torch.bfloat16)}
        with (
            patch.object(envs, "IAXL_DDR_POOL_SIZE_GB", 0.01),
            patch.object(envs, "IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS", 1),
            patch("iaxl.kvstore.kvstore.start_mgmt_server", return_value=None),
        ):
            store = KVStore("layerwise", block_dim=1, kv_caches=tensors)
        self.addCleanup(store.stop)
        labels = ["block5", "block1"]
        indices = [5, 1]
        expected = {name: tensor.clone() for name, tensor in tensors.items()}
        for name, tensor in tensors.items():
            tasks = store.put(indices, labels, layer_names=[name])
            store.put_wait(tasks)
            tensor.fill_(-1)
        self.assertEqual(store.has(labels), [True, True])
        tasks = store.get(indices, labels)
        for name, tensor in tensors.items():
            store.get_wait(tasks, layer_names=[name])
            for index in indices:
                self.assertTrue(torch.equal(tensor.select(1, index), expected[name].select(1, index)))
        self.assertEqual(store.tensorzip.status()["pool_in_use"], 0)

    @unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
    def test_connector_block_axis(self):
        with patch("iaxl.setup_root_logger"):
            from kvshrink.kvshrink_connector import KVShrinkConnector

        connector = object.__new__(KVShrinkConnector)
        connector.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        )
        connector.model_config = SimpleNamespace(model="cpu-layout-test")
        connector.use_mla = False
        connector.global_rank = 0
        connector.tp_size = 1
        cases = (
            ("cpu", (2, 2, 4, 32, 64), 1),
            ("cpu", (2, 7, 4, 32, 64), 1),
            ("cuda", (7, 2, 32, 4, 64), 0),
            ("cuda", (2, 7, 32, 4, 64), 1),
            ("xpu", (2, 7, 32, 4, 64), 1),
        )
        for device, shape, expected_axis in cases:
            with self.subTest(device=device, shape=shape):
                connector.vllm_device = device
                with patch("kvshrink.kvshrink_connector.KVStore") as store:
                    connector.register_kv_caches({"layer0": torch.empty(shape)})
                self.assertEqual(store.call_args.kwargs["block_dim"], expected_axis)

    @unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
    def test_connector_warns_on_overlapping_cpu_sets(self):
        with patch("iaxl.setup_root_logger"):
            from kvshrink import kvshrink_connector

        connector = object.__new__(kvshrink_connector.KVShrinkConnector)
        connector.vllm_device = "cpu"
        connector.global_rank = 0
        connector.tp_size = 1
        cases = (
            ({"IAXL_CPU_AFFINITY": ""}, "0-31", "IAXL_CPU_AFFINITY is unset"),
            ({"IAXL_CPU_AFFINITY": "30-35"}, "0-31", "overlaps rank 0 inference CPUs on [30, 31]"),
        )
        for environment, bind, message in cases:
            with self.subTest(message=message):
                with (
                    patch.dict(os.environ, environment),
                    patch.object(kvshrink_connector.envs, "VLLM_CPU_OMP_THREADS_BIND", bind),
                    patch.object(kvshrink_connector.torch_ext, "codec_threads", 1),
                    self.assertLogs(kvshrink_connector.logger, level="WARNING") as logs,
                ):
                    connector._bind_cpu_affinity()
                self.assertIn(message, "\n".join(logs.output))
        with (
            patch.dict(os.environ, {"IAXL_CPU_AFFINITY": "32-35"}),
            patch.object(kvshrink_connector.envs, "VLLM_CPU_OMP_THREADS_BIND", "0-31"),
            patch.object(kvshrink_connector.torch_ext, "codec_threads", 4),
            self.assertNoLogs(kvshrink_connector.logger, level="WARNING"),
        ):
            connector._bind_cpu_affinity()

    def test_connector_refuses_multiple_codec_threads_on_inference_cores(self):
        # Measured: >1 codec thread sharing GEMM cores halves throughput and triples TPOT.
        with patch("iaxl.setup_root_logger"):
            from kvshrink import kvshrink_connector

        connector = object.__new__(kvshrink_connector.KVShrinkConnector)
        connector.vllm_device = "cpu"
        connector.global_rank = 0
        connector.tp_size = 1
        for environment, message in (
            ({"IAXL_CPU_AFFINITY": ""}, "IAXL_CPU_AFFINITY is unset"),
            ({"IAXL_CPU_AFFINITY": "30-35"}, "overlaps rank 0 inference CPUs on [30, 31]"),
        ):
            with self.subTest(message=message):
                with (
                    patch.dict(os.environ, environment),
                    patch.object(kvshrink_connector.envs, "VLLM_CPU_OMP_THREADS_BIND", "0-31"),
                    patch.object(kvshrink_connector.torch_ext, "codec_threads", 2),
                    self.assertRaisesRegex(ValueError, "2 codec threads") as raised,
                ):
                    connector._bind_cpu_affinity()
                self.assertIn(message, str(raised.exception))

    @unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
    def test_connector_drains_preempted_requests(self):
        with patch("iaxl.setup_root_logger"):
            from kvshrink import kvshrink_connector

        values = torch.arange(2 * 7 * 2 * 32 * 64).reshape(2, 7, 2, 32, 64) % 29
        tensors = {"layer0": values.to(torch.bfloat16)}
        with (
            patch.object(envs, "IAXL_DDR_POOL_SIZE_GB", 0.01),
            patch("iaxl.kvstore.kvstore.start_mgmt_server", return_value=None),
        ):
            store = KVStore("preempt", block_dim=1, kv_caches=tensors)
        self.addCleanup(store.stop)

        connector = object.__new__(kvshrink_connector.KVShrinkConnector)
        connector.kvstore = store
        preempted_put = store.put([1, 2], ["p1", "p2"])
        running_put = store.put([3], ["r3"])
        preempted_get = store.get([1, 2], ["p1", "p2"])
        connector._current_put_tasks = {"preempted": [preempted_put], "running": [running_put]}
        connector._pending_load_tasks = {}
        connector._pending_load_layers = {}
        connector._early_promoted_tasks = {}
        connector._active_promoted_tasks = {"preempted": preempted_get}

        metadata = kvshrink_connector.KVShrinkConnectorMetadata(
            reqs_to_load=kvshrink_connector.RequestMetadata(),
            reqs_to_save=kvshrink_connector.RequestMetadata(),
            preempted_req_ids={"preempted"},
        )
        connector.handle_preemptions(metadata)

        self.assertTrue(all(task.ctx is None for task in preempted_put.values()))
        self.assertTrue(all(task.ctx is None for task in preempted_get.values()))
        self.assertEqual(list(connector._current_put_tasks), ["running"])
        self.assertEqual(connector._active_promoted_tasks, {})
        self.assertTrue(all(task.ctx is not None for task in running_put.values()))
        store.put_wait(running_put)

    @unittest.skipUnless(os.environ.get("IAXL_TEST_VLLM") == "1", "opt-in model smoke test")
    def test_vllm_cold_and_warm_inference(self):
        load_layers = os.environ.get("IAXL_TEST_ASYNC_LOAD_LAYERS", "-1")
        allowed = sorted(os.sched_getaffinity(0))
        self.enterContext(patch.dict(os.environ, {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_CPU_OMP_THREADS_BIND": "nobind",
            "VLLM_CPU_KVCACHE_SPACE": "1",
            # The connector refuses more than one codec thread without a pinned IAXL core set.
            "IAXL_CPU_AFFINITY": ",".join(map(str, allowed[-2:])),
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED": "0" if load_layers == "0" else "1",
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS": "-1" if load_layers == "0" else load_layers,
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC": "0",
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP": "0-:0",
        }))
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig

        model = os.environ.get("IAXL_TEST_MODEL", "Qwen/Qwen3-0.6B")
        if not Path(model).is_dir():
            model = snapshot_download(model, local_files_only=True)
        with socket.socket() as controller_socket, socket.socket() as worker_socket:
            controller_socket.bind(("127.0.0.1", 0))
            worker_socket.bind(("127.0.0.1", 0))
            controller_port = controller_socket.getsockname()[1]
            worker_port = worker_socket.getsockname()[1]
        self.enterContext(patch.object(envs, "IAXL_API_CONTROLLER_PORT", controller_port))
        self.enterContext(patch.object(envs, "IAXL_API_WORKER_BASE_PORT", worker_port))
        self.enterContext(patch.object(envs, "IAXL_DDR_POOL_SIZE_GB", 0.05))
        self.enterContext(patch.object(envs, "IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS", 0))
        llm = LLM(
            model=model,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            block_size=32,
            enable_prefix_caching=False,
            seed=0,
            kv_transfer_config=KVTransferConfig(
                kv_connector="KVShrinkConnector",
                kv_connector_module_path="kvshrink.kvshrink_connector",
                kv_role="kv_both",
            ),
        )
        try:
            def status():
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{worker_port}/v1/cache/status", timeout=5
                ) as response:
                    return json.load(response)

            def settled_status(timeout=30.0):
                # PUTs retire asynchronously behind the response; wait for the write count to
                # stop moving so the assertions see the drained cache, as the harness does.
                import time

                deadline = time.monotonic() + timeout
                last = status()
                quiet_since = time.monotonic()
                while time.monotonic() < deadline:
                    time.sleep(0.2)
                    current = status()
                    if current["puts"] != last["puts"]:
                        last, quiet_since = current, time.monotonic()
                    elif current["puts"] > 0 and time.monotonic() - quiet_since >= 1.0:
                        return current
                return last

            prompt = "Summarize these facts in one sentence. " + (
                "The CPU executes the language model. QAT compresses and decompresses "
                "the reusable key and value cache. "
            ) * 5
            sampling = SamplingParams(temperature=0, max_tokens=4)
            cold = llm.generate([prompt], sampling, use_tqdm=False)[0]
            after_cold = settled_status()
            warm = llm.generate([prompt], sampling, use_tqdm=False)[0]
            after_warm = settled_status()
            self.assertGreater(after_cold["puts"], 0)
            self.assertLess(after_cold["total_zip_bytes"], after_cold["total_unzip_bytes"])
            restored = after_warm["hits"] - after_cold["hits"]
            self.assertGreater(restored, 0)
            self.assertEqual(cold.outputs[0].token_ids, warm.outputs[0].token_ids)
            print(json.dumps({
                "codec": ZIP_BACKEND,
                "cold_puts": after_cold["puts"],
                "restored_chunks": restored,
                "raw_bytes": after_cold["total_unzip_bytes"],
                "stored_bytes": after_cold["total_zip_bytes"],
                "matching_token_ids": warm.outputs[0].token_ids,
            }))
        finally:
            llm.llm_engine.engine_core.shutdown()

    def test_has_only_mode(self):
        controller = KVFlow("roundtrip", cache_size_gb=0)
        self.addCleanup(controller.stop)
        self.assertTrue(controller.status()["has_only_mode"])
        self.assertEqual(controller.has("kv", ["missing"]), [False])


if __name__ == "__main__":
    unittest.main()