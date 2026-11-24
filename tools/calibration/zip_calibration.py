#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rank compression segment sizes by decompression throughput, before vLLM starts.

Calibrates QAT when it is enabled (the result also serves IAA), otherwise IAA. One process per vLLM
worker (TP_SIZE * DP_SIZE) is bound like the KVShrink connector binds that worker, and all workers
decompress at the same time. Prints the sizes in KiB, fastest first, comma separated, on stdout for
examples/kvshrink-vllm-serve.sh.

Example:
    source setvars.sh
    LD_PRELOAD=$IAXL_LD_PRELOAD python3 tools/calibration/zip_calibration.py
"""

import math
import multiprocessing
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import torch

SEGMENT_SIZES_KB = (16, 32, 64, 128, 256, 512, 1024, 2048)
MOCK_SHAPE = (2, 256, 512, 8, 128)
MOCK_DTYPE = "bf16"
WARMUP_DECOMPRESSES = 2
TIMED_DECOMPRESSES = 3
START_SLACK_S = 0.002
WORKER_POLL_S = 1.0
BACKEND_ENABLE_ENV = {
    "QAT": "IAXL_QAT_ZIP_ENABLE",
    "IAA": "IAXL_IAA_ZIP_ENABLE",
    "CPU": "IAXL_CPU_ZIP_ENABLE",
}
BACKEND_INSTANCE_ENV = {"QAT": "IAXL_QAT_INSTANCE_NUM", "IAA": "IAXL_IAA_INSTANCE_NUM"}

SEED = 42
FP8_MAX = 448.0
INT4_QMAX = 7.0
INT4_GROUP_SIZE = 128


@dataclass
class WorkerTimes:
    """Fastest timed decompression of one worker at one segment size."""
    decompressed_bytes: int
    decompress_ms: float


@dataclass
class SegmentSpeed:
    segment_kb: int
    decompress_gbps: float


def quantize_kv(values: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Quantize a float KV tensor [2, ...] the way vLLM does."""
    if dtype_name == "bf16":
        return values.to(torch.bfloat16).contiguous()

    values = values.float()
    if dtype_name == "fp8_e4m3":
        # vLLM stores one per-tensor k_scale/v_scale per layer: scale = amax / 448.
        reduce_dims = tuple(range(1, values.ndim))
        scale = values.abs().amax(dim=reduce_dims, keepdim=True) / FP8_MAX
        scaled = (values / scale.clamp_min(1e-12)).clamp_(-FP8_MAX, FP8_MAX)
        return scaled.to(torch.float8_e4m3fn).contiguous()

    # vLLM int4: symmetric group-wise uint4b8, scale = amax / 7, low nibble first.
    head_dim = values.shape[-1]
    group = INT4_GROUP_SIZE if head_dim % INT4_GROUP_SIZE == 0 else head_dim
    grouped = values.unflatten(-1, (-1, group))
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / INT4_QMAX).clamp_min(1e-6)
    codes = (grouped / scale).round_().clamp_(-8, 7).add_(8)
    codes = codes.to(torch.uint8).flatten(-2)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def generate_mock_kv_cache(shape: Tuple[int, ...], dtype_name: str) -> torch.Tensor:
    kv_count, cache_blocks, block_tokens, kv_heads, head_dim = shape
    latent_dim = min(32, max(8, head_dim // 8))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(SEED)

    block_context = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device=device,
        generator=generator,
    )
    token_innovations = torch.randn(
        (cache_blocks, block_tokens, kv_heads, latent_dim),
        device=device,
        generator=generator,
    )
    block_drift = torch.randn(
        (cache_blocks, 1, kv_heads, latent_dim),
        device=device,
        generator=generator,
    )
    token_positions = torch.linspace(-1.0, 1.0, block_tokens, device=device).view(
        1, block_tokens, 1, 1
    )
    hidden_states = (
        0.65 * block_context
        + 0.70 * token_innovations
        + 0.15 * token_positions * block_drift
    )

    projections = torch.randn(
        (kv_count, kv_heads, latent_dim, head_dim),
        device=device,
        generator=generator,
    ) / math.sqrt(latent_dim)
    kv_cache = torch.einsum("bthr,khrd->kbthd", hidden_states, projections)

    channel_scales = (
        torch.randn(
            (kv_count, 1, 1, kv_heads, head_dim),
            device=device,
            generator=generator,
        )
        .mul_(0.25)
        .exp_()
    )
    kv_cache.mul_(channel_scales)
    return quantize_kv(kv_cache, dtype_name)


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def tested_backend() -> str:
    if env_truthy("IAXL_QAT_ZIP_ENABLE"):
        return "QAT"
    if env_truthy("IAXL_IAA_ZIP_ENABLE"):
        return "IAA"
    raise ValueError("zip calibration needs IAXL_QAT_ZIP_ENABLE or IAXL_IAA_ZIP_ENABLE")


class DecompressTimer:
    """Times decompression on this worker, each decompression started with all workers."""

    def __init__(self, rank: int, barrier, deadline) -> None:
        self.rank = rank
        self.barrier = barrier
        self.deadline = deadline
        self.original = generate_mock_kv_cache(MOCK_SHAPE, MOCK_DTYPE).cpu().flatten()
        self.output = torch.empty_like(self.original)

    def measure(self, segment_kb: int) -> WorkerTimes:
        from iaxl import torch_ext

        elements = segment_kb * 1024 // self.original.element_size()
        count = self.original.numel() // elements
        inputs = []
        outputs = []
        for index in range(count):
            inputs.append(self.original[index * elements:(index + 1) * elements])
            outputs.append(self.output[index * elements:(index + 1) * elements])
        blocks = torch_ext.compress_tensors(inputs)
        decompress_ms = float("inf")
        for decompress_index in range(WARMUP_DECOMPRESSES + TIMED_DECOMPRESSES):
            self.start_together()
            start_time = time.perf_counter()
            torch_ext.decompress_tensors(blocks, outputs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if decompress_index >= WARMUP_DECOMPRESSES:
                decompress_ms = min(decompress_ms, elapsed_ms)
        return WorkerTimes(count * elements * self.original.element_size(), decompress_ms)

    def start_together(self) -> None:
        self.barrier.wait()
        if self.rank == 0:
            self.deadline.value = time.monotonic() + START_SLACK_S
        self.barrier.wait()
        while time.monotonic() < self.deadline.value:
            pass


def worker(rank: int, num_workers: int, gpu: str, backend: str, barrier, deadline,
           times_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    for other_backend, enable_env in BACKEND_ENABLE_ENV.items():
        if other_backend == backend:
            os.environ[enable_env] = "1"
        else:
            os.environ[enable_env] = "0"
    os.environ["IAXL_OMP_THREAD_NUM"] = os.environ[BACKEND_INSTANCE_ENV[backend]]
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

    bind_cpu_affinity(rank, num_workers, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)
    timer = DecompressTimer(rank, barrier, deadline)
    for segment_kb in SEGMENT_SIZES_KB:
        times_queue.put(timer.measure(segment_kb))


def collect_worker_times(workers: list, times_queue) -> list[WorkerTimes]:
    """Returns one WorkerTimes per worker; raises RuntimeError if a worker exited."""
    worker_times = []
    while len(worker_times) < len(workers):
        try:
            worker_times.append(times_queue.get(timeout=WORKER_POLL_S))
        except queue.Empty:
            for process in workers:
                if process.exitcode is not None:
                    raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
    return worker_times


def main() -> None:
    started = time.monotonic()
    backend = tested_backend()
    num_workers = int(os.environ["TP_SIZE"]) * int(os.environ["DP_SIZE"])
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = []
    if visible_gpus:
        gpus = visible_gpus.split(",")
    else:
        for index in range(num_workers):
            gpus.append(str(index))

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(num_workers)
    deadline = context.Value("d", 0.0, lock=False)
    times_queue = context.Queue()
    workers = []
    for rank in range(num_workers):
        process = context.Process(
            target=worker,
            args=(rank, num_workers, gpus[rank], backend, barrier, deadline, times_queue),
            name=f"rank{rank}",
        )
        process.start()
        workers.append(process)

    speeds = []
    try:
        for segment_kb in SEGMENT_SIZES_KB:
            gbps_total = 0.0
            worker_times = collect_worker_times(workers, times_queue)
            for times in worker_times:
                gbps_total += times.decompressed_bytes / times.decompress_ms / 1e6
            speeds.append(SegmentSpeed(segment_kb, gbps_total / len(worker_times)))
    except RuntimeError as error:
        for process in workers:
            process.terminate()
        sys.exit(f"[zip_calibration] failed: {error}")
    for process in workers:
        process.join()

    speeds.sort(key=lambda speed: speed.decompress_gbps, reverse=True)
    ranked_kb = []
    summary_parts = []
    for speed in speeds:
        ranked_kb.append(str(speed.segment_kb))
        summary_parts.append(f"{speed.segment_kb} KB {speed.decompress_gbps:.1f}")
    print(
        f"[zip_calibration] {backend} segment sizes by decompress GB/s: "
        f"{', '.join(summary_parts)}, {num_workers} workers, {time.monotonic() - started:.1f} s",
        file=sys.stderr,
    )
    print(",".join(ranked_kb))


if __name__ == "__main__":
    main()
