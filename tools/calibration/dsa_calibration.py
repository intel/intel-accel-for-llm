#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Find the contiguous copy sizes from which CUDA batch copies beat DSA, before vLLM starts.

One process per vLLM worker (TP_SIZE * DP_SIZE) is bound like the KVShrink connector binds that
worker, and all workers copy at the same time. Prints "<H2D KB> <D2H KB>" on stdout for
examples/kvshrink-vllm-serve.sh; 0 means DSA stayed faster up to 4 MB.

Example:
    source setvars.sh
    LD_PRELOAD=$IAXL_LD_PRELOAD python3 tools/calibration/dsa_calibration.py
"""

import multiprocessing
import os
import queue
import sys
import time
from dataclasses import dataclass

# 128 MiB per copy, 2 warmup + 10 timed copies, keep the fastest: most stable in the 6000d test.
COPY_BYTES = 128 << 20
FIRST_COPY_SIZE = 256 << 10
COPY_SIZE_STEP = 128 << 10
LAST_COPY_SIZE = 4 << 20
WARMUP_COPIES = 2
TIMED_COPIES = 10
START_SLACK_S = 0.002  # covers barrier wake-up jitter before the shared start time
WORKER_POLL_S = 1.0
DIRECTIONS = ("H2D", "D2H")


@dataclass
class WorkerTimes:
    """Fastest timed copy of one worker at one copy size."""
    rank: int
    copied_bytes: int
    cuda_batch_ms: float
    dsa_ms: float


def copy_sizes() -> list[int]:
    sizes = []
    size = FIRST_COPY_SIZE
    while size <= LAST_COPY_SIZE:
        sizes.append(size)
        size += COPY_SIZE_STEP
    return sizes


def pointer_operand(address: int):
    from cuda.bindings import runtime as cuda_runtime

    operand = cuda_runtime.cudaMemcpy3DOperand()
    operand.type = cuda_runtime.cudaMemcpy3DOperandType.cudaMemcpyOperandTypePointer
    union = operand.op
    pointer = union.ptr
    pointer.ptr = address
    pointer.rowLength = 0
    pointer.layerHeight = 0
    union.ptr = pointer
    operand.op = union
    return operand


def build_batch_operations(direction: str, host_blocks, gpu_blocks, gpu_order: list[int],
                           copy_size: int) -> list:
    """One cudaMemcpy3DBatchAsync operation per block: host block i <-> GPU block gpu_order[i]."""
    from cuda.bindings import runtime as cuda_runtime

    stream_order = cuda_runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    operations = []
    for host_index in range(len(gpu_order)):
        host_address = host_blocks.data_ptr() + host_index * copy_size
        gpu_address = gpu_blocks.data_ptr() + gpu_order[host_index] * copy_size
        operation = cuda_runtime.cudaMemcpy3DBatchOp()
        if direction == "H2D":
            operation.src = pointer_operand(host_address)
            operation.dst = pointer_operand(gpu_address)
        else:
            operation.src = pointer_operand(gpu_address)
            operation.dst = pointer_operand(host_address)
        extent = cuda_runtime.cudaExtent()
        extent.width = copy_size
        extent.height = 1
        extent.depth = 1
        operation.extent = extent
        operation.srcAccessOrder = stream_order
        operation.flags = 0
        operations.append(operation)
    return operations


def launch_batch(operations: list, stream) -> None:
    from cuda.bindings import runtime as cuda_runtime

    result = cuda_runtime.cudaMemcpy3DBatchAsync(len(operations), operations, 0, stream.cuda_stream)
    if result[0] != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy3DBatchAsync failed: {result[0]}")


class CopyTimer:
    """Times CUDA batch and DSA copies on this worker's GPU, each copy started with all workers."""

    def __init__(self, rank: int, barrier, deadline) -> None:
        import torch

        self.rank = rank
        self.barrier = barrier
        self.deadline = deadline
        self.host_buffer = torch.empty(COPY_BYTES // 2, dtype=torch.int16, pin_memory=True)
        self.gpu_buffer = torch.empty(COPY_BYTES // 2, dtype=torch.int16, device="cuda")
        self.stream = torch.cuda.Stream()

    def measure(self, direction: str, copy_size: int) -> WorkerTimes:
        import torch
        from iaxl.tensor_ops import SliceCopier

        block_count = COPY_BYTES // copy_size
        block_elements = copy_size // 2
        used_elements = block_count * block_elements
        host_blocks = self.host_buffer[:used_elements].view(block_count, block_elements)
        gpu_blocks = self.gpu_buffer[:used_elements].view(block_count, block_elements)
        # GPU blocks are scattered like a real KV cache; the fixed seed repeats the order every run.
        gpu_order = torch.randperm(block_count, generator=torch.Generator().manual_seed(0)).tolist()

        operations = build_batch_operations(direction, host_blocks, gpu_blocks, gpu_order,
                                            copy_size)
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        cuda_batch_ms = float("inf")
        for copy_index in range(WARMUP_COPIES + TIMED_COPIES):
            self.start_together()
            start_event.record(self.stream)
            launch_batch(operations, self.stream)
            stop_event.record(self.stream)
            stop_event.synchronize()
            if copy_index >= WARMUP_COPIES:
                cuda_batch_ms = min(cuda_batch_ms, start_event.elapsed_time(stop_event))

        if direction == "H2D":
            copier = SliceCopier(host_blocks, 0, gpu_order, out=gpu_blocks)
        else:
            copier = SliceCopier(gpu_blocks, 0, gpu_order, out=host_blocks)
        dsa_ms = float("inf")
        for copy_index in range(WARMUP_COPIES + TIMED_COPIES):
            self.start_together()
            start_time = time.perf_counter()
            copier.copy()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if copy_index >= WARMUP_COPIES:
                dsa_ms = min(dsa_ms, elapsed_ms)
        copier.close()
        return WorkerTimes(self.rank, block_count * copy_size, cuda_batch_ms, dsa_ms)

    def start_together(self) -> None:
        # The first barrier waits for all previous copies; the deadline absorbs wake-up jitter.
        self.barrier.wait()
        if self.rank == 0:
            self.deadline.value = time.monotonic() + START_SLACK_S
        self.barrier.wait()
        while time.monotonic() < self.deadline.value:
            pass


def worker(rank: int, num_workers: int, gpu: str, barrier, deadline, times_queue,
           stop_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # must precede the torch import
    # Only the main process writes to stdout, which the serve script parses.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

    bind_cpu_affinity(rank, num_workers, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)
    timer = CopyTimer(rank, barrier, deadline)
    for direction in DIRECTIONS:
        for copy_size in copy_sizes():
            times_queue.put(timer.measure(direction, copy_size))
            if stop_queue.get():  # the main process found this direction's crossover
                break


def collect_worker_times(workers: list, times_queue) -> list[WorkerTimes]:
    """Returns one WorkerTimes per worker in rank order; raises RuntimeError if a worker exited."""
    times_by_rank: dict[int, WorkerTimes] = {}
    while len(times_by_rank) < len(workers):
        try:
            worker_times = times_queue.get(timeout=WORKER_POLL_S)
        except queue.Empty:
            for process in workers:
                if process.exitcode is not None:
                    raise RuntimeError(f"{process.name} exited with code {process.exitcode}")
            continue
        times_by_rank[worker_times.rank] = worker_times
    ordered_times = []
    for rank in range(len(workers)):
        ordered_times.append(times_by_rank[rank])
    return ordered_times


def calibrate_direction(workers: list, times_queue, stop_queues: list) -> int:
    """Returns the crossover in KB of the direction the workers measure now, or 0 when DSA stays
    faster up to LAST_COPY_SIZE."""
    for copy_size in copy_sizes():
        worker_times = collect_worker_times(workers, times_queue)
        cuda_batch_gbps_total = 0.0
        dsa_gbps_total = 0.0
        for times in worker_times:
            cuda_batch_gbps_total += times.copied_bytes / times.cuda_batch_ms / 1e6
            dsa_gbps_total += times.copied_bytes / times.dsa_ms / 1e6
        cuda_batch_gbps = cuda_batch_gbps_total / len(worker_times)
        dsa_gbps = dsa_gbps_total / len(worker_times)
        crossover_found = cuda_batch_gbps > dsa_gbps
        for stop_queue in stop_queues:
            stop_queue.put(crossover_found)
        if crossover_found:
            return copy_size >> 10
    return 0


def main() -> None:
    started = time.monotonic()
    num_workers = int(os.environ["TP_SIZE"]) * int(os.environ["DP_SIZE"])
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = []
    if visible_gpus:
        gpus = visible_gpus.split(",")
    else:
        for index in range(num_workers):
            gpus.append(str(index))

    context = multiprocessing.get_context("spawn")  # each worker initializes CUDA itself
    barrier = context.Barrier(num_workers)
    deadline = context.Value("d", 0.0, lock=False)
    times_queue = context.Queue()
    stop_queues = []
    workers = []
    for rank in range(num_workers):
        stop_queue = context.Queue()
        stop_queues.append(stop_queue)
        process = context.Process(
            target=worker,
            args=(rank, num_workers, gpus[rank], barrier, deadline, times_queue, stop_queue),
            name=f"rank{rank}",
        )
        process.start()
        workers.append(process)

    crossover_kb = {}
    try:
        for direction in DIRECTIONS:
            crossover_kb[direction] = calibrate_direction(workers, times_queue, stop_queues)
    except RuntimeError as error:
        for process in workers:
            process.terminate()
        sys.exit(f"[dsa_calibration] failed: {error}")
    for process in workers:
        process.join()
    print(
        f"[dsa_calibration] crossover H2D {crossover_kb['H2D']} KB, D2H {crossover_kb['D2H']} KB "
        f"(0 = DSA faster up to {LAST_COPY_SIZE >> 10} KB), {num_workers} workers, "
        f"{time.monotonic() - started:.1f} s",
        file=sys.stderr,
    )
    print(f"{crossover_kb['H2D']} {crossover_kb['D2H']}")


if __name__ == "__main__":
    main()
