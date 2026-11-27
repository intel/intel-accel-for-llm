#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Measure accelerator settings before vLLM starts; run by accel_auto_config in tools/auto_config.sh.

DSA (IAXL_DSA_GD_ENABLE): per direction, the smallest block size from which CUDA batch copies beat
DSA, found by bisection; one process per vLLM worker whose GPU is on rank 0's socket, bound like the
KVShrink connector binds that worker, all copying at the same time.
QAT (IAXL_KV_COMPRESSION and IAXL_QAT_ZIP_ENABLE): segment sizes ranked by decompression throughput,
in one process bound like rank 0; the list also serves IAA. With only IAA: half of IAXL_ZIP_SRC_CAP.
A variable that is already set is kept. Prints the variables to export on one line of stdout, e.g.
"IAXL_DSA_H2D_CROSSOVER_KB=256 IAXL_DSA_D2H_CROSSOVER_KB=256 IAXL_ZIP_SEGMENT_SIZE_KB=1024,512,...".

Example (in the container setvars.sh has already set IAXL_DSA_H2D_CROSSOVER_KB,
IAXL_DSA_D2H_CROSSOVER_KB and IAXL_ZIP_SEGMENT_SIZE_KB, so unset them to measure again):
    source setvars.sh
    env -u IAXL_DSA_H2D_CROSSOVER_KB -u IAXL_DSA_D2H_CROSSOVER_KB -u IAXL_ZIP_SEGMENT_SIZE_KB \
        LD_PRELOAD=$IAXL_LD_PRELOAD python3 tools/accel_auto_config.py
"""

from __future__ import annotations

import math
import multiprocessing
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Tuple

WORKER_POLL_S = 1.0  # how often the main process checks whether a worker died

DSA_BYTES_PER_COPY = 32 * 1024 * 1024
DSA_MIN_BLOCK_KB = 128
DSA_MAX_BLOCK_KB = 4096
DSA_BLOCK_STEP_KB = 128
DSA_WARMUP_COPIES = 2
DSA_TIMED_COPIES = 3
DSA_START_SLACK_S = 0.002  # all ranks start copying 2 ms after the barrier, so ranks woken later still start together
DSA_DIRECTIONS = ("H2D", "D2H")

QAT_SEGMENT_SIZES_KB = (32, 64, 128, 256, 512, 1024)
QAT_SEGMENTS_PER_SIZE = 50
QAT_MOCK_SHAPE = (2, 25, 512, 8, 128)  # bf16: 50 MiB, enough for 50 segments of 1024 KB
QAT_MOCK_DTYPE = "bf16"
QAT_WARMUP_DECOMPRESSES = 2
QAT_TIMED_DECOMPRESSES = 3

# Used by quantize_kv and generate_mock_kv_cache, copied with them from benchmark/kvstore/kvstore_benchmark.py.
SEED = 42
FP8_MAX = 448.0
INT4_QMAX = 7.0
INT4_GROUP_SIZE = 128


@dataclass
class WorkerTimes:
    """Fastest timed copy of one DSA worker at one block size."""
    copied_bytes: int
    cuda_batch_ms: float
    dsa_ms: float


@dataclass
class DsaCrossovers:
    """Block sizes in KB from which CUDA batch copies beat DSA; 0 when DSA stays faster up to 4096 KB."""
    h2d_kb: int
    d2h_kb: int


@dataclass
class SegmentSpeed:
    segment_kb: int
    decompress_gbps: float


def quantize_kv(values: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Quantize a float KV tensor [2, ...] the way vLLM does."""
    import torch

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
    import torch

    kv_count, cache_blocks, block_tokens, kv_heads, head_dim = shape
    latent_dim = min(32, max(8, head_dim // 8))
    device = "cpu"
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


def env_unset(name: str) -> bool:
    return os.environ.get(name, "") == ""


def receive(results, workers: list):
    """Next item from results; raises RuntimeError if a worker exits before sending it."""
    while True:
        try:
            return results.get(timeout=WORKER_POLL_S)
        except queue.Empty:
            for process in workers:
                if process.exitcode is not None:
                    raise RuntimeError(f"{process.name} exited with code {process.exitcode}")


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
                           block_bytes: int) -> list:
    """One cudaMemcpy3DBatchAsync operation per block: host block i <-> GPU block gpu_order[i]."""
    from cuda.bindings import runtime as cuda_runtime

    stream_order = cuda_runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    operations = []
    for host_index in range(len(gpu_order)):
        host_address = host_blocks.data_ptr() + host_index * block_bytes
        gpu_address = gpu_blocks.data_ptr() + gpu_order[host_index] * block_bytes
        operation = cuda_runtime.cudaMemcpy3DBatchOp()
        if direction == "H2D":
            operation.src = pointer_operand(host_address)
            operation.dst = pointer_operand(gpu_address)
        else:
            operation.src = pointer_operand(gpu_address)
            operation.dst = pointer_operand(host_address)
        extent = cuda_runtime.cudaExtent()
        extent.width = block_bytes
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
        self.host_buffer = torch.empty(DSA_BYTES_PER_COPY // 2, dtype=torch.int16, pin_memory=True)
        self.gpu_buffer = torch.empty(DSA_BYTES_PER_COPY // 2, dtype=torch.int16, device="cuda")
        self.stream = torch.cuda.Stream()

    def measure(self, direction: str, block_kb: int) -> WorkerTimes:
        import torch
        from iaxl.tensor_ops import SliceCopier

        block_bytes = block_kb * 1024
        block_count = DSA_BYTES_PER_COPY // block_bytes
        block_elements = block_bytes // 2
        used_elements = block_count * block_elements
        host_blocks = self.host_buffer[:used_elements].view(block_count, block_elements)
        gpu_blocks = self.gpu_buffer[:used_elements].view(block_count, block_elements)
        # GPU blocks are scattered like a real KV cache; the fixed seed repeats the order every run.
        gpu_order = torch.randperm(block_count, generator=torch.Generator().manual_seed(0)).tolist()

        operations = build_batch_operations(direction, host_blocks, gpu_blocks, gpu_order,
                                            block_bytes)
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        cuda_batch_ms = float("inf")
        for copy_index in range(DSA_WARMUP_COPIES + DSA_TIMED_COPIES):
            self.start_together()
            start_event.record(self.stream)
            launch_batch(operations, self.stream)
            stop_event.record(self.stream)
            stop_event.synchronize()
            if copy_index >= DSA_WARMUP_COPIES:
                cuda_batch_ms = min(cuda_batch_ms, start_event.elapsed_time(stop_event))

        if direction == "H2D":
            copier = SliceCopier(host_blocks, 0, gpu_order, out=gpu_blocks)
        else:
            copier = SliceCopier(gpu_blocks, 0, gpu_order, out=host_blocks)
        dsa_ms = float("inf")
        for copy_index in range(DSA_WARMUP_COPIES + DSA_TIMED_COPIES):
            self.start_together()
            start_time = time.perf_counter()
            copier.copy()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if copy_index >= DSA_WARMUP_COPIES:
                dsa_ms = min(dsa_ms, elapsed_ms)
        copier.close()
        return WorkerTimes(block_count * block_bytes, cuda_batch_ms, dsa_ms)

    def start_together(self) -> None:
        # The first barrier waits for all previous copies; the deadline absorbs wake-up jitter.
        self.barrier.wait()
        if self.rank == 0:
            self.deadline.value = time.monotonic() + DSA_START_SLACK_S
        self.barrier.wait()
        while time.monotonic() < self.deadline.value:
            pass


def dsa_worker(rank: int, num_workers: int, gpu: str, barrier, deadline, times_queue,
               block_kb_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # read at the first CUDA call, which comes after the fork
    # Only the main process writes to stdout, which accel_auto_config parses.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

    bind_cpu_affinity(rank, num_workers, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)
    timer = CopyTimer(rank, barrier, deadline)
    for direction in DSA_DIRECTIONS:
        block_kb = block_kb_queue.get()
        while block_kb != 0:
            times_queue.put(timer.measure(direction, block_kb))
            block_kb = block_kb_queue.get()


def cuda_batch_faster(block_kb: int, workers: list, times_queue, block_kb_queues: list) -> bool:
    """Has every worker copy blocks of block_kb; True when CUDA batch GB/s summed over workers beats DSA's."""
    for block_kb_queue in block_kb_queues:
        block_kb_queue.put(block_kb)
    cuda_batch_gbps = 0.0
    dsa_gbps = 0.0
    for _ in range(len(workers)):
        times = receive(times_queue, workers)
        cuda_batch_gbps += times.copied_bytes / times.cuda_batch_ms / 1e6
        dsa_gbps += times.copied_bytes / times.dsa_ms / 1e6
    return cuda_batch_gbps > dsa_gbps


def calibrate_direction(workers: list, times_queue, block_kb_queues: list) -> int:
    """Crossover in KB: smallest block size where CUDA batch beats DSA; 0 if DSA wins at DSA_MAX_BLOCK_KB."""
    candidates_kb = list(range(DSA_MIN_BLOCK_KB, DSA_MAX_BLOCK_KB + 1, DSA_BLOCK_STEP_KB))
    crossover_kb = 0
    if cuda_batch_faster(DSA_MAX_BLOCK_KB, workers, times_queue, block_kb_queues):
        low_index = 0  # DSA_MIN_BLOCK_KB is measured only if the range narrows down to it
        high_index = len(candidates_kb) - 1
        while low_index < high_index:
            middle_index = (low_index + high_index) // 2
            if cuda_batch_faster(candidates_kb[middle_index], workers, times_queue, block_kb_queues):
                high_index = middle_index
            else:
                low_index = middle_index + 1
        crossover_kb = candidates_kb[high_index]
    for block_kb_queue in block_kb_queues:
        block_kb_queue.put(0)
    return crossover_kb


def socket_of_numa_node(numa_node: int) -> int:
    with open(f"/sys/devices/system/node/node{numa_node}/cpulist") as cpulist_file:
        first_cpu = cpulist_file.read().split(",")[0].split("-")[0].strip()
    with open(f"/sys/devices/system/cpu/cpu{first_cpu}/topology/physical_package_id") as package_file:
        return int(package_file.read())


def gpu_sockets(gpus: list[str]) -> list[int]:
    """Socket of each CUDA_VISIBLE_DEVICES entry (index or UUID), found through NVML without initializing CUDA."""
    import pynvml

    pynvml.nvmlInit()
    sockets = []
    for gpu in gpus:
        try:
            if gpu.startswith("GPU-"):
                handle = pynvml.nvmlDeviceGetHandleByUUID(gpu)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
            bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
        except (pynvml.NVMLError, ValueError) as error:
            raise RuntimeError(f"cannot find GPU {gpu} of CUDA_VISIBLE_DEVICES: {error}") from error
        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode()
        pci_address = bus_id.lower()[-12:]  # NVML gives "00000000:38:00.0"; sysfs uses "0000:38:00.0"
        with open(f"/sys/bus/pci/devices/{pci_address}/numa_node") as numa_file:
            sockets.append(socket_of_numa_node(int(numa_file.read())))
    pynvml.nvmlShutdown()
    return sockets


def dsa_socket(rank: int) -> int:
    """Socket of the first DSA work queue KVSHRINK_DSA_DEVICES gives this rank (wq2.0 is on dsa2)."""
    first_wq = os.environ["KVSHRINK_DSA_DEVICES"].split("|")[rank].split(",")[0].strip()
    numa_path = f"/sys/bus/dsa/devices/dsa{first_wq[len('wq'):].split('.')[0]}/numa_node"
    if not os.path.exists(numa_path):
        raise RuntimeError(f"rank{rank}: DSA work queue {first_wq} of KVSHRINK_DSA_DEVICES does not exist")
    with open(numa_path) as numa_file:
        return socket_of_numa_node(int(numa_file.read()))


def calibrate_dsa(context, num_workers: int) -> DsaCrossovers:
    started = time.monotonic()
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = []
    if visible_gpus:
        gpus = visible_gpus.split(",")
    else:
        for index in range(num_workers):
            gpus.append(str(index))
    sockets = gpu_sockets(gpus[:num_workers])
    for rank in range(num_workers):
        wq_socket = dsa_socket(rank)
        if wq_socket != sockets[rank]:
            wq = os.environ["KVSHRINK_DSA_DEVICES"].split("|")[rank]
            raise RuntimeError(f"rank{rank}: GPU {gpus[rank]} is on socket {sockets[rank]} but DSA {wq} is on "
                               f"socket {wq_socket}; fix KVSHRINK_DSA_DEVICES")
    measured_ranks = []
    for rank in range(num_workers):
        if sockets[rank] == sockets[0]:
            measured_ranks.append(rank)
    barrier = context.Barrier(len(measured_ranks))
    deadline = context.Value("d", 0.0, lock=False)
    times_queue = context.Queue()
    block_kb_queues = []
    workers = []
    for rank in measured_ranks:
        block_kb_queue = context.Queue()
        block_kb_queues.append(block_kb_queue)
        process = context.Process(
            target=dsa_worker,
            args=(rank, num_workers, gpus[rank], barrier, deadline, times_queue, block_kb_queue),
            name=f"dsa_rank{rank}",
        )
        process.start()
        workers.append(process)

    crossover_kb = {}
    try:
        for direction in DSA_DIRECTIONS:
            crossover_kb[direction] = calibrate_direction(workers, times_queue, block_kb_queues)
    except RuntimeError:
        for process in workers:
            process.terminate()
        raise
    for process in workers:
        process.join()
    print(
        f"[accel_auto_config] DSA crossover H2D {crossover_kb['H2D']} KB, D2H {crossover_kb['D2H']} KB "
        f"(0 = DSA faster up to {DSA_MAX_BLOCK_KB} KB), {len(measured_ranks)} of {num_workers} workers "
        f"(socket {sockets[0]}), {time.monotonic() - started:.1f} s",
        file=sys.stderr,
    )
    return DsaCrossovers(crossover_kb["H2D"], crossover_kb["D2H"])


def decompress_gbps(original: torch.Tensor, segment_kb: int) -> float:
    """Fastest timed decompression of QAT_SEGMENTS_PER_SIZE segments of segment_kb, in GB/s."""
    import torch
    from iaxl import torch_ext

    elements = segment_kb * 1024 // original.element_size()
    output = torch.empty(QAT_SEGMENTS_PER_SIZE * elements, dtype=original.dtype)
    inputs = []
    outputs = []
    for index in range(QAT_SEGMENTS_PER_SIZE):
        inputs.append(original[index * elements:(index + 1) * elements])
        outputs.append(output[index * elements:(index + 1) * elements])
    blocks = torch_ext.compress_tensors(inputs)
    decompress_s = float("inf")
    for decompress_index in range(QAT_WARMUP_DECOMPRESSES + QAT_TIMED_DECOMPRESSES):
        start_time = time.perf_counter()
        torch_ext.decompress_tensors(blocks, outputs)
        elapsed_s = time.perf_counter() - start_time
        if decompress_index >= QAT_WARMUP_DECOMPRESSES:
            decompress_s = min(decompress_s, elapsed_s)
    return QAT_SEGMENTS_PER_SIZE * segment_kb * 1024 / decompress_s / 1e9


def qat_worker(num_workers: int, speeds_queue) -> None:
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # Read by iaxl when it loads: QAT alone, with one thread per QAT instance.
    os.environ["IAXL_IAA_ZIP_ENABLE"] = "0"
    os.environ["IAXL_CPU_ZIP_ENABLE"] = "0"
    os.environ["IAXL_OMP_THREAD_NUM"] = os.environ["IAXL_QAT_INSTANCE_NUM"]
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

    bind_cpu_affinity(0, num_workers, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(0)
    original = generate_mock_kv_cache(QAT_MOCK_SHAPE, QAT_MOCK_DTYPE).flatten()
    for segment_kb in QAT_SEGMENT_SIZES_KB:
        speeds_queue.put(SegmentSpeed(segment_kb, decompress_gbps(original, segment_kb)))


def calibrate_qat(context, num_workers: int) -> str:
    """Segment sizes in KB, fastest decompression first, comma separated."""
    started = time.monotonic()
    speeds_queue = context.Queue()
    process = context.Process(target=qat_worker, args=(num_workers, speeds_queue), name="qat")
    process.start()
    speeds = []
    try:
        while len(speeds) < len(QAT_SEGMENT_SIZES_KB):
            speeds.append(receive(speeds_queue, [process]))
    except RuntimeError:
        process.terminate()
        raise
    process.join()
    speeds.sort(key=lambda speed: speed.decompress_gbps, reverse=True)
    ranked_kb = []
    summary_parts = []
    for speed in speeds:
        ranked_kb.append(str(speed.segment_kb))
        summary_parts.append(f"{speed.segment_kb} KB {speed.decompress_gbps:.1f}")
    print(
        f"[accel_auto_config] QAT segment sizes by decompress GB/s: {', '.join(summary_parts)}, "
        f"{time.monotonic() - started:.1f} s",
        file=sys.stderr,
    )
    return ",".join(ranked_kb)


def main() -> None:
    started = time.monotonic()
    num_workers = int(os.environ["TP_SIZE"]) * int(os.environ["DP_SIZE"])
    dsa_needed = (
        env_truthy("IAXL_DSA_GD_ENABLE")
        and env_unset("IAXL_DSA_H2D_CROSSOVER_KB")
        and env_unset("IAXL_DSA_D2H_CROSSOVER_KB")
    )
    segment_needed = env_truthy("IAXL_KV_COMPRESSION") and env_unset("IAXL_ZIP_SEGMENT_SIZE_KB")
    qat_needed = segment_needed and env_truthy("IAXL_QAT_ZIP_ENABLE")
    iaa_only = segment_needed and not qat_needed and env_truthy("IAXL_IAA_ZIP_ENABLE")

    if dsa_needed or qat_needed:
        import torch  # loaded once here; the forked workers inherit it
    context = multiprocessing.get_context("fork")
    settings = []
    try:
        if dsa_needed:
            crossovers = calibrate_dsa(context, num_workers)
            settings.append(f"IAXL_DSA_H2D_CROSSOVER_KB={crossovers.h2d_kb}")
            settings.append(f"IAXL_DSA_D2H_CROSSOVER_KB={crossovers.d2h_kb}")
        if qat_needed:
            settings.append(f"IAXL_ZIP_SEGMENT_SIZE_KB={calibrate_qat(context, num_workers)}")
    except RuntimeError as error:
        sys.exit(f"[accel_auto_config] failed: {error}")
    if iaa_only:
        # Half of IAXL_ZIP_SRC_CAP: the remainder joins the last segment, so every segment fits.
        settings.append(f"IAXL_ZIP_SEGMENT_SIZE_KB={int(os.environ['IAXL_ZIP_SRC_CAP']) // 2 // 1024}")
    print(f"[accel_auto_config] done in {time.monotonic() - started:.1f} s", file=sys.stderr)
    print(" ".join(settings))


if __name__ == "__main__":
    main()
