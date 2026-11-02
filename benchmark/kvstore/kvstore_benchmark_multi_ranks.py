#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run kvstore_benchmark.py with one process per TP rank, mirroring vLLM.

The main process is the scheduler: it drives every rank worker through the same
steps (setup, warmup, PUT, GET, verify) over a pipe and prints the report.
Rank r is bound to its CPUs / QAT / DSA via iaxl.utils.affinity (using the
per-rank env from setvars.sh), gets its own GPU through CUDA_VISIBLE_DEVICES
and its own KVStore(rank=r). Timed steps start simultaneously on all ranks and
the aggregate takes the slowest rank as wall time, like a TP forward pass.

Example:
    TP_SIZE=2 source setvars.sh
    python3 benchmark/kvstore/kvstore_benchmark_multi_ranks.py
"""

import logging
import multiprocessing
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

TIMED_STEPS = ("warmup", "put", "get")
RATIO_KEY = "compression_ratio (unzip/zip, higher=better)"
STATUS_SUM_KEYS = (
    "current_bytes",
    "cache_entries",
    "group_count",
    "hits",
    "misses",
    "puts",
    "total_zip_bytes",
    "total_unzip_bytes",
)


def worker(rank, tp_size, gpu, args, conn, barrier) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # must precede the torch import
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel
    from iaxl.utils.logger import setup_root_logger

    setup_root_logger(f"rank{rank}")
    bind_cpu_affinity(rank, tp_size, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)

    import torch

    import kvstore_benchmark as kb

    class RankBenchmark(kb.Benchmark):
        # Each rank's store has its own port / daemon: query it directly, not via REST.
        def metrics(self, **params) -> dict:
            return self.kvstore.metrics(params)

        def status(self) -> dict:
            return self.kvstore.status()

    logger.info("Bound rank %d to GPU %s (%s)", rank, gpu, torch.cuda.get_device_name(0))
    bench = RankBenchmark(args, rank, tp_size)
    if rank == 0:
        bench.print_config()

    while (step := conn.recv()) is not None:
        try:
            if step in TIMED_STEPS:
                barrier.wait()
            value = getattr(bench, step)
            conn.send(("ok", value() if callable(value) else value))
        except Exception:
            conn.send(("error", traceback.format_exc()))
            raise


def aggregate(statuses: list, metrics: list) -> tuple:
    status = {key: sum(s[key] for s in statuses) for key in STATUS_SUM_KEYS}
    zip_bytes = status["total_zip_bytes"]
    status[RATIO_KEY] = status["total_unzip_bytes"] / zip_bytes if zip_bytes else 0.0
    total = {}
    for op in ("compress", "decompress", "h2d", "d2h"):
        # Same bytes / accumulated-ns definition as a single rank's *_gbps.
        num_bytes = sum(m[f"{op}_bytes"] for m in metrics)
        ns = sum(m[f"{op}_ns"] for m in metrics)
        total[f"{op}_bytes"] = num_bytes
        total[f"{op}_ns"] = ns
        total[f"{op}_gbps"] = num_bytes / ns if ns else 0.0
    return status, total


def schedule(kb, tp_size: int, conns: list) -> bool:
    def call(step: str, ranks=range(tp_size)) -> list:
        for rank in ranks:
            conns[rank].send(step)
        results = []
        for rank in ranks:
            try:
                state, value = conns[rank].recv()
            except EOFError:
                sys.exit(f"rank{rank} died during {step}")
            if state != "ok":
                sys.exit(f"rank{rank} failed during {step}:\n{value}")
            results.append(value)
        return results

    call("setup", [0])  # rank 0 writes the model KV cache file the others reuse
    call("setup", range(1, tp_size))
    total_blocks = call("total_blocks", [0])[0]
    total_bytes = call("total_bytes", [0])[0]
    call("warmup")
    put_times = call("put")
    get_times = call("get")
    verified = call("verify")
    statuses = call("status")
    metrics = call("metrics")

    def gibps(seconds: float) -> float:
        return total_bytes / seconds / (1024**3)

    print(f"\nPer rank ({kb.format_bytes(total_bytes)} each)")
    print("-" * 100)
    print(
        f"{'Rank':>4} {'PUT GiB/s':>10} {'GET GiB/s':>10} "
        f"{'Verify':>6} {'Ratio':>7} {'Zip GB/s':>9} {'Unzip GB/s':>10} {'D2H GB/s':>9} {'H2D GB/s':>9}"
    )
    for rank in range(tp_size):
        print(
            f"{rank:>4} {gibps(put_times[rank]):10.3f} {gibps(get_times[rank]):10.3f} "
            f"{'pass' if verified[rank] else 'FAIL':>6} {statuses[rank][RATIO_KEY]:7.3f} "
            f"{metrics[rank]['compress_gbps']:9.3f} {metrics[rank]['decompress_gbps']:10.3f} "
            f"{metrics[rank]['d2h_gbps']:9.3f} {metrics[rank]['h2d_gbps']:9.3f}"
        )
    print(
        f"{'Sum':>4} {sum(map(gibps, put_times)):10.3f} {sum(map(gibps, get_times)):10.3f} "
        f"{'':>6} {'':>7} "
        f"{sum(m['compress_gbps'] for m in metrics):9.3f} "
        f"{sum(m['decompress_gbps'] for m in metrics):10.3f} "
        f"{sum(m['d2h_gbps'] for m in metrics):9.3f} "
        f"{sum(m['h2d_gbps'] for m in metrics):9.3f}"
    )

    status, total = aggregate(statuses, metrics)
    kb.print_report(
        max(put_times),
        max(get_times),
        all(verified),
        status,
        total,
        total_blocks * tp_size,
        total_bytes * tp_size,
        title=f"Aggregate over {tp_size} ranks (wall time = slowest rank)",
    )
    return all(verified)


def main() -> None:
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count < tp_size:
        sys.exit(f"TP_SIZE={tp_size} needs {tp_size} GPUs, but only {gpu_count} are visible")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(gpu_count)]

    import kvstore_benchmark as kb

    args = kb.parse_args()

    ctx = multiprocessing.get_context("spawn")  # each rank must initialize CUDA itself
    barrier = ctx.Barrier(tp_size, timeout=600)
    conns, procs = [], []
    for rank in range(tp_size):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=worker,
            args=(rank, tp_size, gpus[rank], args, child_conn, barrier),
            name=f"rank{rank}",
        )
        proc.start()
        conns.append(parent_conn)
        procs.append(proc)
    try:
        verified = schedule(kb, tp_size, conns)
    finally:
        for conn in conns:
            try:
                conn.send(None)
            except OSError:
                pass
        for proc in procs:
            proc.join(30)
            if proc.is_alive():
                proc.terminate()
    sys.exit(0 if verified else 1)


if __name__ == "__main__":
    main()
