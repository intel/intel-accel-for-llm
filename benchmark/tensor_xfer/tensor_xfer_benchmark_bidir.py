#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Full-duplex variant of tensor_xfer_benchmark.py: H2D and D2H run at the same time.

Each direction gets its own thread and keeps issuing passes for a shared time
window, so the reported GB/s of one direction is measured while the other one
is busy (PCIe is full duplex).  Methods:

    cuda   torch slice copies on two streams
    batch  cudaMemcpy3DBatchAsync on two streams
    iaxl   SliceCopier in both directions with IAXL_KV_XFER_FULL_DUPLEX=1 (the default
           here, export 0 to compare): H2D on Intel DSA, D2H on the CUDA copy engine,
           exactly what the KV cache transfer does with the flag on

Example:
    IAXL_DSA_GD_ENABLE=1 \
    python3 benchmark/tensor_xfer/tensor_xfer_benchmark_bidir.py
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Read by the IAXL native library when it loads, so it must precede the iaxl import.
os.environ.setdefault("IAXL_KV_XFER_FULL_DUPLEX", "1")

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensor_xfer_benchmark as bench  # noqa: E402
from iaxl.tensor_ops import SliceCopier  # noqa: E402
from tensor_xfer_benchmark import Result, build_3d_batch_ops, cuda_runtime, format_size  # noqa: E402

METHODS = {
    "cuda": "CUDA Copy",
    "batch": "CUDA Copy Batch",
    "iaxl": "IAXL DSA",
}
DIRECTIONS = ("BIDIR H2D", "BIDIR D2H")
# Passes queued ahead per stream; keeps the stream busy without a long drain tail.
LOOKAHEAD = 2


def parse_methods(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"invalid choice: {', '.join(unknown)} (choose from {', '.join(METHODS)})"
        )
    return names


def slice_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: list[int],
    dst_indices: list[int],
    frag_elements: int,
) -> None:
    for src_index, dst_index in zip(src_indices, dst_indices):
        src_start = src_index * frag_elements
        dst_start = dst_index * frag_elements
        dst[dst_start : dst_start + frag_elements].copy_(
            src[src_start : src_start + frag_elements], non_blocking=True
        )


def batch_copy(operations: list[Any], stream: torch.cuda.Stream) -> None:
    result = cuda_runtime.cudaMemcpy3DBatchAsync(
        len(operations), operations, 0, stream.cuda_stream
    )
    if result[0] != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy3DBatchAsync failed: {result[0]}")


@dataclass
class Lane:
    """One direction: issue() enqueues a pass and returns a handle, wait(handle) blocks on it."""

    issue: Callable[[], Any]
    wait: Callable[[Any], None]
    lookahead: int = LOOKAHEAD
    stream: torch.cuda.Stream | None = None  # made current on the lane's thread


def stream_lane(run: Callable[[], None], stream: torch.cuda.Stream) -> Lane:
    def issue() -> torch.cuda.Event:
        run()
        event = torch.cuda.Event()
        event.record(stream)
        return event

    return Lane(issue, lambda event: event.synchronize(), stream=stream)


def blocking_lane(run: Callable[[], None]) -> Lane:
    return Lane(run, lambda _: None, lookahead=1)


def measure_bidir(
    lanes: tuple[Lane, Lane], *, seconds: float, warmup: int, label: tuple[str, ...] = ()
) -> list[tuple[float, int]]:
    """Run both lanes concurrently for `seconds`; returns (ms, passes) per lane."""
    barrier = threading.Barrier(len(lanes) + 1)
    timings: list[tuple[float, int]] = [(0.0, 0)] * len(lanes)
    errors: list[BaseException] = []

    def worker(index: int, lane: Lane) -> None:
        pending: deque[Any] = deque()
        count = 0
        try:
            with torch.cuda.stream(lane.stream):
                for _ in range(warmup):
                    lane.wait(lane.issue())
                barrier.wait()
                start = time.perf_counter()
                deadline = start + seconds
                while time.perf_counter() < deadline:
                    if len(pending) >= lane.lookahead:
                        lane.wait(pending.popleft())
                    pending.append(lane.issue())
                    count += 1
                while pending:
                    lane.wait(pending.popleft())
                timings[index] = ((time.perf_counter() - start) * 1000, count)
        except BaseException as error:  # noqa: BLE001 - re-raised on the main thread
            errors.append(error)
            barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(index, lane), daemon=True)
        for index, lane in enumerate(lanes)
    ]
    for thread in threads:
        thread.start()
    if bench.RECORDER is not None:
        bench.RECORDER.start(label)
    if bench.SYNC is not None:
        bench.SYNC()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join()
    if bench.RECORDER is not None:
        bench.RECORDER.stop()
    if errors:
        raise errors[0]
    return timings


def run(frag_bytes: int, args: argparse.Namespace) -> list[Result]:
    total_target = int(args.total_gib * (1 << 30))
    fragments = max(1, total_target // frag_bytes)
    frag_elements = frag_bytes // 2
    pass_bytes = fragments * frag_bytes
    elements = fragments * frag_elements
    label = ("BIDIR", format_size(frag_bytes))

    source = torch.randint(-32768, 32767, (elements,), dtype=torch.int16, pin_memory=True)
    permutation = torch.randperm(fragments)
    src_indices = list(range(fragments))
    dst_indices = permutation.tolist()
    expected = torch.empty_like(source.view(fragments, frag_elements))
    expected[permutation] = source.view(fragments, frag_elements)
    expected = expected.flatten()

    host_src = source
    host_dst = torch.empty(elements, dtype=torch.int16, pin_memory=True)
    gpu_src = source.cuda()
    gpu_dst = torch.empty_like(gpu_src)
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())

    def clear_destinations() -> None:
        gpu_dst.zero_()
        host_dst.zero_()
        torch.cuda.synchronize()

    def h2d_cuda() -> None:
        slice_copy(host_src, gpu_dst, src_indices, dst_indices, frag_elements)

    def d2h_cuda() -> None:
        slice_copy(gpu_src, host_dst, src_indices, dst_indices, frag_elements)

    def record(method: str, timings: list[tuple[float, int]]) -> list[Result]:
        torch.cuda.synchronize()
        checks = (torch.equal(gpu_dst.cpu(), expected), torch.equal(host_dst, expected))
        return [
            Result(method, direction, elapsed, count * pass_bytes, ok)
            for direction, (elapsed, count), ok in zip(DIRECTIONS, timings, checks)
        ]

    results = []
    if "cuda" in args.methods:
        clear_destinations()
        timings = measure_bidir(
            (stream_lane(h2d_cuda, streams[0]), stream_lane(d2h_cuda, streams[1])),
            seconds=args.seconds,
            warmup=args.warmup,
            label=label + ("CUDA Copy",),
        )
        results.extend(record("CUDA Copy", timings))

    if "batch" in args.methods:
        src_byte_offsets = [index * frag_bytes for index in src_indices]
        dst_byte_offsets = [index * frag_bytes for index in dst_indices]
        h2d_operations = build_3d_batch_ops(
            host_src.data_ptr(), gpu_dst.data_ptr(), src_byte_offsets, dst_byte_offsets, frag_bytes
        )
        d2h_operations = build_3d_batch_ops(
            gpu_src.data_ptr(), host_dst.data_ptr(), src_byte_offsets, dst_byte_offsets, frag_bytes
        )
        clear_destinations()
        timings = measure_bidir(
            (
                stream_lane(lambda: batch_copy(h2d_operations, streams[0]), streams[0]),
                stream_lane(lambda: batch_copy(d2h_operations, streams[1]), streams[1]),
            ),
            seconds=args.seconds,
            warmup=args.warmup,
            label=label + ("cudaMemcpy3DBatchAsync",),
        )
        results.extend(record("CUDA Copy Batch", timings))

    if "iaxl" in args.methods:
        h2d_copier = SliceCopier(
            host_src.view(fragments, frag_elements), 0, dst_indices,
            out=gpu_dst.view(fragments, frag_elements),
        )
        d2h_copier = SliceCopier(
            gpu_src.view(fragments, frag_elements), 0, src_indices,
            out=host_dst.view(fragments, frag_elements), cpu_index=dst_indices,
        )
        clear_destinations()
        timings = measure_bidir(
            (blocking_lane(h2d_copier.copy), blocking_lane(d2h_copier.copy)),
            seconds=args.seconds,
            warmup=args.warmup,
            label=label + ("IAXL DSA",),
        )
        h2d_copier.close()
        d2h_copier.close()
        results.extend(record("IAXL DSA", timings))

    return results


def plot_results(
    results: dict[str, dict[int, list[Result]]], frag_sizes: list[int], path: str
) -> None:
    directions = [direction for direction in DIRECTIONS if direction in results]
    figure, axes = plt.subplots(
        len(directions), 1, figsize=(8, 5 * len(directions)), squeeze=False
    )
    for axis, direction in zip(axes[:, 0], directions):
        by_fragment = results[direction]
        for method in METHODS.values():
            points = [
                (size, result.gbps)
                for size in frag_sizes
                for result in by_fragment.get(size, [])
                if result.method == method
            ]
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    color=bench.METHOD_COLORS[method],
                    label=method,
                )
        axis.axvspan(4 << 10, 64 << 10, color="gold", alpha=0.15, label="KV <= 64K")
        axis.set_xscale("log", base=2)
        axis.set_xticks(frag_sizes)
        axis.set_xticklabels([format_size(size) for size in frag_sizes], rotation=45)
        axis.set_xlabel("Fragment size")
        axis.set_ylabel("GB/s")
        axis.set_title(f"{direction} (concurrent with the other direction)")
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrent H2D/D2H fragmented copies using CUDA, CUDA batch, and IAXL "
        "(DSA for H2D, CUDA for D2H with IAXL_KV_XFER_FULL_DUPLEX=1)."
    )
    parser.add_argument("--total-gib", type=float, default=0.1)
    parser.add_argument("--frag-sizes", type=bench.parse_sizes, default=bench.DEFAULT_FRAG_SIZES)
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_methods,
        metavar="NAME",
        help=f"transfer methods to run, space or comma separated: {', '.join(METHODS)}",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=1.0,
        metavar="SEC",
        help="how long both directions run concurrently; the ms column shows this window "
        "and GB/s is per direction",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=bench.DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help="directory for the plot, flame graph and perf data",
    )
    parser.add_argument(
        "--plot",
        default="tensor_xfer_bidir.png",
        metavar="PATH",
        help="plot file name inside the output directory; empty disables plotting",
    )
    parser.add_argument(
        "--flamegraph",
        default="",
        metavar="PATH",
        help="record the measured windows (warmup excluded) with perf and write this "
        "flame graph SVG into the output directory, together with .folded and perf.data",
    )
    parser.add_argument("--flamegraph-freq", type=int, default=999, metavar="HZ")
    parser.add_argument("--flamegraph-call-graph", choices=("fp", "dwarf", "lbr"), default="fp")
    return parser


def parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if isinstance(args.frag_sizes, str):
        args.frag_sizes = bench.parse_sizes(args.frag_sizes)
    args.methods = [name for group in args.methods for name in group] if args.methods else list(METHODS)
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    return args


def main() -> None:
    parser = build_parser()
    args = parse_args(parser)
    bench.setup(args)
    bench.print_header()

    all_results: dict[str, dict[int, list[Result]]] = {}
    for frag_bytes in args.frag_sizes:
        for result in run(frag_bytes, args):
            all_results.setdefault(result.direction, {}).setdefault(frag_bytes, []).append(result)
            bench.print_result(frag_bytes, result)

    if args.plot:
        plot_results(all_results, args.frag_sizes, bench.output_path(args.output_dir, args.plot))
    if bench.RECORDER is not None:
        svg = bench.output_path(args.output_dir, args.flamegraph)
        folded = f"{os.path.splitext(svg)[0]}.folded"
        bench.RECORDER.write_folded(folded)
        bench.RECORDER.write_svg(svg, title="tensor_xfer bidir measured window")
        print(f"Flame graph saved to {svg} ({bench.RECORDER.samples} samples)")


if __name__ == "__main__":
    main()
