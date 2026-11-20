[中文](README.zh-CN.md)

# Benchmarks

Run the following commands from the repository root in the configured development environment.

## KVStore

Measures KVStore PUT/GET bandwidth, compression ratio, and compression/decompression throughput.

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --qat
```

The launcher mirrors vLLM: the main process is the scheduler and drives one worker process per TP rank (`--ranks N`, default 1). Rank `r` uses GPU `r` and is bound to its own CPUs / QAT / DSA from `setvars.sh` (`VLLM_CPU_OMP_THREADS_BIND`, `KVSHRINK_QAT_DEVICES`, `KVSHRINK_DSA_DEVICES`), owns a `KVStore(rank=r)`, and all ranks start each timed step together. The report lists every rank plus a `Sum` row, followed by the aggregate where the wall time is the slowest rank's, like a TP forward pass.

`--qat` and/or `--iaa` select the compression engines; with neither, compression is disabled (`IAXL_KV_COMPRESSION=0`) and a warning is printed. `--dsa` enables DSA gather/scatter and `--nsys` wraps the run in Nsight Systems:

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --ranks 2 --qat --iaa --dsa
bash benchmark/kvstore/kvstore_benchmark.sh --qat --shape 2 1024 16 4 128 --dtype bf16 --num-layers 32
```

By default the KV cache is filled by a real transformer prefill of `$MODEL`. The generated data is stored as `<model>_<dtype>.pt` under `--kv-data-dir` and reused whenever it matches the requested shape (rank 0 generates it, the other ranks reuse it). `--dtype` selects `bf16`, `fp8_e4m3` (per-tensor k/v scale), or `int4` (symmetric group-wise, two 4-bit values per byte), matching vLLM's quantization. Use `--data-source mock` for synthetic data:

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --qat --dtype int4 \
    --prompt-file /path/to/text.txt --model-seq-len 16384
bash benchmark/kvstore/kvstore_benchmark.sh --ranks 2 --qat --data-source mock
```

`kvstore_benchmark.py` can still be run directly for a plain single-process run without CPU / accelerator binding.

## Tensor Transfer

Compares fragmented H2D and D2H transfer performance using CUDA, `cudaMemcpy3DBatchAsync`, Triton, and IAXL, and generates a result plot.

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
```

Use `--direction h2d` or `--direction d2h` to run a single direction, and `--methods` to run only some of `cuda`, `batch`, `triton`, `iaxl`:

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --direction h2d --methods iaxl cuda
```

`--ranks N` runs one process per TP rank (rank `r` on GPU `r`, bound like the KVStore benchmark) and reports the per-rank average.

All generated files go to `/_data/tensor_xfer_benchmark`; change it with `--output-dir`.

Add `--flamegraph` to profile with `perf record`. perf is driven through its control FIFO, so only the timed iterations are sampled (warmup is excluded), and every thread of the process is covered, including the native IAXL DSA workers. The run writes a flame graph SVG plus a `.folded` file for `flamegraph.pl` or speedscope, and keeps the raw `perf.data` for `perf report`.

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --flamegraph tensor_xfer_flame.svg
```

This needs `perf` installed in the container and `kernel.perf_event_paranoid <= 2` on the host. Use `--flamegraph-call-graph dwarf` when binaries are built without frame pointers, and `--flamegraph-freq` to change the sampling rate. Profiling perturbs the timings, so use it for analysis runs rather than for reported numbers.

### Full-duplex H2D/D2H

`tensor_xfer_benchmark_bidir.py` runs both directions at the same time, one thread per direction, for `--seconds` (default 1), and reports each direction's GB/s while the other is active. `iaxl` uses the IAXL transfer path in both directions with `IAXL_KV_XFER_FULL_DUPLEX=1` (the script's default), so H2D goes to DSA and D2H to the CUDA copy engine as for the KV cache; export `IAXL_KV_XFER_FULL_DUPLEX=0` to compare with DSA in both directions. `cuda` and `batch` use two CUDA streams.

```bash
IAXL_DSA_GD_ENABLE=1 source setvars.sh
LD_PRELOAD=$IAXL_LD_PRELOAD numactl --cpunodebind=0 --membind=0 \
    python3 benchmark/tensor_xfer/tensor_xfer_benchmark_bidir.py --methods iaxl batch
```