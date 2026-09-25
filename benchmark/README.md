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

## CPU Serving

`kvstore/serving_benchmark.sh` measures request throughput, TTFT and TPOT for the `raw`,
`sw` and `qat` KVShrink arms and for vLLM's own prefix cache (`vllm`). The load comes
from `vllm bench serve` with the client settings of `tests/vllm-benchmark.sh` (random
dataset, shared prefix = hit rate, `--ignore-eos`, `--request-rate inf`). For every point
it primes only the shared prefix, waits for the connector to finish writing it, runs the
measurement, and reads vLLM's prefix-cache counters to report the hit rate achieved.

It starts one CPU server per cell and arm through `examples/kvshrink-vllm-cpu-serve.sh`,
pinned to `SERVER_CPUS` (the launcher gives the last `IAXL_CORES` of them to IAXL). All
inputs are environment variables; lists are space separated:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | required | Local model directory |
| `ARMS` | `raw sw qat vllm` | Backends to compare |
| `INPUT_LENS` / `OUTPUT_LENS` | `8192` / `128` | Prompt and generated tokens |
| `HIT_RATES` | `80` | Percent of each prompt shared across prompts |
| `CONCURRENCY` | `1 4 16` | Concurrent clients per point |
| `PROMPTS_PER_CLIENT`, `MIN_PROMPTS` | `2`, `8` | Prompts per point = max(clients x 2, 8) |
| `SERVER_CPUS`, `IAXL_CORES`, `NUMA_NODE` | `0-31`, `2`, `0` | Server placement |
| `BLOCK_SIZE`, `CACHE_POOL_GB`, `SEED` | `32`, `96`, `20260925` | vLLM block size, IAXL DDR cache, dataset seed |
| `CLIENT_CPUS` | unset | Cores for `vllm bench serve`, outside `SERVER_CPUS` |
| `SERVER` | `launch` | `external` benchmarks an already running server on `PORT` |
| `RESULTS` | `./results/serving-<time>` | Output directory |

```bash
# All arms, one operating point per concurrency.
MODEL=/models/Qwen3-8B CLIENT_CPUS=32-35 benchmark/kvstore/serving_benchmark.sh
# Restore-heavy comparison of QAT and SW.
MODEL=/models/Qwen3-8B ARMS="qat sw" INPUT_LENS="4096 8192" OUTPUT_LENS=8 HIT_RATES=100 \
    CONCURRENCY="1 4 16" CLIENT_CPUS=32-35 benchmark/kvstore/serving_benchmark.sh
# A GPU server started with examples/kvshrink-vllm-serve.sh, same client and metrics.
SERVER=external PORT=8000 ARMS=qat MODEL=Qwen/Qwen3-32B benchmark/kvstore/serving_benchmark.sh
```

Results: `summary.csv` (one row per point: throughput, median/p95 TTFT and TPOT, measured
hit rate, QAT request delta, DSA state) plus each `vllm bench serve` JSON and server log.

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