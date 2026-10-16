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

### Cross-node remote reuse (remote_pool / RDMA)

The same launcher also measures the cross-node KV cache path described in
[remote-pool-kvstore.en.md](../doc/usage/remote-pool-kvstore.en.md). When
`IAXL_RDMA_ENABLE=1`, `iaxl.kvstore.KVStore` transparently resolves to
`KVStoreRemote`, so each rank's PUT/GET is served by RDMA to the daemon node's
pool instead of a local store. The numbers you get are the end-to-end client
GPU ↔ daemon RDMA bandwidth plus the daemon-side compression ratio.

Compression / CPU / accelerator binding all run on the **daemon**, so on the
client the `--qat` / `--iaa` / `--dsa` flags are ignored; the
"neither --qat nor --iaa" warning on the client is harmless.

Start the daemon first, exactly as in the usage doc (TP=4 example, daemon
`10.10.10.10`, client `10.10.10.11`):

```bash
# ---- daemon node (10.10.10.10), repository root ----
NVIDIA_RUNTIME=none ./start.sh
# ---- inside the container shell opened by start.sh ----
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4
# configure the daemon-side compression backend, e.g. QAT:
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export KVSHRINK_QAT_DEVICES="0|1|4|5"
./examples/kvshrink-daemon.sh
```

Then, on the client node, run the benchmark instead of vLLM. `--ranks` must
equal the daemon's `IAXL_RDMA_TP_SIZE`, and each rank needs one GPU:

```bash
# ---- client node (10.10.10.11), repository root ----
./start.sh
# ---- inside the container shell opened by start.sh ----
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10   # same value as the daemon
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_CLIENT_NIC_IPS=10.10.10.11   # this node's RDMA NIC IP(s)
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4

bash benchmark/kvstore/kvstore_benchmark.sh --ranks 4 --data-source mock
```

Multi-rank runs read metrics straight from each rank's `KVStoreRemote` (RPC to
the daemon), so no REST endpoint is required. Five
`KVStoreRemote connected: ...` lines on the client (and `pool: ...` /
`copy_chunks_batch: using RDMA backend` on the daemon) confirm the RDMA path is
in use. `--data-source model` (needs `MODEL`) or `--dtype` / `--shape` /
`--num-layers` work the same as in the local run. Start the daemon before the
client, or the client's connect will time out (`IAXL_API_TIMEOUT`, default 60s).

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