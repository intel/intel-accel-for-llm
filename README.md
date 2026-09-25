[中文](README.zh-CN.md)

# intel-accel-for-llm (`iaxl`)

`iaxl` uses Intel hardware accelerators to improve LLM inference performance.

## Design Documentation

- [IAXL Design](doc/design/iaxl.md)
- [KVShrink Design](doc/design/kvshrink.md)

## Host Setup

1. Add `intel_iommu=on,sm_on iommu=pt` to the kernel command line, then reboot the host:

```bash
sudo ./tools/setup_kernel_cmdline.sh
sudo reboot
```

2. After rebooting, download and install the QAT driver:

```bash
wget -q https://downloadmirror.intel.com/843052/QAT20.L.1.2.30-00078.tar.gz
tar xf QAT20.L.1.2.30-00078.tar.gz
./configure
make -j$(nproc)
sudo make install
```

Use the following commands to stop or start the QAT service:

```bash
adf_ctl down
adf_ctl up
```

3. Install the GDRCopy driver and configure DSA:

```bash
sudo ./tools/install_gdr_driver.sh
./tools/setup_dsa_cnt.sh
```

## Environment Variables

Common settings in `setvars.sh`:

| Environment variable | Default | Description |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen3-32B` | Hugging Face model ID or local model path |
| `TP_SIZE` | `2` | Required; number of Tensor Parallel workers. CPU, QAT, and DSA resources are configured based on this value |
| `IAXL_KV_COMPRESSION` | `1` | Enable DEFLATE compression (`0`/`1`) |
| `IAXL_QAT_ZIP_ENABLE` | `1` | Enable QAT compression workers (`0`/`1`) |
| `IAXL_IAA_ZIP_ENABLE` | `0` | Enable Intel IAA compression workers via Intel QPL (`0`/`1`). Can be combined with `IAXL_QAT_ZIP_ENABLE`: IAA decodes at most a 4 KB DEFLATE history window while QAT gen4 always compresses with 32 KB, so each block records whether IAA can decode it and IAA only claims those on the way back |
| `IAXL_CPU_ZIP_ENABLE` | `1` | Enable CPU compression workers (`0`/`1`) |
| `IAXL_DSA_GD_ENABLE` | `0` | Enable Intel DSA + GDRCopy transfers (`0`/`1`) |
| `IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS` | `1` | Do not compress the KV cache for the first N layers |
| `PYTHONOPTIMIZE` | `0` | Preserve Python `assert` checks |

> [!WARNING]
> Do not enable `IAXL_DSA_GD_ENABLE` on GPUs that do not support P2P DMA. Keep it set to `0`.

## KVShrink vLLM Example

KVShrink is a vLLM V1 KV connector based on IAXL `KVStore`. Configure `setvars.sh`, then start the container:

```bash
./start.sh
```

`setvars.sh` automatically configures the CPU, QAT, and DSA resources for each rank based on the NUMA topology of the first `TP_SIZE` GPUs.

Inside the container, optionally install the package with pip:

```bash
pip install -e . --verbose --no-build-isolation
```

Start the service inside the container:

```bash
./examples/kvshrink-vllm-serve.sh
```

This script starts vLLM on `localhost:8000`, loads `KVShrinkConnector`, and writes logs to `log.kvshrink-vllm`. Use the `MODEL`, `TP_SIZE`, and per-rank CPU/QAT/DSA settings in the startup log to verify the active topology.

Open the same container from a second host terminal:

```bash
docker exec -it -w "$PWD" iaxl.vllm bash
```

Send a Chat Completions test request:

```bash
./tests/vllm-test.sh
```

## CPU Inference With QAT

Build with `DEVICE=cpu` to run the model on CPU while offloading KV cache
compression and decompression to QAT (or IAA). This uses the same `KVStore`,
asynchronous native queues, codecs, and persist/evict APIs as GPU inference. On
CPU the codec works on the inference tensor in place: PUT gathers a block straight
into the accelerator's staging buffer and GET decompresses straight back into the
KV tensor, so no scratch snapshot is taken and no host-to-host copy of the
uncompressed block is made by the CPU. CUDA, SYCL and GDRCopy are not required;
Intel DSA is optional (see below).

Use a CPU-enabled PyTorch/vLLM environment. Install the Python build requirements
and native development dependencies, including a C/C++ compiler, CMake, NASM,
SQLite, and zlib. NASM is needed by the existing QPL dependency even when IAA is
disabled at runtime.

For hosts using installed QATlib with a configured VFIO driver and accessible QAT
devices, install the development libraries providing `libqat.so` and `libusdm.so`,
then build:

```bash
python -m pip install -r requirements.txt
DEVICE=cpu IAXL_CMAKE_ARGS="-DIAXL_USE_SYSTEM_QAT=ON" \
	python -m pip install -e . --no-build-isolation
```

For the SDK/legacy driver described in Host Setup, omit `IAXL_CMAKE_ARGS` to keep
the existing SDK build. That build additionally needs Boost/Boost Regex, SSL,
udev, and netlink development packages. Match the user-space library to the host
driver: the SDK runtime expects `/dev/qat_dev_processes`; QATlib uses the host's
VFIO setup. Do not install the GPU-only GDRCopy/DSA components for CPU inference.

Start CPU serving with the standalone CPU launcher, without sourcing the
GPU-topology-based `setvars.sh`:

```bash
MODEL=Qwen/Qwen3-0.6B taskset -c 0-31 ./examples/kvshrink-vllm-cpu-serve.sh
```

The launcher gives inference every CPU it may run on except the last `IAXL_CORES`
(default 2), which it gives to IAXL; set `VLLM_CPU_OMP_THREADS_BIND`,
`IAXL_CPU_AFFINITY` and `OMP_NUM_THREADS` together to place them yourself. It uses
every QAT device, QAT-only compression, BF16, one worker, block size 32, and port
8000; `KVSHRINK_CONNECTOR=0` serves plain vLLM with its own prefix cache instead.
`VLLM_CPU_KVCACHE_SPACE` controls the live CPU KV cache size in GiB;
`IAXL_DDR_POOL_SIZE_GB` independently controls the compressed cache. CPU persisted
data lives under `IAXL_CACHE_DIR/cpu/{compressed,raw}` to avoid reusing incompatible
GPU tensor layouts. Compression format and lossless/lossy settings are unchanged.

### Core budget

On CPU every core the cache path occupies is a core the model does not have, so
the two are partitioned explicitly:

- `VLLM_CPU_OMP_THREADS_BIND` places the inference OpenMP threads.
- `IAXL_CPU_AFFINITY` (for example `32-35`) pins every IAXL native thread: the codec
  pollers, the transfer and record queue threads. The connector refuses to start
  when more than one codec thread would share the rank's inference CPUs (unset or
  overlapping), and warns for a single thread. For multiple ranks, set `TP_SIZE`,
  per-rank inference affinity, and `KVSHRINK_QAT_DEVICES` (for example, `0|1`).
- `IAXL_QAT_POLL_THREADS` / `IAXL_IAA_POLL_THREADS` decouple accelerator concurrency
  from CPU threads: poller *p* of *P* drives instances *p, p+P, …* and their queue
  slots. The launcher dedicates `IAXL_CORES` (default 2) cores to IAXL, runs one QAT
  poller per core, and uses four instances on every QAT device, since each device
  decompresses about 6 GB/s. Pollers spin briefly with `pause` and then yield the core.
- The compression OpenMP team is sized from the poller counts and
  `IAXL_CPU_ZIP_THREADS`, independently of inference's `OMP_NUM_THREADS`.

### Intel DSA on CPU

`IAXL_DSA_MEMCPY_ENABLE=1` moves the pure-copy work of the CPU path to the DSA work
queues named in `IAXL_DSA_WQS`, using host virtual addresses (no GDRCopy): raw
(uncompressed) blocks in both directions, including layers excluded by
`IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS`, and the QAT/IAA per-block copies (compressed
payload into the device buffer, unshuffled gather/scatter), which run asynchronously
beside the codec. It needs a user-mode DSA WQ (`/dev/dsa/wqX.Y`, see
`tools/setup_dsa.sh`); if the WQ cannot be used the process logs one warning and
falls back to `memcpy` for the rest of its life. Byte-shuffled blocks are transformed
on the CPU, so `IAXL_KV_DATA_SHUFFLE=1` trades restore throughput for capacity: on
Qwen3-8B it saves 28% of KV bytes instead of 19%, but at 8192 tokens fully cached it
runs at 0.84x raw+DSA, while shuffle off matches raw+DSA. The launcher defaults it off.

A CPU attention block has `2 * num_kv_heads * block_size * head_size * dtype_bytes`
bytes per layer. Keep it within `IAXL_ZIP_SRC_CAP`, and size `IAXL_ZIP_DST_CAP` for
the compressed output, or reduce `BLOCK_SIZE`. Both capacities default to 256 KiB.

Run the CPU regression suite without QAT hardware, or require QAT-only operation:

```bash
python -u -m unittest discover -s tests -p test_cpu_inference.py -v
IAXL_TEST_ZIP_BACKEND=qat \
	python -u -m unittest discover -s tests -p test_cpu_inference.py -v
```

The tests cover block layouts, FP32/FP16/BF16, raw and mixed-layer compression,
asynchronous waits, the direct (scratch-free) codec path, native-thread affinity,
DSA fallback, QAT poller consolidation, and persisted reloads. An opt-in inference
smoke test uses a cached `Qwen/Qwen3-0.6B` snapshot (or `IAXL_TEST_MODEL`, a local
model path or cached model ID), disables vLLM prefix caching, and requires actual
external-cache hits and identical cold/warm generated token IDs:

```bash
IAXL_TEST_ZIP_BACKEND=qat IAXL_TEST_VLLM=1 \
	python -u -m unittest discover -s tests -p test_cpu_inference.py -v
```

The smoke test uses async loading by default; set `IAXL_TEST_ASYNC_LOAD_LAYERS=0`
to check synchronous loading. This is a correctness check, not a performance
benchmark. Check for other workloads before running hardware tests.

### CPU serving benchmark

`benchmark/kvstore/serving_benchmark.sh` is the serving benchmark. It drives
`vllm bench serve` with the same client settings as `tests/vllm-benchmark.sh`, so CPU
and GPU numbers come from the same load generator. Pick the backends with `ARMS`
(`raw`, `sw`, `qat`, `vllm`); see [benchmark/README.md](benchmark/README.md#cpu-serving).

## KVShrink vLLM Benchmark

Keep the KVShrink vLLM service running and execute the online serving benchmark in the second container terminal:

```bash
./tests/vllm-benchmark.sh
```

## REST API

The management API listens on `localhost:18700` by default and forwards requests to each rank.

| Endpoint | Description |
| --- | --- |
| `GET /v1/cache/status` | Query cache status |
| `POST /v1/cache/evict` | Evict cache groups from DDR |
| `POST /v1/cache/persist` | Persist cache groups to disk |

```bash
curl http://localhost:18700/v1/cache/status
```

For `persist` and `evict`, `count` specifies the maximum number of cache groups to process. To preserve cached data, call `persist` before `evict`:

```bash
curl -X POST http://localhost:18700/v1/cache/persist \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
curl -X POST http://localhost:18700/v1/cache/evict \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
```
