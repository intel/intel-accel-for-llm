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

2. After rebooting, install a QAT driver. Use **one** of the following.

   **Out-of-tree driver package** (default):

```bash
wget -q https://downloadmirror.intel.com/843052/QAT20.L.1.2.30-00078.tar.gz
tar xf QAT20.L.1.2.30-00078.tar.gz
./configure
make -j$(nproc)
sudo make install
```

Use the following commands to stop or start the out-of-tree QAT service:

```bash
adf_ctl down
adf_ctl up
```

   **In-tree driver + qatlib** (kernel `qat_4xxx` module, virtual functions bound to `vfio-pci`):

```bash
sudo apt install qatlib-service libqat-dev libusdm-dev   # or build https://github.com/intel/qatlib
printf 'POLICY=0\nServicesEnabled=dc\n' | sudo tee /etc/sysconfig/qat
sudo systemctl enable qat
sudo systemctl restart qat
export IAXL_QATLIB=intree   # before sourcing setvars.sh / building
```

`POLICY=0` gives every process one virtual function from each QAT device, which `IAXL_QAT_DEVICES` / `KVSHRINK_QAT_DEVICES` index. With `POLICY=1` each process only sees device `0`.

With `IAXL_QATLIB=intree` the build links the system `libqat` / `libusdm` found by `pkg-config qatlib` instead of downloading and building the out-of-tree package into `_lib`. For container builds, the qatlib development packages must be installed inside the container, and `/run/qat` must be mounted from the host at runtime.

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
| `IAXL_QATLIB` | unset (`oot`) | Optional. QAT user-space library used by the build: `oot` (out-of-tree package built into `_lib`, used when unset) or `intree` (system qatlib for the in-tree driver) |
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
asynchronous native queues, codecs, and persist/evict APIs as GPU inference.
With the vLLM CPU layout (blocks on axis 0) the codec reads and writes KV blocks
in place, so no scratch buffers are used; byte shuffle or lossy truncation, which
rewrite the codec input, fall back to scratch buffers. CUDA, SYCL and GDRCopy are
not required.

Use a CPU-enabled PyTorch/vLLM environment. Install the Python build requirements
and native development dependencies, including a C/C++ compiler, CMake, NASM,
SQLite, and zlib. NASM is needed by the existing QPL dependency even when IAA is
disabled at runtime.

For the in-tree driver with system qatlib (see Host Setup), build with
`IAXL_QATLIB=intree`:

```bash
python -m pip install -r requirements.txt
DEVICE=cpu IAXL_QATLIB=intree python -m pip install -e . --no-build-isolation
```

For the out-of-tree driver package, leave `IAXL_QATLIB` unset (`oot`). That build
additionally needs Boost/Boost Regex, SSL, udev, and netlink development packages.
Match the user-space library to the host driver: the out-of-tree runtime expects
`/dev/qat_dev_processes`; qatlib uses the host's VFIO setup. Do not install the
GPU-only GDRCopy/DSA components for CPU inference.

Start CPU serving with the standalone CPU launcher, without sourcing the
GPU-topology-based `setvars.sh`:

```bash
MODEL=Qwen/Qwen3-0.6B taskset -c 0-31 ./examples/kvshrink-vllm-cpu-serve.sh
```

The launcher gives inference every CPU it may run on except the last `IAXL_CORES`
(default 2), which it gives to IAXL; set `VLLM_CPU_OMP_THREADS_BIND`,
`IAXL_CPU_AFFINITY` and `OMP_NUM_THREADS` together to place them yourself. It uses
QAT-only compression, BF16, one worker, block size 32, and port
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
- Each QAT/IAA instance is driven by its own codec thread, so the launcher uses one
  QAT instance per IAXL core.
- The compression OpenMP team is sized from the instance counts and
  `IAXL_CPU_ZIP_THREADS`, independently of inference's `OMP_NUM_THREADS`.

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
asynchronous waits, native-thread affinity, and persisted reloads. An opt-in inference
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
