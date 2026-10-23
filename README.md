[中文](README.zh-CN.md)

# intel-accel-for-llm (`iaxl`)

`iaxl` uses Intel hardware accelerators to improve LLM inference performance.

## Design Documentation

- [IAXL Design](doc/design/iaxl.md)
- [KV Zip Design](doc/design/kv-zip.md)
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
wget -q https://downloadmirror.intel.com/921047/QAT20.L.1.2.30-00178.tar.gz
tar xf QAT20.L.1.2.30-00178.tar.gz
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

Common settings in `setvars.sh`. Other variables in `setvars.sh` are advanced settings that most users do not need to change.

| Category | Environment variable | Default | Description |
| --- | --- | --- | --- |
| Accelerators | `IAXL_QAT_ZIP_ENABLE` | `1` | Enable QAT compression workers (`0`/`1`) |
| Accelerators | `IAXL_IAA_ZIP_ENABLE` | `0` | Enable Intel IAA compression workers via Intel QPL (`0`/`1`). Can be combined with QAT; each block records whether IAA can decode it |
| Accelerators | `IAXL_CPU_ZIP_ENABLE` | `0` | Enable CPU compression workers (`0`/`1`). CPU compression uses a lot of CPU time, so leave it off when the QAT/IAA accelerators already provide enough throughput |
| Accelerators | `IAXL_DSA_GD_ENABLE` | `0` | Enable Intel DSA + GDRCopy transfers (`0`/`1`) |
| Compression | `IAXL_KV_COMPRESSION` | `1` | Enable DEFLATE compression of the KV cache (`0`/`1`) |
| Compression | `IAXL_ZIP_LEVEL` | `1` | Compression level for all zip backends, from `1` (fastest) to `9` (best ratio). QAT and CPU use the level as is. IAA hardware only supports level 1, so IAA ignores this setting. Keep the default `1`: higher levels barely improve the KV cache compression ratio but make compression much slower |
| Compression | `IAXL_KV_DATA_SHUFFLE` | `1` | Byte-shuffle BF16 data before compression (`0`/`1`). Slightly lowers compression throughput, but improves the compression ratio and decompression throughput |
| Compression | `IAXL_ZIP_EXPONENT_ONLY` | `IAXL_KV_DATA_SHUFFLE` | With shuffle on, compress only the BF16 high bytes (sign and exponent) and store the low bytes raw (`0`/`1`). Improves the compression ratio and both throughputs; `IAXL_KV_LOSSY_TRUNC` then barely helps the ratio |
| vLLM | `MODEL` | `Qwen/Qwen3-32B` | Hugging Face model ID or local model path |
| vLLM | `TP_SIZE` | `2` | Required; number of Tensor Parallel workers. CPU, QAT, and DSA resources are configured based on this value |
| vLLM | `DP_SIZE` | `1` | Number of Data Parallel groups; total ranks = `TP_SIZE × DP_SIZE` |
| vLLM | `VLLM_CPU_OMP_THREADS_BIND` | Auto-detected from GPU NUMA topology | CPUs for vLLM, one `\|`-separated CPU list per rank |
| IAXL | `KVSHRINK_CPU_OMP_THREADS_BIND` | Same as `VLLM_CPU_OMP_THREADS_BIND` | CPUs for IAXL threads, one entry per rank. Set it for CPU inference; see the [vLLM CPU inference guide](doc/usage/vllm-cpu-inference.en.md) |
| IAXL | `IAXL_RDMA_ENABLE` | `0` | Use a remote KVStore daemon over RDMA (`0`/`1`); see [cross-node KV Cache reuse](#cross-node-kv-cache-reuse) |
| IAXL | `IAXL_RDMA_DAEMON_NIC_IPS` | unset | Daemon RDMA NIC IPs; required when `IAXL_RDMA_ENABLE=1` |
| IAXL | `IAXL_RDMA_CLIENT_NIC_IPS` | unset | Client RDMA NIC IPs; required on the client when `IAXL_RDMA_ENABLE=1` |
| IAXL | `IAXL_OMP_THREAD_NUM` | Number of compression workers | Threads driving compression. Fewer threads use fewer CPU cores |

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

To run the model on the CPU, build IAXL with `DEVICE=cpu` and follow the [vLLM CPU inference guide](doc/usage/vllm-cpu-inference.en.md).

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

## Cross-Node KV Cache Reuse

remote_pool supports storing KV Cache on a separate daemon node and transferring or reusing it with vLLM client nodes over RDMA. See the [cross-node KV Cache guide](doc/usage/remote-pool-kvstore.en.md) for the topology, build and startup steps, per-rank NIC configuration, and operations.
