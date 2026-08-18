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
