# Remote (Cross-Node) KV Cache over NIXL -- Usage

Design: [`../design/remote-nixl-kv-cache.md`](../design/remote-nixl-kv-cache.md).
中文版: [`remote-nixl-kv-cache.zh-CN.md`](remote-nixl-kv-cache.zh-CN.md).

This backend lets a vLLM/KVShrink deployment offload its KV cache pool to a
separate node over NIXL (RDMA/GDR), instead of the local, GPU-attached
pool. Enable it with `KVSHRINK_REMOTE_CACHE_ENABLE=1`; the connector then
swaps `iaxl.kvstore.KVStore` for `iaxl.remote.RemoteKVStore` transparently
(see `kvshrink/kvshrink_connector.py::_make_kvstore`).

---

## 1. Build

**Only needs to be done once, on the GPU node.** The remote cache node does
NOT need any build toolchain or `Dockerfile.dev`; it runs directly on the
same `vllm/vllm-openai:v0.23.0` base image (see section 2, "Remote daemon
node").

```bash
cd /mnt/data/zengjun/intel-accel-for-llm
./start.sh
```

`./start.sh` builds `docker/Dockerfile.dev` on top of
`vllm/vllm-openai:v0.23.0`, enters the container and runs
`pip install -e . --no-build-isolation`, which compiles the `iaxl` package
(including `torch_ext*.so` and the compression entry points
`zip_compress_to_mem` / `zip_decompress_from_mem`).

Once the build succeeds, package the standalone, relocatable release bundle
for the remote daemon node:

```bash
# Run AFTER ./start.sh has finished; works either inside the dev container
# or on the host (the script only reads the already-compiled
# iaxl/torch_ext*.so and _lib/*.so; it does not invoke any compiler).
tools/remote_daemon/build_release.sh
# -> dist/iaxl-remote-daemon-v0.1.tar.gz
```

To stamp a different version tag (e.g. v0.2, or a git sha for a hotfix),
override with `RELEASE_VERSION=v0.2 tools/remote_daemon/build_release.sh`.

---

## 2. Deployment

Start the remote daemon FIRST, then the GPU-side vLLM;
`KVSHRINK_REMOTE_FAIL_IF_UNREACHABLE=1` (default) makes each worker exit
during init if it cannot reach the daemon.

### 2.1 Remote daemon node

Only Docker and the `vllm/vllm-openai:v0.23.0` image are required. If the
image is not already present locally, `docker-run-daemon.sh` will pull it
from Docker Hub on the first `docker run` (for air-gapped nodes, do a
`docker pull` + `docker save` on a networked host and `docker load` on the
target).

```bash
# Copy the release bundle from the GPU node and extract
scp dist/iaxl-remote-daemon-v0.1.tar.gz root@10.10.10.10:/root/
ssh root@10.10.10.10
tar xzf /root/iaxl-remote-daemon-v0.1.tar.gz -C /root
cd /root/iaxl-remote-daemon-v0.1
```

**Recommended example (one daemon instance per GPU rank, dual-rail RDMA):**

```bash
NUM_INSTANCES=4 CONTROL_PORT_BASE=19000 NIXL_PORT_BASE=19100 \
TRANSPORT=nixl \
NIXL_DEVICE=rocep21s0f0:1,rocep21s0f1:1 \
NIXL_HOST=10.10.10.10 \
POOL_SIZE_GB=64 STAGING_SLOTS=256 STAGING_SLOT_MB=8 \
KVSHRINK_REMOTE_QAT_DEVICES="0|1|4|5" \
tools/remote_daemon/run-daemon-multi.sh
```

`run-daemon-multi.sh` auto-detects whether the current directory is a
release bundle (presence of `pysrc/iaxl/`) or an in-repo checkout, and
picks `docker-run-daemon.sh` (release bundle, starts a
`vllm/vllm-openai:v0.23.0` container) or `run-daemon.sh` (in-repo)
accordingly; it prints `USE_DOCKER=0|1` at start. Force either mode with
`USE_DOCKER=1` / `USE_DOCKER=0`.

#### Disable compression (raw bandwidth benchmark)

```bash
# add one line
COMPRESS=0 \
```

With compression off, QAT/IAA devices are unused;
`KVSHRINK_REMOTE_QAT_DEVICES` / `IAXL_QAT_ZIP_ENABLE` can be omitted.

#### Selecting compression devices (QAT / IAA / CPU)

`KVSHRINK_REMOTE_QAT_DEVICES` uses `"|"` to separate instances and commas
to separate devices within an instance; its length must be **exactly**
`NUM_INSTANCES` (an empty segment silently gives that instance no device,
which shows up as `qat_devices=<none>` in the log). With
`NUM_INSTANCES=1`, use a comma-separated list, e.g. `"0,1,4,5"`.

Other compression-backend switches (identical to the local KVStore path;
defaults come from `setvars.sh`):

| Variable | Meaning |
| --- | --- |
| `IAXL_QAT_ZIP_ENABLE` | Enable QAT compression |
| `IAXL_IAA_ZIP_ENABLE` | Enable IAA compression |
| `IAXL_CPU_ZIP_ENABLE` | Enable CPU compression fallback |
| `IAXL_KV_COMPRESSION` | Master switch for the KV compression pipeline |

#### OMP / QAT instance count

The per-device compression instance count (i.e. OMP concurrency) is
`IAXL_QAT_ZIP_INSTANCES_PER_DEVICE` (default 4); `run-daemon.sh` then
derives `IAXL_QAT_INSTANCE_NUM = |IAXL_QAT_DEVICES| ×
IAXL_QAT_ZIP_INSTANCES_PER_DEVICE` automatically. Usually there is no need
to override this, unless you want to force a specific number:

```bash
# 8 instances per device × 4 devices = 32 concurrent codecs
IAXL_QAT_ZIP_INSTANCES_PER_DEVICE=8 \
```

Codec calls inside one daemon process are serialized (GET jumps ahead of
PUT); to increase real concurrency, add QAT devices or `NUM_INSTANCES`
rather than tuning semaphores.

#### Single RDMA device (no multi-rail)

Drop the comma from the device list:

```bash
NIXL_DEVICE=rocep21s0f0:1 \
```

Multi-rail (comma list) lets UCX stripe a single NIXL session across
multiple ports; only useful once one port has saturated a NIC.

#### Single daemon process (shared by all ranks)

```bash
CONTROL_PORT=19000 \
TRANSPORT=nixl \
NIXL_DEVICE=rocep21s0f0:1 \
NIXL_HOST=10.10.10.10 NIXL_PORT=19100 \
POOL_SIZE_GB=64 STAGING_SLOTS=256 STAGING_SLOT_MB=8 \
KVSHRINK_REMOTE_QAT_DEVICES="0,1,4,5" \
tools/remote_daemon/docker-run-daemon.sh
```

All `tp_size` workers then share one daemon's control plane and codec --
useful for light load or debugging; production deployments should prefer
`NUM_INSTANCES=tp_size`.

#### Other common daemon-side variables

| Variable | Default | Description |
| --- | --- | --- |
| `CONTROL_HOST` | `0.0.0.0` | Control-plane listen address |
| `POOL_SIZE_GB` | `8` | Pool budget per `(model, tp_size, tp_rank)` group |
| `CACHE_DIR` | `_data/kvcache/remote` | Root of persistent `chunks.db` + `chunks/` |
| `STAGING_SLOTS` | `0` | NIXL staging slot count; **must** be > 0 when `TRANSPORT=nixl` |
| `STAGING_SLOT_MB` | `4` | Nominal slot size; the daemon repartitions once the real shard size is known |
| `KVSHRINK_REMOTE_STATS_INTERVAL_SEC` | `30` | Periodic throughput/queueing log; `0` disables |
| `DEVICE` | `cpu` | Device backing the NIXL staging buffer |

### 2.2 GPU node (vLLM side)

Set these environment variables before starting vLLM
(`examples/kvshrink-vllm-serve.sh`); they are read by
`iaxl.remote.config.RemoteCacheConfig.from_vllm`.

**Recommended example (4-rank TP, dual-rail RDMA, paired with the
multi-instance daemon above):**

```bash
KVSHRINK_REMOTE_CACHE_ENABLE=1 \
KVSHRINK_REMOTE_TRANSPORT=nixl \
KVSHRINK_REMOTE_DAEMON_ADDR="10.10.10.10:19000|10.10.10.10:19001|10.10.10.10:19002|10.10.10.10:19003" \
KVSHRINK_REMOTE_NIXL_ADDR="10.10.10.10:19100|10.10.10.10:19101|10.10.10.10:19102|10.10.10.10:19103" \
KVSHRINK_REMOTE_NIXL_DEVICE=mlx5_0:1,mlx5_1:1 \
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 \
UCX_IB_GPU_DIRECT_RDMA=y \
MODEL=/mnt/ssd1/model-space/Qwen/Qwen2.5-32B-Instruct \
./examples/kvshrink-vllm-serve.sh
```

`KVSHRINK_REMOTE_DAEMON_ADDR` / `KVSHRINK_REMOTE_NIXL_ADDR`, when
`"|"`-separated, map rank `r` to entry `r`; a single value shares one
daemon across every rank.

Model loading, TP size, KVShrink's own switches and any other generic vLLM
knobs are out of scope for this feature -- follow the top-level
[`../../README.md`](../../README.md) (Chinese
[`../../README.zh-CN.md`](../../README.zh-CN.md)); they are not repeated
here.

#### Single RDMA device

```bash
KVSHRINK_REMOTE_NIXL_DEVICE=mlx5_0:1 \
UCX_NET_DEVICES=mlx5_0:1 \
```

Both must match (the former is forwarded into UCX).

#### Single daemon process

Use a single `host:port` for `KVSHRINK_REMOTE_DAEMON_ADDR` /
`KVSHRINK_REMOTE_NIXL_ADDR` instead of a `"|"`-separated list:

```bash
KVSHRINK_REMOTE_DAEMON_ADDR=10.10.10.10:19000 \
KVSHRINK_REMOTE_NIXL_ADDR=10.10.10.10:19100 \
```

#### Other vLLM-server-side variables

| Variable | Default | Description |
| --- | --- | --- |
| `KVSHRINK_REMOTE_CACHE_ENABLE` | `0` | **Master switch.** `1` enables the cross-node NIXL backend; `0` (default) makes the connector use the original **local-DDR KVShrink** path and never contact any daemon -- any other `KVSHRINK_REMOTE_*` variables are ignored in this mode. Use this as the one-touch rollback to the on-box path. |
| `KVSHRINK_REMOTE_TRANSPORT` | `nixl` | `nixl` (production) or `tcp` (fallback for validation) |
| `KVSHRINK_REMOTE_CONNECT_TIMEOUT_SEC` / `..._RETRY_SEC` | 30 / 2 | Connect retry parameters |
| `KVSHRINK_REMOTE_REQUEST_TIMEOUT_SEC` | 120 | Per-RPC / transfer timeout |
| `KVSHRINK_REMOTE_FAIL_IF_UNREACHABLE` | `1` | Exit at init if the daemon is unreachable |

### 2.3 Network & device selection notes

- `KVSHRINK_REMOTE_NIXL_DEVICE` / `UCX_NET_DEVICES` (GPU side) and
  `NIXL_DEVICE` (remote daemon side) must select ports on the **same RDMA
  subnet**; the device names do not have to match (e.g. `mlx5_0:1` on the
  GPU side, `rocep21s0f0:1` on the remote side) as long as they can talk to
  each other.
- On RoCE, do MTU / PFC / ECN tuning before running any performance test.
- A comma-separated device list on either side enables UCX multi-rail
  striping for that NIXL session; nothing else is needed to actually use
  multiple RDMA links.

---

## 3. Daemon-side CLI

The equivalent of the local KVStore's `/v1/cache/*` HTTP endpoints is
exposed as a sessionless admin RPC on the daemon, dispatched by
[`iaxl.remote.admin`](../../iaxl/remote/admin.py) with a convenience
wrapper at
[`tools/remote_daemon/remote-cli.sh`](../../tools/remote_daemon/remote-cli.sh)
(shipped inside the release bundle by `build_release.sh`). It is
pure control-plane TCP -- no RDMA / QAT / IAA needed, and it does not
interfere with running vLLM workers.

**Where to run it:** anywhere that can reach the daemon's `CONTROL_PORT`
AND can `import iaxl`. The two recommended options:

- **On the remote storage node, via `docker exec` into the daemon
  container** (the release bundle already contains both `remote-cli.sh`
  and `pysrc/iaxl`):

  ```bash
  docker exec iaxl-remote-daemon-0 \
      bash /root/iaxl-remote-daemon-v0.1/tools/remote_daemon/remote-cli.sh \
      --daemon 127.0.0.1:19000 status
  ```

- **On the GPU node** (`pip install -e .` already done), pointing at
  whichever instance's `CONTROL_PORT`:

  ```bash
  cd /mnt/data/zengjun/intel-accel-for-llm
  tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 status
  ```

### Common commands

```bash
# Overall status + pool occupancy / hit / compression ratio per
# (model, tp_size, tp_rank) group
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 status

# One-shot LRU eviction (up to 32 groups; omit --model/--tp-* to hit every group)
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 evict --count 32

# Persist 32 groups to disk, optionally filtered by group
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 persist --count 32 \
    --model Qwen2.5-32B-Instruct --tp-size 4 --tp-rank 0

# List the next batch of persist / evict candidates
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 candidates --which evict --count 10

# Native compression / decompression throughput metrics
# (same data as the local `/v1/cache/metrics`; optionally --reset)
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 metrics --reset
```

Pass `--json` for the raw response (for scripting).

### Multi-instance daemon

With `NUM_INSTANCES>1`, each instance listens on its own `CONTROL_PORT`
and the admin RPCs are per-instance -- one persist / evict call targets
one instance. Loop in shell to sweep all of them:

```bash
for p in 19000 19001 19002 19003; do
    tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:$p status
done
```
