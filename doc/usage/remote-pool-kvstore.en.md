# Cross-Node KV Cache Reuse with remote_pool KVStore

[简体中文](remote-pool-kvstore.zh-CN.md)

**Applies to:** `iaxl/remote_pool/*`. See the [remote_pool design](../design/remote-pool.md).

The complete `KVStore` pipeline (KVFlow, scratch pool, compression, DDR pool, and persistence) runs on the daemon node. Each vLLM worker uses a thin `KVStoreRemote` wrapper with the same interface as the local `KVStore`; requests are forwarded to the daemon over RPC. The daemon initiates data-plane RDMA: `put` uses RDMA READ to read the client's GPU memory, and `get` uses RDMA WRITE to write to the client's GPU memory. The client does not perform data-plane transfers and does not need a GPU copy stream or scratch pool.

---

## 1. Example Topology

The examples below use a TP=4 setup:

- **Daemon node** (no GPU): `10.10.10.10`, an IP address on its RDMA NIC
- **Client node** (vLLM, TP=4): `10.10.10.11`, an IP address on its RDMA NIC
- **Default single-NIC setup:** control, RPC, and RDMA data traffic use the RDMA NIC associated with the IP above. Per-rank NIC selection is described in section 3.3.

Port allocation (`IAXL_RDMA_DAEMON_PORT` defaults to `5555`; `rank_port(port, r) = port + 1 + r`):

| Role | Listening port on daemon |
| --- | ---: |
| Scheduler process (has-only KVStore) | `5555` |
| Rank 0 KVStore | `5556` |
| Rank 1 KVStore | `5557` |
| Rank 2 KVStore | `5558` |
| Rank 3 KVStore | `5559` |

The client has five processes (one vLLM scheduler and four TP workers), each connecting to its corresponding port.

The management REST API (`/v1/cache/*`) listens on the daemon node:

- Controller: `daemon_ip:${IAXL_API_CONTROLLER_PORT:-18700}` (in the scheduler process)
- Worker `r`: `daemon_ip:${IAXL_API_WORKER_BASE_PORT:-18800}+r` (the controller also forwards requests to local `127.0.0.1:18800..18803`)

Use `http://10.10.10.10:18700/v1/cache/*` as the management entry point.

---

## 2. Daemon Node (`10.10.10.10`): Build and Start

Build and start each node from the same code revision. The daemon node has no GPU, so you must set `NVIDIA_RUNTIME=none` when running `start.sh` on the host to omit the NVIDIA container runtime and GPU arguments. On the client node, run `start.sh` normally. This variable only controls the host-side Docker arguments; it is not passed into the container and does not affect image building.

`start.sh` passes `MODEL`, `TP_SIZE`, and the remote-pool connection variables listed in `setvars.sh` into the container. These are runtime settings, not image-build inputs. To keep configuration consistent across both nodes, the examples below set them in the container shell opened by `start.sh`. Configure compression and resource settings there as well, before starting the service. If the client uses a local model directory outside the repository, set `MODEL` on the host before running `start.sh` so the script mounts that directory into the container; for a model ID or a path already available in the container, set it inside the container. Run the steps from the repository root on each node.

### 2.1 Environment Variables and Parameters

The daemon startup script is [`examples/kvshrink-daemon.sh`](../../examples/kvshrink-daemon.sh). It sources `setvars.sh` and runs `python3 -m iaxl.remote_pool.daemon`. It accepts three command-line parameters, each of which can also be set through an environment variable:

| Parameter | Environment variable / default | Description |
| --- | --- | --- |
| `--ip` | `IAXL_RDMA_DAEMON_IP` | Local RDMA NIC IP and NIXL listen address (each rank process listens on its own port) |
| `--port` | `IAXL_RDMA_DAEMON_PORT` (default `5555`) | Scheduler port; rank `r` uses `port + 1 + r` |
| `--tp-size` | `IAXL_RDMA_TP_SIZE` (default `$TP_SIZE`) | Number of rank processes to spawn; **must match** vLLM's `tensor_parallel_size` |

Key environment variables (parsed in `iaxl/envs.py` and `setvars.sh`):

| Variable | Recommended value / default | Purpose |
| --- | --- | --- |
| `IAXL_RDMA_ENABLE` | `1` (required) | Enables the remote_pool path. The daemon refuses to start when set to `0`. |
| `IAXL_RDMA_DAEMON_IP` | `10.10.10.10` | Daemon listen address and the client's connection target; selects the default data-plane NIC when no per-rank list is configured. |
| `IAXL_RDMA_DAEMON_NIC_IPS` | Unset | List of local RDMA NIC IPs for daemon workers. Selects each rank's local NIC without changing the listen address or ports. |
| `IAXL_RDMA_DAEMON_PORT` | `5555` | Scheduler listen port; rank `r` listens on `5555 + 1 + r`. |
| `IAXL_RDMA_TP_SIZE` | `4` | Number of ranks in the TP group. **Must match on client and daemon**; checked during `register_kv_caches`. |
| `VLLM_CPU_OMP_THREADS_BIND` | `cpu_auto_detect $TP_SIZE` | `setvars.sh` creates a `|`-separated list; each daemon rank uses its segment for CPU affinity. |
| `KVSHRINK_QAT_DEVICES` | For example, `"0,1|2,3|4,5|6,7"` | Per-rank QAT devices; rank `r` uses entry `r` as `IAXL_QAT_DEVICES`. |
| `KVSHRINK_DSA_DEVICES` | For example, `"wq0.0|wq1.0|wq2.0|wq3.0"` (optional) | Per-rank DSA work queues, assigned to `IAXL_DSA_WQS`. |
| `IAXL_KV_COMPRESSION` / `IAXL_QAT_ZIP_ENABLE` / `IAXL_IAA_ZIP_ENABLE` / `IAXL_CPU_ZIP_ENABLE` | See `setvars.sh` | Compression backend settings for the daemon. |
| `IAXL_SCRATCH_POOL_SIZE_GB` | `8` | Pinned scratch pool size. The daemon uses it to determine the maximum number of in-flight put/get blocks. |
| `IAXL_DDR_POOL_SIZE_GB` | Unset: one tenth of host RAM | Daemon DDR cache pool budget. |
| `IAXL_CACHE_DIR` | `_data/kvcache` | Root directory for persistence on the daemon (a `chunks.db` and `chunks/` per `(model, tp_size, rank)`). |
| `IAXL_API_CONTROLLER_PORT` / `IAXL_API_WORKER_BASE_PORT` | `18700` / `18800` | Ports for the management REST API on the daemon. |

### 2.2 Build and Start (TP=4 Example)

```bash
# ---- On the daemon node, from the repository root ----
NVIDIA_RUNTIME=none ./start.sh

# ---- Run the following commands in the container shell opened by start.sh ----
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_IP=10.10.10.10
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4

# Configure compression and resources for the daemon hardware; this is a QAT example
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export KVSHRINK_QAT_DEVICES="0|1|4|5"
export IAXL_SCRATCH_POOL_SIZE_GB=16
# Optional: export IAXL_DDR_POOL_SIZE_GB=32

./examples/kvshrink-daemon.sh
```

Startup logs show the listen address for the scheduler and each of the four rank processes. If any child process exits, the launcher terminates the others and exits as well.

### 2.3 Tuning Recommendations

| Area | Recommendation |
| --- | --- |
| Compression | With QAT, set `IAXL_QAT_ZIP_ENABLE=1` and assign devices with `KVSHRINK_QAT_DEVICES`. With IAA only, set `IAXL_IAA_ZIP_ENABLE=1`. For CPU-only compression, set `IAXL_CPU_ZIP_ENABLE=1`. |
| Scratch pool | `IAXL_SCRATCH_POOL_SIZE_GB=16` is a useful starting point. Adjust after checking the `pool: N blocks x M descs of L B` log and its `nblocks` value. |
| DDR pool | For long contexts or a high hit rate, consider increasing `IAXL_DDR_POOL_SIZE_GB` to `RAM/2`; otherwise use the default. |

---

## 3. Client Node (`10.10.10.11`): Build and Start vLLM

When `IAXL_RDMA_ENABLE=1`, `kvshrink_connector` transparently uses `iaxl.remote_pool.kvstore_remote.KVStoreRemote` instead of `iaxl.kvstore.KVStore` (see `_kvstore_cls()` in [`kvshrink/kvshrink_connector.py`](../../kvshrink/kvshrink_connector.py)). The client does not perform compression or bind CPU/accelerator resources for the remote KVStore; these operations run on the daemon.

### 3.1 Environment Variables

| Variable | Value | Purpose |
| --- | --- | --- |
| `IAXL_RDMA_ENABLE` | `1` | Selects `KVStoreRemote`. |
| `IAXL_RDMA_DAEMON_IP` | `10.10.10.10` | Daemon connection target. |
| `IAXL_RDMA_DAEMON_PORT` | `5555` | Must match the daemon; the scheduler connects to `5555`, and worker `r` to `5555 + 1 + r`. |
| `IAXL_RDMA_CLIENT_IP` | `10.10.10.11` | Client scheduler's local RDMA NIC IP; also selects the default data-plane NIC for workers when no per-rank list is configured. |
| `IAXL_RDMA_CLIENT_NIC_IPS` | Unset | List of local RDMA NIC IPs for client workers; does not change the daemon connection target. |
| `IAXL_RDMA_TP_SIZE` | `4` | Must match vLLM's `-tp` value. |
| `MODEL` | `Qwen/Qwen3-32B` | Model ID or local path loaded by vLLM; the client sends the model identifier to the daemon when registering KV caches. |
| `TP_SIZE` | `4` | vLLM tensor-parallel size. |

Do not configure `KVSHRINK_QAT_DEVICES`, `KVSHRINK_DSA_DEVICES`, `IAXL_QAT_*`, `IAXL_CPU_ZIP_*`, `IAXL_IAA_*`, `IAXL_SCRATCH_POOL_SIZE_GB`, or `IAXL_DDR_POOL_SIZE_GB` on the client; these apply only to the daemon.

### 3.2 Build and Start

```bash
# ---- On the client node, from the repository root ----
./start.sh

# ---- Run this command in the container shell opened by start.sh ----
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_IP=10.10.10.10
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_CLIENT_IP=10.10.10.11
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4
export MODEL=Qwen/Qwen3-32B

./examples/kvshrink-vllm-serve.sh
```

After vLLM starts, five `KVStoreRemote connected: peer=... rank=... has_only=...` messages (one has-only process and four workers) indicate that all ranks completed the handshake.

### 3.3 Select an RDMA NIC Per Rank (Optional, TP=4 Example)

`IAXL_RDMA_DAEMON_IP` is required: it sets the listen address for all daemon ranks and is the client connection target (the port remains `port + 1 + rank`). Set `IAXL_RDMA_CLIENT_IP` for the client scheduler's local NIC; otherwise, the system selects one. The `*_NIC_IPS` lists select a local NIC for each worker without changing the metadata handshake listen address or connection target. The scheduler continues to use a single IP.

Each node's list is indexed by rank; rank `r` uses the IP at index `r % list_length`. In this example, ranks 0 and 1 use `10.10.10.x`, while ranks 2 and 3 use `10.10.11.x`:

```bash
# Daemon container: set before starting the daemon
export IAXL_RDMA_DAEMON_IP=10.10.10.10
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10,10.10.10.10,10.10.11.10,10.10.11.10

# Client container: set before starting vLLM
export IAXL_RDMA_DAEMON_IP=10.10.10.10
export IAXL_RDMA_CLIENT_IP=10.10.10.11
export IAXL_RDMA_CLIENT_NIC_IPS=10.10.10.11,10.10.10.11,10.10.11.11,10.10.11.11
```

Each IP must belong to an RDMA NIC on its own node. The selected interfaces for the same rank on both nodes must be reachable over a compatible RoCE network. Interface names may differ between nodes; pair interfaces by network reachability, not by names such as `mlx5_*`. List entries must not be empty. Without a list, the single-NIC setup is used. Check local IP ownership with `ip -j -4 addr`; compare the per-port `port_rcv_data` counter under `/sys/class/infiniband/<device>/ports/1/counters/` to confirm traffic distribution. Evaluate throughput and TTFT together with GPU and NIC NUMA topology.

### 3.4 Startup Order

**Start the daemon before the client.** The client's `KVStoreRemote.__init__` calls `xfer.connect`, which times out after `IAXL_API_TIMEOUT` (60 seconds by default). vLLM exits if the daemon does not become available within that period.

---

## 4. Verification and Operations

- **Check connectivity** (without starting vLLM): after the daemon is up, run this on the client:
  ```bash
  curl -s http://10.10.10.10:18700/v1/cache/status | jq .
  ```
  A response such as `{"workers": [...]}` indicates that the control plane, including all four ranks, is ready.
- **Check data-plane activity:** on the first matching put/get, daemon rank logs should include `[kv_xfer/rdma] pool: N blocks x M descs of L B` and `[kv_xfer] copy_chunks_batch: using RDMA backend`.
- **Clear the cache:** `curl -s -X POST http://10.10.10.10:18700/v1/cache/evict -d '{"count": 500000000000}'`. The controller forwards the request to all four local workers.
- **Read metrics:** `curl -s http://10.10.10.10:18700/v1/cache/metrics`.

---

## 5. Troubleshooting

- `--ip or IAXL_RDMA_DAEMON_IP is required`: the daemon was started without a listen IP.
- `IAXL_RDMA_ENABLE=1 is required on the daemon`: set `IAXL_RDMA_ENABLE=1` in the daemon container shell before running `examples/kvshrink-daemon.sh`.
- `no network interface owns 10.10.10.10`: the IP is not assigned to a local interface. Check with `ip -j -4 addr` before importing NIXL or initializing RDMA.
- `<netdev> (10.10.10.10) is not an RDMA-capable NIC`: the interface exists but has no RDMA device under `/sys/class/net/*/device/infiniband/`. Verify that the interface supports RDMA and that MLNX_OFED or rdma-core is installed on the host.
- `client tp_size N != daemon tp_size M`: vLLM's `-tp` and daemon's `--tp-size` must match exactly.
- `rdma_register_local called twice`: each rank should register the pool once. This can indicate that the KVStore was created more than once.
- Cannot connect to REST on daemon port `18700`: check the daemon firewall and ensure `IAXL_API_CONTROLLER_PORT` matches the client URL.
