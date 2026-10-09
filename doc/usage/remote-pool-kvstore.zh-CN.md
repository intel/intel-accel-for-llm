# 跨节点 KV Cache 复用（remote_pool KVStore 版）—— 使用文档

[English](remote-pool-kvstore.en.md)

**适用版本**：`iaxl/remote_pool/*`，见 [remote_pool 设计文档](../design/remote-pool.md)。

实现的核心思路：**把整个 `KVStore`（KVFlow + ScratchPool + 压缩 + DDR 池 +
持久化）搬到 daemon 节点**；vLLM worker 只保留一个薄壳 `KVStoreRemote`，接口与
本地 `KVStore` 完全一致，内部全部通过 RPC 转发到 daemon。**数据面由 daemon 侧
发起 RDMA**：`put` = daemon RDMA `READ` client 显存，`get` = daemon RDMA `WRITE`
client 显存；client 不参与数据面、不需要 GPU 拷贝流、也不需要 scratch pool。

---

## 一、示例拓扑

本文以下面 TP=4 的通路为例说明：

- **daemon 节点**（无 GPU）：`10.10.10.10`，RDMA NIC 持有该 IP
- **client 节点**（vLLM，TP=4）：`10.10.10.11`，RDMA NIC 持有该 IP
- **网口配置**：两端都用必填的 IP 列表（`IAXL_RDMA_DAEMON_NIC_IPS` /
  `IAXL_RDMA_CLIENT_NIC_IPS`）指定 RDMA 网口；各 rank 按列表选择数据面网口，列表
  第一个 IP 在承担RDMA 数据传输之外也用于控制面。列表只有一个 IP 时即单网口配置，多网口见 3.3 节

端口分配（`IAXL_RDMA_DAEMON_PORT` 默认 `5555`，`rank_port(port, r) = port + 1 + r`）：

| 角色 | daemon 节点监听端口 |
|------|--------------------|
| scheduler 进程（has-only KVStore） | `5555` |
| rank 0 KVStore | `5556` |
| rank 1 KVStore | `5557` |
| rank 2 KVStore | `5558` |
| rank 3 KVStore | `5559` |

client 侧共 5 个进程（vLLM scheduler + 4 个 TP worker）分别连到对应端口。

管理面 REST（`/v1/cache/*`，用于运维/清理），仍走 `KVStore.__init__` 启动的
HTTP server，**监听在 daemon 节点**：

- controller: `daemon_ip:${IAXL_API_CONTROLLER_PORT:-18700}`（在 scheduler 进程里）
- worker r: `daemon_ip:${IAXL_API_WORKER_BASE_PORT:-18800}+r`（controller 会自动
  广播到本机 `127.0.0.1:18800..18803`）

因此**运维只需要一个入口**：`http://10.10.10.10:18700/v1/cache/*`。

---

## 二、daemon 节点（10.10.10.10）：编译并启动

daemon 与 client 节点分别使用同一代码版本构建和启动。daemon 节点没有 GPU，必须在
宿主机运行 `start.sh` 时设置 `NVIDIA_RUNTIME=none`，跳过 NVIDIA 容器 runtime 和
GPU 参数；client 节点按常规方式运行 `start.sh`。该变量只控制宿主机上的 Docker 启动
参数，不会传入容器，也不影响镜像构建。

`start.sh` 会把 `MODEL`、`TP_SIZE` 和 `setvars.sh` 中列出的变量透传给容器；它们是
运行配置，不是镜像构建输入。为便于两侧配置，下面示例在`start.sh` 打开的容器 shell 中
设置这些变量。压缩和资源配置也应在容器内、启动服务前设置。若 client 使用仓库目录之外的
本地模型路径，需在宿主机运行 `start.sh` 前设置`MODEL`，使脚本将该目录挂载进容器；使用
模型 ID 或容器内已有路径时，可在容器内设置。
下面步骤都从各自节点的仓库根目录执行。

### 2.1 环境变量与参数

daemon 侧的启动脚本是 [`examples/kvshrink-daemon.sh`](../../examples/kvshrink-daemon.sh)，
它 `source setvars.sh` 之后调用 `python3 -m iaxl.remote_pool.daemon`。命令行
参数只有 3 个（都可以由环境变量给出默认值）：

| 参数 | 环境变量默认 | 含义 |
|------|-------------|------|
| `--nic-ips` | `IAXL_RDMA_DAEMON_NIC_IPS` | 本机 RDMA NIC IP 列表（逗号分隔，必填，可以是单个IP）；第一个 IP 也是控制面地址，rank r 的数据面使用第 `r % 列表长度` 个 IP |
| `--port` | `IAXL_RDMA_DAEMON_PORT`（默认 `5555`） | scheduler 端口；rank r 用 `port + 1 + r` |
| `--tp-size` | `IAXL_RDMA_TP_SIZE`（默认 `$TP_SIZE`） | 要 spawn 的 rank 进程数，**必须等于** vLLM 的 `tensor_parallel_size` |

关键环境变量（参数解析在 `iaxl/envs.py` + `setvars.sh`）：

| 变量 | 推荐值 / 默认 | 作用 |
|------|--------------|------|
| `IAXL_RDMA_ENABLE` | `1`（必需） | 打开 remote_pool 分支。`0` 时 daemon 会拒绝启动 |
| `IAXL_RDMA_DAEMON_NIC_IPS` | `10.10.10.10`（必填） | daemon 本机 RDMA NIC IP 列表，必填，可以是单个IP。第一个 IP 也是控制面地址（client 连接目标、scheduler 网口、管理 REST 入口）；rank r 使用第 `r % 列表长度` 个 IP 对应的网口 |
| `IAXL_RDMA_DAEMON_PORT` | `5555` | scheduler 监听；rank r 监听 `5555+1+r` |
| `IAXL_RDMA_TP_SIZE` | `4` | TP 组内 rank 进程数。**client 与 daemon 必须一致**，`register_kv_caches` 时会校验 |
| `VLLM_CPU_OMP_THREADS_BIND` | `cpu_auto_detect $TP_SIZE`（由 `setvars.sh` 自动填 `\|` 分段） | daemon rank 进程按 rank 从该 `\|`-分段列表里取自己那一段做 `sched_setaffinity`（复用 connector 里那套 `bind_cpu_affinity`） |
| `KVSHRINK_QAT_DEVICES` | 例如 `"0,1\|2,3\|4,5\|6,7"` | 4 个 rank 各自的 QAT 设备；rank r 取第 r 项赋给 `IAXL_QAT_DEVICES` |
| `KVSHRINK_DSA_DEVICES` | 例如 `"wq0.0\|wq1.0\|wq2.0\|wq3.0"`（可选） | 同上，赋给 `IAXL_DSA_WQS` |
| `IAXL_KV_COMPRESSION` / `IAXL_QAT_ZIP_ENABLE` / `IAXL_IAA_ZIP_ENABLE` / `IAXL_CPU_ZIP_ENABLE` | 见 `setvars.sh` | 与本机 KVStore 完全一致的压缩后端开关 |
| `IAXL_SCRATCH_POOL_SIZE_GB` | `8`（默认） | pinned scratch pool 大小；daemon 侧决定单次 in-flight put/get 的最大块数 |
| `IAXL_DDR_POOL_SIZE_GB` | 未设 = 主机 RAM 的 1/10 | daemon 侧 DDR 缓存池预算 |
| `IAXL_CACHE_DIR` | `_data/kvcache` | daemon 侧持久化根目录（每个 `(model, tp_size, rank)` 一组 `chunks.db + chunks/`） |
| `IAXL_API_CONTROLLER_PORT` / `IAXL_API_WORKER_BASE_PORT` | `18700` / `18800` | 管理 REST（`/v1/cache/*`）监听在 daemon 节点 |

### 2.2 编译并启动（TP=4 示例）

```bash
# ---- daemon 节点：在仓库根目录执行 ----
NVIDIA_RUNTIME=none ./start.sh

# ---- 以下命令在 start.sh 打开的容器 shell 中执行 ----
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10   # 单网口；多网口见 3.3 节
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4

# 按 daemon 节点的硬件配置压缩后端和资源；以下为 QAT 示例
export IAXL_KV_COMPRESSION=1
export IAXL_QAT_ZIP_ENABLE=1
export KVSHRINK_QAT_DEVICES="0|1|4|5"
export IAXL_SCRATCH_POOL_SIZE_GB=16

./examples/kvshrink-daemon.sh
```

启动日志会分别显示 scheduler 和 4 个 rank 进程选中的 RDMA 网口（`RDMA NIC ...`）及监听地址。若任何一个子进程崩溃，
launcher 会自动终止其它子进程并退出。

### 2.3 参数推荐

| 场景 | 推荐 |
|------|------|
| 压缩后端 | 硬件有 QAT → `IAXL_QAT_ZIP_ENABLE=1` + `KVSHRINK_QAT_DEVICES=<按 rank 分>`；没有 QAT 只有 IAA → `IAXL_IAA_ZIP_ENABLE=1`；纯 CPU 兜底 → `IAXL_CPU_ZIP_ENABLE=1` |
| Scratch pool | `IAXL_SCRATCH_POOL_SIZE_GB=16`（每层 in-flight 上限）；日志出现 `pool: N blocks x M descs of L B` 后可根据 `nblocks` 调整 |
| DDR 池 | 长上下文 / 高 hit-rate 场景把 `IAXL_DDR_POOL_SIZE_GB` 加到 `RAM/2`；否则用默认 |

---

## 三、client 节点（10.10.10.11）：编译并启动 vLLM

### 3.1 环境变量

`kvshrink_connector` 在 `IAXL_RDMA_ENABLE=1` 时透明地把 `iaxl.kvstore.KVStore` 换
成 `iaxl.remote_pool.kvstore_remote.KVStoreRemote`（见
[`kvshrink/kvshrink_connector.py`](../../kvshrink/kvshrink_connector.py) 里的
`_kvstore_cls()`）。**client 侧不再做压缩 / 绑核 / 绑加速器**——这些都搬到
daemon 节点了；connector 里同一分支上（L148）会自动跳过
`_bind_cpu_affinity` / `_bind_intel_accel`。

| 变量 | 值 | 作用 |
|------|-----|------|
| `IAXL_RDMA_ENABLE` | `1` | 切到 `KVStoreRemote` |
| `IAXL_RDMA_DAEMON_NIC_IPS` | `10.10.10.10`（必填，可以是单个IP） | 与 daemon 侧取值相同；client 只使用第一个 IP 作为连接目标 |
| `IAXL_RDMA_DAEMON_PORT` | `5555` | 与 daemon 侧一致；scheduler 连 `5555`，worker r 连 `5555+1+r` |
| `IAXL_RDMA_CLIENT_NIC_IPS` | `10.10.10.11`（必填，可以是单个IP） | client 本机 RDMA NIC IP 列表。第一个 IP 也用于 client scheduler；worker r 使用第 `r % 列表长度` 个 IP 对应的网口 |
| `IAXL_RDMA_TP_SIZE` | `4` | 必须等于 vLLM `-tp` |
| `MODEL` | `Qwen/Qwen3-32B` | vLLM 加载的模型 ID 或本地路径；client 注册 KV Cache 时会将模型标识发送给 daemon |
| `TP_SIZE` | `4` | vLLM `-tp` |

**不需要**设置的东西：`KVSHRINK_QAT_DEVICES`、`KVSHRINK_DSA_DEVICES`、
`IAXL_QAT_*`、`IAXL_CPU_ZIP_*`、`IAXL_IAA_*`、`IAXL_SCRATCH_POOL_SIZE_GB`、
`IAXL_DDR_POOL_SIZE_GB`——它们只在 daemon 节点生效。

### 3.2 编译并启动

```bash
# ---- client 节点：在仓库根目录执行 ----
./start.sh

# ---- 以下命令在 start.sh 打开的容器 shell 中执行 ----
export IAXL_RDMA_ENABLE=1
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10   # 与 daemon 侧一致
export IAXL_RDMA_DAEMON_PORT=5555
export IAXL_RDMA_CLIENT_NIC_IPS=10.10.10.11   # 单网口；多网口见 3.3 节
export IAXL_RDMA_TP_SIZE=4
export TP_SIZE=4
export MODEL=Qwen/Qwen3-32B

./examples/kvshrink-vllm-serve.sh
```

vLLM 启动完成后，5 条 `KVStoreRemote connected: peer=... rank=... has_only=...`
日志（1 个 has-only + 4 个 worker）表示所有 rank 都握手成功了。

### 3.3 多网口：按 rank 指定 RDMA 网口（TP=4 示例）

两个列表的规则相同：

- **第一个 IP 也是控制面地址**：它首先是 rank 0 的 RDMA 数据面 IP（单 IP 时则是
  所有 rank 的）；同时 `IAXL_RDMA_DAEMON_NIC_IPS` 的第一个 IP 也是 client 的连接
  目标（scheduler 连 `port`，worker r 连 `port+1+r`）和管理 REST 入口，两端
  scheduler 进程也都使用各自列表第一个 IP 对应的网口。
- **rank `r` 使用第 `r % 列表长度` 个 IP 对应的网口**承载该 rank 的 RPC 通知和
  KV 数据传输。列表只有一个 IP 时所有 rank 共用该网口。
- daemon 节点只需要 `IAXL_RDMA_DAEMON_NIC_IPS`；client 节点两个都需要，且
  `IAXL_RDMA_DAEMON_NIC_IPS` 的设置值须与 daemon 侧一致。

下例中 rank 0/1 走 10.10.10.x，rank 2/3 走 10.10.11.x：

```bash
# daemon 节点容器：启动 daemon 前设置
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10,10.10.10.10,10.10.11.10,10.10.11.10

# client 节点容器：启动 vLLM 前设置
export IAXL_RDMA_DAEMON_NIC_IPS=10.10.10.10,10.10.10.10,10.10.11.10,10.10.11.10
export IAXL_RDMA_CLIENT_NIC_IPS=10.10.10.11,10.10.10.11,10.10.11.11,10.10.11.11
```

IP 必须是**所在节点本机** RDMA 网口的地址；两端同一 rank 所选端口必须在
可互通的 RoCE 网络上。网口设备名可不同，不要直接按 `mlx5_*` 名称配对。
逗号列表不可为空，也不可包含空项。列表会覆盖进程继承的 `UCX_NET_DEVICES`，
并设置 `UCX_MAX_RNDV_RAILS=1`，因此无需（也不应）手动配置 UCX 多 rail。
两端启动日志中每个进程会打印 `RDMA NIC <设备> (<网口>, <IP>)`，用于核对选择结果。
可分别在两端用
`ip -j -4 addr` 核对 IP 归属，并比较两个口的
`/sys/class/infiniband/<设备>/ports/1/counters/port_rcv_data` 增量；确认流量
是否按预期分摊，吞吐/TTFT 还要结合 GPU 与网卡 NUMA 拓扑评估。

### 3.4 启动顺序

**先启 daemon，再启 client**：client 侧 `KVStoreRemote.__init__` 里 `xfer.connect`
默认用 `IAXL_API_TIMEOUT=60s` 超时，60s 内 daemon 没上线 vLLM 会直接崩掉。

---

## 四、验证与运维

- **联通性验证**（不启 vLLM，最快）：daemon 起来后，在 client 节点执行
  ```bash
  curl -s http://10.10.10.10:18700/v1/cache/status | jq .
  ```
  能返回 `{"workers": [...]}` 即控制面（含 4 个 rank）就绪。
- **看流量**：daemon 侧 rank 进程日志会打印 `[kv_xfer/rdma] pool: N blocks x M
  descs of L B` 和 `[kv_xfer] copy_chunks_batch: using RDMA backend`——第一次
  put/get 命中时出现。
- **清缓存**：`curl -s -X POST http://10.10.10.10:18700/v1/cache/evict -d
  '{"count": 500000000000}'`。controller 会自动 fan-out 到本机 4 个 worker。
- **导出 metrics**：`curl -s http://10.10.10.10:18700/v1/cache/metrics`。

---

## 五、常见问题

- `... IAXL_RDMA_DAEMON_NIC_IPS is required ...` / `IAXL_RDMA_CLIENT_NIC_IPS is
  required ...`：对应列表未设置、为空或含空项。
- `no metadata from daemon<r>`，且 daemon 日志有 `no route to ...`：该 rank 两端
  选中的网口之间不可达，检查两端列表中同一位置的 IP 是否在可互通的网络上。
- `IAXL_RDMA_ENABLE=1 is required on the daemon`：在 daemon 容器 shell 中、运行
  `examples/kvshrink-daemon.sh` 前设置 `IAXL_RDMA_ENABLE=1`。
- `no network interface owns 10.10.10.10`：`configure_ucx_env` 在
  `import nixl`/`rdma_init` 之前必须能找到持有该 IP 的网口。检查 `ip -j -4 addr`。
- `<netdev> (10.10.10.10) is not an RDMA-capable NIC`：网口存在但没有
  `/sys/class/net/*/device/infiniband/`。要么该口不是 RDMA，要么该主机没装
  MLNX_OFED / rdma-core。
- `client tp_size N != daemon tp_size M`：显式的 topology 校验，vLLM `-tp` 与
  daemon `--tp-size` 必须严格一致。
- `rdma_register_local called twice`：一个 rank 进程只应该注册 pool 一次；如果
  出现，通常是 KVStore 被创建了两遍——上层调用错了。
- 连不上 daemon 上的 `18700` REST：检查 daemon 节点的防火墙，以及
  `IAXL_API_CONTROLLER_PORT` 配置是否一致。
