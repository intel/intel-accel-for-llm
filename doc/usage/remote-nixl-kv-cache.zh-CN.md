# 跨节点 KV 缓存(NIXL)—— 使用说明

设计文档:[`../design/remote-nixl-kv-cache.zh-CN.md`](../design/remote-nixl-kv-cache.zh-CN.md)。
English: [`remote-nixl-kv-cache.md`](remote-nixl-kv-cache.md)。

该后端让 vLLM/KVShrink 部署把 KV 缓存池通过 NIXL(RDMA/GDR)下沉到独立节点,
替代本机、挂 GPU 的缓存池。通过 `KVSHRINK_REMOTE_CACHE_ENABLE=1` 开启;
connector 会透明地把 `iaxl.kvstore.KVStore` 换成
`iaxl.remote.RemoteKVStore`(见 `kvshrink/kvshrink_connector.py::_make_kvstore`)。

---

## 一、编译

**只需要在 GPU 节点执行一次**。远端缓存节点不需要构建工具链、不需要
`Dockerfile.dev`,直接使用同一个 `vllm/vllm-openai:v0.23.0` 基础镜像即可
(见第二节 "远端 daemon 节点")。

```bash
cd /mnt/data/zengjun/intel-accel-for-llm
./start.sh
```

`./start.sh` 会在 `vllm/vllm-openai:v0.23.0` 之上构建 `docker/Dockerfile.dev`
开发镜像,进入容器并执行 `pip install -e . --no-build-isolation`——同时
把 `iaxl` 包(含 `torch_ext*.so` 及压缩相关的 `zip_compress_to_mem`/
`zip_decompress_from_mem`)编译好。

编译完成后,打独立可迁移 release 包供远端 daemon 节点使用:

```bash
# 在 ./start.sh 执行完毕之后运行;容器内或宿主机上均可(该脚本只读取
# 已经编译好的 iaxl/torch_ext*.so 与 _lib/*.so,不再触发任何编译)。
tools/remote_daemon/build_release.sh
# -> dist/iaxl-remote-daemon-v0.1.tar.gz
```

若需要打其它版本号(例如 v0.2、或某次热修复的 git sha),用
`RELEASE_VERSION=v0.2 tools/remote_daemon/build_release.sh` 覆盖即可。

---

## 二、部署

先启动远端 daemon,再启动 GPU 节点上的 vLLM;`KVSHRINK_REMOTE_FAIL_IF_UNREACHABLE=1`
(默认)时,worker 若在初始化阶段连不上 daemon 会直接退出。

### 2.1 远端 daemon 节点

只需要 Docker 与 `vllm/vllm-openai:v0.23.0` 镜像。若该镜像不在本地,
`docker-run-daemon.sh` 第一次执行 `docker run` 时会自动从 Docker Hub 拉取
(内网无外网访问时,先在有网环境 `docker pull` + `docker save`,拷贝到目标
节点再 `docker load`)。

```bash
# 从 GPU 节点拷贝到远端存储节点并展开
scp dist/iaxl-remote-daemon-v0.1.tar.gz root@10.10.10.10:/root/
ssh root@10.10.10.10
tar xzf /root/iaxl-remote-daemon-v0.1.tar.gz -C /root
cd /root/iaxl-remote-daemon-v0.1
```

**推荐示例(每个 GPU rank 一个 daemon 实例,双轨 RDMA):**

```bash
KVSHRINK_REMOTE_DAEMON_ADDR="10.10.10.10:19000|10.10.10.10:19001|10.10.10.10:19002|10.10.10.10:19003" \
KVSHRINK_REMOTE_NIXL_ADDR="10.10.10.10:19100|10.10.10.10:19101|10.10.10.10:19102|10.10.10.10:19103" \
NUM_INSTANCES=4 CONTROL_PORT_BASE=19000 NIXL_PORT_BASE=19100 \
TRANSPORT=nixl \
NIXL_DEVICE=rocep21s0f0:1,rocep21s0f1:1 \
NIXL_HOST=10.10.10.10 \
POOL_SIZE_GB=64 STAGING_SLOTS=2048 STAGING_SLOT_MB=8 \
KVSHRINK_REMOTE_MAX_STAGING_SLOTS=16384 \
KVSHRINK_REMOTE_QAT_DEVICES="0|1|4|5" \
tools/remote_daemon/run-daemon-multi.sh
```

`run-daemon-multi.sh` 会自动识别当前是 release 包(存在 `pysrc/iaxl/`)还是
仓库内检出,自动选择 `docker-run-daemon.sh`(release 包,基于
`vllm/vllm-openai:v0.23.0` 起容器)或直接 `run-daemon.sh`;启动时会打印
`USE_DOCKER=0|1`。可通过 `USE_DOCKER=1`/`USE_DOCKER=0` 强制指定。

#### 关闭压缩(裸带宽基准)

```bash
# 追加一行
COMPRESS=0 \
```

不压缩时 QAT/IAA 设备不生效,`KVSHRINK_REMOTE_QAT_DEVICES`/
`IAXL_QAT_ZIP_ENABLE` 均可省略。

#### 指定压缩设备(QAT / IAA / CPU)

`KVSHRINK_REMOTE_QAT_DEVICES` 用 `"|"` 分隔实例、逗号分隔设备号,长度必须
**恰好**等于 `NUM_INSTANCES`(空段会让对应实例悄悄拿不到设备,日志显示
`qat_devices=<none>`)。`NUM_INSTANCES=1` 时用逗号即可,例如 `"0,1,4,5"`。

其它压缩后端开关(与本机 KVStore 路径完全一致,默认继承自 `setvars.sh`):

| 变量 | 含义 |
| --- | --- |
| `IAXL_QAT_ZIP_ENABLE` | 是否启用 QAT 压缩 |
| `IAXL_IAA_ZIP_ENABLE` | 是否启用 IAA 压缩 |
| `IAXL_CPU_ZIP_ENABLE` | 是否启用 CPU 压缩兜底 |
| `IAXL_KV_COMPRESSION` | 是否启用 KV 压缩流水线(总开关) |

#### OMP / QAT 并发实例数

单个 QAT 设备可创建的压缩实例数(即 OMP 并发)由
`IAXL_QAT_ZIP_INSTANCES_PER_DEVICE` 控制(默认 4),`IAXL_QAT_INSTANCE_NUM`
会由 `run-daemon.sh` 根据 `IAXL_QAT_DEVICES` × `IAXL_QAT_ZIP_INSTANCES_PER_DEVICE`
自动推导——通常不需要手工设置,除非要显式覆盖:

```bash
# 每设备 8 个实例,共 4 设备,则并发 = 32
IAXL_QAT_ZIP_INSTANCES_PER_DEVICE=8 \
```

daemon 单进程内 codec 调用是串行的(GET 会插队到 PUT 前面),提高并发主要
靠增加 QAT 设备数或增加 `NUM_INSTANCES`,而不是加信号量。

#### 单个 RDMA 设备(不使用多轨)

去掉多轨设备列表中的逗号即可,例如:

```bash
NIXL_DEVICE=rocep21s0f0:1 \
```

多轨(逗号列表)会让 UCX 为同一 NIXL session 启用多路条带化;单口
带宽已经打满 NIC 时才有意义。

#### 单个 remote 进程(共享给所有 rank)

```bash
CONTROL_PORT=19000 \
TRANSPORT=nixl \
NIXL_DEVICE=rocep21s0f0:1 \
NIXL_HOST=10.10.10.10 NIXL_PORT=19100 \
POOL_SIZE_GB=64 STAGING_SLOTS=2048 STAGING_SLOT_MB=8 \
KVSHRINK_REMOTE_MAX_STAGING_SLOTS=16384 \
KVSHRINK_REMOTE_QAT_DEVICES="0,1,4,5" \
tools/remote_daemon/docker-run-daemon.sh
```

`tp_size` 个 worker 会共享同一个 daemon 的控制面与 codec——只在轻负载或
调试场景使用;生产环境推荐 `NUM_INSTANCES=tp_size` 的多进程部署。

#### 其它常用 daemon 侧变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CONTROL_HOST` | `0.0.0.0` | 控制面监听地址 |
| `POOL_SIZE_GB` | `8` | 每个 `(model, tp_size, tp_rank)` 组的池预算 |
| `CACHE_DIR` | `_data/kvcache/remote` | 持久化 chunks.db + chunks/ 根目录 |
| `STAGING_SLOTS` | `0` | NIXL staging 槽数;`TRANSPORT=nixl` 时**必须**设为 >0 |
| `STAGING_SLOT_MB` | `4` | 名义槽大小;daemon 会按真实 shard 大小自动重切分 |
| `KVSHRINK_REMOTE_MAX_STAGING_SLOTS` | `4096` | staging arena 按真实 shard 大小自动重切分后的槽数上限;高并发请求需要更多在飞 shard 时,可配合增大 `STAGING_SLOTS` 一起提高 |
| `KVSHRINK_REMOTE_STATS_INTERVAL_SEC` | `30` | 周期打印吞吐/排队日志;`0` 关闭 |
| `DEVICE` | `cpu` | NIXL staging buffer 所在设备 |

### 2.2 GPU 节点(vLLM 侧)

在启动 vLLM(`examples/kvshrink-vllm-serve.sh`)之前设置以下环境变量,由
`iaxl.remote.config.RemoteCacheConfig.from_vllm` 读取:

**推荐示例(4-rank TP,双轨 RDMA,对应上面的多进程 daemon):**

```bash
KVSHRINK_REMOTE_CACHE_ENABLE=1 \
KVSHRINK_REMOTE_TRANSPORT=nixl \
KVSHRINK_REMOTE_DAEMON_ADDR="10.10.10.10:19000|10.10.10.10:19001|10.10.10.10:19002|10.10.10.10:19003" \
KVSHRINK_REMOTE_NIXL_ADDR="10.10.10.10:19100|10.10.10.10:19101|10.10.10.10:19102|10.10.10.10:19103" \
KVSHRINK_REMOTE_NIXL_DEVICE=mlx5_0:1,mlx5_1:1 \
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 \
KVSHRINK_REMOTE_NIXL_MAX_SHARDS_PER_ROUND=1024 \
KVSHRINK_REMOTE_NIXL_ROUNDS_IN_FLIGHT=4 \
UCX_IB_GPU_DIRECT_RDMA=y \
MODEL=/mnt/ssd1/model-space/Qwen/Qwen2.5-32B-Instruct \
./examples/kvshrink-vllm-serve.sh
```

`KVSHRINK_REMOTE_DAEMON_ADDR` / `KVSHRINK_REMOTE_NIXL_ADDR` 用 `"|"` 分隔时,
rank `r` 对应第 `r` 项;单值时所有 rank 共用同一 daemon。

模型加载、TP size、KVShrink 本身的开关等 vLLM 通用参数不属于本特性范畴,
沿用 [`../../README.zh-CN.md`](../../README.zh-CN.md)(英文
[`../../README.md`](../../README.md))里的做法即可,这里不再罗列。

#### 单个 RDMA 设备

```bash
KVSHRINK_REMOTE_NIXL_DEVICE=mlx5_0:1 \
UCX_NET_DEVICES=mlx5_0:1 \
```

两者必须一致(前者是转发给 UCX 的入口)。

#### 单个 remote 进程

`KVSHRINK_REMOTE_DAEMON_ADDR` / `KVSHRINK_REMOTE_NIXL_ADDR` 只填一个
`host:port`,不使用 `"|"` 列表:

```bash
KVSHRINK_REMOTE_DAEMON_ADDR=10.10.10.10:19000 \
KVSHRINK_REMOTE_NIXL_ADDR=10.10.10.10:19100 \
```

#### 其他 vllm server 侧变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `KVSHRINK_REMOTE_CACHE_ENABLE` | `0` | **总开关**。`1` 启用跨节点 NIXL 后端;`0`(默认)时 connector 走原有的**本地 DDR KVShrink** 逻辑,不连 daemon,即使其它 `KVSHRINK_REMOTE_*` 变量都填了也会被忽略——用作快速回滚到本机路径的开关。 |
| `KVSHRINK_REMOTE_TRANSPORT` | `nixl` | `nixl`(生产)或 `tcp`(回退验证) |
| `KVSHRINK_REMOTE_CONNECT_TIMEOUT_SEC` / `..._RETRY_SEC` | 30 / 2 | 连接重试参数 |
| `KVSHRINK_REMOTE_REQUEST_TIMEOUT_SEC` | 120 | 单次 RPC / 传输超时 |
| `KVSHRINK_REMOTE_FAIL_IF_UNREACHABLE` | `1` | 初始化时连不上 daemon 直接退出 |
| `KVSHRINK_REMOTE_NIXL_MAX_SHARDS_PER_ROUND` | `128` | 单轮 NIXL 传输最多携带的 shard 数;增大后可降低大 cache hit 请求的分轮控制面开销 |
| `KVSHRINK_REMOTE_NIXL_ROUNDS_IN_FLIGHT` | `2` | 单个 rank 目标在飞 NIXL 轮数,用于重叠 RDMA 传输与 daemon 侧 codec 工作 |

### 2.3 网络与设备选择建议

- `KVSHRINK_REMOTE_NIXL_DEVICE` / `UCX_NET_DEVICES`(GPU 侧)与 `NIXL_DEVICE`
  (远端 daemon 侧)必须选择**同一 RDMA 子网**中的端口;两侧设备命名可以
  不同(如 GPU 侧 `mlx5_0:1`,远端侧 `rocep21s0f0:1`),但必须能互通。
- RoCE 环境建议先完成 MTU、PFC/ECN 等网络调优再做性能测试。
- 任一侧使用逗号分隔的设备列表都会让 UCX 为该 NIXL session 启用多轨
  (multi-rail)条带化,同时使用多条 RDMA 链路无需其它配置。

---

## 三、daemon 侧 CLI 命令

与本机 KVStore 的 `/v1/cache/*` HTTP 端点等价的管理命令通过一个不建
session 的 admin RPC 暴露出来,入口是
[`iaxl.remote.admin`](../../iaxl/remote/admin.py),便捷脚本
[`tools/remote_daemon/remote-cli.sh`](../../tools/remote_daemon/remote-cli.sh)
(已随 `build_release.sh` 一并打包进 release 包)。仅走控制面 TCP,
不需要 RDMA/QAT/IAA,也不影响正在跑的 vLLM worker。

**在哪里执行:** 只要能连通 daemon 的 `CONTROL_PORT` 且能 import `iaxl`
即可。推荐的两种方式:

- **在远端存储节点上,通过 daemon 容器 exec 进去执行**(release 包已内置
  `remote-cli.sh` 与 `pysrc/iaxl`):

  ```bash
  docker exec iaxl-remote-daemon-0 \
      bash /root/iaxl-remote-daemon-v0.1/tools/remote_daemon/remote-cli.sh \
      --daemon 127.0.0.1:19000 status
  ```

- **在 GPU 节点(已经 `pip install -e .`)上远程执行**,针对哪个实例就填
  哪个 `CONTROL_PORT`:

  ```bash
  cd /mnt/data/zengjun/intel-accel-for-llm
  tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 status
  ```

### 常用命令

```bash
# 整体状态与每个 (model, tp_size, tp_rank) 组的池占用/命中/压缩比等
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 status

# 触发一次 LRU 淘汰(每组最多 32 个 group;不带 --model/--tp-* 时作用于所有组)
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 evict --count 32

# 将 32 个未持久化 group 落盘,可按组过滤
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 persist --count 32 \
    --model Qwen2.5-32B-Instruct --tp-size 4 --tp-rank 0

# 列出下一批 persist / evict 候选
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 candidates --which evict --count 10

# 原生压缩/解压吞吐 metrics(与本机 `/v1/cache/metrics` 同一份;可选 --reset)
tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 metrics --reset
```

用 `--json` 打印完整原始响应,便于脚本化处理。

### 多进程 daemon

`NUM_INSTANCES>1` 时每个实例监听独立的 `CONTROL_PORT`,admin RPC 是逐
实例的——一次只对一个实例做 persist/evict。需要巡检所有实例时,写一层
shell 循环即可:

```bash
for p in 19000 19001 19002 19003; do
    tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:$p status
done
```
