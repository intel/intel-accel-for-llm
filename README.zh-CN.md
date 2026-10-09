# intel-accel-for-llm (`iaxl`)

`iaxl` 利用 Intel 硬件加速器提升 LLM 推理性能。

## 设计文档

- [IAXL 设计](doc/design/iaxl.md)
- [KVShrink 设计](doc/design/kvshrink.md)

## 配置宿主机

1. 配置 kernel cmdline，加入 `intel_iommu=on,sm_on iommu=pt`，然后重启宿主机：

```bash
sudo ./tools/setup_kernel_cmdline.sh
sudo reboot
```

2. 重启后，安装 QAT 驱动，以下两种方式**任选其一**。

   **Out-of-tree 驱动包**（默认）：

```bash
wget -q https://downloadmirror.intel.com/843052/QAT20.L.1.2.30-00078.tar.gz
tar xf QAT20.L.1.2.30-00078.tar.gz
./configure
make -j$(nproc)
sudo make install
```

可以使用以下命令停止或启动 out-of-tree QAT 服务：

```bash
adf_ctl down
adf_ctl up
```

   **In-tree 驱动 + qatlib**（内核 `qat_4xxx` 模块，VF 绑定到 `vfio-pci`）：

```bash
sudo apt install qatlib-service libqat-dev libusdm-dev   # 或从 https://github.com/intel/qatlib 源码构建
printf 'POLICY=0\nServicesEnabled=dc\n' | sudo tee /etc/sysconfig/qat
sudo systemctl enable qat
sudo systemctl restart qat
export IAXL_QATLIB=intree   # 在 source setvars.sh / 构建之前设置
```

`POLICY=0` 为每个进程从每个 QAT 设备各分配一个 VF，`IAXL_QAT_DEVICES` / `KVSHRINK_QAT_DEVICES` 按此编号；若为 `POLICY=1`，每个进程只能看到设备 `0`。

设置 `IAXL_QATLIB=intree` 后，构建会链接 `pkg-config qatlib` 找到的系统 `libqat` / `libusdm`，而不再下载并构建 out-of-tree 驱动包到 `_lib`。在容器内构建时，容器中需要安装 qatlib 开发包，运行时还需从宿主机挂载 `/run/qat`。

3. 安装 GDRCopy 驱动并配置 DSA：

```bash
sudo ./tools/install_gdr_driver.sh
./tools/setup_dsa_cnt.sh
```

## 环境变量配置

`setvars.sh` 中的常用配置如下：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen3-32B` | Hugging Face 模型 ID 或本地模型路径 |
| `TP_SIZE` | `2` | 必须配置；Tensor Parallel worker 数量，CPU、QAT 和 DSA 资源将据此自动配置 |
| `IAXL_KV_COMPRESSION` | `1` | 启用 DEFLATE 压缩（`0`/`1`） |
| `IAXL_QAT_ZIP_ENABLE` | `1` | 启用 QAT 压缩 worker（`0`/`1`） |
| `IAXL_QATLIB` | 未设置（`oot`） | 可选。构建所用的 QAT 用户态库：`oot`（构建到 `_lib` 的 out-of-tree 驱动包，未设置时使用）或 `intree`（in-tree 驱动对应的系统 qatlib） |
| `IAXL_IAA_ZIP_ENABLE` | `0` | 通过 Intel QPL 启用 Intel IAA 压缩 worker（`0`/`1`）。可与 `IAXL_QAT_ZIP_ENABLE` 同时开启：IAA 最多只能解码 4 KB 的 DEFLATE 历史窗口，而 QAT gen4 固定使用 32 KB，因此每个数据块都会记录 IAA 能否解码，解压时 IAA 只领取这些块 |
| `IAXL_CPU_ZIP_ENABLE` | `1` | 启用 CPU 压缩 worker（`0`/`1`） |
| `IAXL_DSA_GD_ENABLE` | `0` | 启用 Intel DSA + GDRCopy 传输（`0`/`1`） |
| `IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS` | `1` | 前 N 层 KV cache 不进行压缩 |
| `PYTHONOPTIMIZE` | `0` | 保留 Python `assert` 检查 |

> [!WARNING]
> GPU 不支持 P2P DMA 时，请勿启用 `IAXL_DSA_GD_ENABLE`，并保持其值为 `0`。

## KVShrink vLLM Example

KVShrink 是基于 IAXL `KVStore` 的 vLLM V1 KV connector。完成 `setvars.sh` 配置后，直接启动容器：

```bash
./start.sh
```

`setvars.sh` 会根据前 `TP_SIZE` 张 GPU 的 NUMA 拓扑自动配置每个 rank 使用的 CPU、QAT 和 DSA 资源。

进入容器后，可以使用 pip 安装：

```bash
pip install -e . --verbose --no-build-isolation
```

### 离线环境编译

在**联网机器**上先运行 `./start.sh` 构建最新 dev 镜像。镜像会预装 Python/系统依赖和 UCX、NIXL 等库，并保存编译 QAT、QPL 所需的库和头文件。可以通过如下步骤将镜像带到目标机器，例如：

```bash
docker save -o vllm-iaxl-dev.tar vllm-iaxl-dev:latest
# 在离线机器上导入镜像（镜像也已包含 vLLM 基础镜像中的内容）
docker load -i vllm-iaxl-dev.tar
```

在离线机器的工程根目录启动容器，跳过镜像构建和进入容器时的自动安装：

```bash
./start.sh --offline
# 在容器内，从挂载的工程目录编译安装；禁止 pip 访问包索引
pip install -e . --verbose --no-build-isolation --no-index --no-deps
```

修改源码后可在容器中重复上述 `pip install` 命令。离线环境宿主机需要通过手动操作配置 GPU 驱动、Docker 运行时和QAT 驱动。具体使用的模型也需在本地可用（可通过 `MODEL` 指向本地模型目录），运行服务不要依赖在线拉取模型。

启用 IAA 或 DSA 时，需要特别注意宿主机硬件配置：

- IAA 的 QPL 源码及DSA的相关库已预置在 dev 镜像，离线编译无需额外下载，但运行前必须在宿主机启用 IAA 设备或DSA设备及其用户态工作队列，确保 `/sys/bus/dsa/devices/iax*` 和对应的 `/dev/iax/` 设备可见；仅设置 `IAXL_IAA_ZIP_ENABLE=1`或 `IAXL_DSA_GD_ENABLE=1`不会自动配置设备。
- CUDA 下的 DSA/GDRCopy 代码会随源码一起编译，dev 镜像中已有 `gdrapi.h`、`libgdrapi.so` 和 `accel-config`，但宿主机需安装与其内核及 CUDA 版本匹配的 GDRCopy 驱动、加载 `gdrdrv` 并暴露 `/dev/gdrdrv`。在有外网访问条件时时，可以执行`tools/install_gdr_driver.sh` 脚本通过执行 `apt-get` 和 `wget`来安装相关的驱动，但对于离线宿主机必须提前手动准备并安装对应的驱动包及依赖。
- 在导入 dev 镜像后、启动服务前，可在宿主机运行 `./tools/setup_dsa_cnt.sh --offline`，使用镜像内的 `accel-config` 配置 DSA 工作队列。随后设置 `IAXL_DSA_GD_ENABLE=1` 启动容器；`setvars.sh` 会检查与 GPU 对应的 DSA 用户态工作队列是否已启用。

例如要同时使用 IAA 压缩时，在宿主机运行 `IAXL_QAT_ZIP_ENABLE=1 IAXL_IAA_ZIP_ENABLE=1 ./start.sh --offline`；同时启用 DSA 时再添加 `IAXL_DSA_GD_ENABLE=1`。物理设备配置须在执行 `start.sh` 前完成。

完成安装后，在容器内启动服务：

```bash
./examples/kvshrink-vllm-serve.sh
```

该脚本会在 `localhost:8000` 启动 vLLM，加载 `KVShrinkConnector`，并将日志写入 `log.kvshrink-vllm`。启动日志中的 `MODEL`、`TP_SIZE` 和每个 rank 的 CPU/QAT/DSA 配置可用于检查实际生效的拓扑。

在宿主机的第二个终端进入同一个容器：

```bash
docker exec -it -w "$PWD" iaxl.vllm bash
```

发送一个 Chat Completions 测试请求：

```bash
./tests/vllm-test.sh
```

## KVShrink vLLM Benchmark

保持 KVShrink vLLM 服务运行，在第二个容器终端执行 online serving benchmark：

```bash
./tests/vllm-benchmark.sh
```

## REST API

管理接口默认监听 `localhost:18700`，并将请求转发到各个 rank。

| 接口 | 说明 |
| --- | --- |
| `GET /v1/cache/status` | 查询 cache 状态 |
| `POST /v1/cache/evict` | 从 DDR 中淘汰 cache |
| `POST /v1/cache/persist` | 将 cache 持久化到磁盘 |

```bash
curl http://localhost:18700/v1/cache/status
```

`persist` 和 `evict` 通过 `count` 指定最多处理的 cache group 数量。需要保留 cache 数据时，先调用 `persist`，再调用 `evict`：

```bash
curl -X POST http://localhost:18700/v1/cache/persist \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
curl -X POST http://localhost:18700/v1/cache/evict \
	-H 'Content-Type: application/json' \
	-d '{"count":999999}'
```

## 跨节点 KV Cache 复用

remote_pool 支持将 KV Cache 存储在独立的 daemon 节点，并通过 RDMA 与 vLLM client 节点传输和复用。部署拓扑、两侧构建与启动、逐 rank 网口配置及运维步骤见[跨节点 KV Cache 使用文档](doc/usage/remote-pool-kvstore.zh-CN.md)。