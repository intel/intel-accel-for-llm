# intel-accel-for-llm (`iaxl`)

`iaxl` 利用 Intel 硬件加速器提升 LLM 推理性能。

## 设计文档

- [IAXL 设计](doc/design/iaxl.md)
- [KV Zip 设计](doc/design/kv-zip.md)
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
wget -q https://downloadmirror.intel.com/921047/QAT20.L.1.2.30-00178.tar.gz
tar xf QAT20.L.1.2.30-00178.tar.gz
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

`setvars.sh` 中的常用配置如下。`setvars.sh` 中的其它变量属于高级配置，普通用户无需修改。

| 类别 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 加速器 | `IAXL_QAT_ZIP_ENABLE` | `1` | 启用 QAT 压缩 worker（`0`/`1`） |
| 加速器 | `IAXL_IAA_ZIP_ENABLE` | `0` | 通过 Intel QPL 启用 Intel IAA 压缩 worker（`0`/`1`）。可与 QAT 同时开启，每个数据块会记录 IAA 能否解码 |
| 加速器 | `IAXL_CPU_ZIP_ENABLE` | `0` | 启用 CPU 压缩 worker（`0`/`1`）。CPU 压缩会大幅提高 CPU 占用率，QAT/IAA 加速器性能足够时请不要开启 |
| 加速器 | `IAXL_DSA_GD_ENABLE` | `0` | 启用 Intel DSA + GDRCopy 传输（`0`/`1`） |
| 压缩 | `IAXL_KV_COMPRESSION` | `1` | 对 KV cache 启用 DEFLATE 压缩（`0`/`1`） |
| 压缩 | `IAXL_ZIP_LEVEL` | `1` | 所有压缩后端的压缩级别，取值 `1`（最快）到 `9`（压缩率最高）。QAT 和 CPU 直接使用该级别；IAA 硬件只支持 level 1，因此 IAA 忽略此配置。建议保持默认值 `1`：提高级别对 KV cache 压缩率的提升很有限，却会明显降低压缩速度 |
| 压缩 | `IAXL_KV_DATA_SHUFFLE` | `1` | 压缩前对 BF16 数据做字节重排（`0`/`1`）。会稍微降低压缩性能，但可以提升压缩率和解压性能 |
| 压缩 | `IAXL_ZIP_EXPONENT_ONLY` | `IAXL_KV_DATA_SHUFFLE` | 字节重排开启时，只压缩 BF16 的高字节（符号和指数），低字节原样保存（`0`/`1`）。可以提升压缩率和压缩、解压性能；此时 `IAXL_KV_LOSSY_TRUNC` 对压缩率基本没有帮助 |
| vLLM | `MODEL` | `Qwen/Qwen3-32B` | Hugging Face 模型 ID 或本地模型路径 |
| vLLM | `TP_SIZE` | `2` | 必须配置；Tensor Parallel worker 数量，CPU、QAT 和 DSA 资源将据此自动配置 |
| vLLM | `DP_SIZE` | `1` | Data Parallel 组数；rank 总数 = `TP_SIZE × DP_SIZE` |
| vLLM | `VLLM_CPU_OMP_THREADS_BIND` | 根据 GPU 的 NUMA 拓扑自动配置 | vLLM 使用的 CPU，每个 rank 一个 CPU 列表，用 `\|` 分隔 |
| IAXL | `KVSHRINK_CPU_OMP_THREADS_BIND` | 与 `VLLM_CPU_OMP_THREADS_BIND` 相同 | IAXL 线程使用的 CPU，每个 rank 一项。用 CPU 做推理时需要配置，见 [vLLM CPU 推理使用指南](doc/usage/vllm-cpu-inference.zh-CN.md) |
| IAXL | `IAXL_RDMA_ENABLE` | `0` | 通过 RDMA 使用远端 KVStore daemon（`0`/`1`），见[跨节点 KV Cache 复用](#跨节点-kv-cache-复用) |
| IAXL | `IAXL_RDMA_DAEMON_NIC_IPS` | 未设置 | daemon 的 RDMA 网卡 IP；`IAXL_RDMA_ENABLE=1` 时必须配置 |
| IAXL | `IAXL_RDMA_CLIENT_NIC_IPS` | 未设置 | client 的 RDMA 网卡 IP；`IAXL_RDMA_ENABLE=1` 时 client 端必须配置 |
| IAXL | `IAXL_OMP_THREAD_NUM` | 压缩 worker 数 | 驱动压缩的线程数，线程越少占用的 CPU 核越少 |

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

如需在 CPU 上运行模型，请以 `DEVICE=cpu` 构建 IAXL，并参考 [vLLM CPU 推理使用指南](doc/usage/vllm-cpu-inference.zh-CN.md)。

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