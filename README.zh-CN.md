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

2. 重启后，下载并安装 QAT 驱动：

```bash
wget -q https://downloadmirror.intel.com/843052/QAT20.L.1.2.30-00078.tar.gz
tar xf QAT20.L.1.2.30-00078.tar.gz
./configure
make -j$(nproc)
sudo make install
```

可以使用以下命令停止或启动 QAT 服务：

```bash
adf_ctl down
adf_ctl up
```

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