# 使用 KVShrink 进行 vLLM CPU 推理

[English](vllm-cpu-inference.en.md)

本文介绍 vLLM 在 CPU 上推理、KVShrink 使用 Intel QAT 或 IAA 压缩 KV cache 时，如何配置线程和 CPU 绑核。容器中需要安装 CPU 版 vLLM（`VLLM_TARGET_DEVICE=cpu`），并以 `DEVICE=cpu` 构建 IAXL。设计原理见 [KV Zip 设计](../design/kv-zip.md)。

## 1. 构建

`DEVICE=cpu` 构建与 GPU 推理使用相同的 `KVStore`、压缩编解码和 persist/evict API。它不需要 CUDA、SYCL、GDRCopy 或 NIXL，因此不支持 DSA 传输和 RDMA（`remote_pool`）。

与 GPU 推理相同，在仓库根目录用 `start.sh` 构建并启动容器。基础镜像指定为 vLLM CPU 发布镜像，`IAXL_QATLIB` 要与宿主机的 QAT 驱动一致：

```bash
export DEVICE=cpu
export IAXL_BASE_DOCKER_IMAGE=vllm/vllm-openai-cpu:v0.29.0
export IAXL_DEV_DOCKER_IMAGE=vllm-iaxl-dev-cpu
export IAXL_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so
export IAXL_QATLIB=intree # in-tree 驱动（qat_4xxx，VF 绑定到 vfio-pci）；out-of-tree 驱动（/dev/qat_dev_processes）不设置
./start.sh
```

- `DEVICE=cpu` 会在开发镜像中跳过 UCX 和 NIXL，以不带 GPU 参数的方式启动容器，并在容器启动时以 `DEVICE=cpu` 构建 IAXL。
- `IAXL_DEV_DOCKER_IMAGE` 让 CPU 镜像与 GPU 镜像分开。`start.sh` 会复用已存在的同名镜像，不设置时 CPU 运行会复用 GPU 镜像（使用 `--rebuild` 时则会覆盖它）。
- 示例和 benchmark 脚本会把 `IAXL_LD_PRELOAD` 加到 `LD_PRELOAD` 前面。vLLM CPU 后端需要预加载 libtcmalloc 和 libiomp5，而 CPU 镜像的 libiomp5 位于 `/opt/venv/lib`，不在默认的 `/usr/local/lib`。
- 设置 `IAXL_QATLIB=intree` 时，如果 `pkg-config` 找不到系统 qatlib（Ubuntu 22.04 的 CPU 镜像没有），构建会从源码编译 qatlib 到 `_lib`；`start.sh` 会挂载宿主机的 `/run/qat`，各 rank 通过宿主机 `qatmgr` 获取 QAT instance。如果宿主机 qatlib 不是 `26.02.0`，运行 `./start.sh --rebuild` 前 `export IAXL_QATLIB_VERSION=<宿主机版本>`。没有 `qatmgr` 时，第一个进程会占用全部 QAT 设备，因此多 rank 运行需要 `qatmgr`。

> [!NOTE]
> 不使用 Docker 时，也可以在已安装 CPU 版 vLLM 的环境中构建 IAXL。先安装 `docker/Dockerfile.dev` 中安装的软件包（UCX 和 NIXL 除外），再执行 `python -m pip install -r requirements.txt` 和 `DEVICE=cpu python -m pip install -e . --no-build-isolation`；使用 in-tree 驱动时加上 `IAXL_QATLIB=intree`。

## 2. CPU 绑核

两个变量把每个 rank 的 CPU 分给 vLLM 和 IAXL：

| 变量 | 使用者 | 线程 |
| --- | --- | --- |
| `VLLM_CPU_OMP_THREADS_BIND` | vLLM | OpenMP 推理线程，每个 CPU 一个 |
| `KVSHRINK_CPU_OMP_THREADS_BIND` | IAXL | `OMP-Main`、`POOL-<n>`、`H2D`、`D2H` 等 IAXL 线程 |

- **格式**：两个变量都是每个 rank 一项，用 `|` 分隔。每项是一个 CPU 列表，例如 `0-61` 或 `0-7,16`。
- **rank 顺序**：第 `r` 项对应全局 rank `r = dp_rank × TP_SIZE + tp_rank`，因此每个变量需要 `TP_SIZE × DP_SIZE` 项。
- **默认值**：`KVSHRINK_CPU_OMP_THREADS_BIND` 默认等于 `VLLM_CPU_OMP_THREADS_BIND`。这时 IAXL 线程和推理线程共用同一批核，所以 CPU 推理时要显式设置它。
- **启动时**：connector 把本 rank 对应的项导出为 `IAXL_CPU_OMP_THREADS_BIND`，IAXL 的所有线程都绑定到这些 CPU 上。

选择 CPU 时：

- 同一个 rank 的 vLLM 项和 IAXL 项不能有重叠。
- 两项放在同一个 NUMA 节点上，该 rank 的 QAT 设备（`KVSHRINK_QAT_DEVICES`）也选在这个节点。
- 每个物理核只用一个逻辑 CPU，可以用 `lscpu -e=CPU,CORE,NODE` 查看对应关系。
- 每个 rank 给 IAXL 的 CPU 数量不少于 `IAXL_OMP_THREAD_NUM`。

## 3. 线程配置

| 变量 | 推荐值 | 作用 |
| --- | --- | --- |
| `IAXL_USE_OMP` | `0`（`DEVICE=cpu` 时的默认值） | IAXL 不用 OpenMP，改用自己的线程池，OpenMP 留给 vLLM。此时 `setvars.sh` 会 unset `OMP_NUM_THREADS` 和 `OMP_THREAD_LIMIT`。 |
| `IAXL_OMP_THREAD_NUM` | `1` 或 `2` | 驱动压缩 worker 的线程数。一个线程就能让多个 QAT/IAA 实例保持忙碌；线程数超过 worker 数时，各 worker 的在途请求分给多个线程。 |
| `IAXL_CPU_ZIP_ENABLE` | `0` | 不占用 CPU 核跑 zlib，由 QAT 和 IAA 压缩。 |
| `IAXL_QAT_ZIP_ENABLE`、`IAXL_IAA_ZIP_ENABLE` | 有哪种加速器就设为 `1` | worker 数为 `IAXL_QAT_INSTANCE_NUM + IAXL_IAA_INSTANCE_NUM`，`setvars.sh` 会打印这两个值。 |

## 4. 缓存配置

`DEVICE=cpu` 时，`examples/kvshrink-vllm-serve.sh` 不传 `--gpu-memory-utilization`。CPU 推理时通常需要调整下列变量：

| 变量 | 作用 |
| --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | vLLM 运行时 KV cache 大小（GiB）。KVShrink 压缩缓存的大小由 `IAXL_DDR_POOL_SIZE_GB` 单独设置。 |
| `MAX_MODEL_LEN` | `--max-model-len`（默认 `32765`，`vllm-cpu` recipe 中为 `16384`）。CPU KV cache 放不下一个满长度请求时调小。 |
| `VLLM_SERVE_EXTRA_ARGS` | 追加到脚本默认选项之后的其他 `vllm serve` 选项（按空白分隔）。 |
| `BLOCK_SIZE` | `--block-size`，仅在设置时传入。vLLM v0.29 在 CPU 上默认为 `128`（MLA 模型为 `16`），其他取值须为 32 的倍数。CPU 上每层一个 KV block 为 `2 × num_kv_heads × block_size × head_size × dtype_bytes` 字节，必须不超过 `IAXL_ZIP_SRC_CAP`，压缩后不超过 `IAXL_ZIP_DST_CAP`（`setvars.sh` 中都设为 4 MiB）。block 更大时，可以同时调大这两个值，或者设置 `IAXL_ZIP_SEGMENT_SIZE_KB`，按该大小（单位 KiB）分段压缩，这时只需每段（小于该大小的两倍）满足上述限制。 |
| `IAXL_KV_DATA_SHUFFLE`、`IAXL_KV_LOSSY_TRUNC` | 两者都为 `0` 时，编解码直接从 vLLM 的 KV block 压缩、直接解压到 KV block，没有中转拷贝。`setvars.sh` 默认开启字节重排。任一开启都会改写编解码输入，block 需要先拷贝到 scratch buffer。字节重排能提升压缩率，代价是对每个 block 多做一遍处理。 |

## 5. 示例

下面的例子在两个 NUMA 节点上运行两个 rank：节点 0 是 CPU 0-63，节点 1 是 CPU 64-127，每个物理核一个逻辑 CPU。

```bash
# 先执行第 1 节中的 export
./start.sh

# 在 start.sh 打开的容器 shell 中执行
export TP_SIZE=2
export VLLM_CPU_OMP_THREADS_BIND="0-61|64-125"       # vLLM 推理
export KVSHRINK_CPU_OMP_THREADS_BIND="62-63|126-127" # IAXL 线程
export KVSHRINK_QAT_DEVICES="0,1|2,3"                # 每个 rank 所在 NUMA 节点上的 QAT 设备
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_OMP_THREAD_NUM=2

./examples/kvshrink-vllm-serve.sh
```

该脚本会 source `setvars.sh`，并把 `IAXL_LD_PRELOAD`（按照 vLLM CPU 后端的推荐预加载 libtcmalloc 和 libiomp5）加到 `LD_PRELOAD` 前面。

[`vllm-cpu` recipe](../../examples/recipes/vllm-cpu/README.md) 把第 1 到第 4 节的 export 和上面的 CPU 划分打包在一起：在主机上执行 `IAXL_RECIPE=vllm-cpu ./start.sh`，再在容器里执行 `./examples/kvshrink-vllm-serve.sh`，即可启动相同的配置。此外，该 recipe 还设置 `MAX_MODEL_LEN=16384`，从默认 DDR pool 预算中扣除 `VLLM_CPU_KVCACHE_SPACE`，并为容器提供 16 GiB 的 `/dev/shm`；机制见 [recipes 设计文档](../design/recipes.md)。

## 6. 验证

- **`setvars.sh` 输出**：检查 `VLLM_CPU_OMP_THREADS_BIND`、`KVSHRINK_CPU_OMP_THREADS_BIND`、`IAXL_OMP_THREAD_NUM` 和 `IAXL_USE_OMP=0`。
- **vLLM 日志**：每个 rank 会打印 `Bound rank <r>: IAXL_CPU_OMP_THREADS_BIND=<cpus>` 和 `[iaxl] config: ... omp_threads=2 use_omp=OFF ...`。
- **线程绑核**：处理完第一个请求后运行下面的命令。IAXL 线程（`OMP-Main`、`POOL-*`、`H2D`、`D2H`）列出的 CPU 应当只有本 rank 在 `KVSHRINK_CPU_OMP_THREADS_BIND` 中的那一项。

  ```bash
  for f in $(grep -l "^OMP-Main$" /proc/[0-9]*/task/*/comm); do
      pid=$(cut -d/ -f3 <<<"$f"); echo "== pid $pid"
      for t in /proc/$pid/task/*; do
          echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"
      done | sort | uniq -c
  done
  ```

## 7. 常见问题

| 报错 | 处理 |
| --- | --- |
| `KVSHRINK_CPU_OMP_THREADS_BIND has N entries, but rank is R` | 每个 rank 提供一项，共 `TP_SIZE × DP_SIZE` 项。 |
| `invalid IAXL_CPU_OMP_THREADS_BIND` | 某一项不是合法的 CPU 列表。 |
| `cannot bind to IAXL_CPU_OMP_THREADS_BIND` | 列出的 CPU 进程都用不了，例如不在容器的 cpuset 内。 |
| `IAXL_OMP_THREAD_NUM must be a positive integer` | 把 `IAXL_OMP_THREAD_NUM` 设为 1 或更大。 |
| `IAXL_OMP_THREAD_NUM (...) + reserved (...) exceeds the smallest rank CPU allocation` | `VLLM_CPU_OMP_THREADS_BIND` 中最小的一项 CPU 太少。调小 `IAXL_RESERVED_CPU_NUM` 或 `IAXL_OMP_THREAD_NUM`。 |
| vLLM 警告 `LD_PRELOAD` 中没有 libiomp 或 libtcmalloc | 按第 1 节设置 `IAXL_LD_PRELOAD`，并用 `examples/kvshrink-vllm-serve.sh` 启动 vLLM。 |
| `RDMA (remote_pool) needs NIXL, which DEVICE=cpu builds do not link` | 跨节点 KV cache 复用需要 CUDA 或 XPU 构建。 |
