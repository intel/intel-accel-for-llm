# 跨节点 KV 缓存(NIXL)—— 设计文档

> 状态：已实现，位于 `iaxl/remote/`(控制面 + TCP/NIXL 两种数据面)与
> `iaxl/csrc/torch_ext/torch_ext.cpp`(新增两个原生入口函数),并通过
> `KVSHRINK_REMOTE_CACHE_ENABLE` 接入 `kvshrink/kvshrink_connector.py`。英文版：
> [`remote-nixl-kv-cache.md`](remote-nixl-kv-cache.md)。使用说明：
> [`../usage/remote-nixl-kv-cache.zh-CN.md`](../usage/remote-nixl-kv-cache.zh-CN.md)。

## 一、背景与目标

KVShrink 在 vLLM 之上,把 GPU 产生的 KV block 缓存到本机(每个 rank 各自)的
host 内存池(`iaxl.kvstore.KVStore`),用于前缀复用以降低 TTFT。本方案新增第
二套后端 `iaxl.remote.RemoteKVStore`,把这个池子下沉到一个**独立的、仅做缓存
的节点**:vLLM worker 通过 NIXL(RDMA/GDR)把 KV block 直接搬到远端 daemon,
而不是在本机压缩;daemon 完成压缩、入池与持久化,命中时再通过 RDMA 传回。

这是对更早期(基于旧代码库 `KVCacheClip`/`kvclip`)同一思路实现的迭代,主要有
三点变化:

1. **remote daemon 尽量复用 iaxl 已有的池管理/压缩解压/chunk 处理代码**,而不是
   daemon 端另起一套纯 Python 实现。
2. **remote daemon 运行在与 GPU 节点相同的容器镜像**(`vllm/vllm-openai:v0.23.0`)
   中,而不是裁剪出的 CPU-only 专用构建。
3. **避免多对一争用**:所有 GPU worker rank 都对接同一个 daemon 进程,会在控制
   面与压缩并发闸门上产生串行/争用。现在 daemon 也可以以 **每个 GPU rank 一个
   进程** 的方式运行。

其余部分(块级 API 形态、NIXL 两阶段 begin/commit/done 协议、TCP 回退、多轮
staging 配额)基本原样保留,因为它们此前已经验证可用。

## 二、复用范围:"复用 iaxl 的池/压缩/chunk 逻辑"具体指什么

本机路径是:`KVStore` -> `KVFlow` -> 原生 `Context`(GPU 搬运 + 压缩)-> 原生
`Mem`/`Storage`/`Record`(DDR 池 + 磁盘持久化)。`Context` 类**天生绑定 GPU**:
`Context::create()` 需要一个真实的 CUDA/XPU tensor
(`iaxl/csrc/torch_ext/context.h`),`iaxl.kvflow.KVFlow.put()`/`get()` 也会
断言 `tensor.is_cuda or tensor.is_xpu`(`iaxl/kvflow/flow.py`)。远端仅做缓存
的节点没有 GPU,因此 `KVFlow` 无法原样复用。

但有两部分**可以直接复用**,因为它们本来就与 GPU 无关:

- **`iaxl.torch_ext.Mem` / `Storage` / `Record`**(`iaxl/csrc/include/
  kv_pool.h`):分组的 DDR 池(LRU、字节预算)、磁盘持久化(`chunks/` 目录 +
  `chunks.db` SQLite)及其 Python 绑定(`iaxl/csrc/torch_ext/torch_ext.cpp`)
  从不涉及 GPU。`Mem.put(keys, data)` / `Mem.get(keys)` / `Mem.has(keys)`
  本身操作的就是普通字节 blob。
- **`kv_zip::kv_zip_compress_batch` / `kv_zip_decompress_batch`**
  (`iaxl/csrc/kv_zip/kv_zip.cpp`,声明于 `iaxl/csrc/include/kv_zip.h`):
  QAT/IAA/CPU 共享的压缩任务池。两个函数都断言输入 tensor **必须是 CPU 上的
  连续 tensor**(`IAXL_CHECK(tensor.is_contiguous() &&
  tensor.device().type()==CPU, ...)`)——完全不依赖 GPU。此前它们只能通过
  `Context::zip_to_mem()`/`unzip_from_mem()` 间接调用,而这两个函数把压缩/
  解压与远端 daemon 根本不需要的 GPU 搬运步骤耦合在了一起(KV 字节已经通过
  RDMA 直接落在 CPU 的 staging buffer 里了)。

因此唯一需要的原生代码改动,是在 `torch_ext.cpp` 里新增两个很小的 pybind11
自由函数,直接调用上述两部分、不经过 `Context`:

```cpp
// iaxl/csrc/torch_ext/torch_ext.cpp
m.def("zip_compress_to_mem",
      [](Mem &mem, labels, cpu_tensors, compress) {
          kv_zip::kv_zip_compress_batch(cpu_tensors, out_bufs, out_sizes, orig_sizes, compress);
          mem.put(labels, std::move(out_bufs), out_sizes, orig_sizes);
      });
m.def("zip_decompress_from_mem",
      [](Mem &mem, labels, cpu_tensors) {
          auto results = mem.get(labels);
          kv_zip::kv_zip_decompress_batch(data_ptrs, cpu_tensors);
      });
```

这两个函数的函数体分别是 `Context::zip_to_mem`/`unzip_from_mem`
(`iaxl/csrc/torch_ext/zip.cpp`)去掉搬运/Context 部分后的样子。daemon
(`iaxl/remote/server.py`)直接调用它们:**与本机、挂 GPU 的 `KVStore` 完全
相同的原生池、相同的压缩后端(QAT/IAA/CPU)、相同的磁盘 chunk 格式**——由完全
相同的 `IAXL_KV_COMPRESSION` / `IAXL_QAT_ZIP_ENABLE` / `IAXL_QAT_DEVICES` /
`IAXL_QAT_ZIP_INSTANCES_PER_DEVICE` 等环境变量控制。daemon 侧不存在旧实现
中那种独立的“codec”概念(不再有 `codec=qat/zlib/none` 的选择):压缩后端的
选择与本机完全一致,由原生库的配置决定;`iaxl.remote.daemon` 只额外提供一个
`--no-compress` 开关,用于裸带宽基准测试。

由于 `zip_compress_to_mem`/`zip_decompress_from_mem` 完全不触碰 CUDA/XPU,
daemon 可以直接加载 GPU 节点构建出的同一个 `iaxl.torch_ext` 扩展,并在完全
没有 GPU 的主机上运行——上文第 2 点由此自然满足(见第八节)。

### 2.1 每个 (model, tp_size, tp_rank) 一组原生 Mem/Storage/Record

daemon 为每个 `(model_name, tp_size, tp_rank)` 创建一组独立的
`RemoteCacheGroup`(自己的 `Mem`/`Storage`/`Record` 三元组),路径为
`{cache_dir}/{model}_tp{tp_size}_rank{tp_rank}/`——与
`iaxl.kvflow.KVFlow` 本机使用的 `{persist_dir}/{model}_rank{rank}/`
的 `chunks.db` + `chunks/` 布局完全一致(多加入 tp_size 是为了在同一个 daemon
上区分同一模型的不同 TP 配置)。原生 chunk key 使用完全相同的
`{label}:{block_hash}:{tensor_key}` 约定
(`iaxl/csrc/include/kv_pool.h::make_chunk_label`,`label="kv"` 与
`KVStore.LABEL` 一致),因此一个 block 跨**所有层**的分片都落在同一个原生
“group”里(同一个 LRU/持久化单元),与本机路径完全一致。

有一处结构性差异:本机路径中,一层的 K 和 V 是**一起**压缩成一个 blob 的
(`KVFlow.put()` 对每层只调用一次 `zip_to_mem`,传入 D2H 已经重组好的
`[2, ...]` 形状 CPU tensor)。而远端数据面需要把 K、V 作为**两个独立的**
RDMA 描述符处理(当 `block_dim==1` 时,它们在 GPU 显存中并不连续),因此
daemon 把它们存成同一个 block group 下的两个并列条目:
`tensor_key = "{layer_id}"`(MLA / 已融合布局)或
`"{layer_id}.k"` / `"{layer_id}.v"`(拆分布局)——这里用 `.` 而非 `:`,
是因为 `kv_pool.h::validate_label_component` 不允许单个 label 分量里出现
`:` 或 `/`。这会损失一点 K/V 之间的压缩相关性,但与原生池的
group/LRU/持久化语义完全兼容。

## 三、总体架构

```
GPU 节点(vLLM)                                     远端缓存节点(1..N 个)
+----------------------------+                     +----------------------------------+
| Scheduler 进程             |                      | iaxl.remote.daemon(1..N 个进程)   |
|  KVShrinkConnector(SCHED)  |   控制面(TCP)         |  +------------------------------+ |
|   -> RemoteKVStore(仅 has) +--------------------->|  | 控制面 server                 | |
|                            |  has / capability     |  |  session/has/mark_ready       | |
+----------------------------+                      |  +------------------------------+ |
| Worker 进程(rank r)        |                      |  | 每个 (model, tp_size,        | |
|  KVShrinkConnector(WORKER) |   控制面(TCP)         |  | tp_rank) 一组 RemoteCacheGroup:| |
|   -> RemoteKVStore(worker) +--------------------->|  |  原生 Mem/Storage/Record      | |
|      -> RemoteCacheClient  |  put/get begin/commit |  |  (LRU 池 + chunks.db/目录)    | |
|      -> DataPlane(nixl/tcp)|                       |  +------------------------------+ |
|         (注册本地 GPU HBM) |  数据面(RDMA/NIXL)     |        ^  zip_compress_to_mem /   |
|                            +=======================+========+  zip_decompress_from_mem |
+----------------------------+  RDMA WRITE/READ      +----------------------------------+
```

- **Scheduler 侧 `RemoteKVStore`(仅 has)**:只查询 `has()`;不注册 GPU、不建
  NIXL agent。用于 `get_num_new_matched_tokens()`。
- **Worker 侧 `RemoteKVStore`**:在 `register_kv_caches()` 中创建,基于真实
  GPU KV tensor 建立 NIXL 本地 agent,建立 session,驱动 `put`/`get`/`_wait`。
- **daemon**:控制面 TCP server + 每个 session 一组 `RemoteCacheGroup` +
  (可选)供数据面使用的 NIXL staging 池。

## 四、元数据与线上 key

与此前实现的设计基本一致(已验证正确,且与 vLLM 版本/attention backend 无关):

- 一个远端分片由 `(model_name, tp_size, tp_rank, block_hash, layer_id,
  tensor_key)` 唯一标识;非 MLA 的 `[2, num_blocks, ...]` 布局
  (`block_dim==1`)下 `tensor_key` 为 `"k"`/`"v"`,MLA/融合布局下为
  `"kv"`(`iaxl.remote.metadata.tensor_keys_for_layout`,与
  `kvshrink_connector.register_kv_caches` 自身的 `block_dim` 判断一致)。
- `block_descriptors(shape, block_dim, elem_size, block_index)` 对标准
  C-contiguous tensor 计算出一个 block 所占的连续 `(offset, length)` 字节
  区间——NIXL 数据面用它构造传输描述符,TCP 回退用它切片字节,二者共享同一套
  偏移计算。
- 线上的分片 key 是精简的、session 内相对的
  `"{block_hash}|{layer_id}|{tensor_key}"`(不再需要 model/tp/rank 前缀:
  一个 session 本就一一对应一个 `RemoteCacheGroup`)。
- `iaxl.remote.metadata.full_chunk_label()` 把它映射为第 2.1 节所述的原生
  `Mem` key。

## 五、建链与协议

与此前实现相同的四步握手与消息集合
(`iaxl/remote/protocol.py`、`client.py`、`server.py`):

1. **能力协商**:`RemoteCacheClient.connect()` 带重试直到连通(若
   `KVSHRINK_REMOTE_FAIL_IF_UNREACHABLE=1`,默认如此,则连不上直接报错);
   交换协议版本。
2. **创建 session**:worker/scheduler 发送
   `model_name/tp_size/tp_rank/num_layers/tensor_keys/dtype/block_size/
   shard_bytes`;daemon 创建(或复用)对应的 `RemoteCacheGroup`,返回
   `session_id` 及其宣告的 NIXL staging 容量。
3. **NIXL 握手**(仅 worker,且 `transport=nixl` 时):交换 NIXL agent
   metadata;worker 在首次 put/get 时惰性注册自己的 GPU KV tensor。
4. **运行时 `put`/`get`/`has`/`mark_ready`**——见第六节。

帧格式:4 字节长度前缀 + JSON header(+ TCP 数据面场景下可选的二进制 blob)。
控制面为何仍然选择 TCP 而非 RDMA SEND,见 `protocol.py` 里的说明,第九节有
更完整的分析。

## 六、运行时数据流

**SAVE(put)**:`save_kv_layer(layer)` -> `RemoteKVStore.put(block_ids,
block_hashes, [layer])` 为每个 (block, k/v) 构造一个 `ShardRef` -> 提交到
线程池 -> `DataPlane.put()`。NIXL 路径:`begin`(daemon 分配 staging 并返回
描述符)-> client RDMA `WRITE` GPU->staging -> `commit`(daemon 读取
staging、调用 `zip_compress_to_mem`、释放槽位)。某个 block 的最后一层写完
后,worker 调用 `mark_ready()`。

**LOAD(get)**:scheduler 的 `has()` 决定命中前缀;`start_load_kv()` ->
按层调用 `RemoteKVStore.get()` -> `DataPlane.get()`。NIXL 路径:`begin`
(daemon 调用 `zip_decompress_from_mem` 解压到 staging;对确实缺失的 key
做零填充,见下文)-> client RDMA `READ` staging->GPU -> `done`(daemon 释放
槽位,fire-and-forget)。

一个关于缓存未命中的细节:`zip_decompress_from_mem` 在 key 缺失时会
**中止整个 daemon 进程**(原生 `IAXL_CHECK` 语义,不是可捕获的异常——见
`iaxl/csrc/include/iaxl_common.h`)。因此 `server.py` 总是先调用
`mem.has(keys)`,只对存在的子集做解压,对缺失的(可能是在 `has()` 和
`get()` 之间被淘汰,也可能是真正的 bug)做零填充,而不是让 daemon 崩溃。

TCP 回退:分片字节直接内联在控制面帧里传输;daemon 仍然调用完全相同的
`zip_compress_to_mem`/`zip_decompress_from_mem`,只是跳过 RDMA 步骤。

## 七、多路 RDMA 端口/链路

`KVSHRINK_REMOTE_NIXL_DEVICE`(client 侧)/ `NIXL_DEVICE`(daemon 脚本侧)
既可以是单个设备(`"mlx5_0:1"`),也可以是逗号分隔的列表
(`"mlx5_0:1,mlx5_1:1"`)。该值会原样转发给 `UCX_NET_DEVICES`;UCX 本身会把
一个 NIXL session 的 RDMA 流量分摊(striping)到列表中的所有设备上("多轨/
multi-rail")。这一能力从此前的实现中原样保留——不需要、也不应该在应用层
另建一套按设备轮询的抽象,因为 UCX 已经在为一条逻辑 NIXL 连接做这件事了。

## 八、daemon 单进程 vs. 多进程

一个 daemon 进程可以服务一个 TP 组的所有 rank,但这样一来,所有 rank 的
控制面 RPC 与压缩调用都会在这一个进程上产生争用——实测在多卡负载下会
明显影响性能(请求在控制 socket 的分发循环、以及只允许一个 `kv_zip_*`
调用同时在飞的单 worker codec 队列上串行,原因见
`iaxl/remote/server.py::_CodecWorker`)。`iaxl.remote` 同时支持两种模式:

- **单进程**(`--num-instances 1`,默认):一个控制端口、一个 NIXL staging
  池,被所有 rank 共享。`KVSHRINK_REMOTE_QAT_DEVICES`(如果设置)会被
  **拉平合并**:`"0|1|4|5"` 和 `"0,1,4,5"` 都会变成
  `IAXL_QAT_DEVICES=0,1,4,5`,即这一个进程会一起使用列出的所有设备
  (对应 `iaxl.remote.daemon._resolve_qat_devices` 中 `num_instances<=1`
  的分支)。
- **多进程**(`--num-instances N --instance-id i`,或直接使用
  `tools/remote_daemon/run-daemon-multi.sh`):每个 GPU rank 一个进程,各自
  拥有自己的控制/NIXL 端口。每个实例只取 `KVSHRINK_REMOTE_QAT_DEVICES` 中
  **属于自己的那一份**(与 `kvshrink_connector._bind_intel_accel` 中
  `KVSHRINK_QAT_DEVICES`/`KVSHRINK_DSA_DEVICES` 相同的按 rank
  `"|"`分隔约定):实例 `i` 得到 `IAXL_QAT_DEVICES=<第 i 项>`,而**不是**
  所有设备的并集。

这是两种模式中唯一行为真正不同的地方,因此特意单独说明:如果搞反了,要么
让多进程部署里每个实例都只看到设备 0(QAT 设备被闲置),要么让单进程和某个
rank 独占同一个设备产生冲突。

为了把 GPU rank `r` 定向到 daemon 实例 `r`(从而避免多对一争用),
`KVSHRINK_REMOTE_DAEMON_ADDR` 与 `KVSHRINK_REMOTE_NIXL_ADDR` 同样支持按 rank
`"|"`分隔的写法(`iaxl.remote.config._resolve_per_rank`):普通的
`host:port` 会被所有 rank 共享;用 `"|"` 连接的列表会为 rank `r` 选取第 `r`
项。scheduler 角色总是解析为第 `0` 项,这与 "用 rank 0 的就绪记录作为整个
部署命中代理" 的约定(`server.py` 中的 `_h_has`)保持一致。

## 九、控制面为何仍用 TCP,而非 RDMA SEND

我们评估过,也否决了把控制面(能力协商、创建 session、`has`、`mark_ready`、
NIXL 握手、put 的 begin/commit、get 的 begin/done)从 TCP 换成 RDMA
SEND/RECV:

- 控制面帧本身很小(单次请求几百字节到几 KB,即使是一整轮的分片列表),
  相对于同一轮里 NIXL WRITE/READ 已经在搬运的数据量而言微不足道。
- 本地/RoCE 环境下一次 TCP 往返大约是几十微秒量级;RDMA SEND 或许能省下
  其中的几微秒。无论哪种情况,相对于一轮实际的 RDMA 传输 + QAT/IAA
  (解)压缩耗时(根据 daemon 的 `RoundStats` 日志,通常是几百微秒到几毫秒
  量级)都可以忽略不计。
- RDMA SEND 需要在已有的 NIXL 数据面 agent 之外,再搭建一条独立的可靠消息
  通道(自己的 completion queue、消息序号、agent 握手流程)——为了一个
  接近测量噪声量级的收益,引入了实打实的实现复杂度。
- 实际观测到(也是第一节第 3 点变化的动机)的瓶颈,是多对一扇入下的
  **控制面/压缩闸门争用**,这正是第八节的多进程 daemon 直接解决的问题;
  而不是控制面本身的**时延**——后者才是 RDMA SEND 能改善的对象。

结论:**控制面继续使用 TCP**(已经设置 `TCP_NODELAY`;每个 worker 线程
独立一条 socket,已经避免了并发轮次之间的队头阻塞——见
`iaxl/remote/transport/nixl_backend.py`)。

## 十、代码结构

```
iaxl/remote/
  config.py          client 侧配置(环境变量 KVSHRINK_REMOTE_* + kv_connector_extra_config)
  protocol.py         控制面帧格式 + 消息类型常量
  metadata.py         RemoteKey、block_descriptors、原生 full_chunk_label()
  client.py           RemoteCacheClient(控制面 RPC)
  remote_kvstore.py   RemoteKVStore(与 KVStore 接口兼容的门面)
  server.py           RemoteCacheDaemon:控制面 server + RemoteCacheGroup
                       (原生 Mem/Storage/Record)+ zip_compress_to_mem /
                       zip_decompress_from_mem 调用
  daemon.py           CLI 入口(单/多进程 QAT 设备解析)
  transport/
    base.py           DataPlane 接口 + ShardRef
    tcp_backend.py     TCP 数据面(可移植回退)
    nixl_backend.py     NIXL 数据面(生产路径)+ daemon 侧 staging 池

iaxl/csrc/torch_ext/torch_ext.cpp
  zip_compress_to_mem() / zip_decompress_from_mem()   新增、与 GPU 无关的绑定

kvshrink/kvshrink_connector.py
  _make_kvstore() / _remote_cache_config()             本地/远端 KVStore 的选择工厂

tools/remote_daemon/
  run-daemon.sh        单 daemon 进程启动脚本(环境变量驱动)
  run-daemon-multi.sh   N 进程启动脚本,每个 GPU rank 一个实例
```

## 十一、现状与后续

- 已实现:控制面、TCP 与 NIXL 两种数据面、原生压缩解压 + 池复用、单/多进程
  daemon、connector 接入、部署脚本。
- 本次改动尚未在真实 QAT/RDMA 硬件上跑过压测(编写时没有这样的环境可用);
  投入生产前请先用 `start.sh` / `pip install -e .` 编译并验证,并先跑一遍
  使用文档里的自测流程。
- 可能的后续工作(此前实现中也是未完成项):session 重连/容错、跨 rank 的
  严格就绪判定(目前用 rank 0 作为整个部署的命中代理)、远端池向共享存储
  (DAOS 等)的二级持久化。
