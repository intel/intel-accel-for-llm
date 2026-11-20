[English](README.md)

# 性能测试

请在配置好的开发环境中，从仓库根目录运行以下命令。

## KVStore

测试 KVStore 的 PUT/GET 带宽、压缩率以及压缩/解压吞吐量。

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --qat
```

启动脚本模仿 vLLM 的结构：主进程作为 scheduler，为每个 TP rank 启动一个 worker 进程（`--ranks N`，默认 1）。rank `r` 使用 GPU `r`，按 `setvars.sh` 中的 per-rank 配置绑定自己的 CPU / QAT / DSA（`VLLM_CPU_OMP_THREADS_BIND`、`KVSHRINK_QAT_DEVICES`、`KVSHRINK_DSA_DEVICES`），拥有独立的 `KVStore(rank=r)`，所有 rank 同时开始每个计时步骤。报告先列出每个 rank 及 `Sum` 行，再给出汇总结果，其中耗时取最慢的 rank，与 TP 前向一致。

`--qat` 和/或 `--iaa` 选择压缩引擎；两者都不指定时关闭压缩（`IAXL_KV_COMPRESSION=0`）并打印警告。`--dsa` 启用 DSA gather/scatter，`--nsys` 用 Nsight Systems 包裹运行：

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --ranks 2 --qat --iaa --dsa
bash benchmark/kvstore/kvstore_benchmark.sh --qat --shape 2 1024 16 4 128 --dtype bf16 --num-layers 32
```

默认使用 `$MODEL` 真实模型预填充生成 KV cache 数据，结果以 `<model>_<dtype>.pt` 保存在 `--kv-data-dir` 中，shape 匹配时直接复用（由 rank 0 生成，其余 rank 复用）。`--dtype` 支持 `bf16`、`fp8_e4m3`（per-tensor k/v scale）和 `int4`（对称 group-wise，每字节两个 4bit 值），量化方式与 vLLM 保持一致。使用 `--data-source mock` 可改用合成数据：

```bash
bash benchmark/kvstore/kvstore_benchmark.sh --qat --dtype int4 \
    --prompt-file /path/to/text.txt --model-seq-len 16384
bash benchmark/kvstore/kvstore_benchmark.sh --ranks 2 --qat --data-source mock
```

也可以直接运行 `kvstore_benchmark.py`，即不做 CPU / 加速器绑定的单进程测试。

## Tensor Transfer

对比 CUDA、`cudaMemcpy3DBatchAsync`、Triton 和 IAXL 的碎片化 H2D/D2H 传输性能，并生成结果图。

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh
```

用 `--direction h2d` 或 `--direction d2h` 只跑单个方向，用 `--methods` 只跑部分方法（`cuda`、`batch`、`triton`、`iaxl`）：

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --direction h2d --methods iaxl cuda
```

`--ranks N` 为每个 TP rank 启动一个进程（rank `r` 使用 GPU `r`，绑定方式与 KVStore 测试相同），并报告各 rank 的平均值。

所有生成的文件都放在 `/_data/tensor_xfer_benchmark`，可用 `--output-dir` 修改。

加上 `--flamegraph` 可使用 `perf record` 采样。perf 通过 control FIFO 控制，因此只采集计时迭代（不含 warmup），并覆盖进程的所有线程，包括 IAXL DSA 的原生工作线程。运行后会生成火焰图 SVG、供 `flamegraph.pl` 或 speedscope 使用的 `.folded` 文件，并保留原始 `perf.data` 以供 `perf report` 分析。

```bash
bash benchmark/tensor_xfer/tensor_xfer_benchmark.sh --flamegraph tensor_xfer_flame.svg
```

要求容器内已安装 `perf`，且宿主机 `kernel.perf_event_paranoid <= 2`。二进制文件未保留帧指针时可用 `--flamegraph-call-graph dwarf`，`--flamegraph-freq` 用于调整采样频率。采样会干扰计时，因此仅用于分析，不要用于上报性能数据。

### 全双工 H2D/D2H

`tensor_xfer_benchmark_bidir.py` 同时运行两个方向，每个方向一个线程，持续 `--seconds`（默认 1 秒），分别报告另一方向同时在跑时本方向的 GB/s。`iaxl` 两个方向都走 IAXL 传输路径，并默认设置 `IAXL_KV_XFER_FULL_DUPLEX=1`，即与 KV cache 一样 H2D 走 DSA、D2H 走 CUDA 拷贝引擎；导出 `IAXL_KV_XFER_FULL_DUPLEX=0` 可对比两个方向都走 DSA 的情况。`cuda` 和 `batch` 使用两个 CUDA stream。

```bash
IAXL_DSA_GD_ENABLE=1 source setvars.sh
LD_PRELOAD=$IAXL_LD_PRELOAD numactl --cpunodebind=0 --membind=0 \
    python3 benchmark/tensor_xfer/tensor_xfer_benchmark_bidir.py --methods iaxl batch
```
