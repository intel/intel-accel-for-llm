# vLLM CPU Inference with KVShrink

[简体中文](vllm-cpu-inference.zh-CN.md)

This guide shows how to configure threads and CPU binding when vLLM runs inference on the CPU and KVShrink compresses the KV cache on Intel QAT or IAA. It requires a CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`) in the container and IAXL built with `DEVICE=cpu`. For the design behind these settings, see the [KV Zip design](../design/kv-zip.md).

## 1. Build

A `DEVICE=cpu` build uses the same `KVStore`, codecs, and persist/evict APIs as GPU inference. It needs no CUDA, SYCL, GDRCopy, or NIXL, so DSA transfers and RDMA (`remote_pool`) are not available.

As for GPU inference, build and start the container with `start.sh` from the repository root. Point it at the vLLM CPU release image and set `IAXL_QATLIB` to match the host QAT driver:

```bash
export DEVICE=cpu
export IAXL_BASE_DOCKER_IMAGE=vllm/vllm-openai-cpu:v0.29.0
export IAXL_DEV_DOCKER_IMAGE=vllm-iaxl-dev-cpu
export IAXL_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so
export IAXL_QATLIB=intree # in-tree driver (qat_4xxx, VFs bound to vfio-pci); leave unset for the out-of-tree driver (/dev/qat_dev_processes)
./start.sh
```

- `DEVICE=cpu` skips UCX and NIXL in the dev image, starts the container without GPU arguments, and builds IAXL with `DEVICE=cpu` on entry.
- `IAXL_DEV_DOCKER_IMAGE` keeps the CPU image apart from the GPU one. `start.sh` reuses an existing image with that tag, so without it the CPU run would reuse the GPU image (or overwrite it with `--rebuild`).
- `IAXL_LD_PRELOAD` is what the example and benchmark scripts prepend to `LD_PRELOAD`. vLLM's CPU backend wants libtcmalloc and libiomp5 preloaded, and the CPU image ships libiomp5 in `/opt/venv/lib` instead of the default `/usr/local/lib`.
- With `IAXL_QATLIB=intree`, the build compiles qatlib from source into `_lib` when `pkg-config` finds no system qatlib (the Ubuntu 22.04 CPU image has none), and `start.sh` mounts the host `/run/qat`, so each rank gets its QAT instances from the host `qatmgr`. If the host qatlib is not `26.02.0`, `export IAXL_QATLIB_VERSION=<host version>` before `./start.sh --rebuild`. Without `qatmgr`, the first process reserves all QAT devices, so runs with more than one rank need it.

> [!NOTE]
> Without Docker, IAXL also builds in an environment that already has CPU vLLM installed. Install the packages that `docker/Dockerfile.dev` installs (except UCX and NIXL), then run `python -m pip install -r requirements.txt` and `DEVICE=cpu python -m pip install -e . --no-build-isolation`, with `IAXL_QATLIB=intree` for the in-tree driver.

## 2. CPU Binding

Two variables split each rank's CPUs between vLLM and IAXL:

| Variable | Used by | Threads |
| --- | --- | --- |
| `VLLM_CPU_OMP_THREADS_BIND` | vLLM | OpenMP inference threads, one per listed CPU |
| `KVSHRINK_CPU_OMP_THREADS_BIND` | IAXL | `OMP-Main`, `POOL-<n>`, `H2D`, `D2H`, and other IAXL threads |

- **Format:** both variables contain one entry per rank, separated by `|`. Each entry is a CPU list such as `0-61` or `0-7,16`.
- **Rank order:** entry `r` belongs to global rank `r = dp_rank × TP_SIZE + tp_rank`, so each list needs `TP_SIZE × DP_SIZE` entries.
- **Default:** `KVSHRINK_CPU_OMP_THREADS_BIND` defaults to `VLLM_CPU_OMP_THREADS_BIND`. In that case IAXL threads share the inference cores, so set it explicitly for CPU inference.
- **At startup:** the connector exports the rank's entry as `IAXL_CPU_OMP_THREADS_BIND`, and IAXL pins all of its threads to those CPUs.

When choosing CPUs:

- For each rank, keep the vLLM and IAXL entries disjoint.
- Put both entries on the same NUMA node. Choose the rank's QAT devices (`KVSHRINK_QAT_DEVICES`) on that node too.
- Use one logical CPU per physical core. `lscpu -e=CPU,CORE,NODE` shows the mapping.
- Give IAXL at least `IAXL_OMP_THREAD_NUM` CPUs per rank.

## 3. Thread Settings

| Variable | Recommended | Purpose |
| --- | --- | --- |
| `IAXL_USE_OMP` | `0` (default with `DEVICE=cpu`) | IAXL runs on its own thread pool instead of OpenMP, which leaves OpenMP to vLLM. `setvars.sh` then unsets `OMP_NUM_THREADS` and `OMP_THREAD_LIMIT`. |
| `IAXL_OMP_THREAD_NUM` | `1` or `2` | Number of threads driving the compression workers. One thread can keep several QAT/IAA instances busy; more threads than workers split each worker's in-flight requests. |
| `IAXL_CPU_ZIP_ENABLE` | `0` | Spends no cores on zlib; QAT and IAA do the compression. |
| `IAXL_QAT_ZIP_ENABLE`, `IAXL_IAA_ZIP_ENABLE` | `1` for the accelerators you have | The worker count is `IAXL_QAT_INSTANCE_NUM + IAXL_IAA_INSTANCE_NUM`; `setvars.sh` prints both values. |

## 4. Cache Settings

`examples/kvshrink-vllm-serve.sh` drops `--gpu-memory-utilization` when `DEVICE=cpu`. These variables commonly need overriding for CPU inference:

| Variable | Purpose |
| --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | Live vLLM KV cache size in GiB. `IAXL_DDR_POOL_SIZE_GB` sizes the compressed KVShrink cache independently. |
| `MAX_MODEL_LEN` | `--max-model-len` (default `32765`; `16384` in the `vllm-cpu` recipe). Lower it when the CPU KV cache cannot hold one full-length request. |
| `VLLM_SERVE_EXTRA_ARGS` | More `vllm serve` options, appended to the script's defaults (split on whitespace). |
| `BLOCK_SIZE` | `--block-size`, passed only when set. vLLM v0.29 defaults to `128` on CPU (`16` for MLA models) and needs a multiple of 32 otherwise. A CPU KV block holds `2 × num_kv_heads × block_size × head_size × dtype_bytes` bytes per layer and must fit in `IAXL_ZIP_SRC_CAP` and, compressed, in `IAXL_ZIP_DST_CAP` (both 4 MiB in `setvars.sh`). For larger blocks, raise both caps or set `IAXL_ZIP_SEGMENT_SIZE_KB` to compress each block in segments of that many KiB; then only each segment, shorter than twice that size, must fit. |
| `IAXL_KV_DATA_SHUFFLE`, `IAXL_KV_LOSSY_TRUNC` | With both set to `0`, the codec compresses straight from the vLLM KV blocks and decompresses straight into them, with no staging copies. `setvars.sh` enables shuffle by default. Either one rewrites the codec input, so blocks are staged in scratch buffers. Shuffle improves the compression ratio at the cost of one extra pass over every block. |

## 5. Example

This example runs two ranks on a host where NUMA node 0 has CPUs 0-63 and node 1 has CPUs 64-127, with one logical CPU per core:

```bash
# With the exports from section 1
./start.sh

# In the container shell opened by start.sh
export TP_SIZE=2
export VLLM_CPU_OMP_THREADS_BIND="0-61|64-125"       # vLLM inference
export KVSHRINK_CPU_OMP_THREADS_BIND="62-63|126-127" # IAXL threads
export KVSHRINK_QAT_DEVICES="0,1|2,3"                # QAT devices on each rank's NUMA node
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_OMP_THREAD_NUM=2

./examples/kvshrink-vllm-serve.sh
```

The script sources `setvars.sh` and prepends `IAXL_LD_PRELOAD` (libtcmalloc and libiomp5, as vLLM's CPU backend recommends) to `LD_PRELOAD`.

The [`vllm-cpu` recipe](../../examples/recipes/vllm-cpu/README.md) packages the exports of sections 1 to 4 and this CPU split, so `IAXL_RECIPE=vllm-cpu ./start.sh` followed by `./examples/kvshrink-vllm-serve.sh` in the container starts the same setup. The recipe also sets `MAX_MODEL_LEN=16384`, takes `VLLM_CPU_KVCACHE_SPACE` out of the default DDR pool budget and gives the container a 16 GiB `/dev/shm`; see the [recipes design](../design/recipes.md).

### qatlib Standalone Mode

With the in-tree driver, the [`qat-standalone` recipe](../../examples/recipes/qat-standalone/README.md) runs each container in qatlib standalone mode, which qatlib recommends for containers: the container gets the last VF of each QAT PF on its NUMA node and opens them itself, with no host `qatmgr` and no `--privileged`. Each container runs one rank on one NUMA node, and on a host without GPUs the recipe applies the settings of sections 1 to 4 itself:

```bash
IAXL_RECIPE=qat-standalone ./start.sh   # first NUMA node with QAT: iaxl.vllm.numa0, port 8000

# In the container shell opened by start.sh
./examples/kvshrink-vllm-serve.sh

# The same in a second terminal starts the next node (iaxl.vllm.numa1, port 8001)
IAXL_RECIPE=qat-standalone ./start.sh
```

The recipe gives vLLM every core of the node but the last two, which go to IAXL (`RECIPE_IAXL_CPUS_PER_RANK`) with `IAXL_OMP_THREAD_NUM=2`, and sizes the DDR pool from the node's memory minus `VLLM_CPU_KVCACHE_SPACE` (40 GiB) and the model weights, when they are already downloaded. `RECIPE_NUMA_NODE=<n>` selects a node, and explicit `VLLM_CPU_OMP_THREADS_BIND` / `KVSHRINK_CPU_OMP_THREADS_BIND` lists are used as given. The host needs the VFs bound to `vfio-pci` and no running `qatmgr` (`systemctl stop qat` after boot keeps them bound). Containers share the source tree and each builds IAXL on entry, so start them one after another.

## 6. Verification

- **`setvars.sh` output:** check `VLLM_CPU_OMP_THREADS_BIND`, `KVSHRINK_CPU_OMP_THREADS_BIND`, `IAXL_OMP_THREAD_NUM`, and `IAXL_USE_OMP=0`.
- **vLLM log:** each rank prints `Bound rank <r>: IAXL_CPU_OMP_THREADS_BIND=<cpus>` and `[iaxl] config: ... omp_threads=2 use_omp=OFF ...`.
- **Thread affinity:** after the first request, run the command below. IAXL threads (`OMP-Main`, `POOL-*`, `H2D`, `D2H`) should only list the rank's `KVSHRINK_CPU_OMP_THREADS_BIND` entry.

  ```bash
  for f in $(grep -l "^OMP-Main$" /proc/[0-9]*/task/*/comm); do
      pid=$(cut -d/ -f3 <<<"$f"); echo "== pid $pid"
      for t in /proc/$pid/task/*; do
          echo "$(cat $t/comm) $(grep Cpus_allowed_list $t/status | cut -f2)"
      done | sort | uniq -c
  done
  ```

## 7. Troubleshooting

| Message | Fix |
| --- | --- |
| `KVSHRINK_CPU_OMP_THREADS_BIND has N entries, but rank is R` | Provide one entry per rank (`TP_SIZE × DP_SIZE`). |
| `invalid IAXL_CPU_OMP_THREADS_BIND` | An entry is not a valid CPU list. |
| `cannot bind to IAXL_CPU_OMP_THREADS_BIND` | None of the listed CPUs is available to the process, for example because they are outside the container's cpuset. |
| `IAXL_OMP_THREAD_NUM must be a positive integer` | Set `IAXL_OMP_THREAD_NUM` to 1 or more. |
| `IAXL_OMP_THREAD_NUM (...) + reserved (...) exceeds the smallest rank CPU allocation` | The smallest `VLLM_CPU_OMP_THREADS_BIND` entry is too small. Lower `IAXL_RESERVED_CPU_NUM` or `IAXL_OMP_THREAD_NUM`. |
| vLLM warns that libiomp or libtcmalloc is not in `LD_PRELOAD` | Start vLLM with `examples/kvshrink-vllm-serve.sh`, with `IAXL_LD_PRELOAD` set as in section 1. |
| `RDMA (remote_pool) needs NIXL, which DEVICE=cpu builds do not link` | Cross-node KV cache reuse needs a CUDA or XPU build. |
| `containers in qatlib standalone mode run one rank each` | With the `qat-standalone` recipe, start one container per NUMA node with `TP_SIZE=1` (section 5). |
