# Recipe `qat-standalone`: qatlib Standalone Mode, One Rank per Container

qatlib recommends standalone mode for containers (qatlib `INSTALL`, "STANDALONE MODE"): the container gets its own QAT VFs and opens them itself, with no host `qatmgr`. By default, `start.sh` with `IAXL_QATLIB=intree` shares the host `qatmgr` (managed mode), so the container needs `--privileged`, `--pid host` and the host `/dev` to reach any VF `qatmgr` might hand out, and a rank can get VFs on another NUMA node. This recipe passes in only the container's VFs, all on its NUMA node, and drops those flags.

Ranks in one container would race for its VFs, so each container runs one rank (`TP_SIZE=1`). Start one container per GPU, or with `DEVICE=cpu` one per NUMA node. For several ranks in one container, run without this recipe (host `qatmgr`) or with the out-of-tree driver. The mechanism is described in [doc/design/recipes.md](../../../doc/design/recipes.md).

## Run

```bash
# Host: each run starts the container of the next GPU without one; on a host without GPUs,
# DEVICE=cpu and the next NUMA node with QAT
IAXL_RECIPE=qat-standalone ./start.sh

# Container shell opened by start.sh (IAXL_RECIPE is forwarded)
./examples/kvshrink-vllm-serve.sh
```

That is all the input needed. To choose instead: `CUDA_VISIBLE_DEVICES=<gpu>` (XPU: `DEVICE=xpu ZE_AFFINITY_MASK=<gpu>`), `RECIPE_NUMA_NODE=<node>` on CPU, or explicit `VLLM_CPU_OMP_THREADS_BIND` / `KVSHRINK_CPU_OMP_THREADS_BIND` lists, which are used as given. Set `IAXL_DSA_GD_ENABLE=1`, `IAXL_IAA_ZIP_ENABLE=1` or `IAXL_RDMA_ENABLE=1` before `start.sh` to pass the DSA and GDRCopy, IAA or InfiniBand devices. Containers share the source tree and each builds IAXL on entry, so start them one after another.

## Defaults

| Setting | GPU (`DEVICE=cuda`, default when `nvidia-smi` finds a GPU) | CPU (`DEVICE=cpu`, default otherwise) |
| --- | --- | --- |
| Rank placement | The first GPU without a running `iaxl.vllm.gpu<N>`; `--gpus device=<N>` (XPU: `ZE_AFFINITY_MASK` is passed in) | The first NUMA node with CPUs and a QAT PF without a running `iaxl.vllm.numa<N>` |
| CPUs | The CPUs of the GPU's rank in a single container with all GPUs (`cpu_auto_detect`), shared by vLLM and IAXL as without the recipe | One CPU per core of the node: the last `RECIPE_IAXL_CPUS_PER_RANK` (2) for IAXL, the rest for vLLM |
| QAT VFs | One VF of each QAT PF on the node of the first vLLM CPU, counted back from the last VF by GPU index so containers do not overlap, plus `/dev/vfio/vfio` and `QAT_POLICY=<VF count>` | Same, last VF |
| QAT devices, threads | `KVSHRINK_QAT_DEVICES` / `IAXL_QAT_DEVICES` default to all of the container's VFs, so the rank uses every instance; the default `IAXL_OMP_THREAD_NUM` is capped at 8 and at IAXL's CPUs (its own list, or the rank's CPUs minus `IAXL_RESERVED_CPU_NUM`) | Same, so 2 threads on the 2 IAXL CPUs |
| DDR pool | The GPU's share of the default budget, as without the recipe | 80% of the node's memory minus `VLLM_CPU_KVCACHE_SPACE` and the model weights (when found under `_data/hf_home` or in a `MODEL` directory), so nothing spills to another node |
| DSA | The GPU's rank's WQ with `IAXL_DSA_GD_ENABLE=1` | not used |
| Name, ports | `iaxl.vllm.gpu<N>`; `PORT`, `IAXL_API_CONTROLLER_PORT`, `IAXL_API_WORKER_BASE_PORT` = 8000, 18700, 18800 + N | `iaxl.vllm.numa<N>`, ports + node |
| vLLM | `TP_SIZE=1` | `TP_SIZE=1`, the CPU image and `IAXL_LD_PRELOAD` of the `vllm-cpu` recipe, `VLLM_CPU_KVCACHE_SPACE=40`, `MAX_MODEL_LEN=16384` |
| Other flags | Unlimited memlock, 16 GiB `/dev/shm`, `SYS_NICE` and `seccomp=unconfined` (NUMA memory policy calls), instead of `--privileged`, `--pid host`, the host `/dev` and `/run/qat` | Same |

The container sees only its GPU and VFs, so it cannot place its rank again: `setvars.post.sh` forwards the CPU lists, the DDR pool sizes and the ports, and the device settings above are fixed when the container starts.

## Performance Notes

Compared with the default managed mode (host `qatmgr`, `--privileged`):

- Every QAT VF the rank uses is on its NUMA node. With `qatmgr` and `POLICY=0`, a process gets one VF of every PF on the host, and the `KVSHRINK_QAT_DEVICES` split only keeps them local when the PFs happen to be ordered by node.
- `IAXL_QAT_INSTANCE_NUM` and the default `IAXL_OMP_THREAD_NUM` follow the devices the rank really opens. Without the recipe they are computed from `IAXL_QAT_DEVICES=0` (one device) while each rank opens its whole `KVSHRINK_QAT_DEVICES` entry.
- Nothing on the data path needs the dropped privileges: the pinned pools and QAT DMA memory need only the memlock limit, NUMA memory policy calls need `SYS_NICE` (Docker's default seccomp profile blocks them otherwise), and `kernel.numa_balancing=0` is set on the host by `start.sh`.
- On CPU, IAXL's threads stay off vLLM's inference cores and the DDR pool stays within the node.

## Host Requirements

The host needs the VFs bound to `vfio-pci` with the `dc` service enabled, which `qat.service` does at boot. Clients of a running host `qatmgr` can take the same VFs, so use one mode per host: after boot, `systemctl stop qat` stops `qatmgr` and keeps the VFs bound. The recipe warns when it finds the `qatmgr` socket.

## Files

| File | Purpose |
| --- | --- |
| [auto_config.sh](auto_config.sh) | Helpers to pick a free GPU or node (`standalone_free_slot`), find the VFs on the host (`qat_standalone_vfs`) and in the container (`qat_container_vfs`), and size the CPU DDR pool; caps the default `IAXL_OMP_THREAD_NUM` by wrapping `omp_thread_count` |
| [setvars.pre.sh](setvars.pre.sh) | Checks `IAXL_QATLIB=intree` and one rank, picks the device, GPU or NUMA node, CPUs, DDR pool, VFs, QAT devices, ports and container name |
| [setvars.post.sh](setvars.post.sh) | Replaces the shared-host `docker run` flags with the container's devices and forwards the values the container cannot derive |

Check the result without starting anything:

```bash
IAXL_RECIPE=qat-standalone bash -c 'source setvars.sh && printf "%q " "${DOCKER_RUN_ARGS[@]}"; echo'
```
