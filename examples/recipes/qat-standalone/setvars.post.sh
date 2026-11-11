# qat-standalone recipe, hook 3 of 3: docker run flags.
#
# Replaces the default --privileged, --pid host and host /dev with the container's own devices,
# and forwards the values the container cannot derive itself. Runs on the host and again in the
# container, where DOCKER_RUN_ARGS is unused; it only rebuilds arrays, so it is idempotent.

# The container sees only its GPU and VFs, so it cannot place its rank again: forward the CPUs,
# DDR pool share and ports the host picked. ./setvars.pre.sh keeps these when they are set, so
# nothing is adjusted twice.
ENV_VARS+=(
    PORT
    IAXL_API_CONTROLLER_PORT
    IAXL_API_WORKER_BASE_PORT
    VLLM_CPU_OMP_THREADS_BIND
    KVSHRINK_CPU_OMP_THREADS_BIND
    IAXL_DDR_POOL_TOTAL_SIZE_GB
    IAXL_DDR_POOL_SIZE_GB
)
# The recipe's own inputs, for vLLM and kvshrink-vllm-serve.sh in the container.
ENV_VARS+=(RECIPE_NUMA_NODE RECIPE_IAXL_CPUS_PER_RANK VLLM_CPU_KVCACHE_SPACE MAX_MODEL_LEN)

# Drop the flags that share the host's devices and qatmgr, and pass only the container's GPU.
recipe_args=()
for ((recipe_i = 0; recipe_i < ${#DOCKER_RUN_ARGS[@]}; recipe_i++)); do
    recipe_arg=${DOCKER_RUN_ARGS[recipe_i]}
    recipe_next=${DOCKER_RUN_ARGS[recipe_i + 1]:-}
    case "$recipe_arg" in
        --privileged) continue ;;
        --pid) recipe_i=$((recipe_i + 1)); continue ;;
        -v)
            if [[ "$recipe_next" == /dev:/dev || "$recipe_next" == /run/qat:/run/qat ]]; then
                recipe_i=$((recipe_i + 1))
                continue
            fi
            ;;
        --gpus)
            if [[ -n "$recipe_qat_gpu" ]]; then
                recipe_args+=(--gpus "device=$recipe_qat_gpu")
                recipe_i=$((recipe_i + 1))
                continue
            fi
            ;;
    esac
    recipe_args+=("$recipe_arg")
done
DOCKER_RUN_ARGS=("${recipe_args[@]}")
[[ "$DEVICE" == xpu && -n "$recipe_qat_gpu" ]] && DOCKER_RUN_ARGS+=(-e "ZE_AFFINITY_MASK=$recipe_qat_gpu")

# qatlib opens only these VFs; it and the pinned IAXL pools need unlimited memlock. QAT_POLICY lets
# the one process take all of them. SYS_NICE and seccomp=unconfined allow the NUMA memory policy
# calls (mbind, migrate_pages) that vLLM's CPU backend makes. The shared host /dev/shm of the
# default --privileged -v /dev:/dev is gone, so give vLLM's shared-memory queues room.
DOCKER_RUN_ARGS+=(
    --device /dev/vfio/vfio
    --ulimit memlock=-1:-1
    --shm-size 16g
    --cap-add SYS_NICE
    --security-opt seccomp=unconfined
    -e "QAT_POLICY=$(wc -l <<<"$recipe_qat_vfs")"
)
while read -r recipe_vf recipe_group; do
    DOCKER_RUN_ARGS+=(--device "/dev/vfio/$recipe_group")
done <<<"$recipe_qat_vfs"

# DSA + GDRCopy, IAA and InfiniBand are passed only when enabled at start.
if [[ -n "${KVSHRINK_DSA_DEVICES:-}" && -d /dev/dsa ]]; then
    DOCKER_RUN_ARGS+=(--device /dev/dsa)
    [[ -e /dev/gdrdrv ]] && DOCKER_RUN_ARGS+=(--device /dev/gdrdrv)
    ENV_VARS+=(IAXL_DSA_GD_ENABLE KVSHRINK_DSA_DEVICES)
fi
if env_truthy "$IAXL_IAA_ZIP_ENABLE" && [[ -d /dev/iax ]]; then
    DOCKER_RUN_ARGS+=(--device /dev/iax)
    ENV_VARS+=(IAXL_IAA_ZIP_ENABLE IAXL_IAA_DEVICES)
fi
if env_truthy "$IAXL_RDMA_ENABLE" && [[ -d /dev/infiniband ]]; then
    DOCKER_RUN_ARGS+=(--device /dev/infiniband)
    ENV_VARS+=(IAXL_RDMA_ENABLE)
fi

unset recipe_args recipe_i recipe_arg recipe_next recipe_vf recipe_group recipe_qat_gpu recipe_qat_vfs
return 0
