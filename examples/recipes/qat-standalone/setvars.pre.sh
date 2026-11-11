# qat-standalone recipe, hook 2 of 3: environment defaults.
#
# Scenario: the in-tree QAT driver in qatlib standalone mode, which qatlib recommends for
# containers. Each container gets its own QAT VFs and opens them itself, with no host qatmgr, so
# it runs without --privileged, --pid host or the host /dev. Ranks in one container would race for
# its VFs, so every container runs one rank: one container per GPU or, on CPU, per NUMA node.
#
# With no other input, `IAXL_RECIPE=qat-standalone ./start.sh` picks the device (cuda when
# nvidia-smi finds a GPU, else cpu) and the first GPU or QAT NUMA node without a running
# container, so running it again in another terminal starts the next one. CUDA_VISIBLE_DEVICES /
# ZE_AFFINITY_MASK or RECIPE_NUMA_NODE select one explicitly.
#
# Runs on the host (start.sh), where it places the rank, and again in the container, where the
# CPUs, DDR pool and ports come from the host (./setvars.post.sh forwards them) and the VFs from
# /dev/vfio. Every default keeps a value set on the command line.

export IAXL_QATLIB=${IAXL_QATLIB:-intree}
if [[ "$IAXL_QATLIB" != intree ]]; then
    echo "ERROR: the qat-standalone recipe needs the in-tree QAT driver (IAXL_QATLIB=intree), got IAXL_QATLIB='$IAXL_QATLIB'" >&2
    return 1
fi
export TP_SIZE=${TP_SIZE:-1}
recipe_workers=$(iaxl_num_workers) || return 1
if ((recipe_workers != 1)); then
    echo "ERROR: containers in qatlib standalone mode run one rank each; got TP_SIZE=$TP_SIZE DP_SIZE=${DP_SIZE:-1}." \
        "Start one container per GPU or, on CPU, per NUMA node instead," \
        "or run without IAXL_RECIPE=qat-standalone (host qatmgr or out-of-tree driver) for several ranks in one container." >&2
    return 1
fi
unset recipe_workers

# ---- Device ------------------------------------------------------------------
recipe_num_gpus=0
if [[ "${DEVICE:-cuda}" == cuda && "${NVIDIA_RUNTIME:-}" != none ]] && command -v nvidia-smi >/dev/null; then
    recipe_num_gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
fi
if [[ -z "${DEVICE:-}" ]]; then
    export DEVICE=$( ((recipe_num_gpus > 0)) && echo cuda || echo cpu)
fi
if [[ "$DEVICE" == cpu ]]; then
    # vLLM CPU inference, as in the vllm-cpu recipe (doc/usage/vllm-cpu-inference.en.md).
    export IAXL_BASE_DOCKER_IMAGE=${IAXL_BASE_DOCKER_IMAGE:-vllm/vllm-openai-cpu:v0.29.0}
    export IAXL_DEV_DOCKER_IMAGE=${IAXL_DEV_DOCKER_IMAGE:-vllm-iaxl-dev-cpu}
    export IAXL_LD_PRELOAD=${IAXL_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so}
    export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-40} # Live vLLM KV cache in GiB
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
    # vLLM's inference threads take one CPU per core, so IAXL gets the last cores of the node.
    export RECIPE_IAXL_CPUS_PER_RANK=${RECIPE_IAXL_CPUS_PER_RANK:-2}
    export MODEL=${MODEL:-Qwen/Qwen3-32B} # setvars.sh's default, needed below for the DDR pool
fi

# ---- The container's GPU, CPUs and NUMA node -------------------------------
recipe_qat_gpu=
recipe_auto_cpus=
case "$DEVICE" in
    cuda) recipe_qat_gpu=${CUDA_VISIBLE_DEVICES:-} ;;
    xpu) recipe_qat_gpu=${ZE_AFFINITY_MASK:-} ;;
esac
if [[ "$DEVICE" != cpu && -z "$recipe_qat_gpu" ]]; then
    if in_container; then
        recipe_qat_gpu=0 # The container sees only its GPU
    else
        recipe_gpus=$recipe_num_gpus
        [[ "$DEVICE" == xpu ]] && recipe_gpus=$(ls -d /dev/dri/renderD* 2>/dev/null | wc -l)
        ((recipe_gpus > 0)) || recipe_gpus=1
        recipe_qat_gpu=$(standalone_free_slot gpu $(seq 0 $((recipe_gpus - 1)))) || return 1
        echo "qatlib standalone: picked GPU $recipe_qat_gpu (the first without a running container)"
        unset recipe_gpus
    fi
fi
if [[ -n "$recipe_qat_gpu" && ! "$recipe_qat_gpu" =~ ^[0-9]+$ ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK to the index of the container's one GPU, got '$recipe_qat_gpu'" >&2
    return 1
fi
if ((recipe_num_gpus > 0)); then
    # The GPU's container gets the CPUs, DSA WQ and DDR pool share of the GPU's rank in a single
    # container with all GPUs. In the container, which sees only its GPU, these come from the host.
    if ((recipe_qat_gpu >= recipe_num_gpus)); then
        echo "ERROR: CUDA_VISIBLE_DEVICES=$recipe_qat_gpu, but only $recipe_num_gpus GPUs were found" >&2
        return 1
    fi
    if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" ]]; then
        recipe_ranks=$(cpu_auto_detect "$recipe_num_gpus") || return 1
        VLLM_CPU_OMP_THREADS_BIND=$(cut -d'|' -f$((recipe_qat_gpu + 1)) <<<"$recipe_ranks")
        recipe_auto_cpus=1
    fi
    if env_truthy "${IAXL_DSA_GD_ENABLE:-0}" && [[ -z "${KVSHRINK_DSA_DEVICES:-}" ]]; then
        recipe_ranks=$(dsa_auto_detect "$recipe_num_gpus") || return 1
        export KVSHRINK_DSA_DEVICES=$(cut -d'|' -f$((recipe_qat_gpu + 1)) <<<"$recipe_ranks")
    fi
    if [[ -z "${IAXL_DDR_POOL_SIZE_GB:-}${IAXL_DDR_POOL_TOTAL_SIZE_GB:-}" ]]; then
        recipe_pool_gb=$(ddr_pool_total_size_gb "$recipe_num_gpus") || return 1
        export IAXL_DDR_POOL_TOTAL_SIZE_GB=$((recipe_pool_gb / recipe_num_gpus))
    fi
    unset recipe_ranks
fi
if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" ]]; then
    # CPU inference (or a GPU without nvidia-smi): every core of one NUMA node with QAT.
    recipe_node=${RECIPE_NUMA_NODE:-}
    if [[ -z "$recipe_node" ]]; then
        if [[ -z "$recipe_qat_gpu" ]] && ! in_container; then
            recipe_node=$(standalone_free_slot numa $(qat_numa_nodes)) || return 1
            echo "qatlib standalone: picked NUMA node $recipe_node (the first with QAT and without a running container)"
        else
            recipe_node=0
        fi
    fi
    VLLM_CPU_OMP_THREADS_BIND=$(numa_physical_cpus "$recipe_node") || return 1
    recipe_auto_cpus=1
    unset recipe_node
fi
# Split off IAXL's CPUs only from a binding picked here, never from one given on the command line.
if [[ -n "$recipe_auto_cpus" && -n "${RECIPE_IAXL_CPUS_PER_RANK:-}" && -z "${KVSHRINK_CPU_OMP_THREADS_BIND:-}" ]]; then
    recipe_split=$(cpu_list_split "$VLLM_CPU_OMP_THREADS_BIND" "$RECIPE_IAXL_CPUS_PER_RANK") || return 1
    VLLM_CPU_OMP_THREADS_BIND=${recipe_split%%$'\n'*}
    export KVSHRINK_CPU_OMP_THREADS_BIND=${recipe_split##*$'\n'}
    unset recipe_split
fi
export VLLM_CPU_OMP_THREADS_BIND
unset recipe_auto_cpus
# The first CPU of the binding selects the NUMA node, and with it the QAT devices.
recipe_qat_numa=$(cpu_numa_node "${VLLM_CPU_OMP_THREADS_BIND%%[^0-9]*}") || return 1

# ---- DDR pool (CPU inference) ----------------------------------------------
# vLLM's weights and KV cache share the node with the IAXL DDR pool, and memory taken from
# another node would slow everything down: size the pool from the node, not from the host.
if [[ "$DEVICE" == cpu && -z "${IAXL_DDR_POOL_SIZE_GB:-}${IAXL_DDR_POOL_TOTAL_SIZE_GB:-}" ]]; then
    recipe_node_gb=$(numa_mem_gb "$recipe_qat_numa") || return 1
    if in_container; then
        recipe_hub=${HF_HOME:-/_data/hf_home}/hub
    else
        recipe_hub=$PWD/_data/hf_home/hub # start.sh mounts it as HF_HOME
    fi
    if ! recipe_weights_gb=$(model_weights_gb "$MODEL" "$recipe_hub"); then
        recipe_weights_gb=0
        echo "WARNING: the weights of $MODEL were not found locally, so the DDR pool default leaves no room for them;" \
            "download the model first or set IAXL_DDR_POOL_SIZE_GB" >&2
    fi
    recipe_pool_gb=$((recipe_node_gb * 8 / 10 - VLLM_CPU_KVCACHE_SPACE - recipe_weights_gb))
    if ((recipe_pool_gb < 1)); then
        echo "ERROR: NUMA node $recipe_qat_numa has $recipe_node_gb GB; 80% of it leaves no room for the IAXL DDR pool" \
            "after $recipe_weights_gb GB of weights and VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE GiB." \
            "Use a smaller model or KV cache, or set IAXL_DDR_POOL_SIZE_GB" >&2
        return 1
    fi
    echo "qatlib standalone: DDR pool $recipe_pool_gb GB = 80% of node $recipe_qat_numa ($recipe_node_gb GB)" \
        "- KV cache $VLLM_CPU_KVCACHE_SPACE - weights $recipe_weights_gb"
    export IAXL_DDR_POOL_TOTAL_SIZE_GB=$recipe_pool_gb
    unset recipe_node_gb recipe_hub recipe_weights_gb
fi
unset recipe_pool_gb recipe_num_gpus

# ---- QAT VFs ---------------------------------------------------------------
if in_container; then
    recipe_qat_vfs=$(qat_container_vfs) || return 1
else
    recipe_qat_vfs=$(qat_standalone_vfs "$recipe_qat_numa" "${recipe_qat_gpu:-0}") || return 1
fi
# The container sees only these VFs, so device indices count from 0, and the rank uses all of them.
export KVSHRINK_QAT_DEVICES=${KVSHRINK_QAT_DEVICES:-$(seq -s, 0 $(($(wc -l <<<"$recipe_qat_vfs") - 1)))}
export IAXL_QAT_DEVICES=${IAXL_QAT_DEVICES:-$KVSHRINK_QAT_DEVICES}
echo "qatlib standalone: NUMA node $recipe_qat_numa, QAT VFs $(cut -d' ' -f1 <<<"$recipe_qat_vfs" | paste -sd,)"
[[ -S /run/qat/qatmgr.sock ]] && echo "WARNING: the host qatmgr is running; its clients can take the VFs this container uses." \
    "Use one QAT mode per host (systemctl stop qat keeps the VFs bound)." >&2

# ---- Container name and ports ----------------------------------------------
# Containers share the host network, so each GPU (or NUMA node on CPU) gets its own ports.
recipe_offset=${recipe_qat_gpu:-$recipe_qat_numa}
export PORT=${PORT:-$((8000 + recipe_offset))}
export IAXL_API_CONTROLLER_PORT=${IAXL_API_CONTROLLER_PORT:-$((18700 + recipe_offset))}
export IAXL_API_WORKER_BASE_PORT=${IAXL_API_WORKER_BASE_PORT:-$((18800 + recipe_offset))}
CONTAINER_NAME=${CONTAINER_NAME:-iaxl.vllm.$([[ -n "$recipe_qat_gpu" ]] && echo "gpu$recipe_qat_gpu" || echo "numa$recipe_qat_numa")}
unset recipe_offset recipe_qat_numa
# recipe_qat_gpu and recipe_qat_vfs are used and unset by ./setvars.post.sh.
return 0
