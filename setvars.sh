export IAXL_VERSION=${IAXL_VERSION:-1.5.1}
export IAXL_BASE_DOCKER_IMAGE=${IAXL_BASE_DOCKER_IMAGE:-"vllm/vllm-openai:v0.29.0"}
export IAXL_DEV_DOCKER_IMAGE=${IAXL_DEV_DOCKER_IMAGE:-"vllm-iaxl-dev:$IAXL_VERSION"}
export IAXL_BUILDER_DOCKER_IMAGE=${IAXL_BUILDER_DOCKER_IMAGE:-"vllm-iaxl-builder"}

TOP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TOP_DIR/tools/auto_config.sh"
$TOP_DIR/tools/setup_system.sh

# =============================================================================
# Configurable runtime / build environment variables (all IAXL_* prefixed).
# Each keeps its default unless already set in the environment; edit as needed.
# =============================================================================

# ---- Build ------------------------------------------------------------------
export DEVICE=${DEVICE:-cuda}                 # Build backend: cuda | xpu | cpu (vLLM CPU inference)
export IAXL_CMAKE_ARGS=${IAXL_CMAKE_ARGS:-""} # Extra cmake flags, e.g. "-DENABLE_NVTX=OFF"
# export IAXL_QATLIB=intree                   # Optional: qatlib for the in-tree driver (system qatlib if pkg-config finds it, else built into _lib); unset = out-of-tree QAT20 package
# export IAXL_QATLIB_VERSION=26.02.0          # Optional: qatlib tag built into _lib for IAXL_QATLIB=intree; match the host qatmgr
export NVIDIA_RUNTIME=${NVIDIA_RUNTIME:-}     # Set to "none" to skip the nvidia docker runtime/gpus args (e.g. on a storage-only node without GPUs)

# ---- Feature switches -------------------------------------------------------
export IAXL_KV_COMPRESSION=${IAXL_KV_COMPRESSION:-1} # Enable DEFLATE compression (0/1)
export IAXL_QAT_ZIP_ENABLE=${IAXL_QAT_ZIP_ENABLE:-1} # Enable QAT compression workers (0/1)
export IAXL_IAA_ZIP_ENABLE=${IAXL_IAA_ZIP_ENABLE:-0} # Enable IAA (QPL) compression workers (0/1)
export IAXL_CPU_ZIP_ENABLE=${IAXL_CPU_ZIP_ENABLE:-0} # Enable CPU compression workers (0/1); costs a lot of CPU, keep off unless needed
export IAXL_DSA_GD_ENABLE=${IAXL_DSA_GD_ENABLE:-0}   # Use Intel DSA + GDRCopy transfers (0/1)

# ---- Async KV load ----------------------------------------------------------
export KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=${KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED:-1}      # Enable asynchronous KV loading (0/1)
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS:--1}       # -1=wait all layers, N=start prefill after first N layers (used when DYNAMIC=0)
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC:-1} # 0=fixed LAYERS, 1=select layers from DYNAMIC_MAP per request concurrency
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP:-0-3:0,4-6:8,7-:16}" # Contiguous START-END:LAYERS rules from 0; 0 layers means sync and the final range is open-ended

# ---- vLLM ------------------------------------------------------------------
# qatlib standalone mode: containers (start.sh) with the in-tree driver open their own QAT VFs, one rank per container.
# Inside, start.sh's QAT_POLICY marks such a container.
QAT_STANDALONE=
[[ "${IAXL_QATLIB:-}" == intree ]] && [[ -n "${IAXL_START_SH:-}" || (-f /.dockerenv && -n "${QAT_POLICY:-}") ]] && QAT_STANDALONE=1
export MODEL="${MODEL:-Qwen/Qwen3-32B}" # Hugging Face model ID or local model path
export TP_SIZE="${TP_SIZE:-$([[ -n "$QAT_STANDALONE" ]] && echo 1 || echo 2)}" # Tensor-parallel worker count (1 in qatlib standalone mode)
export DP_SIZE="${DP_SIZE:-1}"            # Data-parallel group count
# Total workers = one per (dp, tp) pair; CPU/QAT/DSA specs are indexed per worker.
NUM_WORKERS=$((TP_SIZE * DP_SIZE))
QAT_STANDALONE_VFS=
QAT_GPU=
if [[ -n "$QAT_STANDALONE" ]]; then
    if ((NUM_WORKERS != 1)); then
        echo "ERROR: containers with IAXL_QATLIB=intree run one rank each (qatlib standalone mode); got TP_SIZE=$TP_SIZE DP_SIZE=$DP_SIZE." \
            "Start one container per GPU (CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK) or, on CPU, per NUMA node instead." >&2
        return 1 2>/dev/null || exit 1
    fi
    case "$DEVICE" in
        cuda) QAT_GPU=${CUDA_VISIBLE_DEVICES:-0} ;;
        xpu) QAT_GPU=${ZE_AFFINITY_MASK:-0} ;;
    esac
    if [[ -n "$QAT_GPU" && ! "$QAT_GPU" =~ ^[0-9]+$ ]]; then
        echo "ERROR: set CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK to the index of the container's one GPU, got '$QAT_GPU'" >&2
        return 1 2>/dev/null || exit 1
    fi
    if [[ "$DEVICE" == cuda && "$NVIDIA_RUNTIME" != none ]] && command -v nvidia-smi >/dev/null; then
        # Each GPU container gets the CPUs and DSA WQ of its GPU's rank in a single container with all GPUs.
        QAT_NUM_GPUS=$(nvidia-smi -L | wc -l)
        if ((QAT_GPU >= QAT_NUM_GPUS)); then
            echo "ERROR: CUDA_VISIBLE_DEVICES=$QAT_GPU, but only $QAT_NUM_GPUS GPUs were found" >&2
            return 1 2>/dev/null || exit 1
        fi
        if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" ]]; then
            QAT_ALL_RANKS=$(cpu_auto_detect "$QAT_NUM_GPUS") || return 1 2>/dev/null || exit 1
            VLLM_CPU_OMP_THREADS_BIND=$(cut -d'|' -f$((QAT_GPU + 1)) <<<"$QAT_ALL_RANKS")
        fi
        if env_truthy "$IAXL_DSA_GD_ENABLE" && [[ -z "${KVSHRINK_DSA_DEVICES:-}" ]]; then
            QAT_ALL_RANKS=$(dsa_auto_detect "$QAT_NUM_GPUS") || return 1 2>/dev/null || exit 1
            KVSHRINK_DSA_DEVICES=$(cut -d'|' -f$((QAT_GPU + 1)) <<<"$QAT_ALL_RANKS")
        fi
        IAXL_DDR_POOL_TOTAL_SIZE_GB="${IAXL_DDR_POOL_TOTAL_SIZE_GB:-$(($(ddr_pool_total_size_gb "$QAT_NUM_GPUS") / QAT_NUM_GPUS))}" || return 1 2>/dev/null || exit 1
    fi
    VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-$(numa_physical_cpus 0)}" || return 1 2>/dev/null || exit 1
fi
VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-$(cpu_auto_detect "$NUM_WORKERS")}" || return 1 2>/dev/null || exit 1
export VLLM_CPU_OMP_THREADS_BIND # Per-worker CPU affinity (one entry per global rank)
export KVSHRINK_CPU_OMP_THREADS_BIND="${KVSHRINK_CPU_OMP_THREADS_BIND:-$VLLM_CPU_OMP_THREADS_BIND}" # Per-worker CPUs for iaxl threads (one entry per global rank)
rank_cpu_counts "$VLLM_CPU_OMP_THREADS_BIND" "$NUM_WORKERS" || return 1 2>/dev/null || exit 1
if [[ -n "$QAT_STANDALONE" ]]; then
    QAT_NUMA=$(cpu_numa_node "${VLLM_CPU_OMP_THREADS_BIND%%[^0-9]*}") || return 1 2>/dev/null || exit 1
    if [[ -n "${IAXL_START_SH:-}" ]]; then
        QAT_STANDALONE_VFS=$(qat_standalone_vfs "$QAT_NUMA" "${QAT_GPU:-0}") || return 1 2>/dev/null || exit 1
    else
        QAT_STANDALONE_VFS=$(qat_container_vfs) || return 1 2>/dev/null || exit 1
    fi
    # The container sees only these VFs, so device indices count from 0.
    KVSHRINK_QAT_DEVICES="${KVSHRINK_QAT_DEVICES:-$(seq -s, 0 $(($(wc -l <<<"$QAT_STANDALONE_VFS") - 1)))}"
    # Containers share the host network, so each GPU (or NUMA node on CPU) gets its own ports.
    QAT_PORT_OFFSET=${QAT_GPU:-$QAT_NUMA}
    export PORT="${PORT:-$((8000 + QAT_PORT_OFFSET))}"
    export IAXL_API_CONTROLLER_PORT="${IAXL_API_CONTROLLER_PORT:-$((18700 + QAT_PORT_OFFSET))}"
    export IAXL_API_WORKER_BASE_PORT="${IAXL_API_WORKER_BASE_PORT:-$((18800 + QAT_PORT_OFFSET))}"
    [[ -S /run/qat/qatmgr.sock ]] && echo "WARNING: the host qatmgr is running; its clients can take the VFs this container uses. Use one QAT mode per host (systemctl stop qat keeps the VFs bound)." >&2
fi
case "${IAXL_QAT_ZIP_ENABLE,,}" in
    1|true|yes|on)
        KVSHRINK_QAT_DEVICES="${KVSHRINK_QAT_DEVICES:-$(qat_auto_detect "$NUM_WORKERS")}" || return 1 2>/dev/null || exit 1
        export KVSHRINK_QAT_DEVICES # Per-rank QAT device indices
        ;;
    *)
        unset KVSHRINK_QAT_DEVICES
        ;;
esac
case "${IAXL_DSA_GD_ENABLE,,}" in
    1|true|yes|on)
        KVSHRINK_DSA_DEVICES="${KVSHRINK_DSA_DEVICES:-$(dsa_auto_detect "$NUM_WORKERS")}" || return 1 2>/dev/null || exit 1
        export KVSHRINK_DSA_DEVICES # Per-rank DSA work queues
        export IAXL_DSA_WQS="${IAXL_DSA_WQS:-${KVSHRINK_DSA_DEVICES%%|*}}" # Use rank 0 DSA work queues by default
        ;;
    *)
        unset KVSHRINK_DSA_DEVICES
        ;;
esac

printf '%s\n' \
    "vLLM configuration:" \
    "  MODEL=$MODEL" \
    "  TP_SIZE=$TP_SIZE" \
    "  DP_SIZE=$DP_SIZE" \
    "  IAXL_KV_COMPRESSION=$IAXL_KV_COMPRESSION" \
    "  IAXL_QAT_ZIP_ENABLE=$IAXL_QAT_ZIP_ENABLE" \
    "  IAXL_IAA_ZIP_ENABLE=$IAXL_IAA_ZIP_ENABLE" \
    "  IAXL_CPU_ZIP_ENABLE=$IAXL_CPU_ZIP_ENABLE" \
    "  IAXL_DSA_GD_ENABLE=$IAXL_DSA_GD_ENABLE" \
    "  VLLM_CPU_OMP_THREADS_BIND=$VLLM_CPU_OMP_THREADS_BIND" \
    "  KVSHRINK_CPU_OMP_THREADS_BIND=$KVSHRINK_CPU_OMP_THREADS_BIND" \
    "  KVSHRINK_QAT_DEVICES=${KVSHRINK_QAT_DEVICES:-disabled}${QAT_STANDALONE_VFS:+ (qatlib standalone, NUMA $QAT_NUMA VFs: $(cut -d' ' -f1 <<<"$QAT_STANDALONE_VFS" | paste -sd,))}" \
    "  KVSHRINK_DSA_DEVICES=${KVSHRINK_DSA_DEVICES:-disabled}" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=$KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=$KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=$KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=$KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP"

# ---- Cache / compression ----------------------------------------------------
export IAXL_KV_LOSSY_TRUNC=${IAXL_KV_LOSSY_TRUNC:-0}                     # Lossy LSB truncation: 'auto', 0 (off), or N bits
export IAXL_KV_DATA_SHUFFLE=${IAXL_KV_DATA_SHUFFLE:-1}                   # Byte-shuffle before compression (0/1)
export IAXL_ZIP_EXPONENT_ONLY=${IAXL_ZIP_EXPONENT_ONLY:-$IAXL_KV_DATA_SHUFFLE} # With shuffle, compress only the BF16 high (sign/exponent) bytes and store the low bytes raw (0/1)
export IAXL_ZIP_LEVEL=${IAXL_ZIP_LEVEL:-1}                               # Compression level 1-9: higher = better ratio, slower
export IAXL_ZIP_SRC_CAP=${IAXL_ZIP_SRC_CAP:-4194304}                     # Source/decompressed block capacity (4 MiB)
export IAXL_ZIP_DST_CAP=${IAXL_ZIP_DST_CAP:-4194304}                     # Compressed-output capacity (4 MiB)
export IAXL_ZIP_SEGMENT_SIZE_KB=${IAXL_ZIP_SEGMENT_SIZE_KB:-0}           # Compress blocks in segments of N KiB (0 = whole block)
export IAXL_CACHE_DIR=${IAXL_CACHE_DIR:-_data/kvcache}                   # Base directory for persisted cache files
export IAXL_CACHE_STREAM_SYNC_ON_GET=${IAXL_CACHE_STREAM_SYNC_ON_GET:-0} # CPU-sync the GPU stream on get() (0/1)
export IAXL_CACHE_CACHEGROUP_SIZE=${IAXL_CACHE_CACHEGROUP_SIZE:-100}     # Reserved entries per cache group
export IAXL_CACHE_CACHEGROUP_NUM=${IAXL_CACHE_CACHEGROUP_NUM:-100000}    # Reserved number of cache groups
export IAXL_PREALLOC_LIMIT=${IAXL_PREALLOC_LIMIT:-0}                     # Cap pinned scratch-pool pre-allocation (0 = unlimited)
export IAXL_SCRATCH_POOL_SIZE_GB=${IAXL_SCRATCH_POOL_SIZE_GB:-8}         # Pinned scratch pool (GPU<->CPU staging) size in GB
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=${IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS:-1} # Store the first N KV layers without compression
IAXL_DDR_POOL_TOTAL_SIZE_GB="${IAXL_DDR_POOL_TOTAL_SIZE_GB:-$(ddr_pool_total_size_gb "$NUM_WORKERS")}" || return 1 2>/dev/null || exit 1
export IAXL_DDR_POOL_TOTAL_SIZE_GB # DDR (host) cache pool budget across all ranks in GB (default 80% of host RAM, each rank limited to 80% of one NUMA node)
IAXL_DDR_POOL_SIZE_GB="${IAXL_DDR_POOL_SIZE_GB:-$((IAXL_DDR_POOL_TOTAL_SIZE_GB / NUM_WORKERS))}"
export IAXL_DDR_POOL_SIZE_GB # Per-rank DDR pool size in GB (default TOTAL / NUM_WORKERS)
export IAXL_DDR_POOL_EVICT_HIGH_WATERMARK=${IAXL_DDR_POOL_EVICT_HIGH_WATERMARK:-0.90} # Scheduler evicts LRU groups once the DDR pool reaches this usage ratio
export IAXL_DDR_POOL_EVICT_LOW_WATERMARK=${IAXL_DDR_POOL_EVICT_LOW_WATERMARK:-0.5}    # ... until usage drops to about this ratio, evicting only groups whose lease (LEASE_S) has expired
# WARNING: reaching FORCE may evict the oldest groups that are still within their lease, i.e. groups has()
# reported as present whose get()/put() may not have landed yet (see doc/design/eviction.md). Keep FORCE
# well above LOW so it only kicks in as a last resort under hot load; values below 0.5 are rejected.
export IAXL_DDR_POOL_EVICT_FORCE_WATERMARK=${IAXL_DDR_POOL_EVICT_FORCE_WATERMARK:-0.8} # Usage one pass must reach even if it has to evict groups still within their lease (0.5 <= LOW <= FORCE <= HIGH)
export IAXL_DDR_POOL_EVICT_LEASE_S=${IAXL_DDR_POOL_EVICT_LEASE_S:-300}               # Groups hit by has() within this many seconds are not evicted unless needed to reach FORCE; must exceed has() -> last load/save of a request
export IAXL_DDR_POOL_EVICT_INTERVAL_S=${IAXL_DDR_POOL_EVICT_INTERVAL_S:-30}          # How often the scheduler checks DDR pool usage

# ---- Intel QAT (compression accelerator) ------------------------------------
if [[ -n "$QAT_STANDALONE" && -n "${KVSHRINK_QAT_DEVICES:-}" ]]; then
    # qatlib standalone mode: the container's one rank uses every QAT device it was given, with all of their instances.
    export IAXL_QAT_DEVICES=${IAXL_QAT_DEVICES:-$KVSHRINK_QAT_DEVICES}
fi
export IAXL_QAT_DEVICES=${IAXL_QAT_DEVICES:-0}                                   # Comma-separated QAT device indices, e.g. "0,1"
export IAXL_QAT_ZIP_INSTANCES_PER_DEVICE=${IAXL_QAT_ZIP_INSTANCES_PER_DEVICE:-4} # Instances (driving threads) per device
# QAT driving threads = number of devices x instances-per-device (override to force).
case "${IAXL_QAT_ZIP_ENABLE,,}" in
    1|true|yes|on)
        export IAXL_QAT_INSTANCE_NUM=${IAXL_QAT_INSTANCE_NUM:-$(qat_thread_count "$IAXL_QAT_DEVICES" "$IAXL_QAT_ZIP_INSTANCES_PER_DEVICE")} || return 1 2>/dev/null || exit 1
        ;;
    *) export IAXL_QAT_INSTANCE_NUM=0 ;;
esac
export IAXL_QAT_ZIP_QUEUE_DEPTH=${IAXL_QAT_ZIP_QUEUE_DEPTH:-4} # In-flight requests per instance (<= 4)

# ---- Intel IAA (compression accelerator, via Intel QPL) ---------------------
export IAXL_IAA_DEVICES=${IAXL_IAA_DEVICES:-auto}                                # Comma-separated NUMA nodes for IAA jobs, or "auto"
export IAXL_IAA_ZIP_INSTANCES_PER_DEVICE=${IAXL_IAA_ZIP_INSTANCES_PER_DEVICE:-4} # Instances (driving threads) per IAA device; a NUMA node usually owns several
case "${IAXL_IAA_ZIP_ENABLE,,}" in
    1|true|yes|on)
    	export IAXL_IAA_INSTANCE_NUM=${IAXL_IAA_INSTANCE_NUM:-4}
        ;;
    *) export IAXL_IAA_INSTANCE_NUM=0 ;;
esac
export IAXL_IAA_ZIP_QUEUE_DEPTH=${IAXL_IAA_ZIP_QUEUE_DEPTH:-4} # In-flight requests per instance (<= 4)

# ---- CPU / OpenMP compression workers --------------------------------------
export IAXL_RESERVED_CPU_NUM=${IAXL_RESERVED_CPU_NUM:-4} # CPUs reserved for inference and other tasks
case "${IAXL_CPU_ZIP_ENABLE,,}" in
    1|true|yes|on)
        export IAXL_CPU_ZIP_THREADS=${IAXL_CPU_ZIP_THREADS:-$(cpu_zip_thread_count "$MIN_RANK_CPU_COUNT" "$IAXL_QAT_INSTANCE_NUM" "$IAXL_RESERVED_CPU_NUM" "$IAXL_IAA_INSTANCE_NUM")} || return 1 2>/dev/null || exit 1
        ;;
    *) export IAXL_CPU_ZIP_THREADS=0 ;;
esac
if env_truthy "$IAXL_QAT_ZIP_ENABLE" "$IAXL_IAA_ZIP_ENABLE" "$IAXL_CPU_ZIP_ENABLE"; then
    # Defaults to one thread per zip worker; fewer threads each poll several workers (1 = caller only),
    # more threads split a worker's in-flight requests (up to one thread per request).
    if [[ -z "$IAXL_OMP_THREAD_NUM" ]]; then
        IAXL_OMP_THREAD_NUM=$(omp_thread_count "$IAXL_QAT_INSTANCE_NUM" "$IAXL_CPU_ZIP_THREADS" "$IAXL_IAA_INSTANCE_NUM") || return 1 2>/dev/null || exit 1
        # qatlib standalone mode gives the rank every instance of its QAT devices. Each thread can poll several, and in
        # kvstore_benchmark 16 instances ran fastest on 8 threads: more slowed the host<->GPU copies. Also keep within
        # the rank's CPUs.
        if [[ -n "$QAT_STANDALONE" ]]; then
            QAT_MAX_THREADS=$((MIN_RANK_CPU_COUNT - IAXL_RESERVED_CPU_NUM))
            ((QAT_MAX_THREADS <= 8)) || QAT_MAX_THREADS=8
            ((QAT_MAX_THREADS < 1 || IAXL_OMP_THREAD_NUM <= QAT_MAX_THREADS)) || IAXL_OMP_THREAD_NUM=$QAT_MAX_THREADS
        fi
    fi
    export IAXL_OMP_THREAD_NUM
else
    # No zip worker: OpenMP threads only copy raw KV blocks on the host; cap at 4.
    if [[ -z "$IAXL_OMP_THREAD_NUM" ]]; then
        IAXL_OMP_THREAD_NUM=$(cpu_zip_thread_count "$MIN_RANK_CPU_COUNT" 0 "$IAXL_RESERVED_CPU_NUM" 0) || return 1 2>/dev/null || exit 1
        ((IAXL_OMP_THREAD_NUM > 4)) && IAXL_OMP_THREAD_NUM=4
    fi
    export IAXL_OMP_THREAD_NUM
fi
if [[ "$DEVICE" == cpu ]]; then
    export IAXL_USE_OMP=${IAXL_USE_OMP:-0} # vLLM CPU inference owns the OpenMP runtime
fi
export IAXL_USE_OMP=${IAXL_USE_OMP:-1} # parallel.h runs on OpenMP (1) or iaxl's own thread pool (0), which stays off the OpenMP runtime vLLM and torch share
if env_truthy "$IAXL_USE_OMP"; then
    export OMP_NUM_THREADS=$IAXL_OMP_THREAD_NUM
    export OMP_THREAD_LIMIT=$IAXL_OMP_THREAD_NUM
else
    unset OMP_NUM_THREADS OMP_THREAD_LIMIT # Leave OpenMP to vLLM (CPU inference pins one thread per core)
fi
export OMP_MAX_ACTIVE_LEVELS=2
validate_omp_config "$MIN_RANK_CPU_COUNT" || return 1 2>/dev/null || exit 1
# Prepended to LD_PRELOAD by the example and benchmark scripts. The vLLM CPU image ships libiomp5 in
# its venv instead: IAXL_LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so
export IAXL_LD_PRELOAD=${IAXL_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so}
export TCMALLOC_RELEASE_RATE=${TCMALLOC_RELEASE_RATE:-0} # Keep freed memory in tcmalloc instead of returning it to the OS

# ---- Intel DSA (host<->device copy accelerator, CUDA only) ------------------
export IAXL_DSA_GD_RESET_ON_DESTROY=${IAXL_DSA_GD_RESET_ON_DESTROY:-0} # Large BAR with stable tensor VAs can keep this off for better performance (0/1)
export IAXL_DSA_WQS=${IAXL_DSA_WQS:-wq0.0}                             # Comma/space separated DSA work-queue names

# ---- Debug / profiling ------------------------------------------------------
export PYTHONOPTIMIZE=0                                 # Keep Python assert statements enabled
export IAXL_DEBUG=${IAXL_DEBUG:-0}                      # Python log level (0=INFO, 1=DEBUG)
export IAXL_DEBUG_LOG=${IAXL_DEBUG_LOG:-0}              # Native kv_pool verbose logging (0/1)
export IAXL_PROFILE_MODE=${IAXL_PROFILE_MODE:-disabled} # Profiling: disabled | nvtx | full

# ---- Management REST API ----------------------------------------------------
export IAXL_API_WORKER_BASE_PORT=${IAXL_API_WORKER_BASE_PORT:-18800} # Worker server port base (+ rank)
export IAXL_API_CONTROLLER_PORT=${IAXL_API_CONTROLLER_PORT:-18700}   # Controller server port
export IAXL_API_TIMEOUT=${IAXL_API_TIMEOUT:-60}                      # HTTP request timeout in seconds

# ---- Remote pool (KVStore daemon over RDMA/NIXL) ----------------------------
export IAXL_RDMA_ENABLE=${IAXL_RDMA_ENABLE:-0}                # Use remote KVStore daemon instead of local KVStore (0/1)
export IAXL_RDMA_DAEMON_NIC_IPS=${IAXL_RDMA_DAEMON_NIC_IPS:-} # Daemon RDMA NIC IPs per rank (a single IP is allowed); first is also the control-plane IP (required)
export IAXL_RDMA_CLIENT_NIC_IPS=${IAXL_RDMA_CLIENT_NIC_IPS:-} # Client RDMA NIC IPs per rank (a single IP is allowed); first is also the scheduler IP (required on client)
export IAXL_RDMA_DAEMON_PORT=${IAXL_RDMA_DAEMON_PORT:-5555}   # Scheduler port; rank r listens on port+1+r
export IAXL_RDMA_TP_SIZE=${IAXL_RDMA_TP_SIZE:-$TP_SIZE}       # Daemon rank process count (must equal client TP)

HOST_IP=$(ip route get 1 | awk '{print $7}' | tr -d '\n')
export no_proxy=localhost,127.0.0.1,localaddress,.localdomain.com,.local,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,${HOST_IP}
if env_truthy "$IAXL_RDMA_ENABLE" && [[ -n "$IAXL_RDMA_DAEMON_NIC_IPS" ]]; then
    export no_proxy="$no_proxy,$IAXL_RDMA_DAEMON_NIC_IPS"
fi
export http_proxy="${http_proxy:-}"
export https_proxy=$http_proxy

# docker
CONTAINER_NAME=iaxl.vllm
if [[ -n "$QAT_STANDALONE_VFS" ]]; then
    CONTAINER_NAME+=$([[ -n "$QAT_GPU" ]] && echo ".gpu$QAT_GPU" || echo ".numa$QAT_NUMA")
fi
ENV_VARS=(
    no_proxy
    http_proxy
    https_proxy
    HOST_IP
    MODEL
    TP_SIZE
    DEVICE
    IAXL_QATLIB
    IAXL_QATLIB_VERSION
    PORT
    IAXL_VERSION
    IAXL_LD_PRELOAD
)

case "$DEVICE" in
    cuda)
        if [[ "$NVIDIA_RUNTIME" == "none" ]]; then
            DOCKER_RUN_ARGS=()
        else
            DOCKER_RUN_ARGS=("--runtime" "nvidia" "--gpus" "${QAT_GPU:+device=}${QAT_GPU:-all}")
        fi
        ;;
    xpu)
        DOCKER_RUN_ARGS=("--device" "/dev/dri")
        [[ -n "$QAT_GPU" ]] && DOCKER_RUN_ARGS+=("-e" "ZE_AFFINITY_MASK=$QAT_GPU")
        ;;
    cpu)
        DOCKER_RUN_ARGS=()
        ;;
    *)
        echo "Unsupported DEVICE: $DEVICE (expected cuda, xpu or cpu)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
for var in "${ENV_VARS[@]}"; do
    DOCKER_RUN_ARGS+=("-e" "$var=${!var:-}")
done
DOCKER_RUN_ARGS+=("-e" "HF_HOME=/_data/hf_home")
DOCKER_RUN_ARGS+=("-v" "$PWD/_data:/_data")
if [[ -n "$QAT_STANDALONE_VFS" ]]; then
    # No --privileged: qatlib opens only these VFs; vLLM's CPU backend needs SYS_NICE + seccomp=unconfined (migrate_pages).
    # The CPU binding picks the container's NUMA node, so it and the ports are fixed at start.
    DOCKER_RUN_ARGS+=("--device" "/dev/vfio/vfio" "--ulimit" "memlock=-1:-1" "--shm-size" "4g"
        "--cap-add" "SYS_NICE" "--security-opt" "seccomp=unconfined"
        "-e" "QAT_POLICY=$(wc -l <<<"$QAT_STANDALONE_VFS")"
        "-e" "VLLM_CPU_OMP_THREADS_BIND=$VLLM_CPU_OMP_THREADS_BIND"
        "-e" "KVSHRINK_CPU_OMP_THREADS_BIND=$KVSHRINK_CPU_OMP_THREADS_BIND"
        "-e" "IAXL_API_CONTROLLER_PORT=$IAXL_API_CONTROLLER_PORT"
        "-e" "IAXL_API_WORKER_BASE_PORT=$IAXL_API_WORKER_BASE_PORT"
        "-e" "IAXL_DDR_POOL_TOTAL_SIZE_GB=$IAXL_DDR_POOL_TOTAL_SIZE_GB")
    mapfile -t -O "${#DOCKER_RUN_ARGS[@]}" DOCKER_RUN_ARGS < <(awk '{print "--device"; print "/dev/vfio/" $2}' <<<"$QAT_STANDALONE_VFS")
    if [[ -n "${KVSHRINK_DSA_DEVICES:-}" && -d /dev/dsa ]]; then
        DOCKER_RUN_ARGS+=("--device" "/dev/dsa" "-e" "KVSHRINK_DSA_DEVICES=$KVSHRINK_DSA_DEVICES" "-e" "IAXL_DSA_GD_ENABLE=$IAXL_DSA_GD_ENABLE")
        [[ -e /dev/gdrdrv ]] && DOCKER_RUN_ARGS+=("--device" "/dev/gdrdrv")
    fi
    env_truthy "$IAXL_RDMA_ENABLE" && [[ -d /dev/infiniband ]] && DOCKER_RUN_ARGS+=("--device" "/dev/infiniband" "-e" "IAXL_RDMA_ENABLE=$IAXL_RDMA_ENABLE")
else
    DOCKER_RUN_ARGS+=("--privileged" "--pid" "host" "-v" "/dev:/dev")
    # qatlib (in-tree driver) shares the host's QAT VFs through the host qatmgr socket.
    if [[ -d /run/qat ]]; then
        DOCKER_RUN_ARGS+=("-v" "/run/qat:/run/qat")
    fi
fi
if [[ -d "$MODEL" ]]; then
    MODEL_DIR=$(realpath "$MODEL")
    DOCKER_RUN_ARGS+=("-v" "$MODEL_DIR:$MODEL_DIR:ro")
fi
DOCKER_RUN_ARGS+=("-v" "$PWD:$PWD" "-w" "$PWD")
