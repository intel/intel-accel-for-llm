export IAXL_BASE_DOCKER_IMAGE=${IAXL_BASE_DOCKER_IMAGE:-"vllm/vllm-openai:v0.23.0"}
export IAXL_DEV_DOCKER_IMAGE=${IAXL_DEV_DOCKER_IMAGE:-"vllm-iaxl-dev"}
export IAXL_BUILDER_DOCKER_IMAGE=${IAXL_BUILDER_DOCKER_IMAGE:-"vllm-iaxl-builder"}

TOP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TOP_DIR/tools/auto_config.sh"

# =============================================================================
# Configurable runtime / build environment variables (all IAXL_* prefixed).
# Each keeps its default unless already set in the environment; edit as needed.
# =============================================================================

# ---- Build ------------------------------------------------------------------
export DEVICE=${DEVICE:-cuda}                 # Build backend: cuda | xpu
export IAXL_CMAKE_ARGS=${IAXL_CMAKE_ARGS:-""} # Extra cmake flags, e.g. "-DENABLE_NVTX=OFF"

# ---- Feature switches -------------------------------------------------------
export IAXL_KV_COMPRESSION=${IAXL_KV_COMPRESSION:-1} # Enable DEFLATE compression (0/1)
export IAXL_QAT_ZIP_ENABLE=${IAXL_QAT_ZIP_ENABLE:-1} # Enable QAT compression workers (0/1)
export IAXL_CPU_ZIP_ENABLE=${IAXL_CPU_ZIP_ENABLE:-1} # Enable CPU compression workers (0/1)
export IAXL_DSA_GD_ENABLE=${IAXL_DSA_GD_ENABLE:-0}   # Use Intel DSA + GDRCopy transfers (0/1)

# ---- Async KV load ----------------------------------------------------------
export KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=${KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED:-1}      # Enable asynchronous KV loading (0/1)
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS:--1}       # -1=wait all layers, N=start prefill after first N layers (used when DYNAMIC=0)
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC:-1} # 0=fixed LAYERS, 1=select layers from DYNAMIC_MAP per request concurrency
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP:-0-3:0,4-6:4,7-:8}" # Contiguous START-END:LAYERS rules from 0; 0 layers means sync and the final range is open-ended

# ---- vLLM ------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-32B}" # Hugging Face model ID or local model path
export TP_SIZE="${TP_SIZE:-2}"            # Tensor-parallel worker count
VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-$(cpu_auto_detect "$TP_SIZE")}" || return 1 2>/dev/null || exit 1
export VLLM_CPU_OMP_THREADS_BIND # Per-rank CPU affinity
rank_cpu_counts "$VLLM_CPU_OMP_THREADS_BIND" "$TP_SIZE" || return 1 2>/dev/null || exit 1
case "${IAXL_QAT_ZIP_ENABLE,,}" in
    1|true|yes|on)
        KVSHRINK_QAT_DEVICES="${KVSHRINK_QAT_DEVICES:-$(qat_auto_detect "$TP_SIZE")}" || return 1 2>/dev/null || exit 1
        export KVSHRINK_QAT_DEVICES # Per-rank QAT device indices
        ;;
    *)
        unset KVSHRINK_QAT_DEVICES
        ;;
esac
case "${IAXL_DSA_GD_ENABLE,,}" in
    1|true|yes|on)
        KVSHRINK_DSA_DEVICES="${KVSHRINK_DSA_DEVICES:-$(dsa_auto_detect "$TP_SIZE")}" || return 1 2>/dev/null || exit 1
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
    "  IAXL_KV_COMPRESSION=$IAXL_KV_COMPRESSION" \
    "  IAXL_QAT_ZIP_ENABLE=$IAXL_QAT_ZIP_ENABLE" \
    "  IAXL_CPU_ZIP_ENABLE=$IAXL_CPU_ZIP_ENABLE" \
    "  IAXL_DSA_GD_ENABLE=$IAXL_DSA_GD_ENABLE" \
    "  VLLM_CPU_OMP_THREADS_BIND=$VLLM_CPU_OMP_THREADS_BIND" \
    "  KVSHRINK_QAT_DEVICES=${KVSHRINK_QAT_DEVICES:-disabled}" \
    "  KVSHRINK_DSA_DEVICES=${KVSHRINK_DSA_DEVICES:-disabled}" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=$KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=$KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC=$KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC" \
    "  KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=$KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP"

# ---- Cache / compression ----------------------------------------------------
export IAXL_KV_LOSSY_TRUNC=${IAXL_KV_LOSSY_TRUNC:-0}                     # Lossy LSB truncation: 'auto', 0 (off), or N bits
export IAXL_KV_DATA_SHUFFLE=${IAXL_KV_DATA_SHUFFLE:-0}                   # Byte-shuffle before compression (0/1)
export IAXL_ZIP_SRC_CAP=${IAXL_ZIP_SRC_CAP:-262144}                      # Source/decompressed block capacity (256 KiB)
export IAXL_ZIP_DST_CAP=${IAXL_ZIP_DST_CAP:-262144}                      # Compressed-output capacity (256 KiB)
export IAXL_CACHE_DIR=${IAXL_CACHE_DIR:-_data/kvcache}                   # Base directory for persisted cache files
export IAXL_CACHE_STREAM_SYNC_ON_GET=${IAXL_CACHE_STREAM_SYNC_ON_GET:-0} # CPU-sync the GPU stream on get() (0/1)
export IAXL_CACHE_CACHEGROUP_SIZE=${IAXL_CACHE_CACHEGROUP_SIZE:-100}     # Reserved entries per cache group
export IAXL_CACHE_CACHEGROUP_NUM=${IAXL_CACHE_CACHEGROUP_NUM:-100000}    # Reserved number of cache groups
export IAXL_PREALLOC_LIMIT=${IAXL_PREALLOC_LIMIT:-0}                     # Cap pinned scratch-pool pre-allocation (0 = unlimited)
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=${IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS:-1} # Store the first N KV layers without compression
# export IAXL_DDR_POOL_SIZE_GB=...         # DDR (host) cache pool size in GB (unset = 1/10 of RAM)

# ---- Hybrid (GDN/Mamba) models ----------------------------------------------
# Used only when the model has GDN/Mamba layers (e.g. Qwen3.5); pure
# attention models ignore every knob below.
export KVSHRINK_SAVE=${KVSHRINK_SAVE:-1}                     # Persist newly computed KV/state (0/1)
export KVSHRINK_SAVE_PIPELINED=${KVSHRINK_SAVE_PIPELINED:-1} # Submit attention saves per layer during forward (0 = all after forward)
# export KVSHRINK_PERSIST_DIR=...          # Commit-manifest root (unset = $IAXL_CACHE_DIR/kv4-manifests)
export KVSHRINK_METRICS_PORT=${KVSHRINK_METRICS_PORT:-18801} # Metrics exporter base port, +rank per worker (0 = disabled)
# export KVSHRINK_DEBUG_LOG=1              # Verbose per-request plan logging
# export KVSHRINK_DEBUG_DUMP=1             # Log GDN state page digests after each forward
# Cross-restart cache reuse requires reproducible block hashes: vLLM
# seeds its first-block hash from PYTHONHASHSEED and falls back to a
# random value when unset, which makes every restart miss.
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# ---- Intel QAT (compression accelerator) ------------------------------------
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

# ---- CPU / OpenMP compression workers --------------------------------------
export IAXL_RESERVED_CPU_NUM=${IAXL_RESERVED_CPU_NUM:-4} # CPUs reserved for inference and other tasks
case "${IAXL_CPU_ZIP_ENABLE,,}" in
    1|true|yes|on)
        export IAXL_CPU_ZIP_THREADS=${IAXL_CPU_ZIP_THREADS:-$(cpu_zip_thread_count "$MIN_RANK_CPU_COUNT" "$IAXL_QAT_INSTANCE_NUM" "$IAXL_RESERVED_CPU_NUM")} || return 1 2>/dev/null || exit 1
        ;;
    *) export IAXL_CPU_ZIP_THREADS=0 ;;
esac
export IAXL_OMP_THREAD_NUM=$(omp_thread_count "$IAXL_QAT_INSTANCE_NUM" "$IAXL_CPU_ZIP_THREADS") || return 1 2>/dev/null || exit 1
export OMP_NUM_THREADS=$IAXL_OMP_THREAD_NUM
export OMP_THREAD_LIMIT=$IAXL_OMP_THREAD_NUM
export OMP_MAX_ACTIVE_LEVELS=2
validate_omp_config "$MIN_RANK_CPU_COUNT" || return 1 2>/dev/null || exit 1

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

# Only used to widen no_proxy. Containers often ship without iproute2,
# and that must not stop the engine from starting.
HOST_IP=$(ip route get 1 2>/dev/null | awk '{print $7}' | tr -d '\n') || HOST_IP=""
export no_proxy=localhost,127.0.0.1,localaddress,.localdomain.com,.local,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12${HOST_IP:+,$HOST_IP}
export http_proxy="${http_proxy:-}"
export https_proxy=$http_proxy

# docker
CONTAINER_NAME=iaxl.vllm
ENV_VARS=(
    no_proxy
    http_proxy
    https_proxy
    HOST_IP
    MODEL
    TP_SIZE
    DEVICE
)

case "$DEVICE" in
    cuda)
        DOCKER_RUN_ARGS=("--runtime" "nvidia" "--gpus" "all")
        ;;
    xpu)
        DOCKER_RUN_ARGS=("--device" "/dev/dri")
        ;;
    *)
        echo "Unsupported DEVICE: $DEVICE (expected cuda or xpu)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
for var in "${ENV_VARS[@]}"; do
    DOCKER_RUN_ARGS+=("-e" "$var=${!var:-}")
done
DOCKER_RUN_ARGS+=("-e" "HF_HOME=/_data/hf_home")
DOCKER_RUN_ARGS+=("-v" "$PWD/_data:/_data")
if [[ -d "$MODEL" ]]; then
    MODEL_DIR=$(realpath "$MODEL")
    DOCKER_RUN_ARGS+=("-v" "$MODEL_DIR:$MODEL_DIR:ro")
fi
DOCKER_RUN_ARGS+=("-v" "$PWD:$PWD" "-w" "$PWD")
