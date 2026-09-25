#!/usr/bin/env bash
set -euo pipefail

export DEVICE=cpu
export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export TP_SIZE="${TP_SIZE:-1}"
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"

# --- Core placement -------------------------------------------------------------------------
# Measured: the codec thread must not share a core with inference (a floating poller loses
# 13-21% restore throughput; two pollers on inference cores halve throughput and triple TPOT).
# Default: inference on every visible CPU but the last IAXL_CORES; IAXL pinned to those.
# Two cores: restore throughput scales with pollers (standalone, 4 QAT devices, DSA on).
IAXL_CORES="${IAXL_CORES:-2}"
if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" || -z "${IAXL_CPU_AFFINITY:-}" ]]; then
    mapfile -t CPUS < <(python3 -c 'import os; print(*sorted(os.sched_getaffinity(0)), sep="\n")')
    if (( ${#CPUS[@]} < IAXL_CORES + TP_SIZE )); then
        echo "need at least $((IAXL_CORES + TP_SIZE)) CPUs, have ${#CPUS[@]}" >&2; exit 1
    fi
    PER_RANK=$(( (${#CPUS[@]} - IAXL_CORES) / TP_SIZE ))
    INF_COUNT=$(( PER_RANK * TP_SIZE ))
    BIND=""
    for (( r = 0; r < TP_SIZE; r++ )); do
        lo=${CPUS[$(( r * PER_RANK ))]}; hi=${CPUS[$(( (r + 1) * PER_RANK - 1 ))]}
        BIND+="${BIND:+|}${lo}-${hi}"
    done
    AFF=${CPUS[$INF_COUNT]}
    (( ${#CPUS[@]} - INF_COUNT > 1 )) && AFF+="-${CPUS[$(( ${#CPUS[@]} - 1 ))]}"
    export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-$BIND}"
    export IAXL_CPU_AFFINITY="${IAXL_CPU_AFFINITY:-$AFF}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PER_RANK}"
fi
# OMP_NUM_THREADS must match the per-rank bind width when both are given explicitly.
: "${OMP_NUM_THREADS:?set OMP_NUM_THREADS to the CPUs per rank in VLLM_CPU_OMP_THREADS_BIND}"

# --- Codec ----------------------------------------------------------------------------------
export IAXL_KV_COMPRESSION="${IAXL_KV_COMPRESSION:-1}"
export IAXL_QAT_ZIP_ENABLE="${IAXL_QAT_ZIP_ENABLE:-1}"
export IAXL_CPU_ZIP_ENABLE="${IAXL_CPU_ZIP_ENABLE:-0}"
export IAXL_IAA_ZIP_ENABLE="${IAXL_IAA_ZIP_ENABLE:-0}"
# bf16 byte-plane transform. Off (default): restore matches raw+DSA, 19% of KV bytes saved.
# On: 28% saved, but the CPU unshuffle costs up to ~17% throughput on long fully-cached prompts.
export IAXL_KV_DATA_SHUFFLE="${IAXL_KV_DATA_SHUFFLE:-0}"
export IAXL_KV_LOSSY_TRUNC="${IAXL_KV_LOSSY_TRUNC:-0}"
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS="${IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS:-0}"

# Spread instances over every QAT device: each decompresses ~6 GB/s, so restore scales with devices.
QAT_DEVICE_COUNT=0
for dev in /sys/bus/pci/drivers/{4xxx,420xx}/0000:*; do
    [[ -e "$dev" ]] && QAT_DEVICE_COUNT=$(( QAT_DEVICE_COUNT + 1 ))
done
(( QAT_DEVICE_COUNT > 0 )) || QAT_DEVICE_COUNT=1
export IAXL_QAT_ZIP_INSTANCES_PER_DEVICE="${IAXL_QAT_ZIP_INSTANCES_PER_DEVICE:-4}"
export IAXL_QAT_INSTANCE_NUM="${IAXL_QAT_INSTANCE_NUM:-$(( QAT_DEVICE_COUNT * IAXL_QAT_ZIP_INSTANCES_PER_DEVICE ))}"
# qat_zip takes at most INSTANCES_PER_DEVICE from each listed device.
export IAXL_QAT_DEVICES="${IAXL_QAT_DEVICES:-$(seq -s, 0 $(( (IAXL_QAT_INSTANCE_NUM + IAXL_QAT_ZIP_INSTANCES_PER_DEVICE - 1) / IAXL_QAT_ZIP_INSTANCES_PER_DEVICE - 1 )))}"
# One poller per dedicated IAXL core; poller p drives instances p, p+P, ... (one per device).
export IAXL_QAT_POLL_THREADS="${IAXL_QAT_POLL_THREADS:-$IAXL_CORES}"
export IAXL_IAA_INSTANCE_NUM="${IAXL_IAA_INSTANCE_NUM:-4}"
export IAXL_IAA_POLL_THREADS="${IAXL_IAA_POLL_THREADS:-1}"

# --- DSA ------------------------------------------------------------------------------------
# Host-to-host DSA for bulk block copies (raw path: +18-62% restore throughput) and for the QAT
# per-block staging/scatter copies, which run asynchronously next to the codec. Raw batches under
# IAXL_DSA_MEMCPY_MIN_BYTES stay on memcpy.
export IAXL_DSA_GD_ENABLE=0
export IAXL_DSA_WQS="${IAXL_DSA_WQS:-wq0.0}"
if [[ -z "${IAXL_DSA_MEMCPY_ENABLE:-}" ]]; then
    IAXL_DSA_MEMCPY_ENABLE=0
    [[ -e "/dev/dsa/${IAXL_DSA_WQS%%,*}" ]] && IAXL_DSA_MEMCPY_ENABLE=1
fi
export IAXL_DSA_MEMCPY_ENABLE
export IAXL_DSA_MEMCPY_MIN_BYTES="${IAXL_DSA_MEMCPY_MIN_BYTES:-1048576}"

# --- Cache ----------------------------------------------------------------------------------
export IAXL_DDR_POOL_SIZE_GB="${IAXL_DDR_POOL_SIZE_GB:-4}"
export PYTHONOPTIMIZE=0

# Async layer loading was not part of the CPU benchmark campaign; leave it off until measured.
export KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED="${KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED:-0}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS:--1}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC:-0}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP:-0-:0}"

echo "[launch] inference: OMP_NUM_THREADS=$OMP_NUM_THREADS bind=$VLLM_CPU_OMP_THREADS_BIND | iaxl: affinity=$IAXL_CPU_AFFINITY qat_devices=$IAXL_QAT_DEVICES dsa=$IAXL_DSA_MEMCPY_ENABLE shuffle=$IAXL_KV_DATA_SHUFFLE" >&2

python - <<'PY'
from iaxl import torch_ext
from vllm.platforms import current_platform

if torch_ext.device_type != "cpu":
    raise SystemExit("Build IAXL with DEVICE=cpu before starting CPU inference.")
if not current_platform.is_cpu():
    raise SystemExit("Use a CPU-enabled vLLM and PyTorch installation.")
PY

# KVSHRINK_CONNECTOR=0 serves plain vLLM with its own prefix cache (the baseline arm).
if [[ "${KVSHRINK_CONNECTOR:-1}" == "1" ]]; then
    KV_ARGS=(--kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}'
             --no-enable-prefix-caching)
else
    KV_ARGS=(--enable-prefix-caching)
fi

exec vllm serve "$MODEL" \
    "${KV_ARGS[@]}" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "${DTYPE:-bfloat16}" \
    --enforce-eager \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --block-size "${BLOCK_SIZE:-32}" \
    --port "${PORT:-8000}" \
    "$@"