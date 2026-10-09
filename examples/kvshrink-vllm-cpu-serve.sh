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
# bf16 byte-plane transform. Off (default): 19% of KV bytes saved.
# On: 28% saved, at the cost of a CPU byte-plane pass over every block.
export IAXL_KV_DATA_SHUFFLE="${IAXL_KV_DATA_SHUFFLE:-0}"
export IAXL_KV_LOSSY_TRUNC="${IAXL_KV_LOSSY_TRUNC:-0}"
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS="${IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS:-0}"

# Each QAT/IAA instance runs its own codec thread, so one instance per dedicated IAXL core.
export IAXL_QAT_INSTANCE_NUM="${IAXL_QAT_INSTANCE_NUM:-$IAXL_CORES}"
export IAXL_IAA_INSTANCE_NUM="${IAXL_IAA_INSTANCE_NUM:-1}"

# --- Cache ----------------------------------------------------------------------------------
export IAXL_DDR_POOL_SIZE_GB="${IAXL_DDR_POOL_SIZE_GB:-4}"
export PYTHONOPTIMIZE=0

# Async layer loading was not part of the CPU benchmark campaign; leave it off until measured.
export KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED="${KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED:-0}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS:--1}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC:-0}"
export KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP="${KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP:-0-:0}"

echo "[launch] inference: OMP_NUM_THREADS=$OMP_NUM_THREADS bind=$VLLM_CPU_OMP_THREADS_BIND | iaxl: affinity=$IAXL_CPU_AFFINITY qat_devices=${IAXL_QAT_DEVICES:-0} shuffle=$IAXL_KV_DATA_SHUFFLE" >&2

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