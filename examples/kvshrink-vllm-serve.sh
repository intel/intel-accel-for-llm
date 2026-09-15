#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi
case "${ENABLE_EXPERT_PARALLEL,,}" in
    1 | true | yes | on) parallel_args+=(--enable-expert-parallel) ;;
esac

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
    "${parallel_args[@]}" \
    --max-model-len "${MAX_MODEL_LEN:-32765}" \
    --block-size "${BLOCK_SIZE:-16}" \
    --port "${PORT:-8000}" \
    --no-enable-prefix-caching \
    2>&1 | tee log.kvshrink-vllm
