#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"
# vLLM's V2 model runner corrupts requests recomputed after a KV load failure
# (kv_load_failure_policy=recompute); use the V1 runner unless overridden.
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-0}

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}' \
    --trust-remote-code \
    "${parallel_args[@]}" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32765 \
    --block-size "${BLOCK_SIZE:-16}" \
    --port "${PORT:-8000}" \
    --no-enable-prefix-caching \
    2>&1 | tee log.kvshrink-vllm
