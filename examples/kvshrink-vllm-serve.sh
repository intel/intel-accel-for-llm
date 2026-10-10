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

# Hybrid GDN/Mamba models need prefix caching: with it off vLLM sets
# mamba_cache_mode to 'none' and no block-aligned state is kept. Their
# pages are enlarged to fit the mamba state, beyond the default zip cap.
cache_args=(--no-enable-prefix-caching)
if [[ "${KVSHRINK_HYBRID:-0}" == "1" ]]; then
    cache_args=(--enable-prefix-caching)
    export IAXL_ZIP_SRC_CAP=2097152 IAXL_ZIP_DST_CAP=2097152
fi

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    --trust-remote-code \
    "${parallel_args[@]}" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32765 \
    --block-size "${BLOCK_SIZE:-16}" \
    --port "${PORT:-8000}" \
    "${cache_args[@]}" \
    2>&1 | tee log.kvshrink-vllm
