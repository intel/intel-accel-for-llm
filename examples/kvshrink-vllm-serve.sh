#!/bin/bash -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

# Attention-only models keep vLLM's own prefix cache out of the way:
# block hashes are still computed whenever a connector is registered, so
# the connector sees every request.
#
# GDN/Mamba hybrid models (e.g. Qwen3.5) cannot do that. With prefix
# caching off vLLM rewrites the mamba cache mode to 'none', gives each
# request a single max_model_len block and disables block-aligned chunk
# splitting, leaving GDN state with no addressable boundary. Both flags
# must therefore be explicit -- set KVSHRINK_HYBRID=1 for such models.
if [[ "${KVSHRINK_HYBRID:-0}" == "1" ]]; then
    CACHE_ARGS=(--enable-prefix-caching
                --mamba-cache-mode align
                --no-disable-hybrid-kv-cache-manager)
else
    CACHE_ARGS=(--no-enable-prefix-caching)
fi

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    -tp "$TP_SIZE" \
    --max-model-len 32765 \
    --block-size "${BLOCK_SIZE:-16}" \
    --port "${PORT:-8000}" \
    "${CACHE_ARGS[@]}" \
    2>&1 | tee log.kvshrink-vllm
