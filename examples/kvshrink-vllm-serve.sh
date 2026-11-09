#!/bin/bash -e
# Start vLLM with the KVShrink connector. MODEL, TP_SIZE and DP_SIZE come from setvars.sh and
# BLOCK_SIZE is optional. Every other `vllm serve` option is in VLLM_SERVE_CMDLINE_ARGS; set it
# to replace the default list, e.g. without --gpu-memory-utilization for CPU inference
# (VLLM_CPU_KVCACHE_SPACE sizes the KV cache there).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"
VLLM_SERVE_CMDLINE_ARGS=${VLLM_SERVE_CMDLINE_ARGS:-"--trust-remote-code --max-model-len 32765 --port 8000 --no-enable-prefix-caching --gpu-memory-utilization 0.8"}

export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

# Unset BLOCK_SIZE keeps vLLM's default: 128 on CPU (16 for MLA models), 16 or the attention backend's preference on GPU.
block_args=()
[[ -n "$BLOCK_SIZE" ]] && block_args=(--block-size "$BLOCK_SIZE")

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    "${parallel_args[@]}" \
    "${block_args[@]}" \
    $VLLM_SERVE_CMDLINE_ARGS \
    2>&1 | tee log.kvshrink-vllm
