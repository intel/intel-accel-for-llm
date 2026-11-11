#!/bin/bash -e
# Start vLLM with the KVShrink connector. MODEL, TP_SIZE and DP_SIZE come from setvars.sh; the
# environment or a recipe's setvars.pre.sh can set the other options:
#   PORT (8000), MAX_MODEL_LEN (32765), BLOCK_SIZE (unset: vLLM's default)
#   VLLM_SERVE_EXTRA_ARGS    more `vllm serve` options, appended to the defaults
#   VLLM_SERVE_CMDLINE_ARGS  escape hatch: replaces the default options (everything except the model,
#                            parallelism, --block-size, the connector and VLLM_SERVE_EXTRA_ARGS)
# Both *_ARGS are split on whitespace, so no value may contain a space.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

parallel_args=(-tp "$TP_SIZE")
if [[ "${DP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--data-parallel-size "$DP_SIZE")
fi

# Unset BLOCK_SIZE keeps vLLM's default: 128 on CPU (16 for MLA models), 16 or the attention backend's preference on GPU.
block_args=()
[[ -n "$BLOCK_SIZE" ]] && block_args=(--block-size "$BLOCK_SIZE")

serve_args=(--trust-remote-code --max-model-len "${MAX_MODEL_LEN:-32765}" --port "${PORT:-8000}" --no-enable-prefix-caching)
# vLLM CPU inference sizes its KV cache from VLLM_CPU_KVCACHE_SPACE (GiB) instead.
[[ "$DEVICE" == cpu ]] || serve_args+=(--gpu-memory-utilization 0.8)
if [[ -n "${VLLM_SERVE_CMDLINE_ARGS:-}" ]]; then
    read -r -a serve_args <<<"$VLLM_SERVE_CMDLINE_ARGS"
fi
read -r -a extra_args <<<"${VLLM_SERVE_EXTRA_ARGS:-}"

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    "${parallel_args[@]}" \
    "${block_args[@]}" \
    "${serve_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee log.kvshrink-vllm
