#!/bin/bash -e
# DeepSeek-V4-Flash (DSv4) KVShrink serve script (run INSIDE the iaxl container).
#
# DSv4 is a hybrid MLA model. Differences vs examples/kvshrink-vllm-serve.sh:
#   --kv-cache-dtype fp8                    DSv4 only supports an fp8 KV layout
#                                           (vllm/models/deepseek_v4/attention.py
#                                            asserts kv_cache_dtype startswith fp8).
#   --enforce-eager                         DSv4 attention runs a custom eager op;
#                                           the per-layer KV-transfer hook needs it.
#   --no-disable-hybrid-kv-cache-manager    keep HMA on -- KVShrinkConnector is a
#                                           SupportsHMA connector, so SWA blocks are
#                                           freed and per-group block_ids are passed.
#   (no --block-size)                       DSv4 declares its own per-group block
#                                           sizes; the connector hashes at the max
#                                           group block size (256).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Disable the Intel accelerator compression / DSA transfer paths for this
# bring-up (H20 has no QAT/IAA/DSA hardware). These are set BEFORE sourcing
# setvars.sh so its ${VAR:-default} keeps them and skips QAT/DSA auto-detection.
# The KV cache is stored raw (no DEFLATE). Runs inside the container, so this
# guarantees the switches take effect regardless of how the container is started.
export IAXL_KV_COMPRESSION="${IAXL_KV_COMPRESSION:-0}"
export IAXL_QAT_ZIP_ENABLE="${IAXL_QAT_ZIP_ENABLE:-0}"
export IAXL_IAA_ZIP_ENABLE="${IAXL_IAA_ZIP_ENABLE:-0}"
export IAXL_DSA_GD_ENABLE="${IAXL_DSA_GD_ENABLE:-0}"

source "$SCRIPT_DIR/../setvars.sh"
: "${MODEL:?Set MODEL in setvars.sh or the environment}"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

vllm serve "$MODEL" \
    --kv-transfer-config '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}' \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
    -tp "$TP_SIZE" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --port "${PORT:-8000}" \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    2>&1 | tee log.kvshrink-vllm-dsv4
