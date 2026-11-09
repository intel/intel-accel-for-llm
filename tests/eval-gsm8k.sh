#!/bin/bash -e
#
# Reproduce the customer's gsm8k accuracy test against a running vLLM server.
#
# Usage:
#   bash eval-gsm8k.sh <run_name> [limit] [parallel]
#
# Example (matches the customer's report):
#   bash eval-gsm8k.sh run1 200 50
#
# Prerequisites:
#   - vLLM is already serving the model at $API_URL with name $MODEL.
#   - evalscope is installed (see install.sh in this directory).
#   - The gsm8k dataset is available locally or will be downloaded by evalscope
#     from ModelScope.
#

if [[ -z "${1:-}" ]]; then
    echo "Usage: bash $0 <run_name>" >&2
    echo "  LIMIT=200 PARALLEL=50 bash $0 <run_name>" >&2
    exit 1
fi
RUN_NAME=$1
LIMIT=${LIMIT:-200}
PARALLEL=${PARALLEL:-50}

# --- vLLM endpoint configuration ---------------------------------------------
# Override these via environment variables if your vLLM instance differs.
# NOTE: openai_api eval_type expects the OpenAI base URL (ending in /v1),
# NOT the full /v1/chat/completions path.
export API_URL=${API_URL:-http://127.0.0.1:8000/v1}
export MODEL=${MODEL:-qwen3-32b-fp8}
export API_KEY=${API_KEY:-EMPTY}

# --- Output ------------------------------------------------------------------
WORK_DIR=${WORK_DIR:-/_data/evalscope}
OUTPUT_DIR=${OUTPUT_DIR:-${WORK_DIR}/outputs/${RUN_NAME}}
DATASET_DIR=${DATASET_DIR:-${WORK_DIR}/datasets}

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "Removing existing output directory: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}" "${DATASET_DIR}"

echo "======================================== "
echo "Run name      : ${RUN_NAME}"
echo "Model name    : ${MODEL}"
echo "API URL       : ${API_URL}"
echo "Cases (limit) : ${LIMIT}"
echo "Parallel      : ${PARALLEL}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "======================================== "

evalscope eval \
    --model "${MODEL}" \
    --api-url "${API_URL}" \
    --api-key "${API_KEY}" \
    --eval-type openai_api \
    --datasets gsm8k \
    --dataset-dir "${DATASET_DIR}" \
    --limit "${LIMIT}" \
    --eval-batch-size "${PARALLEL}" \
    --work-dir "${OUTPUT_DIR}" \
    --generation-config '{"max_tokens": 2048, "temperature": 0.0}'

echo "======================================== "
echo "Done: ${RUN_NAME}"
