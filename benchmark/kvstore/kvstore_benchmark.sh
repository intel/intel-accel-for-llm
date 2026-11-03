#!/bin/bash -e
# Usage: kvstore_benchmark.sh [--ranks N] [--qat] [--iaa] [--dsa] [--nsys] [kvstore_benchmark.py options]
#   --qat / --iaa select the compression engines (none -> compression off); --dsa enables DSA gather/scatter.

export IAXL_CPU_ZIP_ENABLE=0
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0
export IAXL_DDR_POOL_SIZE_GB=32

RANKS=1
USE_QAT=0
USE_IAA=0
USE_DSA=0
USE_NSYS=0
PY_ARGS=()

while (($#)); do
    case "$1" in
        --ranks) RANKS=$2; shift 2 ;;
        --ranks=*) RANKS=${1#--ranks=}; shift ;;
        --qat) USE_QAT=1; shift ;;
        --iaa) USE_IAA=1; shift ;;
        --dsa) USE_DSA=1; shift ;;
        --nsys) USE_NSYS=1; shift ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done
if ((!USE_QAT && !USE_IAA)); then
    echo -e "\033[31mWARNING: neither --qat nor --iaa given, running with compression disabled\033[0m" >&2
    export IAXL_KV_COMPRESSION=0
fi
export IAXL_QAT_ZIP_ENABLE=$USE_QAT
export IAXL_IAA_ZIP_ENABLE=$USE_IAA
export IAXL_DSA_GD_ENABLE=$USE_DSA
export TP_SIZE=$RANKS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setvars.sh"

export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

# Each rank process binds its own CPUs / accelerators via iaxl.utils.affinity.
BENCH=(python3 "$SCRIPT_DIR/kvstore_benchmark_multi_ranks.py" "${PY_ARGS[@]}")

if [[ "$USE_NSYS" == "1" ]]; then
    export IAXL_PROFILE_MODE=nvtx
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1

    rm -f /_data/nsys_report*

    nsys profile -o /_data/nsys_report \
        -t cuda,nvtx,nccl,python-gil,osrt \
        --python-sampling=true \
        --python-backtrace=cuda \
        --trace-fork-before-exec=true \
        "${BENCH[@]}"
else
    "${BENCH[@]}"
fi
