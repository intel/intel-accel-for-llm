#!/bin/bash -e
# Usage: tensor_xfer_benchmark.sh [--ranks N] [tensor_xfer_benchmark.py options]

export IAXL_DSA_GD_ENABLE=1
export IAXL_DSA_H2D_CROSSOVER_KB=0 IAXL_DSA_D2H_CROSSOVER_KB=0

RANKS=1
ARGS=()
while (($#)); do
    case "$1" in
        --ranks) RANKS=$2; shift 2 ;;
        --ranks=*) RANKS=${1#--ranks=}; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
export TP_SIZE=$RANKS

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../setvars.sh"
echo SCRIPT_DIR: $SCRIPT_DIR

export LD_PRELOAD="$IAXL_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"

if ((RANKS > 1)); then
    # Each rank process binds its own CPUs / accelerators via iaxl.utils.affinity.
    python3 "$SCRIPT_DIR/tensor_xfer_benchmark_multi_ranks.py" "${ARGS[@]}"
else
    numactl --cpunodebind=0 --membind=0 python3 "$SCRIPT_DIR/tensor_xfer_benchmark.py" "${ARGS[@]}"
fi
