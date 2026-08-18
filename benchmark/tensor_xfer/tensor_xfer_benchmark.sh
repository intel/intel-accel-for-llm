#!/bin/bash -e

export IAXL_DSA_GD_ENABLE=1
export TP_SIZE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../setvars.sh"
echo SCRIPT_DIR: $SCRIPT_DIR

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

numactl --cpunodebind=0 --membind=0 python3 "$SCRIPT_DIR/tensor_xfer_benchmark.py"
