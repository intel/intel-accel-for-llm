#!/usr/bin/env bash

set -euo pipefail

NWQ=${1:-1}

DEVS=()
for s in /sys/bus/dsa/devices/dsa[0-9]*; do
    [[ -d "$s" ]] || continue
    dev=$(basename "$s")
    [[ "$dev" =~ ^dsa[0-9]+$ ]] || continue
    DEVS+=("$dev")
    node=$(cat "$s/numa_node" 2>/dev/null || echo -1)
    echo "numa $node: $dev"
done

if [[ ${#DEVS[@]} -eq 0 ]]; then
    echo "ERROR: no DSA device" >&2
    exit 1
fi
echo "DSA devices: ${DEVS[*]}"

for DEV in "${DEVS[@]}"; do
    ID=${DEV#dsa}
    SYS="/sys/bus/dsa/devices/$DEV"

    TOTAL=$(cat "$SYS/max_work_queues_size")
    MAXWQ=$(cat "$SYS/max_work_queues")
    MAXBATCH=$(cat "$SYS/max_batch_size")
    MAXXFER=$(cat "$SYS/max_transfer_size")
    if ((NWQ > MAXWQ)); then
        echo "ERROR: $DEV supports only $MAXWQ WQs (requested $NWQ)" >&2
        exit 1
    fi

    WQSIZE=$((TOTAL / NWQ))
    if ((WQSIZE < 1)); then
        echo "ERROR: $DEV max_work_queues_size=$TOTAL too small for $NWQ WQs" >&2
        exit 1
    fi

    echo "Configuring $DEV: $NWQ WQs (per-WQ size=$WQSIZE) ..."

    for ((w = 0; w < MAXWQ; w++)); do
        wqdev="wq${ID}.$w"
        if [[ -e "/sys/bus/dsa/devices/$wqdev/driver" ]]; then
            echo "$wqdev" >"/sys/bus/dsa/devices/$wqdev/driver/unbind" 2>/dev/null || true
        fi
        accel-config disable-wq "$DEV/$wqdev" 2>/dev/null || true
    done
    accel-config disable-device "$DEV" 2>/dev/null || true

    for e in "$SYS"/engine${ID}.*; do
        [ -e "$e" ] || continue
        accel-config config-engine "$DEV/$(basename "$e")" --group-id=0
    done

    for ((i = 0; i < NWQ; i++)); do
        accel-config config-wq "$DEV/wq${ID}.$i" \
            --group-id=0 \
            --wq-size="$WQSIZE" \
            --mode=dedicated \
            --type=user \
            --name=dsa_gpu \
            --priority=10 \
            --block-on-fault=1 \
            --max-transfer-size="$MAXXFER" \
            --max-batch-size="$MAXBATCH" \
            --driver-name=user
    done

    accel-config enable-device "$DEV"
    for ((i = 0; i < NWQ; i++)); do
        accel-config enable-wq "$DEV/wq${ID}.$i"
    done
done

echo "Done. Configured ${#DEVS[@]} device(s), $NWQ WQ each:"
IDS=()
WQS=()
for DEV in "${DEVS[@]}"; do
    ID=${DEV#dsa}
    IDS+=("$ID")
    echo "  $DEV: wq${ID}.0 .. wq${ID}.$((NWQ - 1))"
    for ((i = 0; i < NWQ; i++)); do
        WQS+=("wq${ID}.$i")
    done
done
DSA_DEVICES=$(
    IFS=,
    echo "${IDS[*]}"
)
DSA_WQS=$(
    IFS=,
    echo "${WQS[*]}"
)
