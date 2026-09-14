#!/bin/bash -e
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Launch NUM_INSTANCES remote cache daemon processes on this host, one per
# GPU worker rank, each on its own control/NIXL port. Use this instead of
# run-daemon.sh when a single daemon process cannot keep up with tp_size GPU
# workers (many-to-one contention on the control plane / codec gates) -- see
# the design doc's "why multi-process daemon" section.
#
# GPU worker rank r must then be pointed at instance r specifically, e.g.:
#
#   KVSHRINK_REMOTE_DAEMON_ADDR="10.10.10.10:19000|10.10.10.10:19001|10.10.10.10:19002|10.10.10.10:19003" \
#   KVSHRINK_REMOTE_NIXL_ADDR="10.10.10.10:19100|10.10.10.10:19101|10.10.10.10:19102|10.10.10.10:19103" \
#   ...
#
# (KVSHRINK_REMOTE_DAEMON_ADDR/NIXL_ADDR accept the same "|"-separated
# per-rank convention as KVSHRINK_QAT_DEVICES -- see iaxl/remote/config.py).
#
# Example (4 instances, 2 QAT devices each, matching a 4-rank TP group where
# QAT_AUTO_DETECT would have assigned "0,1|2,3|4,5|6,7" to the GPU workers):
#
#   NUM_INSTANCES=4 CONTROL_PORT_BASE=19000 NIXL_PORT_BASE=19100 \
#   KVSHRINK_REMOTE_QAT_DEVICES="0,1|2,3|4,5|6,7" \
#   POOL_SIZE_GB=64 STAGING_SLOTS=256 STAGING_SLOT_MB=8 \
#   TRANSPORT=nixl NIXL_DEVICE=mlx5_0:1 \
#   tools/remote_daemon/run-daemon-multi.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

: "${NUM_INSTANCES:=1}"
: "${CONTROL_PORT_BASE:=19000}"
: "${NIXL_PORT_BASE:=19100}"

# Release bundles (build_release.sh) ship pysrc/iaxl/ instead of a full repo
# checkout with `pip install -e .` already done into the host's own Python,
# so each instance must run inside a container started from the base image
# (docker-run-daemon.sh) -- run-daemon.sh by itself never starts docker.
# Override with USE_DOCKER=0/1 to force either mode.
if [[ -z "${USE_DOCKER:-}" ]]; then
    [[ -d "$BUNDLE_DIR/pysrc/iaxl" ]] && USE_DOCKER=1 || USE_DOCKER=0
fi
LAUNCHER="$SCRIPT_DIR/run-daemon.sh"
if [[ "$USE_DOCKER" == "1" ]]; then
    LAUNCHER="$SCRIPT_DIR/docker-run-daemon.sh"
fi
echo "run-daemon-multi: USE_DOCKER=$USE_DOCKER launcher=$LAUNCHER"

pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for ((i = 0; i < NUM_INSTANCES; i++)); do
    (
        export NUM_INSTANCES INSTANCE_ID="$i"
        export CONTROL_PORT=$((CONTROL_PORT_BASE + i))
        export NIXL_PORT=$((NIXL_PORT_BASE + i))
        # Each instance's own persist directory (never shared -- every
        # instance owns disjoint (model, tp_size, tp_rank) groups anyway
        # since a GPU rank only ever talks to its own instance, but a
        # distinct CACHE_DIR keeps chunks.db files from different instances
        # visually separated on disk for operators).
        export CACHE_DIR="${CACHE_DIR:-_data/kvcache/remote}/instance$i"
        # Distinct container name per instance; docker-run-daemon.sh already
        # defaults CONTAINER_NAME off INSTANCE_ID, but set it explicitly so
        # a caller-provided CONTAINER_NAME_PREFIX also works.
        if [[ "$USE_DOCKER" == "1" ]]; then
            export CONTAINER_NAME="${CONTAINER_NAME_PREFIX:-iaxl-remote-daemon}-$i"
        fi
        exec "$LAUNCHER"
    ) &
    pids+=("$!")
    echo "started daemon instance $i (pid ${pids[-1]}): control_port=$((CONTROL_PORT_BASE + i)) nixl_port=$((NIXL_PORT_BASE + i))"
done

wait -n
echo "a daemon instance exited; stopping the rest" >&2
exit 1
