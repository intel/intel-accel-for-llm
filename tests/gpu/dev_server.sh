#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Starts a long-lived engine for the development loop and leaves it
# running, so the gates do not pay the startup cost on every iteration.
#
#   # once (blocks; Ctrl-C to stop)
#   MODEL=/path/to/Qwen3.5-4B tests/gpu/dev_server.sh
#
#   # then, repeatedly, in seconds instead of minutes
#   GATE_REUSE_SERVER=1 GATE_SERVER_LOG=_data/gate-logs/dev_server.log \
#   GATE_KEEP_CACHE=1 MODEL=/path/to/Qwen3.5-4B \
#       tests/gpu/probe_warm_reuse.sh
#
# The cache directory is kept across iterations on purpose: it is the
# thing under test. Delete $GATE_CACHE_DIR by hand for a clean slate.

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1
export GATE_KEEP_CACHE=1
# Enables POST /reset_prefix_cache, which the warm-reuse gate needs to
# stop vLLM's own prefix cache from answering the second request.
export VLLM_SERVER_DEV_MODE=1

mkdir -p "$GATE_CACHE_DIR" "$GATE_LOG_DIR"

gate_serve dev_server || { log "engine failed to start"; exit 1; }

cat <<EOF

Engine ready on http://127.0.0.1:$GATE_PORT
  log:   $GATE_LAST_LOG
  cache: $GATE_CACHE_DIR

Run gates against it with:
  GATE_REUSE_SERVER=1 GATE_SERVER_LOG=$GATE_LAST_LOG \\
  GATE_PORT=$GATE_PORT MODEL=$MODEL \\
      tests/gpu/probe_warm_reuse.sh

Ctrl-C to stop.
EOF

# gate_cleanup (EXIT trap) stops the engine.
wait "$_SERVER_PID"
