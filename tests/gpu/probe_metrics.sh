#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Metrics gate: the documented kvshrink_* series must be exposed with
# sane values while the hybrid path is running.
#
# The exporter runs inside the engine process on its own port because
# vLLM's /metrics endpoint aggregates a private registry that a
# connector cannot reach. Several series encode invariants that are
# guaranteed by construction (synchronous saves => no deferred blocks),
# so they must be present and zero rather than absent.
#
# Usage:
#   MODEL=/path/to/Qwen3.5-4B tests/gpu/probe_metrics.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1
METRICS_PORT="${KVSHRINK_METRICS_PORT:-18801}"
export KVSHRINK_METRICS_PORT="$METRICS_PORT"
PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-400}")"

gate_reset_cache
gate_serve metrics || { fail "engine startup"; gate_summary; exit 1; }

# Generate traffic twice so both the save and the load side report.
gate_completion "$PROMPT" "${GATE_MAX_TOKENS:-32}" >/dev/null
gate_completion "$PROMPT" "${GATE_MAX_TOKENS:-32}" >/dev/null

SCRAPE="$GATE_LOG_DIR/metrics.txt"
if curl -sf "http://127.0.0.1:$METRICS_PORT/metrics" -o "$SCRAPE"; then
    pass "metrics endpoint reachable on :$METRICS_PORT"
else
    fail "metrics endpoint unreachable on :$METRICS_PORT"
    gate_summary
    exit 1
fi

for series in \
    kvshrink_lookup_boundary_total \
    kvshrink_external_hit_tokens_total \
    kvshrink_state_snapshot_boundary_total \
    kvshrink_transfer_bytes_total \
    kvshrink_job_latency_seconds_sum \
    kvshrink_manifest_incomplete_total \
    kvshrink_checksum_failure_total \
    kvshrink_deferred_blocks \
    kvshrink_pinned_pool_bytes \
    kvshrink_pending_store_jobs \
    kvshrink_inflight_boundaries \
    kvshrink_cursor_rollbacks
do
    check "series present: $series" grep -q "^$series" "$SCRAPE"
done

# Saving happened, so bytes must have moved.
check "transfer bytes are non-zero" \
    bash -c 'awk "/^kvshrink_transfer_bytes_total/ {if (\$NF+0 > 0) found=1} END {exit !found}" "$1"' \
    _ "$SCRAPE"

# Contract gauges: the hybrid save path is synchronous, so nothing may
# ever be pending or deferred at scrape time.
for gauge in kvshrink_deferred_blocks kvshrink_pending_store_jobs \
             kvshrink_inflight_boundaries
do
    check "$gauge is zero (synchronous save contract)" \
        bash -c 'awk -v g="$2" "\$1 == g {if (\$NF+0 != 0) exit 1} END {exit 0}" "$1"' \
        _ "$SCRAPE" "$gauge"
done

check "no checksum failures" \
    bash -c 'awk "/^kvshrink_checksum_failure_total/ {if (\$NF+0 != 0) exit 1} END {exit 0}" "$1"' \
    _ "$SCRAPE"

gate_summary
