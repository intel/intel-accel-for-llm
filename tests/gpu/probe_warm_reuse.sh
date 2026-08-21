#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Warm-reuse gate for hybrid (GDN/Mamba) models -- the fast development
# loop. One engine process, no restart, no disk round-trip.
#
# The obstacle to observing an external cache inside a single process is
# vLLM's own prefix cache: it would serve the second request itself and
# the connector would never be asked. Disabling prefix caching is NOT an
# option for hybrid models -- vLLM then rewrites mamba_cache_mode to
# 'none', gives each request a single max_model_len block and turns off
# block-aligned chunk splitting, so GDN state stops being addressable by
# boundary and there is nothing to key a snapshot on.
#
# Instead we keep prefix caching on and drop only vLLM's copy between the
# two requests, via the dev endpoint POST /reset_prefix_cache (its
# reset_external parameter defaults to false, so our cache survives).
# The second request must then come to the connector, and it is served
# from the host memory pool rather than disk.
#
# Use probe_cold_hot.sh instead when the question is persistence across a
# restart; this gate deliberately does not answer that.
#
# Usage:
#   MODEL=/path/to/Qwen3.5-4B tests/gpu/probe_warm_reuse.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1
# Needed for POST /reset_prefix_cache.
export VLLM_SERVER_DEV_MODE=1

PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-400}")"
MAX_TOKENS="${GATE_MAX_TOKENS:-64}"

gate_reset_cache

gate_serve warm_reuse || { fail "engine startup"; gate_summary; exit 1; }
LOG="$GATE_LAST_LOG"

# ------------------------------------------------------- first request
log "first request: nothing cached anywhere, everything computed"
FIRST_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"

check "hybrid path active" \
    grep -q "kvshrink hybrid path enabled" "$LOG"
check "first request saved boundaries" \
    grep -qE "chunk_save: [1-9][0-9]* pages stored, [1-9][0-9]* boundaries" "$LOG"
check "first request produced output" test -n "$FIRST_OUT"

# --------------------------------------------- drop vLLM's own cache
# Everything after this mark is the second request, so the load below
# cannot be credited to the first one.
MARK=$(wc -l < "$LOG")

log "dropping vLLM's internal prefix cache (external cache kept)"
if curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" >/dev/null; then
    pass "internal prefix cache reset"
else
    fail "internal prefix cache reset (is VLLM_SERVER_DEV_MODE=1 set?)"
fi

# ------------------------------------------------------ second request
log "second request: same prompt, internal cache empty, external cache warm"
SECOND_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"

TAIL_LOG="$GATE_LOG_DIR/warm_reuse.second.log"
tail -n "+$((MARK + 1))" "$LOG" >"$TAIL_LOG"

# Without this the identical output below would prove nothing: a full
# recompute produces the same tokens too.
check "second request hit the external cache" \
    grep -qE "start_load_kv: [1-9][0-9]* pages loaded" "$TAIL_LOG"

if [[ "$FIRST_OUT" == "$SECOND_OUT" ]]; then
    pass "restored output is byte-identical to the computed output"
else
    fail "restored output differs from the computed output"
    printf '  first : %s\n  second: %s\n' "${FIRST_OUT:0:200}" "${SECOND_OUT:0:200}"
fi

check "no unrestored-state errors" \
    bash -c '! grep -q "refusing to enter forward with unrestored state" "$1"' _ "$LOG"
check "no load poison" \
    bash -c '! grep -q "kvshrink load poison" "$1"' _ "$LOG"

gate_stop
gate_summary
