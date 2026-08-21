#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Cold -> hot correctness gate for hybrid (GDN/Mamba) models.
#
# The question this answers: after a request's KV and GDN state have
# been persisted, does a FRESH engine that restores them produce
# byte-identical output to the run that computed them from scratch?
#
# That is the only property that matters for an external KV cache: a
# restored run must be indistinguishable from a recomputed one. The gate
# runs two separate engine processes (a restart, not a warm cache) so
# the data really comes off disk, and checks the logs for evidence that
# the second run actually hit the external cache instead of silently
# recomputing everything (which would trivially produce equal output).
#
# Usage:
#   MODEL=/path/to/Qwen3.5-4B TP_SIZE=2 tests/gpu/probe_cold_hot.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1

# This gate is defined by the restart: reusing a live engine would keep
# the state in memory and prove nothing about persistence.
if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
    echo "probe_cold_hot.sh cannot reuse a running engine; it must" \
         "restart one to prove the data really came off disk." >&2
    echo "Use tests/gpu/probe_warm_reuse.sh for the in-process loop." >&2
    exit 2
fi
PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-400}")"
MAX_TOKENS="${GATE_MAX_TOKENS:-64}"

gate_reset_cache

# ---------------------------------------------------------------- cold
log "cold run: empty cache, everything computed and persisted"
gate_serve cold_hot_cold || { fail "cold engine startup"; gate_summary; exit 1; }
COLD_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_stop

COLD_LOG="$GATE_LAST_LOG"
check "hybrid path active" \
    grep -q "kvshrink hybrid path enabled" "$COLD_LOG"
check "cold run saved boundaries" \
    grep -qE "chunk_save: [1-9][0-9]* pages stored, [1-9][0-9]* boundaries" "$COLD_LOG"
check "cold run produced output" test -n "$COLD_OUT"

# ----------------------------------------------------------------- hot
log "hot run: fresh engine, same prompt, cache on disk"
gate_serve cold_hot_hot || { fail "hot engine startup"; gate_summary; exit 1; }
HOT_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_stop

HOT_LOG="$GATE_LAST_LOG"

# The engine must report external tokens; otherwise the identical output
# below proves nothing (a full recompute would also match).
check "hot run hit the external cache" \
    grep -qE "start_load_kv: [1-9][0-9]* pages loaded" "$HOT_LOG"

if [[ "$COLD_OUT" == "$HOT_OUT" ]]; then
    pass "restored output is byte-identical to the recomputed output"
else
    fail "restored output differs from the recomputed output"
    printf '  cold: %s\n  hot : %s\n' "${COLD_OUT:0:200}" "${HOT_OUT:0:200}"
fi

# A load that failed is fatal by design: the engine must never fall back
# to reading unrestored state.
check "no unrestored-state errors" \
    bash -c '! grep -q "refusing to enter forward with unrestored state" "$1"' _ "$HOT_LOG"
check "no load poison" \
    bash -c '! grep -q "kvshrink load poison" "$1"' _ "$HOT_LOG"

gate_summary
