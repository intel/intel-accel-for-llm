#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Regression gate: a pure-attention model must be completely unaffected
# by the hybrid support.
#
# The hybrid stack is selected from kv_cache_config.has_mamba_layers, so
# a model without GDN/Mamba layers must take the original code path:
# no hybrid initialisation, no behaviour change, correct output.
#
# Usage:
#   MODEL=/path/to/Qwen3 TP_SIZE=2 tests/gpu/probe_pure_attention.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=0
PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-200}")"
MAX_TOKENS="${GATE_MAX_TOKENS:-64}"

gate_reset_cache

log "first run: populate the cache"
gate_serve pure_attention_first || {
    fail "engine startup"; gate_summary; exit 1; }
FIRST_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_stop

FIRST_LOG="$GATE_LOG_DIR/pure_attention_first.log"
check "hybrid path NOT taken on a pure-attention model" \
    bash -c '! grep -q "kvshrink hybrid path enabled" "$1"' _ "$FIRST_LOG"
check "pure-attention KV store registered" \
    grep -q "Registered .* KV cache layers" "$FIRST_LOG"
check "first run produced output" test -n "$FIRST_OUT"

log "second run: fresh engine reuses the cache"
gate_serve pure_attention_second || {
    fail "engine startup"; gate_summary; exit 1; }
SECOND_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_stop

SECOND_LOG="$GATE_LOG_DIR/pure_attention_second.log"
if [[ "$FIRST_OUT" == "$SECOND_OUT" ]]; then
    pass "output identical across runs"
else
    fail "output changed across runs"
    printf '  first : %s\n  second: %s\n' \
        "${FIRST_OUT:0:200}" "${SECOND_OUT:0:200}"
fi
check "no connector errors" \
    bash -c '! grep -qE "Failed to load KV cache|kvshrink load poison" "$1"' \
    _ "$SECOND_LOG"

gate_summary
