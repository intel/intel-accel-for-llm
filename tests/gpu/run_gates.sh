#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Runs the GPU gates that apply to the configured model.
#
#   MODEL=/path/to/Qwen3.5-4B TP_SIZE=2 tests/gpu/run_gates.sh
#
# Hybrid gates need a GDN/Mamba model; the pure-attention regression
# gate needs an attention-only model. Set GATE_MODEL_HYBRID and/or
# GATE_MODEL_ATTENTION to run both in one pass:
#
#   GATE_MODEL_HYBRID=/models/Qwen3.5-4B \
#   GATE_MODEL_ATTENTION=/models/Qwen3-14B \
#   TP_SIZE=2 tests/gpu/run_gates.sh

set -uo pipefail
GATE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

HYBRID_MODEL="${GATE_MODEL_HYBRID:-${MODEL:-}}"
ATTENTION_MODEL="${GATE_MODEL_ATTENTION:-}"
STATUS=0

run_gate() {
    local model="$1" script="$2"
    echo
    echo "=============================================================="
    echo "  $(basename "$script")"
    echo "=============================================================="
    if MODEL="$model" bash "$GATE_DIR/$script"; then
        echo "-> $(basename "$script"): PASS"
    else
        echo "-> $(basename "$script"): FAIL"
        STATUS=1
    fi
}

if [[ -n "$HYBRID_MODEL" ]]; then
    run_gate "$HYBRID_MODEL" probe_warm_reuse.sh
    run_gate "$HYBRID_MODEL" probe_cold_hot.sh
    run_gate "$HYBRID_MODEL" probe_metrics.sh
else
    echo "skipping hybrid gates: set MODEL or GATE_MODEL_HYBRID"
fi

if [[ -n "$ATTENTION_MODEL" ]]; then
    run_gate "$ATTENTION_MODEL" probe_pure_attention.sh
else
    echo "skipping the pure-attention regression gate:" \
         "set GATE_MODEL_ATTENTION to an attention-only model"
fi

echo
if (( STATUS == 0 )); then
    echo "ALL GATES PASSED"
else
    echo "SOME GATES FAILED"
fi
exit $STATUS
