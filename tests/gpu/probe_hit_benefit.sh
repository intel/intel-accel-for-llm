#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# What the hybrid external cache is WORTH: time-to-first-token when a
# prefix is restored versus when it is recomputed.
#
# There is no historical baseline to compare against here -- hybrid
# support is new -- so the honest comparison is against the same engine
# doing the work itself. Both measurements run in ONE process against
# the same weights and the same prompt; the only difference is whether
# the external cache can answer.
#
# Method, and why it is shaped this way:
#   1. warm the external cache with the prompt (this run also pays for
#      the saves, so it is NOT the recompute baseline);
#   2. drop vLLM's own prefix cache so it cannot answer either request;
#   3. RECOMPUTE baseline: ask for a prompt the cache has never seen,
#      so the engine computes the whole prefix;
#   4. RESTORE: ask for the warmed prompt again, served from the
#      external cache.
# Steps 3 and 4 are interleaved and repeated, so drift in machine state
# cannot masquerade as a difference between the two.
#
# Reported: median TTFT of each arm. TTFT is the metric that matters --
# a restored prefix removes prefill work, it does not speed up decode.
#
# Usage:
#   MODEL=/path/to/Qwen3.5-4B TP_SIZE=2 tests/gpu/probe_hit_benefit.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1
export VLLM_SERVER_DEV_MODE=1

ROUNDS="${GATE_BENEFIT_ROUNDS:-5}"
# Saving is synchronous within a pass, and the recompute arm stores a
# whole new prefix (~600ms observed for 8 boundaries). Without a settle
# gap the next request queues behind that work and the measurement
# reports save cost as restore cost. Loads themselves are steady at
# ~13ms, so this gap is about isolating the arms, not hiding a defect.
SETTLE="${GATE_BENEFIT_SETTLE_S:-3}"
SEGMENTS="${GATE_PROMPT_SEGMENTS:-400}"
WARM_PROMPT="$(gate_long_prompt "$SEGMENTS")"

gate_reset_cache
gate_serve hit_benefit || { fail "engine startup"; gate_summary; exit 1; }

# TTFT via streaming: time until the first chunk arrives.
ttft_ms() {
    local prompt="$1"
    python3 - "$GATE_PORT" "$MODEL" "$prompt" <<'PY'
import json, sys, time, urllib.request
port, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 8,
                   "temperature": 0, "seed": 0, "stream": True}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                             data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req) as r:
    for line in r:
        if line.startswith(b"data:") and b"[DONE]" not in line:
            print(f"{(time.perf_counter() - t0) * 1000:.1f}")
            break
PY
}

reset_internal() {
    curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" \
        >/dev/null || true
}

log "warming the external cache (this run also pays for the saves)"
reset_internal
ttft_ms "$WARM_PROMPT" >/dev/null

RESTORE=() ; RECOMPUTE=()
for i in $(seq 1 "$ROUNDS"); do
    # Recompute arm: a prompt never seen before, so neither cache helps.
    reset_internal
    FRESH="$(gate_long_prompt "$SEGMENTS" | sed "s/alpha/alpha-r$i-$RANDOM/")"
    RECOMPUTE+=("$(ttft_ms "$FRESH")")
    sleep "$SETTLE"

    # Restore arm: the warmed prompt, external cache only.
    reset_internal
    RESTORE+=("$(ttft_ms "$WARM_PROMPT")")
    sleep "$SETTLE"
    log "round $i: recompute=${RECOMPUTE[-1]}ms restore=${RESTORE[-1]}ms"
done

gate_stop

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1}
    END{print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

MED_RE="$(median "${RECOMPUTE[@]}")"
MED_ST="$(median "${RESTORE[@]}")"

echo
echo "=============================================================="
printf '  TTFT median, recompute : %8s ms\n' "$MED_RE"
printf '  TTFT median, restored  : %8s ms\n' "$MED_ST"
awk -v a="$MED_RE" -v b="$MED_ST" 'BEGIN{
    if (a > 0) printf "  saved                  : %8.1f ms (%.1f%%)\n",
                      a-b, 100*(a-b)/a }'
echo "=============================================================="

# A restore that is not faster than recomputing is not necessarily a
# bug -- a short prompt or a slow storage tier can erase the gain -- but
# it does mean the feature is not paying for itself on this workload.
awk -v a="$MED_RE" -v b="$MED_ST" 'BEGIN{ exit !(b < a) }' \
    && pass "restoring is faster than recomputing" \
    || fail "restoring is NOT faster than recomputing on this workload"

gate_summary
