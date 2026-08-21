#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the GPU gates. Everything that varies per machine
# comes from an environment variable with a default, so the gates run
# unchanged anywhere.

set -euo pipefail

GATE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$GATE_DIR/../.." && pwd)

: "${MODEL:?set MODEL to a model path or HF id}"

# The connector requires the full runtime environment and refuses to
# start without it, so the gates go through the same entry point an
# operator would. Every variable in setvars.sh honours a value already
# present in the environment, so the caller stays in control; a machine
# without the Intel accelerators just turns them off, e.g.
#   IAXL_QAT_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=0 tests/gpu/run_gates.sh
source "$REPO_DIR/setvars.sh" >/dev/null

GATE_PORT="${GATE_PORT:-8000}"
GATE_TP="${TP_SIZE:-1}"
GATE_LOG_DIR="${GATE_LOG_DIR:-$REPO_DIR/_data/gate-logs}"
GATE_CACHE_DIR="${GATE_CACHE_DIR:-$REPO_DIR/_data/gate-cache}"
GATE_KEEP_CACHE="${GATE_KEEP_CACHE:-0}"
GATE_STARTUP_TIMEOUT="${GATE_STARTUP_TIMEOUT:-600}"
GATE_MAX_MODEL_LEN="${GATE_MAX_MODEL_LEN:-8192}"
GATE_GPU_UTIL="${GATE_GPU_UTIL:-0.85}"
# Reproducible block hashes: vLLM seeds its first-block hash from
# PYTHONHASHSEED and randomizes it when unset, which makes every
# restart miss.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

_SERVER_PID=""
FAILURES=0

log()  { printf '[gate] %s\n' "$*"; }
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

# check <description> <condition-exit-code>
check() {
    local desc="$1"; shift
    if "$@"; then pass "$desc"; else fail "$desc"; fi
}

gate_reset_cache() {
    # A reused engine already opened the cache; wiping it underneath
    # would be a lie about what the run then proves.
    if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
        log "reusing an engine: leaving the existing cache in place"
        mkdir -p "$GATE_CACHE_DIR" "$GATE_LOG_DIR"
        return 0
    fi
    rm -rf "$GATE_CACHE_DIR"
    mkdir -p "$GATE_CACHE_DIR" "$GATE_LOG_DIR"
}

gate_cleanup() {
    if [[ -n "$_SERVER_PID" ]] && kill -0 "$_SERVER_PID" 2>/dev/null; then
        kill "$_SERVER_PID" 2>/dev/null || true
        wait "$_SERVER_PID" 2>/dev/null || true
    fi
    _SERVER_PID=""
    if [[ "$GATE_KEEP_CACHE" != "1" && "${GATE_REUSE_SERVER:-0}" != "1" ]]; then
        rm -rf "$GATE_CACHE_DIR"
    fi
}
trap gate_cleanup EXIT

# gate_serve <log-name> [extra vllm args...]
# Starts an engine with the KVShrink connector and waits until it is
# ready. Hybrid-specific flags are added only when GATE_HYBRID=1.
#
# During development, engine startup dominates the loop. Set
# GATE_REUSE_SERVER=1 to run the checks against an engine that is
# already listening on GATE_PORT (see tests/gpu/dev_server.sh); then
# point GATE_SERVER_LOG at that engine's log so the log assertions have
# something to read.
gate_serve() {
    local log_name="$1"; shift
    local log_file="$GATE_LOG_DIR/$log_name.log"
    mkdir -p "$GATE_LOG_DIR"

    if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
        GATE_LAST_LOG="${GATE_SERVER_LOG:-$log_file}"
        if [[ ! -r "$GATE_LAST_LOG" ]]; then
            log "GATE_REUSE_SERVER=1 needs a readable GATE_SERVER_LOG" \
                "(got '${GATE_SERVER_LOG:-unset}')"
            return 1
        fi
        if ! curl -sf "http://127.0.0.1:$GATE_PORT/health" >/dev/null 2>&1; then
            log "no engine answering on port $GATE_PORT"
            return 1
        fi
        log "reusing the engine on port $GATE_PORT (log: $GATE_LAST_LOG)"
        return 0
    fi

    local -a args=(
        serve "$MODEL"
        --kv-transfer-config
        '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}'
        --trust-remote-code
        --tensor-parallel-size "$GATE_TP"
        --max-model-len "$GATE_MAX_MODEL_LEN"
        --gpu-memory-utilization "$GATE_GPU_UTIL"
        --port "$GATE_PORT"
        --enforce-eager
    )
    if [[ "${GATE_HYBRID:-0}" == "1" ]]; then
        # GDN snapshots are only addressable on aligned boundaries, and
        # the hybrid memory allocator must stay enabled. Prefix caching
        # must be requested explicitly: vLLM defaults it OFF for hybrid
        # models and then silently rewrites the cache mode to 'none'.
        args+=(--enable-prefix-caching
               --mamba-cache-mode align
               --no-disable-hybrid-kv-cache-manager)
    fi
    args+=("$@")

    log "starting engine -> $log_file"
    GATE_LAST_LOG="$log_file"
    IAXL_CACHE_DIR="$GATE_CACHE_DIR" vllm "${args[@]}" >"$log_file" 2>&1 &
    _SERVER_PID=$!

    local waited=0
    until curl -sf "http://127.0.0.1:$GATE_PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$_SERVER_PID" 2>/dev/null; then
            log "engine exited during startup; last lines:"
            tail -30 "$log_file" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        if (( waited >= GATE_STARTUP_TIMEOUT )); then
            log "engine did not become ready in ${GATE_STARTUP_TIMEOUT}s"
            tail -30 "$log_file" >&2
            return 1
        fi
    done
    log "engine ready after ${waited}s"
}

gate_stop() {
    # Never kill an engine we did not start.
    if [[ -n "$_SERVER_PID" ]]; then
        kill "$_SERVER_PID" 2>/dev/null || true
        wait "$_SERVER_PID" 2>/dev/null || true
        _SERVER_PID=""
    fi
}

# gate_completion <prompt> [max_tokens]
# Greedy completion (temperature 0) so cold and hot runs are comparable.
gate_completion() {
    local prompt="$1"
    local max_tokens="${2:-64}"
    curl -sf "http://127.0.0.1:$GATE_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "prompt": sys.argv[2],
                  "max_tokens": int(sys.argv[3]), "temperature": 0,
                  "seed": 0}))' "$MODEL" "$prompt" "$max_tokens")" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
}

# A prompt long enough to cross several GDN alignment boundaries.
gate_long_prompt() {
    local repeats="${1:-400}"
    python3 -c '
import sys
n = int(sys.argv[1])
print(" ".join("segment %d carries token payload alpha beta gamma." % i
               for i in range(n)))' "$repeats"
}

gate_summary() {
    echo
    if (( FAILURES == 0 )); then
        echo "RESULT: ALL PASS"
    else
        echo "RESULT: $FAILURES FAILED"
    fi
    return $(( FAILURES > 0 ))
}
