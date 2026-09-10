#!/bin/bash -e
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Launch a single remote (NIXL/RDMA) KV cache daemon process.
#
# Runs `python -m iaxl.remote.daemon` with env-var-driven configuration, the
# same way `examples/kvshrink-vllm-serve.sh` drives the local vLLM server.
# Intended to run inside the SAME docker image as the GPU node
# (vllm/vllm-openai:v0.23.0, see ../../setvars.sh IAXL_BASE_DOCKER_IMAGE and
# ../../start.sh) with `pip install -e .` already done -- the daemon reuses
# the exact same installed iaxl package (native Mem/Storage/Record + the
# QAT/IAA/CPU zip pipeline), so there is nothing extra to build for it.
#
# See doc/usage/remote-nixl-kv-cache.md for the full deployment walkthrough
# and doc/design/remote-nixl-kv-cache.md for the architecture.
#
# Example (single daemon process serving every rank of one TP group):
#
#   CONTROL_PORT=19000 NIXL_HOST=10.10.10.10 NIXL_PORT=19100 \
#   TRANSPORT=nixl NIXL_DEVICE=mlx5_0:1 POOL_SIZE_GB=64 \
#   STAGING_SLOTS=256 STAGING_SLOT_MB=8 \
#   tools/remote_daemon/run-daemon.sh
#
# For a multi-process deployment (one daemon instance per GPU rank, see the
# design doc's "why multi-process daemon" section) use run-daemon-multi.sh
# instead, which drives this script once per instance.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

# Standalone release bundles (see build_release.sh) ship the compiled iaxl
# package under pysrc/ instead of a full repo checkout + `pip install -e .`;
# a full checkout already has iaxl importable via the editable install, so
# this is a no-op there.
if [[ -d "$REPO_DIR/pysrc/iaxl" ]]; then
    export PYTHONPATH="$REPO_DIR/pysrc${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$REPO_DIR/pysrc/iaxl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # Runtime-only Python deps missing from the vllm/vllm-openai base image
    # (currently just xxhash; see build_release.sh). Idempotent: pip exits
    # ~instantly once satisfied. Disable if the image is already prepared.
    if [[ -f "$REPO_DIR/requirements-runtime.txt" && "${IAXL_SKIP_PIP_INSTALL:-0}" != "1" ]]; then
        "${PYTHON:-python3}" -m pip install --disable-pip-version-check -q \
            -r "$REPO_DIR/requirements-runtime.txt" \
            || echo "WARNING: pip install -r requirements-runtime.txt failed; set IAXL_SKIP_PIP_INSTALL=1 to skip" >&2
    fi
fi

# tools/auto_config.sh is a pure function library (no GPU/NIC auto-detection
# runs merely by sourcing it), so it is safe to source on a GPU-less cache
# node just to reuse qat_thread_count().
source tools/auto_config.sh

: "${PYTHON:=python3}"
: "${CONTROL_HOST:=0.0.0.0}"
: "${CONTROL_PORT:=19000}"
: "${TRANSPORT:=tcp}"           # tcp | nixl
: "${NIXL_HOST:=}"
: "${NIXL_PORT:=19100}"
: "${NIXL_DEVICE:=}"            # e.g. mlx5_0:1, or mlx5_0:1,mlx5_1:1 for multi-rail
: "${POOL_SIZE_GB:=8}"
: "${CACHE_DIR:=_data/kvcache/remote}"
: "${COMPRESS:=1}"              # 0 = --no-compress (raw bandwidth benchmarking)
: "${STAGING_SLOTS:=0}"         # 0 disables NIXL staging even if TRANSPORT=nixl
: "${STAGING_SLOT_MB:=4}"
: "${DEVICE:=cpu}"
: "${LOG_LEVEL:=INFO}"
: "${INSTANCE_ID:=}"            # set by run-daemon-multi.sh; empty = single-process
: "${NUM_INSTANCES:=1}"

# ---- Feature switches (defaults match setvars.sh) --------------------------
export IAXL_KV_COMPRESSION=${IAXL_KV_COMPRESSION:-1}
export IAXL_QAT_ZIP_ENABLE=${IAXL_QAT_ZIP_ENABLE:-1}
export IAXL_IAA_ZIP_ENABLE=${IAXL_IAA_ZIP_ENABLE:-0}
export IAXL_CPU_ZIP_ENABLE=${IAXL_CPU_ZIP_ENABLE:-1}
export IAXL_CACHE_DIR=${IAXL_CACHE_DIR:-_data/kvcache}

# ---- Resolve QAT device(s) + derive IAXL_QAT_INSTANCE_NUM ------------------
# Mirrors kvshrink_connector._bind_intel_accel (per-rank slice) / setvars.sh
# (qat_thread_count) so the daemon satisfies the same
# `worker_count == IAXL_OMP_THREAD_NUM` invariant the native zip pipeline
# checks (iaxl/csrc/kv_zip/kv_zip.cpp:zip_pipeline). KVSHRINK_REMOTE_QAT_DEVICES
# is "|"-separated per instance (single process: every listed device is used
# together; multi-process: instance i uses only entry i) -- see
# iaxl/remote/daemon.py:_resolve_qat_devices for the same logic in Python
# (kept as a fallback there for direct `python -m iaxl.remote.daemon` use).
if [[ "${IAXL_QAT_ZIP_ENABLE,,}" =~ ^(1|true|yes|on)$ ]] && [[ -z "${IAXL_QAT_DEVICES:-}" ]] \
        && [[ -n "${KVSHRINK_REMOTE_QAT_DEVICES:-}" ]]; then
    if [[ "$NUM_INSTANCES" -le 1 ]]; then
        IAXL_QAT_DEVICES=$(echo "$KVSHRINK_REMOTE_QAT_DEVICES" | tr '|' ',' | tr ',' '\n' | awk '!seen[$0]++' | paste -sd, -)
    else
        [[ -n "$INSTANCE_ID" ]] || { echo "ERROR: INSTANCE_ID is required when NUM_INSTANCES>1" >&2; exit 1; }
        IFS='|' read -r -a _qat_parts <<<"$KVSHRINK_REMOTE_QAT_DEVICES"
        if ((${#_qat_parts[@]} == 1)); then
            IAXL_QAT_DEVICES="${_qat_parts[0]}"
        else
            IAXL_QAT_DEVICES="${_qat_parts[$INSTANCE_ID]}"
        fi
    fi
    export IAXL_QAT_DEVICES
fi
if [[ "${IAXL_QAT_ZIP_ENABLE,,}" =~ ^(1|true|yes|on)$ ]] && [[ -n "${IAXL_QAT_DEVICES:-}" ]] \
        && [[ -z "${IAXL_QAT_INSTANCE_NUM:-}" ]]; then
    export IAXL_QAT_INSTANCE_NUM=$(qat_thread_count "$IAXL_QAT_DEVICES" "${IAXL_QAT_ZIP_INSTANCES_PER_DEVICE:-4}")
fi

echo "remote daemon config: instance=${INSTANCE_ID:-<single>}/${NUM_INSTANCES} " \
     "control=$CONTROL_HOST:$CONTROL_PORT transport=$TRANSPORT nixl=$NIXL_HOST:$NIXL_PORT " \
     "device=$NIXL_DEVICE pool=${POOL_SIZE_GB}GiB compress=$COMPRESS " \
     "qat_devices=${IAXL_QAT_DEVICES:-<none>} qat_instances=${IAXL_QAT_INSTANCE_NUM:-<default>}"

if [[ "$TRANSPORT" == "nixl" && "$STAGING_SLOTS" -le 0 ]]; then
    echo "WARNING: TRANSPORT=nixl but STAGING_SLOTS=0 disables NIXL staging;" \
         "the daemon will only serve TCP clients. Set STAGING_SLOTS (e.g. 256)." >&2
fi

if [[ -n "$NIXL_DEVICE" ]]; then
    # Forwarded verbatim; a comma-separated list enables UCX multi-rail
    # across several RDMA ports for one session (see the design doc).
    export UCX_NET_DEVICES="$NIXL_DEVICE"
fi

compress_flag="--compress"
[[ "$COMPRESS" =~ ^(0|false|no|off)$ ]] && compress_flag="--no-compress"

instance_args=()
[[ -n "$INSTANCE_ID" ]] && instance_args+=(--instance-id "$INSTANCE_ID")

exec "$PYTHON" -m iaxl.remote.daemon \
    --control-host "$CONTROL_HOST" \
    --control-port "$CONTROL_PORT" \
    --nixl-host "$NIXL_HOST" \
    --nixl-port "$NIXL_PORT" \
    --pool-size-gb "$POOL_SIZE_GB" \
    --cache-dir "$CACHE_DIR" \
    "$compress_flag" \
    --staging-slots "$STAGING_SLOTS" \
    --staging-slot-mb "$STAGING_SLOT_MB" \
    --device "$DEVICE" \
    --num-instances "$NUM_INSTANCES" \
    "${instance_args[@]}" \
    --log-level "$LOG_LEVEL"
