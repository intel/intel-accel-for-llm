#!/bin/bash -e
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Run a remote-daemon release bundle (see build_release.sh) inside a
# container started from the SAME base image as the GPU node. Nothing is
# compiled here -- the bundle already contains the compiled iaxl extension
# and its runtime libraries -- so only IAXL_BASE_DOCKER_IMAGE needs to be
# available (pulled from a registry, or `docker load`-ed from a saved
# image) on this node, not the full iaxl dev/build image.
#
# All CONTROL_*/NIXL_*/POOL_SIZE_GB/CACHE_DIR/COMPRESS/STAGING_*/DEVICE/
# INSTANCE_ID/NUM_INSTANCES/KVSHRINK_REMOTE_*/IAXL_*/UCX_* environment
# variables set before calling this script are forwarded into the container
# unchanged; see run-daemon.sh and doc/usage/remote-nixl-kv-cache.md for
# what each one does.
#
# Usage (single daemon process):
#   CONTROL_PORT=19000 TRANSPORT=nixl NIXL_DEVICE=mlx5_1:1 NIXL_HOST=10.10.10.10 \
#   POOL_SIZE_GB=64 STAGING_SLOTS=256 STAGING_SLOT_MB=8 \
#   tools/remote_daemon/docker-run-daemon.sh
#
# For a multi-process deployment, run this script once per instance with
# INSTANCE_ID/NUM_INSTANCES/CONTROL_PORT/NIXL_PORT set accordingly (or adapt
# run-daemon-multi.sh's loop to call `docker run` instead of running
# run-daemon.sh directly).

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE="${IAXL_BASE_DOCKER_IMAGE:-vllm/vllm-openai:v0.23.0}"
CONTAINER_NAME="${CONTAINER_NAME:-iaxl-remote-daemon-${INSTANCE_ID:-0}}"

# Forward every relevant env var into the container by name (not by value
# substitution), so secrets/quoting are never re-interpreted by this shell.
FORWARD_VARS=()
while IFS='=' read -r name _; do
    case "$name" in
        CONTROL_*|NIXL_*|STAGING_*|KVSHRINK_REMOTE_*|IAXL_*|UCX_*)
            FORWARD_VARS+=(-e "$name") ;;
        POOL_SIZE_GB|CACHE_DIR|COMPRESS|DEVICE|INSTANCE_ID|NUM_INSTANCES)
            FORWARD_VARS+=(-e "$name") ;;
        TRANSPORT|LOG_LEVEL|PYTHON)
            FORWARD_VARS+=(-e "$name") ;;
        http_proxy|https_proxy|no_proxy|HTTP_PROXY|HTTPS_PROXY|NO_PROXY|all_proxy|ALL_PROXY)
            # Needed so run-daemon.sh's `pip install -r requirements-runtime.txt`
            # (and any other outbound traffic from the container) can reach the
            # corporate mirror. Docker does not inherit these from the host env
            # automatically -- they must be passed with -e explicitly.
            FORWARD_VARS+=(-e "$name") ;;
    esac
done < <(env)

# vllm/vllm-openai bakes in an ENTRYPOINT that runs the `vllm` CLI; without
# overriding it, our command below would be appended as arguments to that
# entrypoint instead of replacing it (this is what start.sh does too).
#
# TRANSPORT=nixl uses UCX + Mellanox verbs to move KV cache over RDMA, which
# requires the container to see the RDMA uverbs/rdma_cm char devices and to
# be able to pin memory. Without this, UCX only discovers TCP netdevs inside
# the container and NIXL createBackend fails with NIXL_ERR_BACKEND. We mirror
# the GPU-side start.sh setup (`--privileged -v /dev:/dev`) so the same
# vllm/vllm-openai base image works unchanged on the remote node.
exec docker run --rm --net host --ipc host --pid host \
    --privileged \
    -v /dev:/dev \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --name "$CONTAINER_NAME" \
    -v "$BUNDLE_DIR:$BUNDLE_DIR" -w "$BUNDLE_DIR" \
    --entrypoint "" \
    "${FORWARD_VARS[@]}" \
    "$IMAGE" \
    bash "$BUNDLE_DIR/tools/remote_daemon/run-daemon.sh"
