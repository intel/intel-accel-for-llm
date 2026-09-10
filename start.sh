#!/bin/bash -e
source setvars.sh

docker build -f docker/Dockerfile.dev -t "$IAXL_DEV_DOCKER_IMAGE" . \
    --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy

docker run \
    "${DOCKER_RUN_ARGS[@]}" \
    --rm \
    --privileged \
    --pid host \
    --net host \
    -it \
    --name "$CONTAINER_NAME" \
    -v /dev:/dev \
    --entrypoint "" \
    "$IAXL_DEV_DOCKER_IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" >/dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2>/dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation
        # vllm/vllm-openai:v0.23.0 ships nixl 1.2.0, whose bundled UCX does not
        # enable dma-buf CUDA memory registration and falls back to nvidia_peermem;
        # on inbox-kernel hosts this floods the log with ibv_reg_mr Bad address
        # errors and silently disables remote KV cache offload. 1.4+ uses dma-buf
        # and needs no peer-memory kernel module.
        pip install -U "nixl>=1.4"
        exec bash'
