#!/bin/bash -e
# NVIDIA_RUNTIME=none ./start.sh  -> skip nvidia docker runtime/gpus args (e.g. on a storage-only node)
# ./start.sh --rebuild            -> rebuild the dev image even if it already exists
REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        *) echo "Usage: $0 [--rebuild]" >&2; exit 1 ;;
    esac
done
set --

source setvars.sh

if [[ $REBUILD == 1 ]] || ! docker image inspect "$IAXL_DEV_DOCKER_IMAGE" >/dev/null 2>&1; then
    docker build -f docker/Dockerfile.dev -t "$IAXL_DEV_DOCKER_IMAGE" . \
        --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
        --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy
fi

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
        exec bash'
