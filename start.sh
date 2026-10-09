#!/bin/bash -e
# NVIDIA_RUNTIME=none ./start.sh  -> skip nvidia docker runtime/gpus args (e.g. on a storage-only node)
case "${1:-}" in
    --offline)
        OFFLINE=1
        shift
        ;;
    --help|-h)
        echo "Usage: $0 [--offline]"
        echo "  --offline  Use the existing dev image and enter the container without building or installing iaxl"
        exit 0
        ;;
    "") OFFLINE=0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (( $# )); then
    echo "Unexpected argument: $1" >&2
    exit 2
fi

source setvars.sh

if (( OFFLINE )); then
    docker image inspect "$IAXL_DEV_DOCKER_IMAGE" >/dev/null || {
        echo "Dev image $IAXL_DEV_DOCKER_IMAGE is not available locally" >&2
        exit 1
    }
else
    docker build -f docker/Dockerfile.dev -t "$IAXL_DEV_DOCKER_IMAGE" . \
        --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
        --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy
fi

if (( OFFLINE )); then
    DOCKER_RUN_ARGS+=("-e" "IAXL_OFFLINE=1")
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
        if [[ "$1" != 1 ]]; then
            pip install -e . --verbose --no-build-isolation
        fi
        exec bash' _ "$OFFLINE"
