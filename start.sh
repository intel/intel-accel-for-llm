#!/bin/bash -e
# NVIDIA_RUNTIME=none ./start.sh  -> skip nvidia docker runtime/gpus args (e.g. on a storage-only node)
# ./start.sh --rebuild            -> rebuild the dev image even if it already exists
# DEVICE=cpu ./start.sh           -> vLLM CPU inference; also set the CPU base image (see doc/usage/vllm-cpu-inference.en.md)
# IAXL_QATLIB=intree ./start.sh   -> qatlib standalone mode, one rank per container: the GPU of CUDA_VISIBLE_DEVICES /
#                                    ZE_AFFINITY_MASK (default 0), or on CPU the NUMA node of VLLM_CPU_OMP_THREADS_BIND
REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        *) echo "Usage: $0 [--rebuild]" >&2; exit 1 ;;
    esac
done
set --

IAXL_START_SH=1 # setvars.sh: this shell starts a container
source setvars.sh

if [[ $REBUILD == 1 ]] || ! docker image inspect "$IAXL_DEV_DOCKER_IMAGE" >/dev/null 2>&1; then
    docker build -f docker/Dockerfile.dev -t "$IAXL_DEV_DOCKER_IMAGE" . \
        --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
        --build-arg "DEVICE=$DEVICE" \
        --build-arg IAXL_QATLIB_VERSION \
        --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy
fi

docker run \
    "${DOCKER_RUN_ARGS[@]}" \
    --rm \
    --net host \
    -it \
    --name "$CONTAINER_NAME" \
    --entrypoint "" \
    "$IAXL_DEV_DOCKER_IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" >/dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2>/dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation
        exec bash'
