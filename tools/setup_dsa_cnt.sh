#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OFFLINE=0
if [[ "${1:-}" == "--offline" ]]; then
    OFFLINE=1
    shift
fi
if (( OFFLINE )); then
    IMAGE="${IMAGE:-${IAXL_DEV_DOCKER_IMAGE:-vllm-iaxl-dev:latest}}"
else
    IMAGE="${IMAGE:-ubuntu:24.04}"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required but not found." >&2
    exit 1
fi
if (( OFFLINE )); then
    docker image inspect "$IMAGE" >/dev/null || {
        echo "ERROR: offline DSA image $IMAGE is not available locally." >&2
        exit 1
    }
fi

if [[ ! -d /etc/accel-config ]]; then
    echo "==> Creating /etc/accel-config on host"
    sudo mkdir -p /etc/accel-config
fi

echo "==> Launching ${IMAGE} container to configure DSA"
docker run --rm --privileged \
    --entrypoint "" \
    -e "IAXL_OFFLINE=$OFFLINE" \
    -e https_proxy \
    -e http_proxy \
    -e no_proxy \
    -v /sys:/sys \
    -v /etc/accel-config:/etc/accel-config \
    -v "${SCRIPT_DIR}:/scripts:ro" \
    "${IMAGE}" \
    bash -c '
    set -euo pipefail
    if [[ "$IAXL_OFFLINE" == 1 ]]; then
        command -v accel-config >/dev/null || { echo "ERROR: accel-config is missing from the offline image" >&2; exit 1; }
    else
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y accel-config
    fi
    exec bash /scripts/setup_dsa.sh "$@"
  ' -- "$@"
