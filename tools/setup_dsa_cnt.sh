#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-${IAXL_DEV_DOCKER_IMAGE:-vllm-iaxl-dev:${IAXL_VERSION:-1.5.1}}}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required but not found." >&2
    exit 1
fi
docker image inspect "$IMAGE" >/dev/null || {
    echo "ERROR: DSA image $IMAGE is not available locally." >&2
    exit 1
}

if [[ ! -d /etc/accel-config ]]; then
    echo "==> Creating /etc/accel-config on host"
    sudo mkdir -p /etc/accel-config
fi

echo "==> Launching ${IMAGE} container to configure DSA"
docker run --rm --privileged \
    --entrypoint "" \
    -v /sys:/sys \
    -v /etc/accel-config:/etc/accel-config \
    -v "${SCRIPT_DIR}:/scripts:ro" \
    "${IMAGE}" \
    bash -c '
    set -euo pipefail
    command -v accel-config >/dev/null || { echo "ERROR: accel-config is missing from the DSA image" >&2; exit 1; }
    exec bash /scripts/setup_dsa.sh "$@"
  ' -- "$@"
