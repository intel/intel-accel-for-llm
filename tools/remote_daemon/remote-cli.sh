#!/bin/bash -e
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper around `python -m iaxl.remote.admin`.
#
#   tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 status
#   tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 persist --count 32
#   tools/remote_daemon/remote-cli.sh --daemon 10.10.10.10:19000 evict --count 32 \
#       --model Qwen2.5-32B-Instruct --tp-size 4 --tp-rank 0
#
# The wrapper only handles PYTHONPATH so it works both from a normal repo
# checkout (`pip install -e .` already done) and from a release bundle
# produced by `build_release.sh` (which ships iaxl under pysrc/), matching
# run-daemon.sh's behaviour. No RDMA / QAT / IAA environment is needed --
# admin RPCs are pure control-plane traffic over TCP.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

if [[ -d "$REPO_DIR/pysrc/iaxl" ]]; then
    export PYTHONPATH="$REPO_DIR/pysrc${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$REPO_DIR/pysrc/iaxl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

: "${PYTHON:=python3}"
exec "$PYTHON" -m iaxl.remote.admin "$@"
