#!/bin/bash -e
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Package a standalone, relocatable release bundle for the remote (NIXL)
# cache daemon. Run this AFTER `pip install -e .` has succeeded (inside
# start.sh's dev container, or any environment where `iaxl/torch_ext*.so`
# and `_lib/*.so` were just built).
#
# What gets bundled and why it is enough to run elsewhere without a build
# toolchain: the compiled iaxl/torch_ext*.so's INSTALL_RPATH is
# "${IAXL_LIB_DIR}:$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib" (see CMakeLists.txt),
# so placing the 3 libraries iaxl itself builds (libqat_s.so,
# libusdm_drv_s.so, libqpl.so -- Intel QAT/QPL user-space libs, NOT part of
# any stock vLLM image) at pysrc/iaxl/lib/ next to the extension resolves
# them via $ORIGIN/lib with zero extra configuration. Every OTHER shared
# library torch_ext links against (torch, torch_python, CUDA runtime libs,
# libgdrapi) is already present in the vllm/vllm-openai:v0.23.0-based image
# used on the GPU node -- see the design doc's "why the same docker image"
# reasoning (requirement B). This bundle is therefore only meant to run
# inside a container started from that SAME base image (or the exact image
# used to build it); it is not a portable CPU-only build like an older,
# superseded implementation of this feature had to produce.
#
# Output: dist/iaxl-remote-daemon-<version>.tar.gz (default version: v0.1;
#         override with RELEASE_VERSION=...)
#
# Usage:
#   tools/remote_daemon/build_release.sh
#   scp dist/iaxl-remote-daemon-v0.1.tar.gz root@<remote-storage-node>:/root/
#   ssh root@<remote-storage-node>
#   tar xzf iaxl-remote-daemon-v0.1.tar.gz && cd iaxl-remote-daemon-v0.1
#   CONTROL_PORT=19000 ... tools/remote_daemon/docker-run-daemon.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

SO=$(ls iaxl/torch_ext*.so 2>/dev/null | head -1 || true)
if [[ -z "$SO" ]]; then
    echo "ERROR: iaxl/torch_ext*.so not found. Run 'pip install -e . --no-build-isolation'" \
         "(e.g. via ./start.sh) before building the release package." >&2
    exit 1
fi

# Bundle version tag. Override with RELEASE_VERSION=... to stamp a specific
# release (e.g. v0.2, or a git sha for a hotfix). Default is a stable
# marketing version instead of the git sha so operators can `scp` the same
# filename across nodes without rewriting scripts on every rebuild.
VERSION="${RELEASE_VERSION:-v0.1}"
NAME="iaxl-remote-daemon-$VERSION"
OUT="dist/$NAME"
rm -rf "$OUT"
mkdir -p "$OUT/pysrc" "$OUT/tools/remote_daemon"

# Whole compiled python package, minus C++ sources (not needed at runtime)
# and bytecode caches.
if command -v rsync >/dev/null; then
    rsync -a --exclude 'csrc' --exclude '__pycache__' --exclude '*.pyc' iaxl/ "$OUT/pysrc/iaxl/"
else
    mkdir -p "$OUT/pysrc/iaxl"
    (cd iaxl && tar cf - --exclude csrc --exclude __pycache__ .) | (cd "$OUT/pysrc/iaxl" && tar xf -)
fi

mkdir -p "$OUT/pysrc/iaxl/lib"
bundled=0
for lib in libqat_s.so libusdm_drv_s.so libqpl.so; do
    if [[ -f "_lib/$lib" ]]; then
        # cp -a preserves the full SONAME symlink chain (e.g.
        # libqpl.so -> libqpl.so.1 -> libqpl.so.1.9.0); torch_ext.so's
        # DT_NEEDED entry is the SONAME (libqpl.so.1), not the unversioned
        # name, so bundling only the unversioned symlink target would make
        # dlopen fail with "libqpl.so.1: cannot open shared object file".
        cp -a "_lib/$lib"* "$OUT/pysrc/iaxl/lib/"
        bundled=$((bundled + 1))
    else
        echo "WARNING: _lib/$lib not found; the daemon will fail to import" \
             "torch_ext unless this library is otherwise on the target's linker path." >&2
    fi
done
echo "bundled $bundled/3 iaxl-built runtime libraries into pysrc/iaxl/lib/"

cp tools/auto_config.sh "$OUT/tools/"
cp tools/remote_daemon/run-daemon.sh \
   tools/remote_daemon/run-daemon-multi.sh \
   tools/remote_daemon/docker-run-daemon.sh \
   tools/remote_daemon/remote-cli.sh \
   tools/remote_daemon/README.md \
   "$OUT/tools/remote_daemon/"

# Runtime-only Python deps NOT provided by the vllm/vllm-openai base image
# (numpy/nvtx/psutil/torch already come with it; xxhash does not). Kept
# separate from the repo's requirements.txt so the release bundle does not
# drag in build-time deps (cmake/pybind11/...) or dev deps (matplotlib) on
# the remote node. run-daemon.sh pip-installs this before starting the
# daemon (idempotent no-op once satisfied).
#
# nixl>=1.4: vllm/vllm-openai:v0.23.0 ships nixl 1.2.0, whose bundled UCX
# does not enable the dma-buf CUDA memory path, so registering GPU KV cache
# with the local NIXL agent falls back to the legacy nvidia_peermem path
# and fails on inbox-kernel hosts with
#   ib_md.c: ibv_reg_mr(...) failed: Bad address
#   ucp_mm.c: failed to register address ... (cuda) on md[N]=mlx5_x:
#            Input/output error (md supports: host)
# 1.4+ registers CUDA via dma-buf and needs no peer-memory kernel module.
cat > "$OUT/requirements-runtime.txt" <<'EOF'
xxhash
nixl>=1.4
EOF

echo "$VERSION" > "$OUT/VERSION"
cat > "$OUT/README.md" <<EOF
# iaxl remote (NIXL) cache daemon -- release bundle $VERSION

Standalone bundle for a remote storage node: the compiled \`iaxl\` python
package (including \`torch_ext*.so\` and its bundled QAT/QPL runtime
libraries) plus the daemon launcher scripts. No build toolchain is required
to run it -- see \`../../doc/usage/remote-nixl-kv-cache.md\` ("release
package" section) for the full deployment walkthrough. Quick start:

\`\`\`bash
CONTROL_PORT=19000 TRANSPORT=nixl NIXL_DEVICE=mlx5_1:1 NIXL_HOST=<this-host> \\
POOL_SIZE_GB=64 STAGING_SLOTS=256 STAGING_SLOT_MB=8 \\
tools/remote_daemon/docker-run-daemon.sh
\`\`\`

This must run inside a container based on the SAME image as the GPU node
(default \`vllm/vllm-openai:v0.23.0\`, override via \`IAXL_BASE_DOCKER_IMAGE\`);
\`docker-run-daemon.sh\` starts that container for you. Built from git commit
\`$VERSION\`.
EOF

# Best-effort sanity check: list the extension's NEEDED shared libraries so
# an operator can compare against what the target image/container provides.
if command -v readelf >/dev/null; then
    echo "torch_ext NEEDED libraries (verify these resolve on the target node/image):"
    readelf -d "$SO" | awk '/\(NEEDED\)/ {print "  " $5}' | tr -d '[]'
fi

mkdir -p dist
tar czf "dist/$NAME.tar.gz" -C dist "$NAME"
echo "release package: dist/$NAME.tar.gz"
