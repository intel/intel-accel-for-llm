# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Cross-node KV cache backend (NIXL/RDMA data plane + TCP control plane).

This package lets a vLLM deployment offload KVShrink's KV cache pool to a
separate "remote cache daemon" process running on another node, instead of
(or as a fallback for) the local, GPU-attached ``iaxl.kvstore.KVStore``. The
daemon reuses the *same* native pool/compression/persistence code as the
local path (``iaxl.torch_ext``'s ``Mem`` / ``Storage`` / ``Record`` and the
QAT/IAA/CPU zip pipeline) -- see ``server.py`` -- so the two backends share
identical on-disk chunk semantics; only the transport differs.

See ``doc/design/remote-nixl-kv-cache.md`` for the full design and
``doc/usage/remote-nixl-kv-cache.md`` for deployment instructions.
"""

from .config import RemoteCacheConfig
from .remote_kvstore import RemoteKVStore

__all__ = ["RemoteCacheConfig", "RemoteKVStore"]
