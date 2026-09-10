# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Data-plane interface shared by the TCP and NIXL backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ShardRef:
    """One compressible unit: a (block, layer, tensor_key) shard.

    ``tensor`` is the full per-layer KV cache tensor; ``offset``/``length``
    are the byte range within its contiguous storage that holds this shard.
    ``key`` is the session-relative part of a shard identifier
    (``block_hash|layer_id|tensor_key``); the daemon prepends the session's
    model/tp prefix and maps it to the native ``iaxl.torch_ext.Mem`` key (see
    ``iaxl.remote.metadata.full_chunk_label``).
    """

    key: str
    tensor: torch.Tensor
    offset: int
    length: int
    dtype: torch.dtype


def dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def str_to_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def tensor_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a 1-D uint8 view sharing storage with a contiguous tensor."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.reshape(-1).view(torch.uint8)


class DataPlane:
    """Abstract data plane. Backends move shard bytes for put/get."""

    name = "base"

    def start(self) -> None:
        pass

    def put(self, session_id: str, shards: List[ShardRef]) -> None:
        raise NotImplementedError

    def get(self, session_id: str, shards: List[ShardRef]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_data_plane(config, session_id: str, staging_slots: int = 0,
                    tp_size: int = 1) -> "DataPlane":
    from ..config import TRANSPORT_NIXL, TRANSPORT_TCP

    if config.transport == TRANSPORT_TCP:
        from .tcp_backend import TcpDataPlane

        return TcpDataPlane(config, session_id)
    if config.transport == TRANSPORT_NIXL:
        from .nixl_backend import NixlDataPlane

        return NixlDataPlane(config, session_id, staging_slots=staging_slots,
                             tp_size=tp_size)
    raise ValueError(f"unknown transport: {config.transport!r}")
