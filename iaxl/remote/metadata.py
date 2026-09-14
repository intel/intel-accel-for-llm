# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Remote metadata keys and block descriptor math.

A remote cache entry is uniquely identified across a multi-GPU (TP)
deployment by six fields:

    model_name, tp_size, tp_rank, block_hash, layer_id, tensor_key

For non-MLA layouts each block splits into K and V shards (``tensor_key`` "k"
and "v", so RDMA can address the two non-contiguous outer-dimension ranges
separately); MLA and fused layouts use a single ``tensor_key`` "kv". This
mirrors ``iaxl.kvshrink``'s own ``block_dim``/K-V-split convention (see
``kvshrink_connector.register_kv_caches``).

The descriptor helpers translate a (tensor shape, block_dim, block_index)
into the contiguous byte ranges that hold that block, independent of the
vLLM version or attention backend. This is what both the NIXL data plane (to
build transfer descriptors) and the TCP fallback (to slice bytes) rely on.

``full_chunk_label`` builds the native ``iaxl.torch_ext.Mem`` key for one
(block, layer, k/v) shard on the daemon side, using exactly the same
``label:chunk_id:tensor_key`` convention as the local, GPU-attached
``iaxl.kvstore.KVStore`` (see ``iaxl/csrc/include/kv_pool.h``), so a block's
shards for every layer share one on-disk/DDR "group" (one LRU/persist unit)
just like the local path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import List, Sequence, Tuple

# Matches iaxl.kvstore.KVStore.LABEL -- kept identical so the remote and local
# pools use the same chunk-label vocabulary (useful if a chunk store is ever
# copied between the two, and just for consistency).
LABEL = "kv"


@dataclass(frozen=True)
class RemoteKey:
    model_name: str
    tp_size: int
    tp_rank: int
    block_hash: str
    layer_id: str
    tensor_key: str

    def encode(self) -> str:
        # '|' is safe: block hashes are decimal/hex strings, layer ids are
        # plain names without '|'.
        return (
            f"{self.model_name}|{self.tp_size}|{self.tp_rank}|"
            f"{self.block_hash}|{self.layer_id}|{self.tensor_key}"
        )

    @staticmethod
    def decode(s: str) -> "RemoteKey":
        model, tp, rank, bh, layer, tk = s.split("|", 5)
        return RemoteKey(model, int(tp), int(rank), bh, layer, tk)


def block_descriptors(
    shape: Sequence[int],
    block_dim: int,
    elem_size: int,
    block_index: int,
) -> List[Tuple[int, int]]:
    """Return ``[(offset_bytes, length_bytes), ...]`` for one block.

    Assumes a standard C-contiguous tensor. Dimensions outer to ``block_dim``
    each yield one contiguous range (their strides interleave the block
    data); dimensions inner to ``block_dim`` are contiguous within a range.

      block_dim == 0                  -> 1 range per block
      [K/V=2, num_blocks, ...] dim 1  -> 2 ranges per block (K and V)
    """
    shape = list(shape)
    inner_elems = prod(shape[block_dim + 1:]) if block_dim + 1 < len(shape) else 1
    outer_count = prod(shape[:block_dim]) if block_dim > 0 else 1
    dim_size = shape[block_dim]

    span = inner_elems * elem_size
    ranges: List[Tuple[int, int]] = []
    for o in range(outer_count):
        base_elems = o * dim_size * inner_elems + block_index * inner_elems
        ranges.append((base_elems * elem_size, span))
    return ranges


def block_nbytes(shape: Sequence[int], block_dim: int, elem_size: int) -> int:
    """Total bytes occupied by a single block across all its descriptors."""
    shape = list(shape)
    inner_elems = prod(shape[block_dim + 1:]) if block_dim + 1 < len(shape) else 1
    outer_count = prod(shape[:block_dim]) if block_dim > 0 else 1
    return outer_count * inner_elems * elem_size


def tensor_keys_for_layout(shape: Sequence[int], block_dim: int, use_mla: bool) -> List[str]:
    """Names of the shards a single (block, layer) splits into on the wire.

    Non-MLA layouts with a leading K/V dimension (block_dim == 1) expose the
    two outer ranges as separate ``k``/``v`` shards so the daemon can
    compress, store and RDMA-address them independently. All other layouts
    use a single ``kv`` shard.
    """
    if not use_mla and block_dim == 1 and len(shape) > 0 and shape[0] == 2:
        return ["k", "v"]
    return ["kv"]


def native_tensor_key(layer_id: str, wire_tensor_key: str) -> str:
    """Map a wire ``tensor_key`` ("k"/"v"/"kv") + layer id to a native Mem
    tensor_key component.

    Locally, ``iaxl.kvflow.KVFlow`` uses the *layer name* as its tensor_key
    (K and V of one layer are transferred and compressed together, since the
    GPU->CPU copy already reassembles them contiguously). The remote path has
    to move K and V as separate RDMA descriptors, so it stores them as two
    sibling entries within the same block's native cache "group"
    (``kv:{block_hash}``) using ``{layer_id}`` (kv/mla) or ``{layer_id}.k`` /
    ``{layer_id}.v`` as the native tensor_key -- '.' is a legal label
    character while ':' and '/' are not (see ``kv_pool.h::validate_label_
    component``).
    """
    if wire_tensor_key == "kv":
        return layer_id
    return f"{layer_id}.{wire_tensor_key}"


def full_chunk_label(block_hash: str, layer_id: str, wire_tensor_key: str) -> str:
    """Native ``iaxl.torch_ext.Mem`` key for one (block, layer, k/v) shard."""
    return f"{LABEL}:{block_hash}:{native_tensor_key(layer_id, wire_tensor_key)}"
