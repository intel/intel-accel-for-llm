# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KVShrink hybrid (GDN) KV cache metadata structures.

These structures mirror the vLLM v0.23.0 HMA (Hybrid Memory Allocator)
KV cache layout:

- ``KVCacheConfig.num_blocks`` is the GLOBAL shared block pool size. All
  KV cache groups share one block id space; each layer has its own block
  table.
- Physical page for (layer, block_id) = ``layer_canonical_view[block_id]``
  where the view is ``(num_blocks, page_size_bytes) int8`` starting at
  storage offset 0.
- Mamba layers expose ``kv_caches[layer_name]`` as a LIST of tensors
  (conv_state, ssm_state) sharing one storage; the canonical page view
  concatenates them (conv at [0, conv_bytes), ssm after).
- Attention layers expose a single tensor; canonical view is
  ``(num_blocks, page_size_bytes)`` int8.

GDN slot contract (v0.23.0): ``preprocess_mamba`` (the prev->curr slot
copy) runs in ``execute_model`` BEFORE the connector's
``bind_connector_metadata``/``start_load_kv``, so every external GDN
snapshot is written directly into the CURR slot during forward
(piggybacked on the preceding attention layer's
``wait_for_layer_load``). There is no "prev" write path and no vLLM
patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = 4


class SchemaMismatchError(ValueError):
    """Raised when a persisted schema version does not equal
    SCHEMA_VERSION: foreign manifests cannot be interpreted safely."""
    pass


def validate_schema(version: int) -> None:
    """Refuse any schema version other than the supported one by raising
    SchemaMismatchError (fail closed: wrong-schema manifests must never
    be read as if they matched)."""
    if version != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"KVShrink schema version mismatch: got {version}, "
            f"expected {SCHEMA_VERSION}")


@dataclass(frozen=True)
class StateRegion:
    """A named region inside a Mamba/GDN state page (validation/stats).

    Pages are opaque bytes for DMA; regions are descriptive only.
    """
    name: str  # "conv" | "ssm"
    offset: int  # byte offset within the page
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class LayerPageInfo:
    """Canonical page info for one layer (as seen by the connector).

    ``block_stride_bytes`` is the byte distance between consecutive
    blocks: equals ``page_size_bytes`` for contiguous layouts, larger
    for packed layouts, and may differ per layer for heterogeneous
    pages.
    """
    layer_name: str
    group_idx: int
    spec_kind: str  # "attention" | "mamba" | "sliding_window" | "mla"
    num_blocks: int  # global block pool size for this layer's view
    page_size_bytes: int
    unpadded_page_size_bytes: int
    block_stride_bytes: int
    storage_offset_bytes: int
    dtype: str
    state_regions: tuple[StateRegion, ...] = ()


@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage spec.

    vLLM buckets layers that share the same storage spec into "KV cache
    groups" (``KVCacheConfig.kv_cache_groups``), each with its own
    independent block pool. A typical hybrid model has two: full-
    attention layers (block-sliced pages, arbitrarily offsettable) and
    GDN/mamba layers (fixed-size recurrent state, whole-snapshot access
    only at segment boundaries).

    What we do with it:

    - Isolation: ``group_idx`` is part of every CacheKey / boundary
      identity, so the same prefix hash in the attention group and the
      mamba group can never alias each other.
    - Bookkeeping: the scheduler tracks per-request block_ids per group
      (each group's block pool is allocated independently).
    - Behavior dispatch: ``kind`` selects the access pattern --
      attention pages are sliceable per ``block_size`` tokens, mamba
      groups are stored/loaded whole at boundaries using
      ``mamba_cache_mode`` / ``mamba_align_size``.
    - Validation: the store fail-closed checks the group exists and
      page sizes match before any chunk move.
    """

    group_idx: int
    kind: str  # "attention" | "mamba" | "sliding_window" | "mla"
    layer_names: tuple[str, ...]
    block_size: int  # tokens per block for this group
    page_size_bytes: int
    mamba_cache_mode: Optional[str]  # None for attention groups
    mamba_align_size: Optional[int]  # offload chunk alignment for mamba


@dataclass(frozen=True)
class CacheKey:
    """Logical key for one page (or one boundary manifest).

    The manifest for boundary ``block_hash`` lives at the key with
    ``layer_name == ""``. Full isolation requires namespace, tp, rank,
    hash and group to be part of the identity.
    """
    namespace: str
    tp_size: int
    rank: int
    block_hash: object  # int (unit tests) or bytes/str (vLLM)
    group_idx: int
    layer_name: str  # "" for the boundary manifest key

    @property
    def hash_str(self) -> str:
        """Stable string form for paths / JSON (bytes -> hex)."""
        h = self.block_hash
        if isinstance(h, bytes):
            return h.hex()
        return str(h)

    @property
    def boundary_key(self) -> tuple[str, int, int, str, int]:
        """Isolation-safe identity for a boundary (namespace/tp/rank/hash/group)."""
        return (self.namespace, self.tp_size, self.rank, self.hash_str,
                self.group_idx)


def make_boundary_key(namespace: str, tp_size: int, rank: int,
                      group_idx: int, block_hash) -> "CacheKey":
    """Boundary-manifest key for one group (shared by the scheduler's
    load/save builders and the hit policy; layer_name="" identity)."""
    return CacheKey(
        namespace=namespace, tp_size=tp_size, rank=rank,
        block_hash=block_hash, group_idx=group_idx, layer_name="")


@dataclass
class GroupTransferMeta:
    """Per-group transfer instructions for one request (one step).

    The worker receives ONLY this metadata -- it never sees the
    scheduler's bookkeeping. So each op must fully describe one data
    movement: WHICH group it belongs to, WHERE the data lives in the
    external store, and WHICH GPU blocks are involved.

    Fields:
    - group_idx: which KV cache group this op targets. Each group has
      its own independent GPU block pool and storage rules, so the
      worker must know the group to interpret gpu_block_ids and keys.
      The group's kind (attention/mamba) is deliberately NOT duplicated
      here: the worker derives it from its own registered GroupInfo
      (``self._groups[op.group_idx].kind``), keeping a single source of
      truth.
    - keys / gpu_block_ids: parallel tuples pairing store address with
      GPU destination -- keys[i] is the external-store identity (which
      chunks to read or write), gpu_block_ids[i] is the GPU block the
      page is loaded into (LOAD) or drained from (SAVE).
    - snapshot_boundary_tokens: for mamba ops, the token position this
      snapshot represents. Written into the commit manifest so a later
      lookup can prove the snapshot covers exactly the boundary a
      request needs.

    GDN loads always target the CURR state slot (see module docstring);
    there is no slot field because there is no choice to make.
    """
    group_idx: int
    keys: tuple[CacheKey, ...] = ()
    gpu_block_ids: tuple[int, ...] = ()
    snapshot_boundary_tokens: Optional[int] = None  # mamba restore point


@dataclass
class ReqMeta:
    """All transfer instructions for one request in one step.

    The unit the worker iterates over: for each ReqMeta it executes
    every GroupTransferMeta (loads before forward, saves after).

    Fields:
    - req_id: which request this plan belongs to. The worker matches it
      against the requests it is about to run and reports completion
      per request id through get_finished.
    - external_hit_tokens: how many tokens the core accepted as
      externally backed for this request. Used for evidence/metrics and
      sanity checks (a LOAD plan with accepted external tokens but zero
      ops is the fail-closed case).
    - group_ops: one GroupTransferMeta per KV cache group. Requests on
      hybrid models always have per-group plans: attention blocks and
      the mamba snapshot move independently.
    """
    req_id: str
    external_hit_tokens: int = 0
    group_ops: tuple[GroupTransferMeta, ...] = ()


@dataclass
class RequestGroupState:
    """Per-group mutable state for one request (scheduler side).

    - block_ids: our copy of vLLM's block table for this group --
      block ids are indices into the group's GPU block pool. Kept in
      sync via update_state_after_alloc (full replace, for new/resumed
      requests) and the scheduled_cached_reqs.new_block_ids append in
      build_connector_meta (for running requests). See the class
      docstring of HybridRequestScheduler for the two sync channels and
      their ordering.
    - next_stored_chunk_idx: incremental-save cursor. Block indices
      below it were already emitted in earlier save plans; on
      preemption resume (or any progress regression) it rolls back so
      blocks whose saves may never have landed are emitted again.
    """
    block_ids: list[int] = field(default_factory=list)
    next_stored_chunk_idx: int = 0


@dataclass
class RequestState:
    request: Any = None
    block_hashes: list[int] = field(default_factory=list)
    num_locally_computed_tokens: int = 0
    snapshot_boundary: int = 0
    groups: tuple[RequestGroupState, ...] = ()
    # External tokens accepted by the core in the CURRENT scheduling
    # pass (recorded by update_state_after_alloc): tokens the core will
    # skip recompute for, so the worker MUST restore them before
    # forward. Consumed by build_resumed_load_meta's fail-closed guard:
    # a resumed request with pending external tokens but no restorable
    # pages must fail-stop, never enter forward reading unrestored KV.
    pending_load_tokens: int = 0
    # Last authoritative progress seen by the save path
    # (num_computed + scheduled of the last save plan). Used for
    # fail-closed regression detection: any drop below this value rolls
    # save cursors back even if the resumed flag is missing.
    last_known_progress: int = 0
