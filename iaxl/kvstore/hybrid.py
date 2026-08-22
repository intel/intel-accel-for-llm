# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""HybridStore: atomic chunk-tier group store for hybrid (GDN) models.

Hybrid models (e.g. Qwen3.5: interleaved full-attention + GDN/mamba
state groups) cannot use the plain KVStore block API: a restorable
prefix is only meaningful when EVERY group's pages exist at the same
token boundary. This module adds the missing group/boundary semantics
on top of the KVFlow engine:

  - a boundary becomes visible ONLY through an atomic manifest commit
    (unique tmp + fsync + os.replace + dir fsync); a crash can only
    leave a stray ``.tmp`` file, never a visible partial manifest;
  - commit is idempotent: a full re-commit overwrites the manifest with
    fresh checksums (overwrite semantics);
  - committed boundaries are content-addressed and NEVER deleted by
    request abort; orphaned pages are reclaimed by offline sweep only;
  - unknown layer/group/spec or malformed checksums are fail-closed:
    refuse commit / refuse hit.

Page payloads live in the KVFlow chunk store (Mem/Record/Storage,
GPU-direct async transfer); this class only owns the schema-4 commit
manifests under the persist dir. The engine's compression is DEFLATE
(lossless); the lossy transforms stay off by default
(IAXL_KV_LOSSY_TRUNC=0), and hybrid callers must not enable them.
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from ..kvflow import KVFlow, Task

# vLLM serving only configures the "vllm" logger (dictConfig,
# propagate=False): a bare iaxl.* namespace drops INFO evidence.
# Prefix with "vllm." so runtime save/load evidence is visible in
# serving containers; in non-vLLM processes (tests, CLI) the records
# still propagate to root.
logger = logging.getLogger("vllm." + __name__)


class PageMissingError(RuntimeError):
    """A required page is absent (fail-closed: refuse load / refuse hit)."""


class PageSizeError(RuntimeError):
    """A page does not match the layout's page_size_bytes (fail-closed)."""


class ChecksumError(RuntimeError):
    """Stored bytes do not match the manifest checksum (fail-closed)."""


@dataclass(frozen=True)
class CacheLayer:
    """Per-layer canonical page description (mirrors LayerPageInfo)."""
    layer_name: str
    group_idx: int
    kind: str                 # "attention" | "mamba" | "sliding_window" | "mla"
    page_size_bytes: int

    @property
    def is_state(self) -> bool:
        return self.kind in ("mamba", "state")


@dataclass(frozen=True)
class CacheGroup:
    """Per-group description (mirrors GroupInfo)."""
    group_idx: int
    kind: str                 # "attention" | "mamba" | "sliding_window" | "mla"
    layer_names: tuple
    page_size_bytes: int


@dataclass(frozen=True)
class CacheLayout:
    """Layer/page/group description for one hybrid cache layout."""
    namespace: str
    tp_size: int
    rank: int
    groups: tuple = ()
    layers: tuple = ()

    @classmethod
    def from_group_infos(cls, namespace: str, tp_size: int, rank: int,
                         group_infos, layer_infos) -> "CacheLayout":
        """Build a CacheLayout from GroupInfo/LayerPageInfo-like objects."""
        groups = tuple(
            CacheGroup(
                group_idx=g.group_idx, kind=g.kind,
                layer_names=tuple(g.layer_names),
                page_size_bytes=g.page_size_bytes,
            ) for g in group_infos)
        layers = tuple(
            CacheLayer(
                layer_name=li.layer_name, group_idx=li.group_idx,
                kind=li.spec_kind, page_size_bytes=li.page_size_bytes,
            ) for li in layer_infos)
        return cls(namespace=namespace, tp_size=tp_size, rank=rank,
                   groups=groups, layers=layers)

    def _layer_map(self) -> Dict[str, CacheLayer]:
        return {l.layer_name: l for l in self.layers}

    def _group_map(self) -> Dict[int, CacheGroup]:
        return {g.group_idx: g for g in self.groups}

    def layer(self, name: str) -> Optional[CacheLayer]:
        return self._layer_map().get(name)

    def require_layer(self, name: str) -> CacheLayer:
        """Fail-closed: unknown layer in a hybrid op refuses to proceed."""
        l = self._layer_map().get(name)
        if l is None:
            raise RuntimeError(
                f"kvshrink layout: unknown layer {name!r} (schema/topology "
                "mismatch); refusing hybrid op")
        return l

    def group(self, idx: int) -> Optional[CacheGroup]:
        return self._group_map().get(idx)

    def require_group(self, idx: int) -> CacheGroup:
        """Fail-closed: unknown group in a hybrid op refuses to proceed."""
        g = self._group_map().get(idx)
        if g is None:
            raise RuntimeError(
                f"kvshrink layout: unknown group {idx!r} (schema/topology "
                "mismatch); refusing hybrid op")
        return g

    def group_layers(self, idx: int) -> Tuple[str, ...]:
        return self.require_group(idx).layer_names


@dataclass(frozen=True)
class CacheBoundary:
    """Identity of one hash boundary for one group (manifest + page set)."""
    namespace: str
    tp_size: int
    rank: int
    block_hash: Any          # int (unit tests) or bytes/str (vLLM)
    group_idx: int

    @property
    def hash_str(self) -> str:
        h = self.block_hash
        if isinstance(h, bytes):
            return h.hex()
        return str(h)

    @property
    def identity(self) -> tuple:
        """Isolation-safe boundary identity (namespace/tp/rank/hash/group)."""
        return (self.namespace, self.tp_size, self.rank, self.hash_str,
                self.group_idx)

    def path_prefix(self, schema: int = 4) -> str:
        """On-disk namespace for this boundary. schema 4 = KVFlow
        chunk-tier manifest (page payloads live in the KVFlow
        Mem/Storage chunk store, only the commit manifest lands here)."""
        tier = {4: "kv4"}.get(schema, "kv4")
        return os.path.join(
            tier, self.namespace, f"tp{self.tp_size}", f"rank{self.rank}",
            f"h{self.hash_str}", f"g{self.group_idx}")

    def __str__(self) -> str:
        return (f"boundary(ns={self.namespace},tp={self.tp_size},"
                f"rank={self.rank},h={self.hash_str[:12]},g={self.group_idx})")


@dataclass
class GroupJob:
    """Barrier job: a group transfer expected to touch ``expected_layers``.

    Completion is judged by counting expected page refs -- never by
    ``layer_names[-1]``. A job whose boundary spans multiple layers only
    reports ``complete`` once every expected layer has been received by
    the store.
    """
    job_id: int
    boundary: CacheBoundary
    expected_layers: Tuple[str, ...]

    @property
    def expected_refs(self) -> int:
        return len(self.expected_layers)


class HybridStore:
    """Atomic chunk-tier group store for hybrid (GDN) models.

    Fail-closed rules:
    - manifest is the commit point: the manifest is durably written
      (atomic tmp+fsync+os.replace+dir fsync) before the in-memory
      manifest is updated, so a crash can never expose a partial
      boundary;
    - ``commit_boundary_chunks`` refuses when any expected layer /
      checksum is missing or invalid; a COMMITTED boundary is
      content-addressed and NEVER deleted;
    - ``is_committed`` exposes only internally-valid COMMITTED
      boundaries whose chunk groups are known to the KVFlow Record
      (fail-closed).

    When ``flow`` is None the store is manifest/metadata-only
    (scheduler-side lookups); a worker store wraps a full KVFlow
    engine. Exactly ONE live process may write a given rank's shard
    under a persist root (single-writer flock lease).
    """

    def __init__(self, layout: CacheLayout, persist_dir: Optional[str] = None,
                 writer_rank: Optional[int] = None,
                 flow: Optional[KVFlow] = None):
        self._layout = layout
        self._persist_dir = persist_dir
        self._lock = threading.RLock()
        self._manifests: Dict[tuple, dict] = {}    # boundary_id -> manifest
        self._writer_lock_fd: Optional[int] = None

        # KVFlow chunk engine. Page bytes are staged through the real
        # Mem/Record/Storage chain (memory -> persist -> evict -> disk
        # hit); persist_dir is used ONLY for the manifest commit point.
        self.flow = flow

        if persist_dir and writer_rank is not None:
            # Single-writer lease: exactly ONE live process may write a
            # given rank's shard under this persist root. A second writer
            # fails CLOSED at startup instead of racing tmp files.
            # Read-only backends (scheduler-side lookups) do not take
            # the lease.
            import fcntl
            os.makedirs(persist_dir, exist_ok=True)
            lock_path = os.path.join(
                persist_dir, f".kvshrink_writer.rank{writer_rank}.lock")
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                raise RuntimeError(
                    f"kvshrink: another writer holds {lock_path}; refusing "
                    f"to start a second writer for the same persist "
                    f"root/rank ({e})")
            self._writer_lock_fd = fd  # held for process lifetime
        if persist_dir:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # topology fingerprint fail-closed
    # ------------------------------------------------------------------

    def _check_boundary(self, boundary: CacheBoundary, op: str) -> None:
        """Refuse any boundary whose namespace / tp_size / rank differ
        from the layout (topology fingerprint fail-closed).

        A boundary key carries the topology it was saved under; storing
        or looking it up through a store whose layout disagrees would
        silently mix shards of different models / TP / rank slices.
        """
        if boundary.namespace != self._layout.namespace:
            raise RuntimeError(
                f"kvshrink topology mismatch ({op}): boundary namespace "
                f"{boundary.namespace!r} != layout namespace "
                f"{self._layout.namespace!r}; refusing {boundary}")
        if boundary.tp_size != self._layout.tp_size:
            raise RuntimeError(
                f"kvshrink topology mismatch ({op}): boundary tp_size "
                f"{boundary.tp_size} != layout tp_size "
                f"{self._layout.tp_size}; refusing {boundary}")
        if boundary.rank != self._layout.rank:
            raise RuntimeError(
                f"kvshrink topology mismatch ({op}): boundary rank "
                f"{boundary.rank} != layout rank {self._layout.rank}; "
                f"refusing {boundary}")

    # ------------------------------------------------------------------
    # KVFlow chunk path -- async GPU-direct page store/load
    # (layer-wise pipelining sunk into the engine)
    # ------------------------------------------------------------------
    #
    # Storage identity (two-level label scheme, no ':' inside parts),
    # aligned with the pure-attention KVStore.put/get scheme:
    #   label      = fixed per group (namespace/tp/rank/group)
    #   chunk_id   = str(block_hash)             (content address of one
    #                block; identical tokens => identical chunk, so
    #                repeated prefixes dedup naturally across requests).
    #                Pages larger than the codec's source-buffer limit
    #                are sub-chunked: chunk_id = f"{hash}#{s}" (see
    #                _split_factor); the factor is uniform within a
    #                group (groups have a single page_size_bytes), so
    #                chunk groups stay coherent across layers/parts.
    #   tensor_key = f"{layer_name}.{part}"      (part: "page" | "k" | "v")
    #
    # ``chunk_indices`` are the GPU block ids (rows of the canonical page
    # views); both sides iterate the op's key list in the same order, so
    # index i pairs with chunk_labels[i] symmetrically. The engine does
    # D2H+zip (put) / unzip+H2D (get) asynchronously on its own streams;
    # callers host-block per layer via wait_layer_*.

    # Codec source-buffer safety margin: the native zip pipeline refuses
    # inputs above IAXL_ZIP_SRC_CAP (default 256 KiB, env-tunable) with a
    # fatal check. The split factor must be a FIXED constant so that the
    # same content always derives the same engine chunk ids regardless of
    # env settings; 128 KiB stays under any plausible cap.
    _SAFE_CHUNK_BYTES = 128 * 1024

    @staticmethod
    def _split_factor(page_size_bytes: int) -> int:
        """Smallest power-of-2 S with page/S <= _SAFE_CHUNK_BYTES and
        page % S == 0. 1 = no sub-chunking (batched engine calls).
        Fail-closed: a page that cannot be evenly split raises (no
        padding games with lossless opaque bytes)."""
        if page_size_bytes <= HybridStore._SAFE_CHUNK_BYTES:
            return 1
        s = 2
        while s <= 4096:
            if page_size_bytes % s == 0 and \
                    page_size_bytes // s <= HybridStore._SAFE_CHUNK_BYTES:
                return s
            s *= 2
        raise RuntimeError(
            f"hybrid chunk tier: page_size {page_size_bytes} cannot be "
            f"evenly sub-chunked below {HybridStore._SAFE_CHUNK_BYTES} "
            "bytes (codec source-buffer limit); refusing to store")

    @staticmethod
    def _expanded_labels(chunk_labels: List[str], split: int) -> List[str]:
        if split == 1:
            return list(chunk_labels)
        return [f"{h}#{s}" for h in chunk_labels for s in range(split)]

    class LayerChunkTasks(NamedTuple):
        """Per-layer put tasks, self-describing for wait_layer_puts.

        parts: tensor_key -> task list. split == 1 (batched): one Task
        whose cpu_tensors align 1:1 with the logical blocks. split > 1
        (sub-chunked): one Task per block, each holding that block's
        split sub-chunk tensors in page order.
        """
        parts: Dict[str, List[Task]]
        n_blocks: int
        split: int

    def _flow_group_label(self, group_idx: int) -> str:
        """Fixed KVFlow label for one group (no ':' so the Storage path
        parser can split full labels at first/last ':' reliably)."""
        self._layout.require_group(group_idx)  # fail-closed
        return (f"kvshr.{self._layout.namespace}.tp{self._layout.tp_size}."
                f"rank{self._layout.rank}.g{group_idx}")

    def submit_layer_puts(
            self, group_idx: int,
            layer_views: Dict[str, Dict[str, torch.Tensor]],
            chunk_indices: List[int],
            chunk_labels: List[str]):
        """Async-store blocks of one group straight from GPU page views.

        ``layer_views`` maps layer_name -> {part_key: (num_blocks,
        page_bytes) int8 GPU view} (canonical page views).
        ``chunk_indices``/``chunk_labels`` pair gpu block ids with their
        content hashes. Pages above the codec source-buffer limit are
        sub-chunked deterministically (per-block calls over the reshaped
        row); smaller pages use one batched call per part.

        Returns ``(tasks, expanded_labels)``: tasks maps layer_name ->
        LayerChunkTasks (see there); ``expanded_labels`` are the real
        engine chunk ids (with ``#s`` suffixes when sub-chunked), in
        canonical page order, for commit_boundary_chunks. Pair with
        wait_layer_puts.
        """
        if self.flow is None:
            raise RuntimeError(
                "submit_layer_puts: KVFlow engine required (chunk path)")
        if not chunk_indices or len(chunk_indices) != len(chunk_labels):
            raise ValueError(
                "submit_layer_puts: chunk_indices/chunk_labels mismatch")
        label = self._flow_group_label(group_idx)
        split = self._split_factor(
            self._layout.require_group(group_idx).page_size_bytes)
        expanded = self._expanded_labels(chunk_labels, split)
        tasks: Dict[str, "HybridStore.LayerChunkTasks"] = {}
        for layer_name, parts in layer_views.items():
            layer = self._layout.require_layer(layer_name)
            if layer.group_idx != group_idx:
                raise RuntimeError(
                    f"submit_layer_puts: layer {layer_name!r} belongs to "
                    f"group {layer.group_idx}, not {group_idx}")
            layer_tasks: Dict[str, List[Task]] = {}
            for pk, view in parts.items():
                tkey = f"{layer_name}.{pk}"
                if split == 1:
                    res = self.flow.put(
                        label=label, tensors={tkey: view}, chunk_dim=0,
                        chunk_indices=chunk_indices,
                        chunk_labels=chunk_labels,
                        description=f"hybrid-put g{group_idx}")
                    layer_tasks[tkey] = [res[tkey]]
                else:
                    # fail-closed: a group page may split evenly in BYTES
                    # while this part's row does not in ELEMENTS (e.g.
                    # odd half-page); .view would fail later with a
                    # cryptic error.
                    if view.shape[1] % split != 0:
                        raise RuntimeError(
                            f"submit_layer_puts: part {tkey} row has "
                            f"{view.shape[1]} elements, not divisible by "
                            f"split factor {split}")
                    sub = view.shape[1] // split
                    per_block: List[Task] = []
                    for blk, h in zip(chunk_indices, chunk_labels):
                        row = view[blk].view(split, sub)
                        res = self.flow.put(
                            label=label, tensors={tkey: row}, chunk_dim=0,
                            chunk_indices=list(range(split)),
                            chunk_labels=[f"{h}#{s}" for s in range(split)],
                            description=f"hybrid-put g{group_idx}")
                        per_block.append(res[tkey])
                    layer_tasks[tkey] = per_block
            tasks[layer_name] = HybridStore.LayerChunkTasks(
                parts=layer_tasks, n_blocks=len(chunk_indices), split=split)
        # record.submit AFTER all layers of this batch are submitted so a
        # chunk group only becomes visible once fully put.
        self.flow.put_finish(label, expanded)
        self.flow.record_flush()
        return tasks, expanded

    def wait_layer_puts(self, tasks: Dict[str, "HybridStore.LayerChunkTasks"]
                        ) -> Dict[str, List[str]]:
        """Host-block until all puts complete; return layer -> per-block
        sha256 list (ordered exactly like the chunk_labels passed to
        submit_layer_puts).

        The checksums are harvested from the engine's own pinned staging
        chunks BEFORE they are released back to the pool, each over the
        canonical page bytes of that block (part keys sorted -- "k"
        before "v"; sub-chunks in page order when the layer was
        sub-chunked), so each entry is exactly the sha256 of the bytes
        that will land on the GPU for that block at load time. This
        preserves the manifest's raw-checksum semantics
        (commit_boundary_chunks) without a separate D2H pass.
        """
        checksums: Dict[str, List[str]] = {}
        flow = self.flow
        for layer_name, lt in tasks.items():
            ordered = [lt.parts[k] for k in sorted(lt.parts)]
            for tl in ordered:
                for t in tl:
                    assert t.ctx is not None
                    t.ctx.zip_wait()
            per_block: List[str] = []
            for i in range(lt.n_blocks):
                h = hashlib.sha256()
                for tl in ordered:
                    if lt.split == 1:
                        h.update(tl[0].cpu_tensors[i].numpy().tobytes())
                    else:
                        for ct in tl[i].cpu_tensors:
                            h.update(ct.numpy().tobytes())
                per_block.append(h.hexdigest())
            for tl in ordered:
                for t in tl:
                    t.ctx = None
                    flow.chunk_pool.release(t.cpu_tensors)
                    t.cpu_tensors = None
            checksums[layer_name] = per_block
        return checksums

    def submit_layer_gets(
            self, group_idx: int,
            layer_views: Dict[str, Dict[str, torch.Tensor]],
            chunk_indices: List[int],
            chunk_labels: List[str]) -> Dict[str, List[Task]]:
        """Async-load blocks of one group straight into GPU page views.

        Mirror of submit_layer_puts (same sub-chunking rules -- the split
        factor is deterministic from the group's page size, so both sides
        derive identical engine chunk ids). Returns layer_name ->
        flattened Task list; pair with wait_layer_gets per layer.
        """
        if self.flow is None:
            raise RuntimeError(
                "submit_layer_gets: KVFlow engine required (chunk path)")
        if not chunk_indices or len(chunk_indices) != len(chunk_labels):
            raise ValueError(
                "submit_layer_gets: chunk_indices/chunk_labels mismatch")
        label = self._flow_group_label(group_idx)
        split = self._split_factor(
            self._layout.require_group(group_idx).page_size_bytes)
        tasks: Dict[str, List[Task]] = {}
        for layer_name, parts in layer_views.items():
            layer = self._layout.require_layer(layer_name)
            if layer.group_idx != group_idx:
                raise RuntimeError(
                    f"submit_layer_gets: layer {layer_name!r} belongs to "
                    f"group {layer.group_idx}, not {group_idx}")
            layer_tasks: List[Task] = []
            for pk, view in parts.items():
                tkey = f"{layer_name}.{pk}"
                if split == 1:
                    res = self.flow.get(
                        label=label, tensors={tkey: view}, chunk_dim=0,
                        chunk_indices=chunk_indices,
                        chunk_labels=chunk_labels,
                        description=f"hybrid-get g{group_idx}")
                    layer_tasks.append(res[tkey])
                else:
                    if view.shape[1] % split != 0:
                        raise RuntimeError(
                            f"submit_layer_gets: part {tkey} row has "
                            f"{view.shape[1]} elements, not divisible by "
                            f"split factor {split}")
                    sub = view.shape[1] // split
                    for blk, h in zip(chunk_indices, chunk_labels):
                        row = view[blk].view(split, sub)
                        res = self.flow.get(
                            label=label, tensors={tkey: row}, chunk_dim=0,
                            chunk_indices=list(range(split)),
                            chunk_labels=[f"{h}#{s}" for s in range(split)],
                            description=f"hybrid-get g{group_idx}")
                        layer_tasks.append(res[tkey])
            tasks[layer_name] = layer_tasks
        return tasks

    def wait_layer_gets(self, layer_tasks: List[Task]) -> None:
        """Host-block until one layer's unzip+H2D has completed.

        Correctness: the host only launches the layer's kernels after
        this returns, so GPU execution order is guaranteed without
        cross-stream events; overlap comes from the engine having
        transferred the OTHER layers concurrently in the background.
        """
        self.flow.get_wait({f"t{i}": t for i, t in enumerate(layer_tasks)})

    def commit_boundary_chunks(self, boundary: CacheBoundary,
                               expected_layers: List[str],
                               checksums: Dict[str, str],
                               chunk_labels: List[str],
                               boundary_tokens: Optional[int] = None
                               ) -> bool:
        """Commit a boundary whose pages went through submit_layer_puts.

        This does NOT re-read and re-hash stored pages: the checksums
        were harvested in wait_layer_puts from the engine's pinned
        staging chunks -- they ARE the bytes zipped into the cache, so
        the TOCTOU re-read guard has nothing left to check.
        ``chunk_labels`` are the content-hash chunk ids of this
        boundary's blocks under the group's KVFlow label; they are
        recorded in the (schema 4, kv4) manifest so is_committed can
        verify presence through the KVFlow Record. Validation stays
        fail-closed on spec (topology / group membership / sha256
        format / layer set); the manifest write is the same atomic
        commit point.
        """
        self._check_boundary(boundary, "commit_boundary_chunks")
        if not chunk_labels:
            raise RuntimeError("commit_boundary_chunks: empty chunk_labels")
        group = self._layout.require_group(boundary.group_idx)
        group_layers = set(group.layer_names)
        if set(expected_layers) != group_layers:
            raise RuntimeError(
                f"commit_boundary_chunks: expected_layers "
                f"{sorted(set(expected_layers))} != group "
                f"{boundary.group_idx} real layers {sorted(group_layers)} "
                "(unknown spec / topology; refusing)")
        if not expected_layers:
            raise RuntimeError("commit_boundary_chunks: empty expected_layers")
        for ln in expected_layers:
            layer = self._layout.require_layer(ln)
            if layer.group_idx != boundary.group_idx:
                raise RuntimeError(
                    f"commit_boundary_chunks: layer {ln!r} group mismatch")
            cs = checksums.get(ln)
            if not self._valid_sha256(cs):
                raise RuntimeError(
                    f"commit_boundary_chunks: invalid or missing sha256 for "
                    f"layer {ln!r} (got {cs!r}); refusing commit")

        info = {
            "schema": 4,
            "tier": "chunks",
            "state": "COMMITTED",
            "namespace": boundary.namespace,
            "tp_size": boundary.tp_size,
            "rank": boundary.rank,
            "hash": boundary.hash_str,
            "group_idx": boundary.group_idx,
            "layers": sorted(expected_layers),
            # AUDIT-ONLY: harvested at save time from the staged bytes,
            # validated for FORMAT at is_committed, never compared
            # against loaded bytes. The load-side integrity gate for the
            # chunk tier is Record presence + engine unzip success.
            "checksums": dict(checksums),
            "chunks": list(chunk_labels),
        }
        if boundary_tokens is not None:
            info["boundary_tokens"] = boundary_tokens

        if self._persist_dir:
            path = self._manifest_path_kv4(boundary)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # manifest is the commit POINT: persist FIRST, update the
            # in-memory manifest only after the durable write succeeded.
            self._atomic_write(path, json.dumps(info, sort_keys=True).encode())
        with self._lock:
            self._manifests[boundary.identity] = info
        return True

    def _chunks_present(self, boundary: CacheBoundary,
                        chunk_labels: List[str]) -> bool:
        """Chunk-tier presence: every chunk group of the boundary is
        known to the KVFlow Record (SQLite; covers both the in-memory
        Mem and the persisted Storage tier -- unpersisted entries are
        cleaned from the DB at startup, so a restart never reports a
        chunk whose bytes are gone). The caller passes the boundary's
        logical block hashes; the engine keys are the sub-chunk-expanded
        forms (deterministic _split_factor of the group's page size)."""
        if self.flow is None:
            return False
        label = self._flow_group_label(boundary.group_idx)
        split = self._split_factor(
            self._layout.require_group(boundary.group_idx).page_size_bytes)
        expanded = self._expanded_labels(list(chunk_labels), split)
        try:
            return all(self.flow.has(label, expanded))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # paths / atomic writes
    # ------------------------------------------------------------------

    def _manifest_path_kv4(self, boundary: CacheBoundary) -> str:
        return os.path.join(
            self._persist_dir, boundary.path_prefix(schema=4), "manifest")

    @staticmethod
    def _atomic_write(path: str, data: bytes) -> None:
        """Unique tmp (mkstemp) + fsync + os.replace + dir fsync.

        The tmp name is UNIQUE per call, so concurrent writers to the
        same target can never consume each other's tmp file. A crash can
        only leave a stray .tmp file (ignored at load), never a
        partially-written visible page or manifest. Directory fsync
        failures PROPAGATE (fail-stop): a rename whose durability cannot
        be proven is not swallowed.
        """
        d = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(
            dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            dfd = os.open(d, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            # only ever remove OUR OWN tmp; on success it is already
            # renamed
            try:
                os.unlink(tmp)
            except OSError:
                pass

    @staticmethod
    def _valid_sha256(value) -> bool:
        return (isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value))

    # ------------------------------------------------------------------
    # disk load (crash remnants are invisible)
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load chunk-tier commit manifests written by a previous
        instance. ``.tmp`` remnants of a crashed atomic write and
        non-schema-4 manifests are explicitly skipped."""
        if not os.path.isdir(self._persist_dir):
            return
        count_m = 0
        for root, _, files in os.walk(self._persist_dir):
            for fn in files:
                if fn.endswith(".tmp"):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "rb") as f:
                        if fn != "manifest":
                            continue
                        info = json.load(f)
                        identity = (info.get("namespace"),
                                    info.get("tp_size"),
                                    info.get("rank"), str(info.get("hash")),
                                    info.get("group_idx"))
                        schema = info.get("schema")
                        if schema != 4:
                            continue  # only kv4 manifests are readable
                        if (info.get("state") == "COMMITTED"
                                and None not in identity):
                            self._manifests[identity] = info
                            count_m += 1
                except Exception:
                    continue  # unreadable remnant: treat as invisible
        logger.info("HybridStore loaded from %s: %d manifests",
                    self._persist_dir, count_m)

    # ------------------------------------------------------------------
    # manifest commit / lookup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the writer lease and stop the KVFlow engine (record
        sync + transfer-backend cleanup). Fail-open: close must never
        mask an in-flight error."""
        if self.flow is not None:
            try:
                self.flow.stop()
            except Exception:  # pragma: no cover - fail-open
                logger.exception("HybridStore close: flow stop failed")
        if self._writer_lock_fd is not None:
            try:
                os.close(self._writer_lock_fd)
            except OSError:  # pragma: no cover - fail-open
                pass
            self._writer_lock_fd = None

    def is_committed(self, boundary: CacheBoundary,
                     expected_layers: Optional[List[str]] = None,
                     expected_boundary_tokens: Optional[int] = None) -> bool:
        """A boundary is HIT iff its chunk-tier manifest is internally
        valid AND every recorded chunk group is known to the KVFlow
        Record (fail-closed presence + integrity)."""
        self._check_boundary(boundary, "is_committed")
        if boundary.identity not in self._manifests and self._persist_dir:
            path = self._manifest_path_kv4(boundary)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        info = json.load(f)
                except Exception:
                    return False
                if (info.get("state") == "COMMITTED"
                        and info.get("hash") == boundary.hash_str
                        and info.get("group_idx") == boundary.group_idx
                        and info.get("schema") == 4):
                    with self._lock:
                        self._manifests[boundary.identity] = info
        info = self._manifests.get(boundary.identity)
        if info is None:
            return False
        if info.get("state") != "COMMITTED":
            return False
        if info.get("schema") != 4:
            return False
        layers = info.get("layers") or []
        if not layers:
            return False
        if expected_layers is not None and \
                sorted(layers) != sorted(expected_layers):
            return False
        checksums = info.get("checksums") or {}
        for ln in layers:
            if not self._valid_sha256(checksums.get(ln)):
                return False
        if expected_boundary_tokens is not None:
            recorded = info.get("boundary_tokens")
            if recorded is None or recorded != expected_boundary_tokens:
                return False
        chunk_labels = info.get("chunks")
        if not chunk_labels:
            return False
        return self._chunks_present(boundary, chunk_labels)

    def orphan_stats(self) -> dict:
        """Observability: crash-remnant .tmp files and on-disk pages
        without a committed sibling manifest. Orphans are reclaimed only
        by offline sweep, never at request_finished time."""
        tmp_files = tmp_bytes = 0
        orphan_pages = orphan_bytes = 0
        if self._persist_dir and os.path.isdir(self._persist_dir):
            manifests = set()
            pages = []
            for root, _, files in os.walk(self._persist_dir):
                for fn in files:
                    path = os.path.join(root, fn)
                    rel = os.path.relpath(path, self._persist_dir)
                    if fn.endswith(".tmp"):
                        tmp_files += 1
                        tmp_bytes += os.path.getsize(path)
                    elif fn == "manifest":
                        manifests.add(rel)
                    else:
                        pages.append((rel, path))
            for rel, path in pages:
                mdir = os.path.dirname(rel)
                if os.path.join(mdir, "manifest") not in manifests:
                    orphan_pages += 1
                    orphan_bytes += os.path.getsize(path)
        return {
            "tmp_files": tmp_files, "tmp_bytes": tmp_bytes,
            "orphan_pages": orphan_pages, "orphan_bytes": orphan_bytes,
        }

    # ------------------------------------------------------------------
    # memory -> persist -> evict -> disk-hit
    # ------------------------------------------------------------------
    #
    # ``persist`` flushes the engine's in-memory Mem groups to its
    # Storage tier; ``evict`` drops the evicted entries from memory
    # (they remain recoverable from Storage + Record -- a subsequent
    # load is a disk hit).

    def persist(self, max_count: int) -> dict:
        if self.flow is not None:
            return self.flow.persist(max_count)
        return {"persisted": 0, "bytes_written": 0, "files": 0,
                "labels": [], "error": "metadata-only store"}

    def evict(self, max_count: int) -> dict:
        if self.flow is not None:
            return self.flow.evict(max_count)
        return {"evicted": 0, "bytes_freed": 0, "labels": [],
                "error": "metadata-only store"}

    def evict_over_watermark(self, high_water: float = 0.8,
                             low_water: float = 0.6) -> dict:
        """Keep the chunk engine's Mem under ``high_water`` of its
        budget by evicting oldest groups down to ``low_water``.

        Only safe after a persist drain: persisted groups lose just
        their memory copy (the Record/storage entries survive and are
        served back from disk on demand); an unpersisted group would be
        removed from the Record (fail-closed MISS). No-op for has-only
        engines.
        """
        flow = self.flow
        if flow is None or getattr(flow, "has_only_mode", False):
            return {"evicted": 0, "bytes_freed": 0, "labels": []}
        budget = getattr(flow, "cache_size_bytes", 0)
        if not budget:
            return {"evicted": 0, "bytes_freed": 0, "labels": []}
        if flow.mem.current_bytes <= int(budget * high_water):
            return {"evicted": 0, "bytes_freed": 0, "labels": []}
        return flow.evict_to_size(int(budget * low_water))
