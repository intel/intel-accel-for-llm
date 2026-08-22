"""Canonical page views over the vLLM GPU KV blocks.

Layout verified on vLLM v0.23.0 + Qwen3.5-4B TP2:

- Attention layer: single tensor; canonical view is built per the
  LayerPageInfo descriptor (block_stride_bytes / storage_offset_bytes).
- Mamba layer: ``kv_caches[layer]`` is a LIST of tensors (conv, ssm) sharing
  one storage. The canonical page view is rebuilt from the first tensor's
  storage (same approach as vLLM's offloading worker).
- Physical page for block_id i = view[i]; with
  ``block_stride_bytes > page_size_bytes`` the layout is packed.
- The block pool is GLOBAL (HMA): block ids are shared across groups; each
  layer simply indexes its own canonical view by block id.

The worker connector exposes these views to the KVFlow chunk engine
(GPU-direct put/get). Durability lives in iaxl.kvstore.HybridStore,
not here.
"""

from __future__ import annotations

import logging

import torch

# log under the vllm.* namespace: vLLM only configures the
# "vllm" logger (handler+level); an unconfigured logger would drop
# INFO evidence lines that the GPU probes grep for.
logger = logging.getLogger("vllm." + __name__)


class Canonicalizer:
    """Builds canonical (num_blocks, page_size_bytes) int8 views per layer."""

    def __init__(self, layer_infos: dict, num_blocks: int):
        """Record per-layer descriptors and the GLOBAL block-pool size
        (shared across groups); views are built by ``register``."""
        self._layer_infos = layer_infos
        self._num_blocks = num_blocks
        self._views: dict[str, torch.Tensor] = {}

    def register(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Build the canonical (num_blocks, page_size_bytes) int8 views
        over the vLLM kv_caches, handling Mamba list-of-tensors storage
        and split-K/V attention layouts. Raises ValueError on any
        descriptor/storage mismatch (fail closed). Called once per layer
        set."""
        for layer_name, info in self._layer_infos.items():
            raw = kv_caches[layer_name]
            if isinstance(raw, (list, tuple)):
                if len(raw) == 0:
                    raise ValueError(
                        f"Mamba layer {layer_name} has empty state list")
                first = raw[0]
                if first.storage_offset() != 0:
                    raise ValueError(
                        f"Mamba layer {layer_name} first state tensor has "
                        f"non-zero storage offset {first.storage_offset()}")
                storage = first.untyped_storage()
                tensor = torch.empty(
                    0, dtype=torch.int8, device=first.device
                ).set_(storage)
            else:
                tensor = torch.empty(
                    0, dtype=torch.int8, device=raw.device
                ).set_(raw.untyped_storage())

            # FlashAttention split-K/V layout fix: a pure
            # attention kv_cache is shaped [2, N, block_size, H, D] whose
            # physical storage is [K block0..N-1][V block0..N-1]. A logical
            # block's K and V are NOT contiguous, so a single flat
            # (num_blocks, page_size) view with stride=page_size strides
            # over TWO ADJACENT K blocks instead of block b's K+V. Detect
            # the physical dim that holds num_blocks (same algorithm as
            # vLLM's offloading worker) and, when K and V are split, keep a
            # (k_view, v_view) pair of (num_blocks, half_page) views; each
            # logical page is then K||V.
            if (not isinstance(raw, (list, tuple))
                    and self._is_split_kv_layout(raw, info)):
                N = info.num_blocks
                half = info.page_size_bytes // 2
                storage = raw.untyped_storage()
                base = torch.empty(
                    0, dtype=torch.int8, device=raw.device).set_(storage)
                flat = base.view(2, N, half)
                k_view, v_view = flat.unbind(0)  # each (num_blocks, half)
                self._views[layer_name] = (k_view, v_view)
                logger.info(
                    "Canonicalized split-K/V layer %s: N=%d half_page=%d",
                    layer_name, N, half)
                continue

            stride = info.block_stride_bytes
            # last page end must fit in storage:
            # offset + stride*(num_blocks-1) + page_size
            needed = (info.storage_offset_bytes
                      + stride * (info.num_blocks - 1)
                      + info.page_size_bytes)
            if needed > storage_size_bytes(raw):
                raise ValueError(
                    f"Layer {layer_name}: descriptor requires {needed} bytes "
                    f"but storage has {storage_size_bytes(raw)}")
            view = torch.as_strided(
                tensor,
                size=(info.num_blocks, info.page_size_bytes),
                stride=(stride, 1),
                storage_offset=info.storage_offset_bytes,
            )
            self._views[layer_name] = view
        logger.info(
            "Canonicalized %d layers, %d blocks, page=%d bytes",
            len(self._views), self._num_blocks,
            next(iter(self._layer_infos.values())).page_size_bytes)

    @staticmethod
    def _is_split_kv_layout(raw: torch.Tensor, info) -> bool:
        """True when num_blocks lives in a non-leading physical dim, i.e.
        the K/V-split [2, N, ...] FlashAttention layout. Mirrors the
        physical-to-logical stride mapping in vLLM's offloading worker."""
        if raw.dim() < 2 or raw.shape[0] != 2:
            return False
        # logical num_blocks dim: find which logical dim equals num_blocks
        strides = raw.stride()
        physical_to_logical = sorted(
            range(len(strides)), key=lambda i: strides[i], reverse=True)
        # the logical dim carrying num_blocks is dim 1 in [2, N, ...]
        try:
            logical_nb_dim = list(raw.shape).index(info.num_blocks)
        except ValueError:
            return False
        physical_pos = physical_to_logical.index(logical_nb_dim)
        return physical_pos != 0

    def _page_parts(self, layer_name: str, block_id: int):
        """Physical tensor(s) holding logical block ``block_id``: a single
        view for contiguous layouts, or (K, V) halves for split layouts."""
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return (v[0][block_id], v[1][block_id])
        return (v[block_id],)

    def page_view_parts(self, layer_name: str):
        """Full-pool canonical page views for the KVFlow chunk engine.

        Returns ``(parts, chunk_dim)``: ``parts`` maps a stable part key to
        a ``(num_blocks, page_bytes)`` int8 GPU view whose dim-0 rows are the
        per-block pages (regular strides, as GpuTransferContext requires);
        ``chunk_dim`` is always 0. Split-K/V attention layers yield
        ``{"k": k_view, "v": v_view}`` (logical page = K||V); contiguous
        layouts (mamba state, packed attention) yield ``{"page": view}``.

        This is the ONLY interface the KVFlow put/get path
        needs -- the engine chunks along dim 0 with
        ``chunk_indices = gpu block ids`` and writes/reads rows directly.
        """
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return {"k": v[0], "v": v[1]}, 0
        return {"page": v}, 0

    def get_page(self, layer_name: str, block_id: int) -> torch.Tensor:
        """Single tensor for one logical page: the canonical view row,
        or a concatenated K||V view for split-K/V layers. Used by the
        read/zero paths."""
        parts = self._page_parts(layer_name, block_id)
        if len(parts) == 1:
            return parts[0]
        # split-K/V: return a concatenated K||V view for read/zero paths
        return torch.cat([p.reshape(-1) for p in parts])

def storage_size_bytes(t: torch.Tensor) -> int:
    """Bytes of the underlying untyped storage for a kv_cache entry;
    Mamba entries (list/tuple of tensors sharing one storage) report
    the first tensor's storage. Used for descriptor bounds checks."""
    if isinstance(t, (list, tuple)):
        return t[0].untyped_storage().size()
    return t.untyped_storage().size()
