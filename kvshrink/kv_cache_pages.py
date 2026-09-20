# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bind vLLM KV cache tensors to logical pages for the connector.

This is connector-side housekeeping: the store receives tensors whose
dimension 0 is already the logical KV block, so it needs no knowledge of
vLLM's cache layout. The geometry comes from `KVCacheConfig` -- the block
count and the group's page size -- never from the tensors, matching the
source vLLM's own connectors read (`offloading/worker.py`,
`nixl/worker.py`).
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .kvshrink_connector import GroupInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageLayout:
    """Model-level page geometry plus the per-layer cache kinds.

    vLLM gives every KV cache group the same page size and the pool a single
    block count, so only the kind varies per layer.
    """

    num_blocks: int
    page_bytes: int
    kinds: Mapping[str, str]  # layer name -> "attention" | "mamba"


def opaque_pages(
    tensor: torch.Tensor, num_blocks: int, page_bytes: int
) -> torch.Tensor:
    """One opaque byte page per block, over the layer's whole storage."""
    base = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage()
    )
    return base.view(num_blocks, page_bytes)


def bind_kv_caches(
    kv_caches: Dict[str, torch.Tensor | list],
    groups: Iterable["GroupInfo"],
    num_blocks: int,
    page_bytes: int,
) -> Tuple[Dict[str, torch.Tensor], PageLayout]:
    """Order and bind vLLM's KV caches for the connector.

    Mamba layers are placed first: the leading window selected by the async
    config must contain every mamba layer (they have no per-layer load hook),
    and only attention layers are waited on demand. Order within each kind is
    preserved. Returns the bound views and the `PageLayout` used, whose `kinds`
    keys are in the new order.
    """
    kinds = {ln: group.kind for group in groups
             for ln in group.layer_names if ln in kv_caches}
    ordered = [ln for ln in kv_caches if kinds.get(ln) == "mamba"]
    ordered += [ln for ln in kv_caches if kinds.get(ln) != "mamba"]
    kv_caches = {ln: kv_caches[ln] for ln in ordered}
    layout = PageLayout(
        num_blocks=num_blocks,
        page_bytes=page_bytes,
        kinds={ln: kinds.get(ln, "attention") for ln in ordered},
    )
    return bind_pages(kv_caches, layout), layout


def bind_pages(
    kv_caches: Optional[Dict[str, torch.Tensor | list]],
    layout: Optional[PageLayout] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Give every layer a view whose dimension 0 is the logical KV block.

    Mamba keeps both states of a page in one backing allocation, so it binds
    as one opaque page per block; attention re-views dim 0 along the logical
    page, keeping the trailing dims (a logical page spans several kernel
    blocks on a hybrid model). Both are plain views: nothing is copied, and
    the scheduler's block IDs index them.
    """
    if kv_caches is None:
        return None
    pools: Dict[str, torch.Tensor] = {}
    for layer_name, entry in kv_caches.items():
        kind = layout.kinds.get(layer_name) if layout is not None else None
        if kind == "mamba":
            tensor = entry[0] if isinstance(entry, (list, tuple)) else entry
            pools[layer_name] = opaque_pages(
                tensor, layout.num_blocks, layout.page_bytes)
        elif kind == "attention" and entry.shape[0] != layout.num_blocks:
            ratio = entry.shape[0] // layout.num_blocks
            pools[layer_name] = entry.view(
                layout.num_blocks, ratio, *entry.shape[1:])
        else:
            pools[layer_name] = entry
    return pools
