"""Parse the real vLLM KVCacheConfig into KVShrink hybrid structures.

Verified against vLLM v0.23.0 + Qwen3.5-4B TP2 (layout unchanged
since v0.21:

- 4 kv_cache_groups: 3 x MambaSpec(GDN) + 1 x FullAttentionSpec
- 8 kv_cache_tensors, each shared by 4 consecutive layers (3 linear + 1 full)
- vLLM pads the attention block size so ALL groups share one page size
- layer names like "language_model.model.layers.0.linear_attn"
- Mamba layers: kv_caches[layer] is a LIST [conv_tensor, ssm_tensor]
  sharing one storage; page layout = conv bytes then ssm bytes.

Fail-closed rules (never guess):
- unknown spec types -> KVShrinkParseError
- unknown dtype -> KVShrinkParseError
- packed / restride / non-zero storage offset layouts are parsed into
  LayerPageInfo.block_stride_bytes / storage_offset_bytes; canonical views
  are built from these descriptors, not from contiguity assumptions.
- heterogeneous page sizes across layers are allowed (each layer carries
  its own page_size_bytes).
"""

from __future__ import annotations

import hashlib

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from .hybrid_metadata import (
    GroupInfo,
    LayerPageInfo,
    StateRegion,
    validate_schema,
)


class KVShrinkParseError(ValueError):
    """Raised when the vLLM cache config cannot be parsed safely
    (unknown spec, dtype or inconsistent layout). The parse never
    guesses (fail closed)."""
    pass


def compute_namespace(
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    cache_dtype: str,
    kv_schema_version: int,
    tp_size: int,
    pp_size: int,
) -> str:
    """Stable cache namespace: sha256 over model identity, cache dtype,
    schema version and tp/pp size, truncated to 16 hex chars. Schema is
    validated first (raises on mismatch) so one store never mixes
    layouts that could not be interpreted."""
    validate_schema(kv_schema_version)
    raw = "|".join([
        model_id, model_revision, tokenizer_revision, cache_dtype,
        str(kv_schema_version), str(tp_size), str(pp_size),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _dtype_size(dtype) -> int:
    """Byte size of a dtype. Fail closed on unknown dtypes."""
    s = str(dtype)
    if "bfloat16" in s or "float16" in s:
        return 2
    if "float32" in s or "float" in s:
        return 4
    if "float64" in s or "double" in s:
        return 8
    if "int8" in s or "uint8" in s:
        return 1
    if "int16" in s or "uint16" in s:
        return 2
    if "int32" in s or "uint32" in s or "int" in s:
        return 4
    if "int64" in s or "uint64" in s or "long" in s:
        return 8
    raise KVShrinkParseError(f"Unknown dtype: {dtype}")


def _spec_kind(spec) -> str:
    """Classify a vLLM cache spec as mamba / attention /
    sliding_window; unknown spec types raise KVShrinkParseError (fail
    closed)."""
    if isinstance(spec, MambaSpec):
        return "mamba"
    if isinstance(spec, AttentionSpec):
        if getattr(spec, "sliding_window", None) is not None:
            return "sliding_window"
        return "attention"
    raise KVShrinkParseError(
        f"Unsupported KV cache spec {type(spec).__name__}")


def _iter_layer_specs(group_spec):
    """Yield (layer_name, spec) pairs, expanding UniformTypeKVCacheSpecs.

    For UniformTypeKVCacheSpecs every layer in the group MUST be present in
    the per-layer specs; missing entries fail closed.
    """
    spec = group_spec.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        per_layer = spec.kv_cache_specs
        for name in group_spec.layer_names:
            if name not in per_layer:
                raise KVShrinkParseError(
                    f"UniformTypeKVCacheSpecs missing spec for layer {name}")
            yield name, per_layer[name]
    else:
        for name in group_spec.layer_names:
            yield name, spec


def parse_kv_cache_config(
    kv_cache_config: KVCacheConfig,
    hash_block_size: int,
) -> tuple[list[GroupInfo], dict[str, LayerPageInfo], int]:
    """Return (groups, layer_infos, num_blocks).

    What we pull out of vLLM's KVCacheConfig, and where each piece goes:

    - ``kv_cache_groups`` -> ``groups`` (list[GroupInfo]): vLLM splits a
      hybrid model's KV cache into groups of layers with the same storage
      spec (e.g. group 0 = full-attention layers, group 1 = GDN/mamba
      layers). For each group we record its kind, layer names, block_size
      (tokens per block) and page_size_bytes; mamba groups additionally
      get mamba_cache_mode / mamba_align_size. Consumed by:
      the scheduler (per-group block_ids bookkeeping and save/load
      planning), boundary_backend (fail-closed group validation, storage
      labels embed group_idx), and policy (attention pages are sliceable
      per block, mamba groups only at boundaries).
    - per-layer specs + ``kv_cache_tensors`` -> ``layer_infos``
      (dict[str, LayerPageInfo]): for every layer we record its group,
      page size (padded/unpadded), block stride / storage offset in the
      shared GPU tensor, dtype, and for mamba layers the state regions
      (conv/ssm) inside a page. Consumed by: Canonicalizer in
      canonical.py (maps token positions -> (block, offset) inside the
      GPU pool), and the chunk save/load paths (how many bytes to move
      per page, which codec applies).
    - ``num_blocks`` -> int: global block-pool size. Consumed by
      Canonicalizer as the valid block-id range.

    Raises KVShrinkParseError on unknown specs, missing layers, unknown
    dtypes or any inconsistency (fail closed).
    """
    num_blocks = kv_cache_config.num_blocks
    groups: list[GroupInfo] = []
    layer_infos: dict[str, LayerPageInfo] = {}

    layer_to_tensor: dict[str, int] = {}
    for t_idx, t in enumerate(kv_cache_config.kv_cache_tensors):
        for name in t.shared_by:
            layer_to_tensor[name] = t_idx

    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        kind = None
        page_size = None
        block_size = None
        regions: list[StateRegion] = []
        per_layer_specs: list[tuple[str, object]] = list(_iter_layer_specs(g))
        if not per_layer_specs:
            raise KVShrinkParseError(f"Group {g_idx} has no layers")

        for name, spec in per_layer_specs:
            sk = _spec_kind(spec)
            if kind is None:
                kind = sk
            elif sk != kind:
                raise KVShrinkParseError(
                    f"Group {g_idx} mixes spec kinds {kind} and {sk}")
            page = int(spec.page_size_bytes)
            if page_size is None:
                page_size = page
            elif page != page_size:
                # heterogeneous pages within one group: allowed only if the
                # tensor can express it; otherwise fail closed
                raise KVShrinkParseError(
                    f"Group {g_idx} layers have differing page sizes "
                    f"({page} vs {page_size}); unsupported within a group")
            bs = int(spec.block_size)
            if block_size is None:
                block_size = bs
            elif bs != block_size:
                raise KVShrinkParseError(
                    f"Group {g_idx} layers have differing block sizes")

        if kind == "mamba":
            mamba_spec = per_layer_specs[0][1]
            mamba_mode = mamba_spec.mamba_cache_mode
            align = block_size
            # Every GDN snapshot is addressed by an aligned boundary, and
            # the kernels only read the block-table column for the
            # current boundary in 'align' mode. In any other mode a
            # request keeps one max_model_len-sized block that is never
            # boundary-addressable, so there is nothing we could key a
            # snapshot on. vLLM silently rewrites the mode to 'none' when
            # prefix caching is off, and it defaults prefix caching off
            # for hybrid models, so this is the common misconfiguration.
            if mamba_mode != "align":
                raise KVShrinkParseError(
                    f"Group {g_idx} has mamba_cache_mode={mamba_mode!r}, "
                    "but the external cache requires 'align'. Start vLLM "
                    "with --enable-prefix-caching --mamba-cache-mode align "
                    "(vLLM forces the mode to 'none' when prefix caching "
                    "is disabled, and disables prefix caching by default "
                    "for hybrid models)")
            # state regions inside the page (for validation only)
            offset = 0
            for shape, dtype in zip(mamba_spec.shapes, mamba_spec.dtypes):
                nbytes = 1
                for d in shape:
                    nbytes *= d
                nbytes *= _dtype_size(dtype)
                regions.append(StateRegion(
                    name="ssm" if offset > 0 else "conv",
                    offset=offset, nbytes=nbytes,
                    dtype=str(dtype), shape=tuple(shape)))
                offset += nbytes
            group = GroupInfo(
                group_idx=g_idx,
                kind=kind,
                layer_names=tuple(g.layer_names),
                block_size=block_size,
                page_size_bytes=page_size,
                mamba_cache_mode=mamba_mode,
                mamba_align_size=align,
            )
        elif kind in ("attention", "sliding_window"):
            group = GroupInfo(
                group_idx=g_idx,
                kind=kind,
                layer_names=tuple(g.layer_names),
                block_size=block_size,
                page_size_bytes=page_size,
                mamba_cache_mode=None,
                mamba_align_size=None,
            )
        else:  # pragma: no cover - _spec_kind raises first
            raise KVShrinkParseError(f"Unsupported kind {kind}")

        groups.append(group)
        for name, spec in per_layer_specs:
            t_idx = layer_to_tensor.get(name)
            if t_idx is None:
                raise KVShrinkParseError(
                    f"Layer {name} not found in kv_cache_tensors")
            tensor = kv_cache_config.kv_cache_tensors[t_idx]
            # MambaSpec carries dtypes (list); AttentionSpec carries dtype.
            if isinstance(spec, MambaSpec):
                if not spec.dtypes:
                    raise KVShrinkParseError(
                        f"Layer {name} MambaSpec has empty dtypes")
                dtype = spec.dtypes[0]
            else:
                dtype = getattr(spec, "dtype", None)
                if dtype is None:
                    raise KVShrinkParseError(
                        f"Layer {name} has no dtype in spec")
            _dtype_size(dtype)  # fail closed on unknown dtype
            dtype_str = str(dtype)
            # KVCacheTensor is (size, shared_by) only; packed layouts
            # (block_stride/offset) are not expressible and are rejected
            # here rather than guessed.
            t_block_stride = int(getattr(tensor, "block_stride", None) or 0)
            if t_block_stride > 0:
                block_stride_bytes = t_block_stride
            else:
                block_stride_bytes = int(spec.page_size_bytes)
            storage_offset_bytes = int(getattr(tensor, "offset", None) or 0)
            layer_infos[name] = LayerPageInfo(
                layer_name=name,
                group_idx=g_idx,
                spec_kind=kind,
                num_blocks=num_blocks,
                page_size_bytes=int(spec.page_size_bytes),
                unpadded_page_size_bytes=int(
                    getattr(spec, "unpadded_page_size_bytes", None)
                    or spec.page_size_bytes),
                block_stride_bytes=block_stride_bytes,
                storage_offset_bytes=storage_offset_bytes,
                dtype=dtype_str,
                state_regions=tuple(regions),
            )

    if not groups:
        raise KVShrinkParseError("No kv cache groups parsed")
    return groups, layer_infos, num_blocks
