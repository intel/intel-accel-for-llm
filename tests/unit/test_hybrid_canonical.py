"""Canonical page view tests (M1, CPU tensors).

Mirrors the verified vLLM v0.21.0 layout:
- attention layer: single (num_blocks, page) int8 tensor
- mamba layer: LIST of (conv, ssm) tensors sharing one storage, conv at
  page offset 0, ssm after conv bytes
"""
import torch

from kvshrink.hybrid_canonical import Canonicalizer

NUM_BLOCKS = 1843
PAGE = 1081344
CONV_BYTES = 3 * 4096 * 2  # bf16
SSM_BYTES = 16 * 128 * 128 * 4  # fp32


def _info(layer_name, spec_kind, group_idx):
    from kvshrink.hybrid_metadata import LayerPageInfo
    return LayerPageInfo(
        layer_name=layer_name, group_idx=group_idx, spec_kind=spec_kind,
        num_blocks=NUM_BLOCKS,
        page_size_bytes=PAGE, unpadded_page_size_bytes=PAGE,
        block_stride_bytes=PAGE, storage_offset_bytes=0,
        dtype="torch.int8",
    )


def _make_attention_kv():
    t = torch.zeros(NUM_BLOCKS * PAGE, dtype=torch.int8)
    return t.view(NUM_BLOCKS, PAGE)


def _make_mamba_kv():
    """Two state tensors sharing one storage, conv first, ssm after.

    Mirrors gpu_model_runner._reshape_kv_cache_tensors for MambaSpec:
    conv  = as_strided(raw, (B, 3, 4096),    (page_elems, 4096, 1), 0)
    ssm   = as_strided(raw, (B, 16,128,128), (page_elems, 16384, 128, 1), 6144)
    """
    storage = torch.zeros(NUM_BLOCKS * PAGE, dtype=torch.int8)
    raw = storage.view(torch.int8).view(NUM_BLOCKS, PAGE)
    conv = torch.as_strided(
        raw,
        size=(NUM_BLOCKS, 3, 4096),
        stride=(PAGE, 4096, 1),
        storage_offset=0,
    )
    ssm = torch.as_strided(
        raw,
        size=(NUM_BLOCKS, 16, 128, 128),
        stride=(PAGE, 16384, 128, 1),
        storage_offset=6144,
    )
    return [conv, ssm]


def test_canonical_attention_view():
    infos = {"attn.0": _info("attn.0", "attention", 0)}
    c = Canonicalizer(infos, NUM_BLOCKS)
    kv = {"attn.0": _make_attention_kv()}
    c.register(kv)
    view = c.get_page("attn.0", 5)
    assert view.shape == (PAGE,)
    assert view.device == torch.device("cpu")


def test_canonical_mamba_view():
    infos = {"mamba.0": _info("mamba.0", "mamba", 0)}
    c = Canonicalizer(infos, NUM_BLOCKS)
    kv = {"mamba.0": _make_mamba_kv()}
    c.register(kv)
    view = c.get_page("mamba.0", 7)
    assert view.shape == (PAGE,)
