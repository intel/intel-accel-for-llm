"""Config parser tests against the real Qwen3.5-4B TP2 KVCacheConfig dump."""
import json
import os


from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KVCacheTensor, KVCacheGroupSpec,
    MambaSpec, FullAttentionSpec, MambaAttentionBackendEnum,
)

from kvshrink.hybrid_config import (
    parse_kv_cache_config, compute_namespace, KVShrinkParseError)
from kvshrink.hybrid_metadata import SCHEMA_VERSION

FIXTURE = os.path.join(os.path.dirname(__file__),
                       "fixture_kvconfig_4b_tp2.json")


def _mamba_spec():
    import torch
    return MambaSpec(
        block_size=528,
        shapes=((3, 4096), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=1081344,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=0,
    )


def _attn_spec():
    import torch
    return FullAttentionSpec(
        block_size=528,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        page_size_padded=1081344,
    )


def _real_config():
    """Rebuild KVCacheConfig from the M0 dump of Qwen3.5-4B TP2."""
    with open(FIXTURE) as f:
        d = json.load(f)
    tensors = [KVCacheTensor(size=t["size"], shared_by=t["shared_by"])
               for t in d["kv_cache_tensors"]]
    groups = []
    for g in d["kv_cache_groups"]:
        spec = g["spec"]
        if spec["type"] == "MambaSpec":
            s = _mamba_spec()
        else:
            s = _attn_spec()
        groups.append(KVCacheGroupSpec(
            layer_names=g["layer_names"], kv_cache_spec=s))
    return KVCacheConfig(
        num_blocks=d["num_blocks"],
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def test_fixture_shape():
    cfg = _real_config()
    assert cfg.num_blocks == 1843
    assert len(cfg.kv_cache_tensors) == 8
    assert len(cfg.kv_cache_groups) == 4


def test_parse_real_config():
    cfg = _real_config()
    groups, layer_infos, num_blocks = parse_kv_cache_config(
        cfg, hash_block_size=16)
    assert num_blocks == 1843
    assert len(groups) == 4
    kinds = [g.kind for g in groups]
    assert kinds == ["mamba", "mamba", "mamba", "attention"]
    # 32 layers, all mapped
    assert len(layer_infos) == 32
    for g in groups:
        assert len(g.layer_names) == 8
        assert g.page_size_bytes == 1081344
    mamba = groups[0]
    assert mamba.mamba_cache_mode == "align"
    assert mamba.mamba_align_size == 528
    # real layer name format
    assert "language_model.model.layers.3.self_attn.attn" in layer_infos
    assert "language_model.model.layers.0.linear_attn" in layer_infos


def test_state_regions():
    cfg = _real_config()
    _, layer_infos, _ = parse_kv_cache_config(
        cfg, hash_block_size=16)
    lin = layer_infos["language_model.model.layers.0.linear_attn"]
    assert len(lin.state_regions) == 2
    conv, ssm = lin.state_regions
    assert conv.name == "conv"
    assert conv.nbytes == 3 * 4096 * 2  # bf16
    assert ssm.name == "ssm"
    assert ssm.nbytes == 16 * 128 * 128 * 4  # fp32
    assert conv.offset == 0
    assert ssm.offset == conv.nbytes


def test_fail_closed_unknown_spec():
    cfg = _real_config()
    # swap one group's spec for an unsupported type
    class Weird:
        block_size = 1
        page_size_bytes = 1

    bad_groups = list(cfg.kv_cache_groups)
    bad_groups[1] = KVCacheGroupSpec(
        layer_names=bad_groups[1].layer_names, kv_cache_spec=Weird())
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad, hash_block_size=16)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError:
        pass


def test_fail_closed_mamba_cache_mode_not_align():
    """A non-'align' mamba cache mode must be refused at startup.

    vLLM defaults prefix caching OFF for hybrid models and then silently
    rewrites --mamba-cache-mode to 'none'. In that mode a request keeps a
    single max_model_len block that no boundary can address, so the
    connector would quietly cache nothing. Refuse loudly instead.
    """
    import dataclasses
    cfg = _real_config()
    bad_groups = []
    for g in cfg.kv_cache_groups:
        spec = g.kv_cache_spec
        if type(spec).__name__ == "MambaSpec":
            spec = dataclasses.replace(spec, mamba_cache_mode="none")
        bad_groups.append(KVCacheGroupSpec(
            layer_names=g.layer_names, kv_cache_spec=spec))
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad, hash_block_size=16)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError as e:
        assert "align" in str(e), e


def test_fail_closed_uniform_missing_layer():
    """UniformTypeKVCacheSpecs missing a layer must raise."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    cfg = _real_config()
    spec = UniformTypeKVCacheSpecs(  # only 7 of 8 layers registered
        block_size=528,
        kv_cache_specs={
            n: _attn_spec()
            for n in cfg.kv_cache_groups[3].layer_names[:-1]
        })
    bad_groups = list(cfg.kv_cache_groups)
    bad_groups[3] = KVCacheGroupSpec(
        layer_names=cfg.kv_cache_groups[3].layer_names, kv_cache_spec=spec)
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad, hash_block_size=16)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError:
        pass


def test_fail_closed_unknown_dtype():
    """_dtype_size must reject unknown dtypes (parser fail-closed helper)."""
    from kvshrink.hybrid_config import _dtype_size
    assert _dtype_size("torch.bfloat16") == 2
    assert _dtype_size("torch.float32") == 4
    try:
        _dtype_size("torch.bogus")
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError as e:
        assert "Unknown dtype" in str(e), str(e)


def test_fail_closed_heterogeneous_pages_in_group():
    """Layers within one group with different page sizes must raise.

    Uses REAL FullAttentionSpec instances with differing page sizes so the
    heterogeneous-page check (not the spec-kind check) fires.
    """
    import torch
    cfg = _real_config()
    layers = list(cfg.kv_cache_groups[3].layer_names)
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    per_layer = {}
    for i, n in enumerate(layers):
        per_layer[n] = FullAttentionSpec(
            block_size=528, num_kv_heads=2, head_size=256,
            dtype=torch.bfloat16,
            page_size_padded=1114112 if i == 0 else 1081344)
    bad_groups = list(cfg.kv_cache_groups)
    bad_groups[3] = KVCacheGroupSpec(
        layer_names=layers,
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=528, kv_cache_specs=per_layer))
    bad = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=cfg.kv_cache_tensors,
        kv_cache_groups=bad_groups,
    )
    try:
        parse_kv_cache_config(bad, hash_block_size=16)
        raise AssertionError("expected KVShrinkParseError")
    except KVShrinkParseError as e:
        assert "differing page sizes" in str(e), str(e)


def test_parse_real_config_layout_descriptors():
    """Real 4B TP2 config: contiguous pages, zero offsets (v0.21 semantics)."""
    cfg = _real_config()
    _, layer_infos, _ = parse_kv_cache_config(
        cfg, hash_block_size=16)
    info = layer_infos["language_model.model.layers.0.linear_attn"]
    assert info.block_stride_bytes == info.page_size_bytes == 1081344
    assert info.storage_offset_bytes == 0
    a = layer_infos["language_model.model.layers.3.self_attn.attn"]
    assert a.spec_kind == "attention"
    assert a.dtype == "torch.bfloat16"


def test_namespace_stability():
    a = compute_namespace("m", "r", "t", "auto", SCHEMA_VERSION, 2, 1)
    b = compute_namespace("m", "r", "t", "auto", SCHEMA_VERSION, 2, 1)
    c = compute_namespace("m", "r", "t", "auto", SCHEMA_VERSION, 4, 1)
    assert a == b
    assert a != c
