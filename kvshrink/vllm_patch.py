# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin (entry point group ``vllm.general_plugins``)."""

import dataclasses
import os
import sys

from vllm.logger import init_logger

# Under the "vllm" namespace so vLLM's log handler prints it.
logger = init_logger(f"vllm.{__name__}")


def _prefix_caching_disabled(vllm_config) -> bool:
    # vLLM resolves --enable-prefix-caching / --no-enable-prefix-caching to a
    # bool before any config hook runs.
    return not vllm_config.cache_config.enable_prefix_caching


def _patch_mamba_cache_mode() -> None:
    from vllm.model_executor.models.config import MambaModelConfig

    orig = MambaModelConfig.verify_and_update_config.__func__

    def verify_and_update_config(cls, vllm_config) -> None:
        orig(cls, vllm_config)
        cache_config = vllm_config.cache_config
        if not _prefix_caching_disabled(vllm_config):
            return
        # The platform then aligns mamba_block_size to the attention block size.
        logger.info(
            "Mamba align patch: prefix caching disabled, overriding "
            "mamba_cache_mode %r -> 'align' (mamba_block_size=%s)",
            cache_config.mamba_cache_mode,
            cache_config.mamba_block_size,
        )
        cache_config.mamba_cache_mode = "align"

    MambaModelConfig.verify_and_update_config = classmethod(verify_and_update_config)


def _patch_mamba_block_size_validator() -> None:
    from pydantic.dataclasses import rebuild_dataclass
    from vllm.config import VllmConfig

    validators = VllmConfig.__pydantic_decorators__.model_validators
    decorator = validators["validate_mamba_block_size"]
    orig = decorator.func

    def validate_mamba_block_size(self):
        if (
            self.model_config is not None
            and _prefix_caching_disabled(self)
            and self.cache_config.mamba_cache_mode == "align"
        ):
            logger.info_once(
                "Mamba align patch: prefix caching disabled, skipping "
                "validate_mamba_block_size (mamba_block_size=%s, max_model_len=%d)",
                self.cache_config.mamba_block_size,
                self.model_config.max_model_len,
            )
            return self
        return orig(self)

    # Pydantic compiles validators into the schema, so rebuild it.
    validators["validate_mamba_block_size"] = dataclasses.replace(
        decorator, func=validate_mamba_block_size
    )
    rebuild_dataclass(VllmConfig, force=True)


def _force_external() -> bool:
    """Every KV block must come from the connector, not the local prefix cache."""
    return os.getenv("KVSHRINK_FORCE_EXTERNAL", "0") == "1"


def _patch_force_external() -> None:
    """Zero the local prefix lookup so the connector supplies the whole prefix.

    Prefix caching stays on, so vLLM still hashes blocks and still produces the
    mamba boundary hand-offs the connector saves.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    orig_connector = KVCacheManager.get_computed_blocks_for_connector
    orig = KVCacheManager.get_computed_blocks

    def get_computed_blocks_for_connector(self, request, max_cache_hit_length=None):
        if _force_external():
            return self.empty_kv_cache_blocks, 0, 0, False
        return orig_connector(self, request, max_cache_hit_length)

    def get_computed_blocks(self, request, max_cache_hit_length=None):
        if _force_external():
            return self.empty_kv_cache_blocks, 0, 0
        return orig(self, request, max_cache_hit_length)

    KVCacheManager.get_computed_blocks_for_connector = get_computed_blocks_for_connector
    KVCacheManager.get_computed_blocks = get_computed_blocks
    logger.info(
        "KVShrink force-external patch registered (active when "
        "KVSHRINK_FORCE_EXTERNAL=1): KVCacheManager local prefix lookup disabled"
    )


def _patch_qsa_per_layer() -> None:
    """Wrap Qwen4Exp QSA attention with per-layer load/save hooks."""
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )
        from vllm.models.qwen4_exp.common.qsa_cache import QSACompressedKeyCache
    except Exception:  # vLLM too old for these helpers
        return

    # qsa.py and model.py import each other, so importing the submodule before
    # the model exists raises. Once the model is built it is already cached.
    module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
    if module is None:
        try:
            from vllm.models.qwen4_exp.nvidia import qsa as module  # type: ignore[no-redef]
        except Exception:  # not this model, or not imported yet
            return

    cls = getattr(module, "Qwen4ExpQSAAttention", None)
    if cls is None:
        return
    original = cls._run_qsa
    if getattr(original, "_iaxl_per_layer", False):
        return

    @eager_break_during_capture  # outermost: wait/save run in the eager segment
    def _run_qsa(self, projected_qk, positions, query, key, value, output,
                 output_gate, qkv):
        if not has_kv_transfer_group():
            return original(self, projected_qk, positions, query, key, value,
                            output, output_gate, qkv)
        connector = get_kv_transfer_group()
        if not connector.has_connector_metadata():
            return original(self, projected_qk, positions, query, key, value,
                            output, output_gate, qkv)
        layer_name = self.layer_name
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        if attn_metadata is None:
            return original(self, projected_qk, positions, query, key, value,
                            output, output_gate, qkv)
        connector.wait_for_layer_load(layer_name)
        result = original(self, projected_qk, positions, query, key, value,
                          output, output_gate, qkv)
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
        return result

    _run_qsa._iaxl_per_layer = True
    cls._run_qsa = _run_qsa
    # The layer and the compressed K cache it owns are both written inside
    # _run_qsa, so one hook covers them together.
    declare_hooked_class(cls)
    declare_hooked_class(QSACompressedKeyCache)
    declare_cache_class(QSACompressedKeyCache)
    logger.info("QSA per-layer KV transfer patch installed")


def _patch_indexer_per_layer() -> None:
    """Hook the DSA sparse indexer so its compressed K cache moves per layer.

    The indexer cache is its own prefix-cacheable layer (``tokens_per_state``
    is the pool ratio, not 1), and its kernels read and write it inside
    ``Indexer.forward``, which runs before MLA attention in the same layer. So a
    wait at the top and a save at the bottom cover it exactly like an attention
    layer's own KV.
    """
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    except Exception:
        return

    for module_name in (
        "vllm.models.glm5next.common.attention",
        "vllm.model_executor.models.deepseek_v2",
    ):
        module = sys.modules.get(module_name)
        cls = getattr(module, "Indexer", None) if module is not None else None
        if cls is None:
            continue
        if getattr(cls.forward, "_iaxl_per_layer", False):
            continue
        original = cls.forward

        @eager_break_during_capture  # outermost: wait/save run in the eager segment
        def forward(self, *args, original=original, **kwargs):
            cache = getattr(self, "k_cache", None)
            if (
                cache is None
                or not has_kv_transfer_group()
                or not get_kv_transfer_group().has_connector_metadata()
            ):
                return original(self, *args, **kwargs)
            connector = get_kv_transfer_group()
            connector.wait_for_layer_load(cache.prefix)
            result = original(self, *args, **kwargs)
            connector.save_kv_layer(cache.prefix, None, None)
            return result

        forward._iaxl_per_layer = True
        cls.forward = forward
        # The registered layer is the cache, not the indexer that owns it.
        declare_hooked_class(DeepseekV32IndexerCache)
        declare_cache_class(DeepseekV32IndexerCache)
        logger.info("Indexer per-layer KV transfer patch installed on %s", module_name)


def install_per_layer_patches() -> None:
    """Call once the model is built; the worker connector does this."""
    _patch_qsa_per_layer()
    _patch_indexer_per_layer()


def declare_cache_class(module_class: type) -> None:
    """Mark a class whose instances are passive caches (no forward of their own).

    The connector must not let such a layer's hook capture a sibling cache that
    another layer's forward has not written yet.
    """
    module_class._iaxl_cache = True


def declare_hooked_class(module_class: type) -> None:
    """Mark a class whose forward a patch wired to the per-layer hook.

    The mark lives on the class, so the connector sees it even if the patch
    module was imported twice (plugin entry point and package import).
    """
    module_class._iaxl_per_layer = True


def patch_kimi_k3_mla_kv_transfer() -> None:
    """Hook Kimi-K3's MLA layers into the KV connector; no-op unless the model loaded them."""
    mla_module = sys.modules.get("vllm.models.kimi_k3.nvidia.mla")
    if mla_module is None:
        return
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.model_executor.layers.attention.kv_transfer_utils import (
        maybe_transfer_kv_layer,
    )

    attention_class = mla_module.MultiHeadLatentAttention
    # eager_break_during_capture keeps the undecorated function in __wrapped__.
    raw_attention = getattr(
        attention_class._attention,
        "__wrapped__",
        attention_class._attention,
    )

    @maybe_transfer_kv_layer
    def attention_with_kv_transfer(
        layer, positions, q, kv_c_normed, k_pe, attn_out, layer_name
    ) -> None:
        raw_attention(layer, positions, q, kv_c_normed, k_pe, attn_out)

    def _attention(self, positions, q, kv_c_normed, k_pe, attn_out) -> None:
        attention_with_kv_transfer(
            self, positions, q, kv_c_normed, k_pe, attn_out, self.layer_name
        )

    # The hook sits inside the eager break so CUDA graph replay still runs it.
    attention_class._attention = eager_break_during_capture(_attention)
    logger.info("Kimi-K3 MLA KV transfer hook registered")


def register() -> None:
    from vllm.model_executor.models.config import MambaModelConfig

    if getattr(MambaModelConfig, "_mamba_align_patched", False):
        return
    _patch_mamba_cache_mode()
    _patch_mamba_block_size_validator()
    MambaModelConfig._mamba_align_patched = True
    _patch_force_external()
    logger.info(
        "Mamba align patch registered: MambaModelConfig.verify_and_update_config, "
        "VllmConfig.validate_mamba_block_size"
    )
