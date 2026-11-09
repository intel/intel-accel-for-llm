# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin (entry point group ``vllm.general_plugins``)."""

import dataclasses

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


def register() -> None:
    from vllm.model_executor.models.config import MambaModelConfig

    if getattr(MambaModelConfig, "_mamba_align_patched", False):
        return
    _patch_mamba_cache_mode()
    _patch_mamba_block_size_validator()
    MambaModelConfig._mamba_align_patched = True
    logger.info(
        "Mamba align patch registered: MambaModelConfig.verify_and_update_config, "
        "VllmConfig.validate_mamba_block_size"
    )
