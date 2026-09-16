# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 per-layer KV-transfer hook (port of KVCacheClip's
``dsv4-layer-hook.patch`` to vLLM 0.23.0).

Background
----------
Standard attention in vLLM runs through
``unified_attention_with_output`` which is decorated with
``@maybe_transfer_kv_layer``; that decorator calls the KV connector's
``wait_for_layer_load`` (before the layer reads the cache) and
``save_kv_layer`` (after it writes the cache).

DeepSeek-V4 does NOT use that path. Its attention runs through
``DeepseekV4Attention.attention_impl`` (decorated only with
``@eager_break_during_capture``), so the connector's per-layer hooks are
never called. KVCacheClip solved this on vLLM 0.20.2 by patching the
``deepseek_v4_attention`` custom op to bracket the attention with the
connector hooks. In 0.23.0 the model lives in ``vllm/models/deepseek_v4``
and there is no such op, so we wrap ``attention_impl`` at runtime instead —
same effect, no fragile source patch.

``self.prefix`` (e.g. ``"model.layers.5.attn"``) is used as the connector
``layer_name``; it matches the KV-cache prefixes the connector maps to
storable tensors.

Requires ``--enforce-eager`` (the DSv4 deployment uses it). Under breakable
cudagraph the host-side hooks would be captured into the graph.
"""

import functools
import logging

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_dsv4_attention_hook() -> bool:
    """Wrap ``DeepseekV4Attention.attention_impl`` with connector hooks.

    Idempotent and best-effort: returns True once the hook is installed (or was
    already), False if the deepseek_v4 attention module is unavailable.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    try:
        from vllm.models.deepseek_v4.attention import DeepseekV4Attention
    except Exception as exc:  # pragma: no cover - only on non-dsv4 builds
        logger.warning("DSv4 attention hook not installed: %s", exc)
        return False

    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )

    original_attention_impl = DeepseekV4Attention.attention_impl

    @functools.wraps(original_attention_impl)
    def attention_impl_with_kv_transfer(self, *args, **kwargs):
        connector = None
        if has_kv_transfer_group() and is_v1_kv_transfer_group():
            candidate = get_kv_transfer_group()
            if candidate.has_connector_metadata():
                connector = candidate

        # Load this layer's cached KV (H2D) before attention reads the cache.
        if connector is not None:
            connector.wait_for_layer_load(self.prefix)

        result = original_attention_impl(self, *args, **kwargs)

        # Save this layer's KV (D2H) after attention writes the cache.
        if connector is not None:
            connector.save_kv_layer(self.prefix, None, None)

        return result

    DeepseekV4Attention.attention_impl = attention_impl_with_kv_transfer
    _INSTALLED = True
    logger.info(
        "Installed DSv4 attention KV-transfer hook (per-layer save/load)"
    )
    return True
