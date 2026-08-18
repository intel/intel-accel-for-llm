# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration policy for KVShrink asynchronous KV loading."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncLoadLayerConfig:
    """Select the number of leading KV layers required before prefill."""

    dynamic: bool = False
    fixed_layers: int = -1
    dynamic_rules: tuple[tuple[int, int], ...] = ()
    dynamic_fallback: int = -1

    def select(self, concurrency: int) -> int:
        """Return the layer count selected for the request concurrency."""
        if not self.dynamic:
            return self.fixed_layers
        for max_concurrency, layers in self.dynamic_rules:
            if concurrency <= max_concurrency:
                return layers
        return self.dynamic_fallback


def _parse_dynamic_layer_map(
    specification: str,
    num_layers: int,
    load_threshold: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
    rules: list[tuple[int, int]] = []
    fallback: Optional[int] = None
    previous_max = 0
    entries = [entry.strip() for entry in specification.split(",")]

    if not entries or any(not entry for entry in entries):
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP must not be empty"
        )

    for index, entry in enumerate(entries):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP entries must "
                f"use MAX_CONCURRENCY:LAYERS, got {entry!r}"
            )
        max_concurrency_text, layers_text = parts
        try:
            layers = int(layers_text)
        except ValueError as error:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP layer values "
                f"must be integers, got {layers_text!r}"
            ) from error
        if not 1 <= layers < num_layers:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP layer values "
                f"must be in [1, {num_layers}), got {layers}"
            )

        if max_concurrency_text == "*":
            if index != len(entries) - 1:
                raise ValueError(
                    "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP wildcard "
                    "must be the final entry"
                )
            fallback = layers
            continue

        try:
            max_concurrency = int(max_concurrency_text)
        except ValueError as error:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP concurrency "
                f"bounds must be positive integers or '*', got "
                f"{max_concurrency_text!r}"
            ) from error
        if max_concurrency <= previous_max:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP concurrency "
                "bounds must be positive and strictly increasing"
            )
        if max_concurrency < load_threshold:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP concurrency "
                f"bounds must be at least the async load threshold "
                f"{load_threshold}, got {max_concurrency}"
            )
        rules.append((max_concurrency, layers))
        previous_max = max_concurrency

    if fallback is None:
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP must end with a "
            "wildcard entry such as '*:8'"
        )
    return tuple(rules), fallback


def build_async_load_layer_config(
    load_threshold: int,
    fixed_layers: int,
    dynamic_enabled: int,
    dynamic_map: str,
    num_layers: int,
    dynamic_map_configured: bool = True,
) -> AsyncLoadLayerConfig:
    """Validate async-load layer settings and build the selection policy."""
    if dynamic_enabled not in (0, 1):
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC must be 0 or 1, got "
            f"{dynamic_enabled}"
        )

    if load_threshold == -1:
        if fixed_layers != -1 or dynamic_enabled:
            logger.warning(
                "Ignoring async load layer configuration because "
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_THRESHOLD=-1 disables async "
                "loading"
            )
        return AsyncLoadLayerConfig()

    if dynamic_enabled:
        if fixed_layers != -1:
            logger.warning(
                "Ignoring KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=%d because "
                "dynamic async load layers are enabled",
                fixed_layers,
            )
        rules, fallback = _parse_dynamic_layer_map(
            dynamic_map,
            num_layers,
            load_threshold,
        )
        return AsyncLoadLayerConfig(
            dynamic=True,
            dynamic_rules=rules,
            dynamic_fallback=fallback,
        )

    if dynamic_map_configured:
        logger.warning(
            "Ignoring KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP=%s "
            "because dynamic async load layers are disabled",
            dynamic_map,
        )
    if fixed_layers != -1 and not 1 <= fixed_layers < num_layers:
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS must be -1 or in "
            f"[1, {num_layers}), got {fixed_layers}"
        )
    return AsyncLoadLayerConfig(fixed_layers=fixed_layers)
