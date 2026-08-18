# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration policy for KVShrink asynchronous KV loading."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncLoadLayerConfig:
    """Select the number of leading KV layers required before prefill."""

    enabled: bool
    dynamic: bool = False
    fixed_layers: int = -1
    dynamic_rules: tuple[tuple[int, Optional[int], int], ...] = ()

    def select(self, concurrency: int) -> int:
        """Return the layer count selected for the request concurrency.

        A return value of zero selects synchronous loading for the request.
        """
        if not self.enabled:
            return 0
        if not self.dynamic:
            return self.fixed_layers
        for start, end, layers in self.dynamic_rules:
            if concurrency >= start and (end is None or concurrency <= end):
                return layers
        raise RuntimeError(f"No async load layer rule for concurrency {concurrency}")


def _parse_dynamic_layer_map(
    specification: str,
    num_layers: int,
) -> tuple[tuple[int, Optional[int], int], ...]:
    rules: list[tuple[int, Optional[int], int]] = []
    expected_start = 0
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
                f"use START-END:LAYERS, got {entry!r}"
            )
        concurrency_range, layers_text = (part.strip() for part in parts)
        try:
            layers = int(layers_text)
        except ValueError as error:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP layer values "
                f"must be integers, got {layers_text!r}"
            ) from error
        if not 0 <= layers < num_layers:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP layer values "
                f"must be in [0, {num_layers}), got {layers}"
            )

        range_parts = concurrency_range.split("-")
        if len(range_parts) != 2:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP ranges must "
                f"use START-END:LAYERS, got {entry!r}"
            )
        start_text, end_text = (part.strip() for part in range_parts)
        try:
            start = int(start_text)
            end = int(end_text) if end_text else None
        except ValueError as error:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP concurrency "
                f"range bounds must be non-negative integers, got "
                f"{concurrency_range!r}"
            ) from error
        if start < 0 or (end is not None and end < 0):
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP concurrency "
                "range bounds must be non-negative"
            )
        if start != expected_start:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP ranges must "
                f"start at 0 and be contiguous; expected start "
                f"{expected_start}, got {start}"
            )
        if end is None:
            if index != len(entries) - 1:
                raise ValueError(
                    "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP open "
                    "range must be the final entry"
                )
        elif end < start:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP range end "
                f"must be at least its start, got {concurrency_range!r}"
            )
        else:
            expected_start = end + 1
        rules.append((start, end, layers))

    if rules[-1][1] is not None:
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP must end with an "
            "open range such as '7-:8'"
        )
    return tuple(rules)


def build_async_load_layer_config(
    async_enabled: int,
    fixed_layers: int,
    dynamic_enabled: int,
    dynamic_map: str,
    num_layers: int,
    dynamic_map_configured: bool = True,
) -> AsyncLoadLayerConfig:
    """Validate async-load layer settings and build the selection policy."""
    if async_enabled not in (0, 1):
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED must be 0 or 1, got "
            f"{async_enabled}"
        )
    if dynamic_enabled not in (0, 1):
        raise ValueError(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC must be 0 or 1, got "
            f"{dynamic_enabled}"
        )

    if not async_enabled:
        logger.warning(
            "Ignoring async load layer configuration because "
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED=0 disables async "
            "loading"
        )
        return AsyncLoadLayerConfig(enabled=False)

    if dynamic_enabled:
        if fixed_layers != -1:
            logger.warning(
                "Ignoring KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS=%d because "
                "dynamic async load layers are enabled",
                fixed_layers,
            )
        rules = _parse_dynamic_layer_map(
            dynamic_map,
            num_layers,
        )
        return AsyncLoadLayerConfig(
            enabled=True,
            dynamic=True,
            dynamic_rules=rules,
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
    return AsyncLoadLayerConfig(
        enabled=True,
        fixed_layers=fixed_layers,
    )


def load_async_load_layer_config_from_env(
    num_layers: int,
    environ: Mapping[str, str] | None = None,
) -> AsyncLoadLayerConfig:
    """Load async KV settings exported by ``setvars.sh``.

    No defaults are supplied here. This keeps ``setvars.sh`` as the single
    source of runtime defaults and makes missing configuration explicit.
    """
    source = os.environ if environ is None else environ

    def required(name: str) -> str:
        try:
            value = source[name]
        except KeyError as error:
            raise ValueError(
                f"{name} must be set; source setvars.sh before starting vLLM"
            ) from error
        if not value:
            raise ValueError(f"{name} must not be empty")
        return value

    def required_int(name: str) -> int:
        value = required(name)
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, got {value!r}") from error

    return build_async_load_layer_config(
        async_enabled=required_int(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED"
        ),
        fixed_layers=required_int("KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS"),
        dynamic_enabled=required_int(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC"
        ),
        dynamic_map=required(
            "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP"
        ),
        num_layers=num_layers,
        dynamic_map_configured=True,
    )
