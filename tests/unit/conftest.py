# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the KVShrink hybrid unit tests.

These tests are pure logic: no GPU, no disk, no model, no machine
specifics. Storage and transfer engines are always faked, so the suite
runs anywhere vLLM and PyTorch import.
"""

from __future__ import annotations

import os
import sys

import pytest

# Import the package from the repository checkout without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Knobs that change connector behaviour. Cleared for every test so a
# developer's shell environment can never alter the results.
_KVSHRINK_ENV = (
    "KVSHRINK_SAVE",
    "KVSHRINK_SAVE_PIPELINED",
    "KVSHRINK_DEBUG_AUTOSAVE",
    "KVSHRINK_DEBUG_LOG",
    "KVSHRINK_DEBUG_DUMP",
    "KVSHRINK_PERSIST_DIR",
    "KVSHRINK_METRICS_PORT",
)


@pytest.fixture(autouse=True)
def _clean_kvshrink_env(monkeypatch):
    for name in _KVSHRINK_ENV:
        monkeypatch.delenv(name, raising=False)
    # The exporter binds a port; unit tests never need it.
    monkeypatch.setenv("KVSHRINK_METRICS_PORT", "0")
