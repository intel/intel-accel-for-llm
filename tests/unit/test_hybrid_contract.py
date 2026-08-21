"""Contract tests: connector hooks against vLLM v0.21.0 (M1).

Covers the M0-verified contract points:
1. build_connector_meta must return a KVConnectorMetadata instance (None
   crashes the worker with `assert kv_connector_metadata is not None`).
2. __init__ requires kv_cache_config.
3. get_num_new_matched_tokens returns tuple[int|None, bool].
4. Connector must subclass SupportsHMA to use --no-disable-hybrid-kv-cache-manager.
"""
import inspect
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA, supports_hma)


def test_build_connector_meta_returns_metadata():
    """A real minimal connector must return a metadata INSTANCE from
    build_connector_meta (None crashes the worker)."""

    class C(KVConnectorBase_V1, SupportsHMA):
        def __init__(self, vllm_config, role, kv_cache_config):
            super().__init__(vllm_config, role, kv_cache_config)

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False

        def update_state_after_alloc(self, request, blocks,
                                     num_external_tokens):
            pass

        def build_connector_meta(self, scheduler_output):
            return KVConnectorMetadata()

        def request_finished(self, request, block_ids):
            return True, None

        def request_finished_all_groups(self, request, block_ids):
            return True, None

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            pass

        def start_load_kv(self, forward_context, **kwargs):
            pass

        def wait_for_layer_load(self, layer_name):
            pass

        def wait_for_save(self):
            pass

    cfg = SimpleNamespace(kv_transfer_config=SimpleNamespace())
    conn = C(vllm_config=cfg, role="kv_both",
             kv_cache_config=SimpleNamespace())
    meta = conn.build_connector_meta(scheduler_output=object())
    assert isinstance(meta, KVConnectorMetadata), \
        f"build_connector_meta must return KVConnectorMetadata, got {type(meta)}"


def test_metadata_empty_instance_usable():
    m = KVConnectorMetadata()
    assert m is not None


def test_supports_hma_protocol():
    class Plain(KVConnectorBase_V1):
        pass

    class Hybrid(Plain, SupportsHMA):
        def request_finished_all_groups(self, request, block_ids):
            return True, None

    assert supports_hma(Hybrid)
    assert not supports_hma(Plain)


def test_get_num_new_matched_tokens_signature():
    sig = inspect.signature(KVConnectorBase_V1.get_num_new_matched_tokens)
    params = list(sig.parameters)
    assert params[:3] == ["self", "request", "num_computed_tokens"], params
    assert sig.return_annotation is not inspect.Signature.empty


def test_init_requires_kv_cache_config():
    """__init__ signature has kv_cache_config as a required positional."""
    sig = inspect.signature(KVConnectorBase_V1.__init__)
    params = list(sig.parameters)
    assert params[:4] == ["self", "vllm_config", "role", "kv_cache_config"], \
        params
    p = sig.parameters["kv_cache_config"]
    assert p.default is inspect.Parameter.empty
