# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, setup_root_logger
from iaxl.envs import envs as iaxl_envs
from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

from .async_load_config import load_async_load_layer_config_from_env
from .vllm_patch import install_per_layer_patches, patch_kimi_k3_mla_kv_transfer

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str

# OMP-Main TaskQueue levels: P0 loads a forward waits on, P1 first N layers, P2 remaining layers.
LOAD_PRIORITY_P0 = 0
LOAD_PRIORITY_P1 = 1
LOAD_PRIORITY_P2 = 2


@dataclass
class ReqMeta:
    # One block table per KV cache group, aligned with block_hashes; 0 marks
    # a block that is not transferred for that group.
    block_ids: tuple[list[int], ...] = ()
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1

    def for_group(self, group_idx: int) -> tuple[list[int], list[str]]:
        pairs = [
            (block_id, block_hash)
            for block_id, block_hash in zip(
                self.block_ids[group_idx], self.block_hashes
            )
            if block_id != 0
        ]
        return [p[0] for p in pairs], [p[1] for p in pairs]


@dataclass
class ReqState:
    num_computed_tokens: int = 0
    existence_cache: list[bool] = field(default_factory=list)
    mamba_existence_cache: list[bool] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    # Worker-side block tables per group, mirrored from the scheduler output.
    block_ids: list[list[int]] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        block_ids: tuple[list[int], ...],
        block_hashes: list[str],
        is_async: bool = False,
        async_load_layers: int = -1,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            block_ids,
            block_hashes,
            is_async,
            async_load_layers,
        )


# Kernels decorated with @maybe_transfer_kv_layer. A forward that names one of
# them reaches the per-layer hook, whatever class hierarchy it belongs to.
_HOOKED_KERNELS = frozenset(
    {"unified_attention_with_output", "unified_mla_attention_with_output"}
)


def _is_cache_module(module: Any) -> bool:
    """Whether this layer is a passive cache (a patch declares it)."""
    return bool(getattr(type(module), "_iaxl_cache", False))


def _forward_calls_hook(module: Any) -> bool:
    for klass in type(module).__mro__:
        forward = klass.__dict__.get("forward")
        if forward is None:
            continue
        code = getattr(forward, "__code__", None)
        return bool(code and _HOOKED_KERNELS.intersection(code.co_names))
    return False


@dataclass
class Group:
    """One KV cache group; its position in ``connector.groups`` is the group id."""

    is_mamba: bool
    layer_names: list[str]
    # KVStore namespace: attention keeps the store default "kv", mamba its own.
    label: Optional[str]


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata
    scheduled_req_ids: set[ReqId] = field(default_factory=set)


class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(
        cls, extra_config: dict[str, Any]
    ) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.model_config = vllm_config.model_config
        # (is_mamba, layer_names) per KV cache group; all groups share one
        # block size, which is the enlarged one on hybrid models. Groups that
        # are not transferable keep their id but carry no layers.
        self.num_groups = len(kv_cache_config.kv_cache_groups)
        self.groups: dict[int, Group] = {}
        for group_id, kv_group in enumerate(kv_cache_config.kv_cache_groups):
            spec = kv_group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            is_mamba = isinstance(spec, MambaSpec)
            if not (kv_group.enable_kv_transfer and kv_group.kv_cache_spec.prefix_cacheable):
                continue
            self.groups[group_id] = Group(
                is_mamba=is_mamba,
                layer_names=list(kv_group.layer_names),
                label="mamba" if is_mamba else None,
            )
        self.has_mamba = any(group.is_mamba for group in self.groups.values())
        if self.has_mamba and vllm_config.cache_config.mamba_cache_mode != "align":
            raise ValueError(
                "KVShrinkConnector on hybrid models requires "
                "mamba_cache_mode='align', got "
                f"{vllm_config.cache_config.mamba_cache_mode!r}"
            )
        block_sizes = {
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size
            for group_id in self.groups
        }
        assert len(block_sizes) == 1, block_sizes
        self.block_size = block_sizes.pop()
        logger.info(
            "KVShrinkConnector(%s): block_size=%d, has_mamba=%s, "
            "mamba_cache_mode=%r, prefix caching=%s",
            role.name,
            self.block_size,
            self.has_mamba,
            vllm_config.cache_config.mamba_cache_mode,
            vllm_config.cache_config.enable_prefix_caching,
        )
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device = vllm_config.device_config.device_type
        parallel_config = vllm_config.parallel_config
        # data_parallel_index, not data_parallel_rank (vLLM resets the latter
        # to 0 for dense models).
        self.dp_rank = parallel_config.data_parallel_index
        if self.model_config.is_moe:
            self.dp_size = parallel_config.data_parallel_size
        else:
            self.dp_size = int(os.environ.get("DP_SIZE", "1"))
            logger.warning(
                "Dense model: reading DP size from the DP_SIZE env var (%d) "
                "because vLLM resets data_parallel_size to 1 for dense models.",
                self.dp_size,
            )
        self.tp_rank = (
            get_tensor_model_parallel_rank()
            if model_parallel_is_initialized()
            else 0
        )
        # Globally-unique worker index / total worker count: KVStore/CPU/port identity.
        self.global_rank = self.dp_rank * self.tp_size + self.tp_rank
        self.global_size = self.dp_size * self.tp_size

        self._req_states: dict[ReqId, ReqState] = {}
        self._reqs_to_load = RequestMetadata()
        self._reqs_to_save = RequestMetadata()
        self._current_get_tasks: Optional[dict[str, Any]] = None
        self._current_put_tasks: dict[ReqId, list[dict[str, Any]]] = {}
        self._deferred_finished_req_ids: set[ReqId] = set()
        self._last_layer_name: Optional[str] = None
        # vLLM layer name -> KVStore key: the transformer layer index, plus an
        # ordinal when one layer owns several caches.
        self._layer_key: dict[str, str] = {}
        self._layer_group: dict[str, int] = {}
        self.num_hookless_layers = 0
        # Ordered worker-side KVStore keys (populated in register_kv_caches),
        # used to select the first N layers for async early-start.
        self._layer_names: list[str] = []
        # Async load bookkeeping (worker side).
        # Per-request tasks still loading across scheduler steps.
        self._pending_load_tasks: dict[ReqId, dict[str, Any]] = {}
        # Early-start layer count selected for each pending async request.
        self._pending_load_layers: dict[ReqId, int] = {}
        # Tasks early-promoted (first N layers done) whose remaining layers are
        # waited on-demand in wait_for_layer_load during the prefill forward.
        self._early_promoted_tasks: dict[ReqId, dict[str, Any]] = {}
        # Early-promoted tasks active for the current forward pass.
        self._active_promoted_tasks: dict[ReqId, dict[str, Any]] = {}
        # TaskQueue bundle id of each async request's remaining layers until they are raised.
        self._load_bundle_ids: dict[ReqId, int] = {}
        # 0 means no bundle in the TaskQueue.
        self._next_load_bundle_id = 1

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
        )

        if role == KVConnectorRole.SCHEDULER:
            _, hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
            if hash_block_size != self.block_size:
                raise ValueError(
                    "KVShrinkConnector needs one vLLM block hash per "
                    f"{self.block_size} tokens, got hash_block_size={hash_block_size}"
                )
            hash_algo = vllm_config.cache_config.prefix_caching_hash_algo
            if (
                hash_algo in ("xxhash", "xxhash_cbor")
                and os.getenv("PYTHONHASHSEED") is None
            ):
                raise ValueError(
                    f"KVShrinkConnector with prefix_caching_hash_algo={hash_algo!r} "
                    "requires PYTHONHASHSEED, otherwise block hashes differ "
                    "between processes"
                )
            # Scheduler store reads its DP group's tp0; offset mgmt ports by DP
            # group so DP schedulers on one host don't clash.
            iaxl_envs.IAXL_API_CONTROLLER_PORT += self.dp_rank
            iaxl_envs.IAXL_API_WORKER_BASE_PORT += self.dp_rank * self.tp_size
            self.kvstore: Optional[KVStore] = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(index) for index in range(self.num_layers)],
                rank=self.dp_rank * self.tp_size,
                tp_size=self.tp_size,
            )
        else:
            self.kvstore = None
            # The model is loaded by now and CUDA graphs are captured after this.
            patch_kimi_k3_mla_kv_transfer()
            if not iaxl_envs.IAXL_RDMA_ENABLE:  # compression/DSA run on the daemon node
                self._bind_cpu_affinity()
                self._bind_intel_accel()

    def _bind_cpu_affinity(self) -> None:
        spec = os.getenv("KVSHRINK_CPU_OMP_THREADS_BIND")
        if spec:
            cpus = spec.split("|")
            if len(cpus) <= self.global_rank:
                raise ValueError(
                    f"KVSHRINK_CPU_OMP_THREADS_BIND has {len(cpus)} entries, "
                    f"but rank is {self.global_rank}"
                )
            os.environ["IAXL_CPU_OMP_THREADS_BIND"] = cpus[self.global_rank]
            logger.info(
                "Bound rank %d: IAXL_CPU_OMP_THREADS_BIND=%s",
                self.global_rank,
                cpus[self.global_rank],
            )
        if self.vllm_device == "cpu":  # vLLM's CPU worker binds itself
            return
        bind_cpu_affinity(
            self.global_rank, self.global_size, envs.VLLM_CPU_OMP_THREADS_BIND
        )

    def _bind_intel_accel(self) -> None:
        bind_intel_accel(self.global_rank)

    def _store(self) -> KVStore:
        if self.kvstore is None:
            raise RuntimeError("KVStore has not been initialized")
        return self.kvstore

    @staticmethod
    def _current_block_ids(
        scheduler_output: SchedulerOutput, req_id: str
    ) -> list[list[int]]:
        # The scheduler's own block table, copied so later CoW moves cannot
        # change the snapshot this step saves from.
        return [
            list(ids)
            for ids in scheduler_output.kv_connector_block_state.get_block_ids(req_id)
        ]

    @staticmethod
    def _boundary_offloads(scheduler_output: SchedulerOutput, req_id: str):
        return list(
            scheduler_output.kv_connector_block_state.boundary_state_offloads.get(
                req_id, []
            )
        )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self._req_states.pop(request.request_id, None) is not None:
            logger.warning("Discarded stale state for request %s", request.request_id)

        # vLLM always computes the last prompt token, so it is never loaded.
        num_blocks = (request.num_tokens - 1) // self.block_size
        block_hashes: list[str] = []
        for block_hash in request.block_hashes[:num_blocks]:
            block_hashes.append(block_hash.hex())
        existence_cache = self._store().has(block_hashes)
        state = ReqState(
            num_computed_tokens=num_computed_tokens,
            existence_cache=existence_cache,
            block_hashes=block_hashes,
        )
        self._req_states[request.request_id] = state

        matched_blocks = next(
            (index for index, exists in enumerate(existence_cache) if not exists),
            len(existence_cache),
        )
        if self.has_mamba:
            attention_matched_blocks = matched_blocks
            # A hit must end at a block whose mamba state was saved.
            state.mamba_existence_cache = self._store().has(
                block_hashes, label="mamba", truncate=False
            )
            while (
                matched_blocks > 0
                and not state.mamba_existence_cache[matched_blocks - 1]
            ):
                matched_blocks -= 1
            # Stop the chunk at the end of the stored attention KV to save its mamba state.
            if attention_matched_blocks > matched_blocks:
                request.shared_prefix_boundary = (
                    attention_matched_blocks * self.block_size
                )
        matched_tokens = matched_blocks * self.block_size
        num_new_tokens = max(0, matched_tokens - num_computed_tokens)

        # Decide sync vs async for this request. The load can only be async when
        # there are external tokens to load and async is enabled. Concurrency is
        # approximated by the number of in-flight requests (this one included).
        selected_layers = self._async_load_layer_config.select(
            len(self._req_states)
        )
        # A synchronous load lets vLLM zero the blocks on the compute stream
        # under our copy, which a hybrid model cannot survive; otherwise the
        # policy may still ask for it. The worker widens the first-loaded layer
        # count when the load is asynchronous.
        use_async = num_new_tokens > 0 and (selected_layers != 0 or self.has_mamba)
        state.is_async = use_async
        if use_async:
            state.async_load_layers = selected_layers

        logger.info(
            f"get_num_new_matched_tokens, req-{request.request_id}, "
            f"externally-cached tokens: {num_new_tokens}, "
            f"locally-cached tokens: {num_computed_tokens}, async={use_async}, "
            f"selected_load_layers={selected_layers}, "
            f"async_load_layers={state.async_load_layers}, "
            f"shared_prefix_boundary={request.shared_prefix_boundary}"
        )
        return num_new_tokens, use_async

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        state = self._req_states.get(request.request_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {request.request_id}")

        if num_external_tokens == 0:
            return
        if num_external_tokens % self.block_size != 0:
            raise ValueError("External token count must be block aligned")

        block_ids = blocks.get_block_ids()
        load_start = state.num_computed_tokens // self.block_size
        load_end = load_start + num_external_tokens // self.block_size
        assert load_end <= len(state.block_hashes)

        all_ids: list[list[int]] = [[] for _ in range(self.num_groups)]
        for group_id, group in self.groups.items():
            group_ids = list(block_ids[group_id])[load_start:load_end]
            if group.is_mamba:
                # A mamba table is null up to the hit boundary, so the same
                # slice leaves only the slot that receives the saved state.
                group_ids[:-1] = [0] * (len(group_ids) - 1)
            all_ids[group_id] = group_ids

        self._reqs_to_load.add_request(
            request.request_id,
            tuple(all_ids),
            state.block_hashes[load_start:load_end],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def _add_request_to_save(
        self,
        req_id: ReqId,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        boundary_offloads: list[tuple[int, int, int]],
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        num_tokens = num_computed_tokens + num_scheduled_tokens
        start = num_computed_tokens // self.block_size
        end = min(num_tokens // self.block_size, len(state.block_hashes))
        if end <= start:
            return
        boundaries: dict[int, list[tuple[int, int]]] = {}
        for gid, block_id, tokens in boundary_offloads:
            boundaries.setdefault(gid, []).append((block_id, tokens))

        all_ids: list[list[int]] = [[] for _ in range(self.num_groups)]
        for group_id, group in self.groups.items():
            blocks = [0] * len(state.block_hashes)
            exists = (
                state.mamba_existence_cache
                if group.is_mamba
                else state.existence_cache
            )
            if group.is_mamba:
                # boundary_offloads carries the exact GPU block id of every
                # committed boundary; the request still owns it this step.
                candidates = (
                    (tokens // self.block_size - 1, block_id)
                    for block_id, tokens in boundaries.get(group_id, ())
                )
            else:
                ids = state.block_ids[group_id]
                candidates = (
                    (index, ids[index])
                    for index in range(start, min(len(ids), end, len(blocks)))
                )
            for index, block_id in candidates:
                if block_id and start <= index < end and not exists[index]:
                    blocks[index] = block_id
            all_ids[group_id] = blocks

        if any(any(ids) for ids in all_ids):
            self._reqs_to_save.add_request(
                req_id, tuple(all_ids), state.block_hashes
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # True = defer freeing to get_finished() (async load/save may still run).
        self._req_states.pop(request.request_id, None)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return self.request_finished(request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for request in scheduler_output.scheduled_new_reqs:
            state = self._req_states[request.req_id]
            state.block_ids = self._current_block_ids(scheduler_output, request.req_id)
            self._add_request_to_save(
                request.req_id,
                request.num_computed_tokens,
                num_scheduled_tokens[request.req_id],
                self._boundary_offloads(scheduler_output, request.req_id),
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                raise RuntimeError("Resuming from preemption is not supported")

            state = self._req_states[req_id]
            state.block_ids = self._current_block_ids(scheduler_output, req_id)
            self._add_request_to_save(
                req_id,
                cached_reqs.num_computed_tokens[index],
                num_scheduled_tokens[req_id],
                self._boundary_offloads(scheduler_output, req_id),
            )

        # kvshrink loads asynchronously, but its start_load_kv only submits
        # host-side work; running it before the forward keeps the 0.23-style
        # placement (0.29 would otherwise defer it to post_forward).
        scheduler_output.has_sync_kv_loads = True
        metadata = KVShrinkConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=self._reqs_to_save,
            # scheduler_output.num_scheduled_tokens is a dict keyed by req id.
            scheduled_req_ids=set(scheduler_output.num_scheduled_tokens.keys()),
        )
        self._reqs_to_load = RequestMetadata()
        self._reqs_to_save = RequestMetadata()
        return metadata

    ############################################################
    # Worker Side Methods
    ############################################################

    def _view_as_blocks(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """View each layer's KV cache as contiguous ``(num_blocks, page)`` rows
        per logical block, taken from the ``KVCacheConfig`` geometry."""
        num_blocks = self.kv_cache_config.num_blocks
        spec_by_layer: dict[str, Any] = {}
        for group in self.kv_cache_config.kv_cache_groups:
            specs = (
                group.kv_cache_spec.kv_cache_specs
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else {}
            )
            for layer_name in group.layer_names:
                spec_by_layer[layer_name] = specs.get(
                    layer_name, group.kv_cache_spec
                )

        attn_dtype = next(
            (
                spec.dtype
                for spec in spec_by_layer.values()
                if isinstance(spec, AttentionSpec)
            ),
            None,
        )
        assert attn_dtype is not None

        views: dict[str, torch.Tensor] = {}
        for layer_name, cache in kv_caches.items():
            ref = group_kernel_blocks(cache, num_blocks)
            # vLLM hands mamba layers int8 views of the shared buffer (their
            # MambaSpec carries per-state dtypes, so the view is not cast), so
            # offset and stride are in int8 units: rescale them by bytes.
            itemsize = attn_dtype.itemsize
            offset = ref.storage_offset() * ref.element_size()
            stride = ref.stride(0) * ref.element_size()
            page = spec_by_layer[layer_name].page_size_bytes
            assert offset % itemsize == 0 and stride % itemsize == 0
            assert page % itemsize == 0 and page <= stride
            views[layer_name] = torch.tensor(
                [], dtype=attn_dtype, device=ref.device
            ).set_(
                ref.untyped_storage(),
                offset // itemsize,
                (num_blocks, page // itemsize),
                (stride // itemsize, 1),
            )
        return views

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Model-side per-layer patches need the model classes, which exist by
        # the time the worker registers its caches.
        install_per_layer_patches()

        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        hookless_layers = {
            ln
            for ln in kv_caches
            if ln not in layers
            or not (
                getattr(type(layers[ln]), "_iaxl_per_layer", False)
                or _forward_calls_hook(layers[ln])
            )
        }
        self.num_hookless_layers = len(hookless_layers)
        logger.info(
            "KVShrinkConnector: %d of %d registered layers have no per-layer hook",
            self.num_hookless_layers,
            len(kv_caches),
        )

        # Skip draft (MTP) layers; put layers without a hook first so the
        # early start loads them before the forward.
        enabled_layers = {
            ln for group in self.groups.values() for ln in group.layer_names
        }
        kv_caches = {
            ln: kv_caches[ln]
            for ln in sorted(kv_caches, key=lambda ln: ln not in hookless_layers)
            if ln in enabled_layers and extract_layer_index(ln) < self.num_layers
        }

        # A transformer layer can own more than one cache module (e.g. MLA and
        # its indexer). Append an ordinal to keep the short KVStore keys unique.
        by_index: dict[int, list[str]] = {}
        for ln in kv_caches:
            by_index.setdefault(extract_layer_index(ln), []).append(ln)
        self._layer_key = {}
        for index, names in by_index.items():
            if len(names) == 1:
                self._layer_key[names[0]] = str(index)
            else:
                for ordinal, ln in enumerate(sorted(names)):
                    self._layer_key[ln] = f"{index}.{ordinal}"

        # Which caches a hook covers: the layer's own KV plus the caches its
        # module owns. Ownership comes from the module graph, so a cache read by
        # another layer's forward (an indexer ahead of MLA) is never captured
        # before that layer wrote it.
        cache_names = {id(module): ln for ln, module in layers.items() if ln in kv_caches}
        owned: dict[str, list[str]] = {}
        for ln, module in layers.items():
            # A registered layer may hold its cache on a submodule.
            for holder in module.modules():
                for value in vars(holder).values():
                    cache_ln = cache_names.get(id(value))
                    if cache_ln is not None and cache_ln != ln:
                        owned.setdefault(ln, []).append(cache_ln)

        self._hook_keys: dict[str, list[str]] = {}
        for ln in kv_caches:
            keys = [self._layer_key[ln]]
            keys += [
                self._layer_key[cache_ln]
                for cache_ln in owned.get(ln, ())
                if cache_ln in kv_caches
            ]
            module = layers.get(ln)
            if module is not None and not _is_cache_module(module):
                # A layer that runs real work also covers the sibling caches of
                # its transformer layer; a cache module covers only itself.
                keys += [
                    self._layer_key[sibling]
                    for sibling in by_index[extract_layer_index(ln)]
                    if sibling != ln and sibling not in owned.get(ln, ())
                ]
            self._hook_keys[ln] = keys

        self._layer_group_all = {
            ln: group_id
            for group_id, group in self.groups.items()
            for ln in group.layer_names
            if ln in kv_caches
        }

        # Layers whose attention runs the per-layer transfer hook; anything
        # else is left to wait_for_save().
        self._layer_group = {
            ln: group_id
            for group_id, group in self.groups.items()
            for ln in group.layer_names
            if ln in kv_caches and ln not in hookless_layers
        }

        # Layers the per-layer hook never covers: wait_for_save submits these.
        self._hookless_keys: dict[int, list[str]] = {}
        for ln in hookless_layers:
            key = self._layer_key.get(ln)
            group_id = self._layer_group_all.get(ln)
            if key is not None and group_id is not None:
                self._hookless_keys.setdefault(group_id, []).append(key)

        self._key_group = {
            self._layer_key[ln]: group_id for ln, group_id in self._layer_group.items()
        }

        # Replace scheduler layer names with the short keys the KVStore uses.
        for group in self.groups.values():
            group.layer_names = [
                self._layer_key[ln] for ln in group.layer_names if ln in kv_caches
            ]
        self._last_layer_name = next(reversed(kv_caches)) if kv_caches else None
        self._layer_names = [self._layer_key[ln] for ln in kv_caches]
        self.num_mamba_layers = sum(
            len(group.layer_names)
            for group in self.groups.values()
            if group.is_mamba
        )
        logger.info(
            "KVShrinkConnector: %d of %d registered layers are mamba state",
            self.num_mamba_layers,
            len(kv_caches),
        )

        if not kv_caches:
            logger.info("No enabled KV cache layers; connector is a no-op")
            self.kvstore = None
            return

        kv_caches = self._view_as_blocks(kv_caches)
        first_kv_cache = next(iter(kv_caches.values()))
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            block_dim=0,
            kv_caches={self._layer_key[ln]: t for ln, t in kv_caches.items()},
            rank=self.global_rank,
            tp_size=self.tp_size,
        )
        logger.info(
            "Registered %d KV cache layers (%d mamba) with shape %s",
            len(kv_caches),
            self.num_mamba_layers,
            list(first_kv_cache.shape),
        )

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        sync_reqs: list[tuple[ReqId, ReqMeta]] = []
        async_reqs: list[tuple[ReqId, ReqMeta]] = []
        for req_id, request in metadata.reqs_to_load.requests.items():
            assert all(
                len(ids) == len(request.block_hashes)
                for group_id, ids in enumerate(request.block_ids)
                if group_id in self.groups
            ), req_id
            if request.is_async:
                async_reqs.append((req_id, request))
            else:
                # Hybrid loads are always async: vLLM still zeroes the blocks
                # of a synchronous load on the compute stream.
                sync_reqs.append((req_id, request))

        # Submit synchronous (blocking) loads first, batched per group so
        # multiple sync requests do not overwrite each other's layer tasks.
        sync_batches: dict[int, tuple[list[int], list[str]]] = {}
        for _, request in sync_reqs:
            for group_id, group in self.groups.items():
                if not group.layer_names:
                    continue
                block_ids, block_hashes = request.for_group(group_id)
                batch = sync_batches.setdefault(group_id, ([], []))
                batch[0].extend(block_ids)
                batch[1].extend(block_hashes)

        self._current_get_tasks = None
        for group_id, (block_ids, block_hashes) in sync_batches.items():
            tasks = self._store().get(
                block_indices=block_ids,
                block_hashs=block_hashes,
                layer_names=self.groups[group_id].layer_names,
                label=self.groups[group_id].label,
            )
            if self._current_get_tasks is None:
                self._current_get_tasks = {}
            self._current_get_tasks.update(tasks)

        # Wait only on promoted requests scheduled in this forward; a no-forward batch waits on none.
        if forward_context.attn_metadata is not None:
            duplicates = (
                self._active_promoted_tasks.keys()
                & self._early_promoted_tasks.keys()
            )
            if duplicates:
                raise RuntimeError(
                    f"Duplicate promoted load tasks for requests {duplicates}"
                )
            scheduled: list[ReqId] = []
            for req_id in self._early_promoted_tasks:
                if req_id in metadata.scheduled_req_ids:
                    scheduled.append(req_id)
            raise_load_bundle_ids: list[int] = []
            for req_id in scheduled:
                self._active_promoted_tasks[req_id] = self._early_promoted_tasks.pop(
                    req_id
                )
                load_bundle_id = self._load_bundle_ids.pop(req_id, None)
                if load_bundle_id is not None:
                    raise_load_bundle_ids.append(load_bundle_id)
            if raise_load_bundle_ids:
                # Before the async gets below; sorting P0 by layer interleaves these with the sync batch.
                self._store().set_bundles_priority(raise_load_bundle_ids, LOAD_PRIORITY_P0)
                self._store().sort_priority_level(LOAD_PRIORITY_P0)

        # Submit asynchronous loads per request; they are polled across
        # scheduler steps in get_finished().
        for req_id, request in async_reqs:
            # First N layers at P1 overtake waiting remaining layers, which stay at P2 until raised.
            # Layers without a per-layer hook must be loaded before the
            # forward; they lead the order, so the count has to cover them.
            prefetch_layers = request.async_load_layers
            if prefetch_layers >= 0:
                prefetch_layers = max(prefetch_layers, self.num_hookless_layers)
            priorities: dict[str, int] = {}
            has_remaining_layers = False
            for layer_index, layer_name in enumerate(self._layer_names):
                # -1 promotes only after every layer is loaded.
                if prefetch_layers == -1 or layer_index < prefetch_layers:
                    priorities[layer_name] = LOAD_PRIORITY_P1
                else:
                    priorities[layer_name] = LOAD_PRIORITY_P2
                    has_remaining_layers = True
            load_bundle_id = self._next_load_bundle_id
            self._next_load_bundle_id += 1
            tasks: dict[str, Any] = {}
            for group_id, group in self.groups.items():
                if not group.layer_names:
                    continue
                block_ids, block_hashes = request.for_group(group_id)
                tasks.update(self._store().get(
                    block_indices=block_ids,
                    block_hashs=block_hashes,
                    layer_names=group.layer_names,
                    description=req_id,
                    label=group.label,
                    priorities=priorities,
                    bundle_id=load_bundle_id,
                ))
            self._pending_load_tasks[req_id] = tasks
            self._pending_load_layers[req_id] = prefetch_layers
            if has_remaining_layers:
                self._load_bundle_ids[req_id] = load_bundle_id

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._current_get_tasks and not self._active_promoted_tasks:
            return
        # Draft (MTP) layers past the base model's hidden layers are not
        # registered, but they do run attention and therefore reach this hook.
        if layer_name not in self._layer_key:
            return

        layer_keys = self._hook_keys[layer_name]
        # Wait for the synchronous (batched) loads for this layer.
        if self._current_get_tasks:
            success = self._store().get_wait(
                get_results=self._current_get_tasks,
                layer_names=layer_keys,
            )
            if not success:
                raise RuntimeError(
                    f"Failed to load KV cache for layer {layer_name}"
                )

        # Wait for the remaining layers of early-promoted async loads. Their
        # first N layers were already finalized in get_finished(); waiting on an
        # already-finalized layer is a no-op.
        for tasks in self._active_promoted_tasks.values():
            success = self._store().get_wait(
                get_results=tasks,
                layer_names=layer_keys,
            )
            if not success:
                raise RuntimeError(
                    f"Failed to load promoted KV cache for layer {layer_name}"
                )

        if layer_name == self._last_layer_name:
            self._current_get_tasks = None
            self._active_promoted_tasks = {}

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # Draft (MTP) layers past the base model's hidden layers are not
        # registered, but they do run attention and therefore reach this hook.
        if self._connector_metadata is None or layer_name not in self._layer_group:
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        # The hook covers this layer's own KV and the caches its module owns;
        # each of them lives in its own group.
        keys_by_group: dict[int, list[str]] = {}
        for key in self._hook_keys[layer_name]:
            group_id = self._key_group.get(key)
            if group_id is not None:
                keys_by_group.setdefault(group_id, []).append(key)
        for req_id, request in metadata.reqs_to_save.requests.items():
            for group_id, layer_keys in keys_by_group.items():
                block_ids, block_hashes = request.for_group(group_id)
                if not block_ids:
                    continue
                tasks = self._store().put(
                    block_indices=block_ids,
                    block_hashs=block_hashes,
                    layer_names=layer_keys,
                    label=self.groups[group_id].label,
                )
                self._current_put_tasks.setdefault(req_id, []).append(tasks)

    def wait_for_save(self) -> None:
        # Layers without a per-layer hook (mamba state, unclaimed caches) are
        # submitted after the forward; nothing waits here: request_finished()
        # keeps the blocks alive until every queued copy lands.
        if self._connector_metadata is None or not self._hookless_keys:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        for req_id, request in metadata.reqs_to_save.requests.items():
            for group_id, layer_keys in self._hookless_keys.items():
                block_ids, block_hashes = request.for_group(group_id)
                if not block_ids:
                    continue
                tasks = self._store().put(
                    block_indices=block_ids,
                    block_hashs=block_hashes,
                    layer_names=layer_keys,
                    label=self.groups[group_id].label,
                )
                self._current_put_tasks.setdefault(req_id, []).append(tasks)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # Poll asynchronous load tasks submitted in start_load_kv().
        finished_recving: set[str] = set()
        for req_id in list(self._pending_load_tasks.keys()):
            tasks = self._pending_load_tasks[req_id]
            async_load_layers = self._pending_load_layers[req_id]
            if async_load_layers == -1:
                # Require all layers before marking the load finished.
                if self._store().get_wait(get_results=tasks, wait=False):
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    finished_recving.add(req_id)
            else:
                # Early promote once the first N layers are loaded; the remaining
                # layers are waited on-demand in wait_for_layer_load().
                first_n_layers = self._layer_names[:async_load_layers]
                if self._store().get_wait(
                    get_results=tasks, layer_names=first_n_layers, wait=False
                ):
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    self._early_promoted_tasks[req_id] = tasks
                    finished_recving.add(req_id)

        self._deferred_finished_req_ids.update(finished_req_ids)
        completed: set[str] = set()

        for req_id in self._deferred_finished_req_ids:
            load_tasks = (
                self._pending_load_tasks.get(req_id)
                or self._early_promoted_tasks.get(req_id)
                or self._active_promoted_tasks.get(req_id)
            )
            if load_tasks is not None:
                if not self._store().get_wait(
                    get_results=load_tasks, wait=False
                ):
                    continue
                self._pending_load_tasks.pop(req_id, None)
                self._pending_load_layers.pop(req_id, None)
                self._early_promoted_tasks.pop(req_id, None)
                self._active_promoted_tasks.pop(req_id, None)
                # Finished before entering a forward: its bundle is never raised.
                self._load_bundle_ids.pop(req_id, None)

            tasks = self._current_put_tasks.get(req_id)
            if tasks is None:
                completed.add(req_id)
                continue

            while tasks and self._store().put_wait(tasks[0], wait=False):
                tasks.pop(0)
            if not tasks:
                self._current_put_tasks.pop(req_id)
                completed.add(req_id)

        self._deferred_finished_req_ids.difference_update(completed)
        return (completed or None), (finished_recving or None)
