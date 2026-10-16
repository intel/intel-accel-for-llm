# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, generate_block_hashs, setup_root_logger
from iaxl.envs import envs as iaxl_envs
from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

from .async_load_config import (
    AsyncLoadScheme,
    load_async_load_layer_config_from_env,
    load_async_load_scheme_from_env,
)

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


@dataclass
class SplitLoadMeta:
    """Blocks of one async request loaded under split_async_load_submit."""
    block_ids: list[int]
    block_hashes: list[str]
    # Layers [0, num_first_n_layers) are submitted on arrival and must finish before
    # promotion; layers [num_first_n_layers, L) are submitted after promotion.
    num_first_n_layers: int


@dataclass
class LayerSegment:
    """Layers [start, end) loaded by one get() for the blocks of ``req_ids``."""
    start: int
    end: int
    req_ids: list[ReqId]


def _layer_ranges(
    loads: dict[ReqId, SplitLoadMeta], num_layers: int
) -> list[tuple[int, int]]:
    # Boundaries are 0, L and every request's N. Between two adjacent boundaries,
    # each request needs either all of the layers or none of them.
    boundaries = {0, num_layers}
    for load in loads.values():
        boundaries.add(load.num_first_n_layers)
    sorted_boundaries = sorted(boundaries)
    ranges = []
    for index in range(len(sorted_boundaries) - 1):
        ranges.append((sorted_boundaries[index], sorted_boundaries[index + 1]))
    return ranges


def _first_n_layer_segments(
    loads: dict[ReqId, SplitLoadMeta], num_layers: int
) -> list[LayerSegment]:
    """Segments covering layers [0, N) of every request, in layer order."""
    segments = []
    for start, end in _layer_ranges(loads, num_layers):
        # Layers [start, end) are among a request's first N layers iff N >= end.
        req_ids = []
        for req_id, load in loads.items():
            if load.num_first_n_layers >= end:
                req_ids.append(req_id)
        if req_ids:
            segments.append(LayerSegment(start, end, req_ids))
    return segments


def _remaining_layer_segments(
    loads: dict[ReqId, SplitLoadMeta], num_layers: int
) -> list[LayerSegment]:
    """Segments covering layers [N, L) of every request, in layer order."""
    segments = []
    for start, end in _layer_ranges(loads, num_layers):
        # Layers [start, end) are among a request's remaining layers iff N <= start.
        req_ids = []
        for req_id, load in loads.items():
            if load.num_first_n_layers <= start:
                req_ids.append(req_id)
        if req_ids:
            segments.append(LayerSegment(start, end, req_ids))
    return segments


@dataclass
class ReqMeta:
    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqState:
    num_seen_blocks: int = 0
    num_computed_tokens: int = 0
    existence_cache: list[bool] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        block_ids: list[int],
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


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata
    # Requests scheduled in this step (keys of SchedulerOutput.num_scheduled_tokens).
    scheduled_req_ids: set[ReqId] = field(default_factory=set)


class KVShrinkConnector(KVConnectorBase_V1):
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
        self.block_size = vllm_config.cache_config.block_size
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
        # Ordered worker-side layer names (populated in register_kv_caches),
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
        # split_async_load_submit: blocks of async requests whose remaining layers
        # are not submitted yet. Kept because reqs_to_load no longer carries a
        # request after its first submission.
        self._remaining_load_meta: dict[ReqId, SplitLoadMeta] = {}
        # Promoted requests whose remaining layers the next start_load_kv() submits.
        self._pending_remaining_load_req_ids: list[ReqId] = []
        # Worker-side step counter for the async_load debug logs.
        self._load_step = 0

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
        )
        self._async_load_scheme = load_async_load_scheme_from_env()
        logger.info(
            "KVShrink async load scheme: %s, role=%s",
            self._async_load_scheme.value,
            role.name,
        )

        if role == KVConnectorRole.SCHEDULER:
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

        block_hashes = [
            str(block_hash)
            for block_hash in generate_block_hashs(
                request.all_token_ids[:-1], self.block_size
            )
        ]
        existence_cache = self._store().has(block_hashes)
        state = ReqState(
            num_computed_tokens=num_computed_tokens,
            existence_cache=existence_cache,
            block_hashes=block_hashes,
        )
        self._req_states[request.request_id] = state

        matched_blocks = next(
            (
                index
                for index, exists in enumerate(existence_cache)
                if not exists
            ),
            len(existence_cache),
        )
        matched_tokens = matched_blocks * self.block_size
        num_new_tokens = max(0, matched_tokens - num_computed_tokens)

        # Decide sync vs async for this request. The load can only be async when
        # there are external tokens to load and async is enabled. Concurrency is
        # approximated by the number of in-flight requests (this one included).
        selected_layers = self._async_load_layer_config.select(
            len(self._req_states)
        )
        # A dynamic-map layer value of 0 selects synchronous loading. It is not
        # an async request that resumes before layer 0.
        use_async = num_new_tokens > 0 and selected_layers != 0
        state.is_async = use_async
        if use_async:
            state.async_load_layers = selected_layers

        logger.info(
            f"get_num_new_matched_tokens, req-{request.request_id}, "
            f"externally-cached tokens: {num_new_tokens}, "
            f"locally-cached tokens: {num_computed_tokens}, async={use_async}, "
            f"selected_load_layers={selected_layers}, "
            f"async_load_layers={state.async_load_layers}"
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

        block_ids = blocks.get_block_ids()[0]
        load_start = state.num_computed_tokens // self.block_size
        load_end = min(
            load_start + num_external_tokens // self.block_size,
            len(block_ids),
            len(state.block_hashes),
        )
        if load_end <= load_start:
            return

        self._reqs_to_load.add_request(
            request.request_id,
            list(block_ids[load_start:load_end]),
            state.block_hashes[load_start:load_end],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def _add_request_to_save(
        self, req_id: ReqId, new_block_ids: list[int]
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        start = state.num_seen_blocks
        end = start + len(new_block_ids)
        block_hashes = state.block_hashes[start:end]
        existence = state.existence_cache[start:end]
        missing = [
            (block_id, block_hash)
            for block_id, block_hash, exists in zip(
                new_block_ids, block_hashes, existence
            )
            if not exists
        ]
        if missing:
            block_ids, hashes = zip(*missing)
            self._reqs_to_save.add_request(
                req_id, list(block_ids), list(hashes)
            )
        state.num_seen_blocks = end

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # True = defer freeing to get_finished() (async load/save may still run).
        self._req_states.pop(request.request_id, None)
        return True, None

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        for request in scheduler_output.scheduled_new_reqs:
            if request.block_ids:
                self._add_request_to_save(request.req_id, request.block_ids[0])

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                raise RuntimeError("Resuming from preemption is not supported")

            block_ids = cached_reqs.new_block_ids[index]
            is_prefill = scheduler_output.num_scheduled_tokens[req_id] > 1
            if block_ids and block_ids[0] and is_prefill:
                self._add_request_to_save(req_id, block_ids[0])

        # kvshrink loads asynchronously, but its start_load_kv only submits
        # host-side work; running it before the forward keeps the 0.23-style
        # placement (0.29 would otherwise defer it to post_forward).
        scheduler_output.has_sync_kv_loads = True
        metadata = KVShrinkConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=self._reqs_to_save,
            scheduled_req_ids=set(scheduler_output.num_scheduled_tokens),
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
            spec = spec_by_layer[layer_name]
            ref = group_kernel_blocks(cache, num_blocks)
            page_bytes = spec.page_size_bytes
            assert page_bytes % attn_dtype.itemsize == 0
            views[layer_name] = torch.tensor(
                [], dtype=attn_dtype, device=ref.device
            ).set_(
                ref.untyped_storage(),
                ref.storage_offset(),
                (num_blocks, page_bytes // attn_dtype.itemsize),
                (
                    ref.stride(0)
                    if isinstance(spec, AttentionSpec)
                    else page_bytes // attn_dtype.itemsize,
                    1,
                ),
            )
        return views

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        static_context = self.vllm_config.compilation_config.static_forward_context
        for layer in static_context.values():
            get_backend = getattr(layer, "get_attn_backend", None)
            if get_backend is not None:
                if "FLASHINFER" in get_backend().get_name().upper():
                    raise RuntimeError("FlashInfer is not supported")
                break

        kv_caches = self._view_as_blocks(kv_caches)
        first_kv_cache = next(iter(kv_caches.values()))
        self._last_layer_name = next(reversed(kv_caches))
        self._layer_names = list(kv_caches.keys())
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            block_dim=0,
            kv_caches=kv_caches,
            rank=self.global_rank,
            tp_size=self.tp_size,
        )
        logger.info(
            "Registered %d KV cache layers with shape %s",
            len(kv_caches),
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
        self._load_step += 1

        sync_block_ids: list[int] = []
        sync_block_hashes: list[str] = []
        async_reqs: list[tuple[ReqId, ReqMeta]] = []
        for req_id, request in metadata.reqs_to_load.requests.items():
            if len(request.block_ids) != len(request.block_hashes):
                raise ValueError(f"Mismatched block metadata for request {req_id}")
            if not request.block_ids:
                continue
            if request.is_async:
                async_reqs.append((req_id, request))
            else:
                sync_block_ids.extend(request.block_ids)
                sync_block_hashes.extend(request.block_hashes)

        # Submit synchronous (blocking) loads first as a single merged batch so
        # they are enqueued ahead of the asynchronous loads for this pass.
        self._current_get_tasks = None
        if sync_block_ids:
            self._current_get_tasks = self._store().get(
                block_indices=sync_block_ids,
                block_hashs=sync_block_hashes,
            )

        # Asynchronous loads are polled across scheduler steps in get_finished().
        if self._async_load_scheme is AsyncLoadScheme.SPLIT_ASYNC_LOAD_SUBMIT:
            self._submit_split_loads(async_reqs)
        elif self._async_load_scheme is AsyncLoadScheme.NAIVE:
            # One get() per request: each request's layers are queued contiguously.
            for req_id, request in async_reqs:
                self._pending_load_tasks[req_id] = self._store().get(
                    block_indices=request.block_ids,
                    block_hashs=request.block_hashes,
                    description=req_id,
                )
                self._pending_load_layers[req_id] = request.async_load_layers
        elif async_reqs:
            # batch_reqs_async_load_submit: like the sync path, one get() for all
            # async requests of this step, so each layer is a single task covering
            # all of them. Every request references the same Task dict;
            # KVFlow.get_wait() skips tasks already waited on, so per-request
            # promotion and cleanup stay unchanged.
            async_block_ids: list[int] = []
            async_block_hashes: list[str] = []
            for req_id, request in async_reqs:
                async_block_ids.extend(request.block_ids)
                async_block_hashes.extend(request.block_hashes)
            tasks = self._store().get(
                block_indices=async_block_ids,
                block_hashs=async_block_hashes,
            )
            for req_id, request in async_reqs:
                self._pending_load_tasks[req_id] = tasks
                self._pending_load_layers[req_id] = request.async_load_layers

        # Only promoted requests scheduled in this forward are waited on in
        # wait_for_layer_load(); the rest stay early-promoted until the step that
        # schedules them. A no-forward batch waits on nothing. Done after all
        # submissions, so a promoted request is always in _early_promoted_tasks
        # when _submit_split_loads() submits its remaining layers.
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
            for req_id in scheduled:
                self._active_promoted_tasks[req_id] = self._early_promoted_tasks.pop(
                    req_id
                )
            if scheduled or self._early_promoted_tasks:
                logger.debug(
                    "async_load step=%d t=%d activate reqs=%s held=%s",
                    self._load_step,
                    time.monotonic_ns(),
                    ",".join(scheduled),
                    ",".join(self._early_promoted_tasks),
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._current_get_tasks and not self._active_promoted_tasks:
            return

        # Wait for the synchronous (batched) loads for this layer.
        if self._current_get_tasks:
            success = self._store().get_wait(
                get_results=self._current_get_tasks,
                layer_names=[layer_name],
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
                layer_names=[layer_name],
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
        if self._connector_metadata is None:
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        for req_id, request in metadata.reqs_to_save.requests.items():
            if not request.block_ids:
                continue
            tasks = self._store().put(
                block_indices=request.block_ids,
                block_hashs=request.block_hashes,
                layer_names=[layer_name],
            )
            self._current_put_tasks.setdefault(req_id, []).append(tasks)

    def wait_for_save(self) -> None:
        return

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
                    self._store().get_wait(get_results=tasks, wait=True)
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
                    self._store().get_wait(
                        get_results=tasks, layer_names=first_n_layers, wait=True
                    )
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    self._early_promoted_tasks[req_id] = tasks
                    finished_recving.add(req_id)
                    # Only split requests that still have remaining layers to
                    # submit are in _remaining_load_meta.
                    if req_id in self._remaining_load_meta:
                        self._pending_remaining_load_req_ids.append(req_id)

        if finished_recving:
            logger.debug(
                "async_load step=%d t=%d promote reqs=%s",
                self._load_step,
                time.monotonic_ns(),
                ",".join(sorted(finished_recving)),
            )

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
                self._store().get_wait(get_results=load_tasks, wait=True)
                self._pending_load_tasks.pop(req_id, None)
                self._pending_load_layers.pop(req_id, None)
                self._early_promoted_tasks.pop(req_id, None)
                self._active_promoted_tasks.pop(req_id, None)
                # Finished before its remaining layers were submitted: never
                # submit them.
                self._remaining_load_meta.pop(req_id, None)
                if req_id in self._pending_remaining_load_req_ids:
                    self._pending_remaining_load_req_ids.remove(req_id)

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

    def _submit_split_loads(self, async_reqs: list[tuple[ReqId, ReqMeta]]) -> None:
        # Remaining layers of the requests promoted in the last get_finished() are
        # submitted before the first N layers of new requests: a forward of this
        # step may already wait on the former, while the latter are not needed
        # before their promotion.
        if self._pending_remaining_load_req_ids:
            remaining_layer_loads: dict[ReqId, SplitLoadMeta] = {}
            for req_id in self._pending_remaining_load_req_ids:
                # Removed here: the metadata is not needed after this submission.
                remaining_layer_loads[req_id] = self._remaining_load_meta.pop(req_id)
            self._pending_remaining_load_req_ids = []
            remaining_layer_tasks = self._submit_layer_segments(
                _remaining_layer_segments(
                    remaining_layer_loads, len(self._layer_names)
                ),
                remaining_layer_loads,
            )
            # The request's Task dict holds its first N layers and is referenced
            # by _early_promoted_tasks; adding the remaining layers in place lets
            # wait_for_layer_load() wait on every layer.
            for req_id, tasks in remaining_layer_tasks.items():
                self._early_promoted_tasks[req_id].update(tasks)

        if not async_reqs:
            return
        num_layers = len(self._layer_names)
        first_n_layer_loads: dict[ReqId, SplitLoadMeta] = {}
        for req_id, request in async_reqs:
            if request.async_load_layers == -1:
                # Promoted only after all layers are loaded: nothing remains.
                num_first_n_layers = num_layers
            elif 0 < request.async_load_layers < num_layers:
                num_first_n_layers = request.async_load_layers
            else:
                raise ValueError(
                    f"async_load_layers of request {req_id} must be -1 or in "
                    f"[1, {num_layers}), got {request.async_load_layers}"
                )
            first_n_layer_loads[req_id] = SplitLoadMeta(
                block_ids=request.block_ids,
                block_hashes=request.block_hashes,
                num_first_n_layers=num_first_n_layers,
            )

        first_n_layer_tasks = self._submit_layer_segments(
            _first_n_layer_segments(first_n_layer_loads, num_layers),
            first_n_layer_loads,
        )
        for req_id, request in async_reqs:
            # get_finished() polls these tasks and promotes the request once its
            # first N layers are loaded.
            self._pending_load_tasks[req_id] = first_n_layer_tasks[req_id]
            self._pending_load_layers[req_id] = request.async_load_layers
            if first_n_layer_loads[req_id].num_first_n_layers < num_layers:
                self._remaining_load_meta[req_id] = first_n_layer_loads[req_id]

    def _submit_layer_segments(
        self,
        segments: list[LayerSegment],
        loads: dict[ReqId, SplitLoadMeta],
    ) -> dict[ReqId, dict[str, Any]]:
        # One get() per segment. Returns each request's tasks (layer name -> Task)
        # over all segments it is in; requests of one segment share its tasks.
        tasks_by_req: dict[ReqId, dict[str, Any]] = {}
        for req_id in loads:
            tasks_by_req[req_id] = {}
        for segment in segments:
            block_ids: list[int] = []
            block_hashes: list[str] = []
            for req_id in segment.req_ids:
                block_ids.extend(loads[req_id].block_ids)
                block_hashes.extend(loads[req_id].block_hashes)
            tasks = self._store().get(
                block_indices=block_ids,
                block_hashs=block_hashes,
                layer_names=self._layer_names[segment.start : segment.end],
            )
            for req_id in segment.req_ids:
                tasks_by_req[req_id].update(tasks)
        return tasks_by_req
