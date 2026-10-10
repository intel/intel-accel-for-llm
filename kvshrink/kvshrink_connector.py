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

from iaxl import KVStore, generate_block_hashs, setup_root_logger
from iaxl.envs import envs as iaxl_envs
from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

from .async_load_config import load_async_load_layer_config_from_env

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


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


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata


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
        # block size, which is the enlarged one on hybrid models.
        self.groups: list[tuple[bool, list[str]]] = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            self.groups.append((isinstance(spec, MambaSpec), list(group.layer_names)))
        block_sizes = {g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups}
        assert len(block_sizes) == 1, block_sizes
        self.block_size = block_sizes.pop()
        self.has_mamba = any(is_mamba for is_mamba, _ in self.groups)
        self.num_mamba_layers = sum(
            len(names) for is_mamba, names in self.groups if is_mamba
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

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
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
        if self.vllm_device == "cpu":
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
        if self.has_mamba:
            # A hit must end at a block whose mamba state was saved.
            state.mamba_existence_cache = self._store().has(
                block_hashes, label="mamba", truncate=False
            )
            while (
                matched_blocks > 0
                and not state.mamba_existence_cache[matched_blocks - 1]
            ):
                matched_blocks -= 1
        matched_tokens = matched_blocks * self.block_size
        num_new_tokens = max(0, matched_tokens - num_computed_tokens)

        # Decide sync vs async for this request. The load can only be async when
        # there are external tokens to load and async is enabled. Concurrency is
        # approximated by the number of in-flight requests (this one included).
        selected_layers = self._async_load_layer_config.select(
            len(self._req_states)
        )
        # Mamba layers lead the worker's layer order and have no layer-load
        # hook, so the early-start window must cover them (and hybrid loads are
        # never synchronous).
        if self.has_mamba and selected_layers >= 0:
            selected_layers += self.num_mamba_layers
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

        block_ids = blocks.get_block_ids()
        load_start = state.num_computed_tokens // self.block_size
        load_end = load_start + num_external_tokens // self.block_size
        assert load_end <= len(state.block_hashes)

        # A mamba table is null up to the hit boundary, so the same slice
        # leaves only the slot that receives the saved state.
        self._reqs_to_load.add_request(
            request.request_id,
            tuple(list(ids[load_start:load_end]) for ids in block_ids),
            state.block_hashes[load_start:load_end],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def _add_request_to_save(
        self, req_id: ReqId, num_computed_tokens: int, num_scheduled_tokens: int
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        num_tokens = num_computed_tokens + num_scheduled_tokens
        start = num_computed_tokens // self.block_size
        end = min(num_tokens // self.block_size, len(state.block_hashes))
        if end <= start:
            return

        # Attention saves every full block; mamba only holds the state at the
        # end of a block-aligned chunk, in slot end - 1.
        save_mamba = (
            self.has_mamba
            and end * self.block_size == num_tokens
            and not state.mamba_existence_cache[end - 1]
        )
        block_ids = tuple(
            [0] * (end - start - 1) + [ids[end - 1] if save_mamba else 0]
            if is_mamba
            else [
                0 if state.existence_cache[index] else ids[index]
                for index in range(start, end)
            ]
            for (is_mamba, _), ids in zip(self.groups, state.block_ids)
        )
        if any(any(ids) for ids in block_ids):
            self._reqs_to_save.add_request(
                req_id, block_ids, state.block_hashes[start:end]
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
            state.block_ids = [list(ids) for ids in request.block_ids]
            self._add_request_to_save(
                request.req_id,
                request.num_computed_tokens,
                num_scheduled_tokens[request.req_id],
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                raise RuntimeError("Resuming from preemption is not supported")

            state = self._req_states[req_id]
            new_block_ids = cached_reqs.new_block_ids[index]
            if new_block_ids:
                for ids, new_ids in zip(state.block_ids, new_block_ids):
                    ids.extend(new_ids)
            # Decode steps start past the prompt's hashes and save nothing.
            self._add_request_to_save(
                req_id,
                cached_reqs.num_computed_tokens[index],
                num_scheduled_tokens[req_id],
            )

        # kvshrink loads asynchronously, but its start_load_kv only submits
        # host-side work; running it before the forward keeps the 0.23-style
        # placement (0.29 would otherwise defer it to post_forward).
        scheduler_output.has_sync_kv_loads = True
        metadata = KVShrinkConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=self._reqs_to_save,
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
            # Mamba caches are int8 views: rescale offset/stride by bytes.
            itemsize = attn_dtype.itemsize
            offset = ref.storage_offset() * ref.element_size()
            stride = ref.stride(0) * ref.element_size()
            page = spec_by_layer[layer_name].page_size_bytes
            assert offset % itemsize == 0 and stride % itemsize == 0
            assert page % itemsize == 0
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
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        # Skip draft (MTP) layers; put mamba layers first so the async
        # early-start window covers them.
        mamba_layers = {
            ln for is_mamba, names in self.groups if is_mamba for ln in names
        }
        kv_caches = {
            ln: kv_caches[ln]
            for ln in sorted(kv_caches, key=lambda ln: ln not in mamba_layers)
            if extract_layer_index(ln) < self.num_layers
        }
        self.groups = [
            (is_mamba, [ln for ln in names if ln in kv_caches])
            for is_mamba, names in self.groups
        ]
        self.layer_group = {
            ln: index for index, (_, names) in enumerate(self.groups) for ln in names
        }

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

        # A no-forward batch cannot consume promoted tasks layer by layer.
        if forward_context.attn_metadata is not None:
            duplicates = (
                self._active_promoted_tasks.keys()
                & self._early_promoted_tasks.keys()
            )
            if duplicates:
                raise RuntimeError(
                    f"Duplicate promoted load tasks for requests {duplicates}"
                )
            self._active_promoted_tasks.update(self._early_promoted_tasks)
            self._early_promoted_tasks = {}

        sync_block_ids: list[int] = []
        sync_block_hashes: list[str] = []
        async_reqs: list[tuple[ReqId, ReqMeta]] = []
        for req_id, request in metadata.reqs_to_load.requests.items():
            assert all(
                len(ids) == len(request.block_hashes) for ids in request.block_ids
            ), req_id
            if request.is_async:
                async_reqs.append((req_id, request))
            else:
                # Hybrid loads are always async: vLLM still zeroes the blocks
                # of a synchronous load on the compute stream.
                assert not self.has_mamba
                sync_block_ids.extend(request.block_ids[0])
                sync_block_hashes.extend(request.block_hashes)

        # Submit synchronous (blocking) loads first as a single merged batch so
        # they are enqueued ahead of the asynchronous loads for this pass.
        self._current_get_tasks = None
        if sync_block_ids:
            self._current_get_tasks = self._store().get(
                block_indices=sync_block_ids,
                block_hashs=sync_block_hashes,
            )

        # Submit asynchronous loads per request; they are polled across
        # scheduler steps in get_finished().
        for req_id, request in async_reqs:
            tasks: dict[str, Any] = {}
            for index, (is_mamba, layer_names) in enumerate(self.groups):
                if not layer_names:  # draft-only group
                    continue
                block_ids, block_hashes = request.for_group(index)
                tasks.update(self._store().get(
                    block_indices=block_ids,
                    block_hashs=block_hashes,
                    layer_names=layer_names,
                    description=req_id,
                    label="mamba" if is_mamba else None,
                ))
            self._pending_load_tasks[req_id] = tasks
            self._pending_load_layers[req_id] = request.async_load_layers

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
        # Draft (MTP) layers are not registered.
        if self._connector_metadata is None or layer_name not in self.layer_group:
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        for req_id, request in metadata.reqs_to_save.requests.items():
            block_ids, block_hashes = request.for_group(self.layer_group[layer_name])
            if not block_ids:
                continue
            tasks = self._store().put(
                block_indices=block_ids,
                block_hashs=block_hashes,
                layer_names=[layer_name],
            )
            self._current_put_tasks.setdefault(req_id, []).append(tasks)

    def wait_for_save(self) -> None:
        # Mamba layers have no save hook: submit their states after forward.
        if self._connector_metadata is None or not self.has_mamba:
            return
        metadata = self._get_connector_metadata()
        for req_id, request in metadata.reqs_to_save.requests.items():
            for index, (is_mamba, layer_names) in enumerate(self.groups):
                block_ids, block_hashes = request.for_group(index)
                if not is_mamba or not block_ids:
                    continue
                tasks = self._store().put(
                    block_indices=block_ids,
                    block_hashs=block_hashes,
                    layer_names=layer_names,
                    label="mamba",
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
