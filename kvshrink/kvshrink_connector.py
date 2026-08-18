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
)
from vllm.distributed.parallel_state import (
    get_world_group,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, generate_block_hashs, setup_root_logger

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

# Async KV load concurrency threshold:
#   -1 = always sync (default), 0 = always async,
#   N (>0) = use async load when the number of in-flight requests >= N.
LOAD_KV_ASYNC_THRESHOLD = int(
    os.getenv("KVSHRINK_VLLM_KV_ASYNC_LOAD_THRESHOLD", "-1")
)
# Async early-start layer count (requires LOAD_KV_ASYNC_THRESHOLD >= 0):
#   -1 = disabled (wait for all layers before marking the load finished),
#   N (>=1) = mark the load finished once the first N layers are loaded; the
#   remaining layers are waited on-demand during the prefill forward pass.
ASYNC_LOAD_LAYER = int(os.getenv("KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS", "-1"))

ReqId = str

@dataclass
class ReqMeta:
    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False


@dataclass
class ReqState:
    num_seen_blocks: int = 0
    num_computed_tokens: int = 0
    existence_cache: list[bool] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        block_ids: list[int],
        block_hashes: list[str],
        is_async: bool = False,
    ) -> None:
        self.requests[req_id] = ReqMeta(block_ids, block_hashes, is_async)


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata


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
        self.model_config = vllm_config.model_config
        self.block_size = vllm_config.cache_config.block_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.use_mla = self.model_config.use_mla
        self.vllm_device = vllm_config.device_config.device_type
        self.rank = get_world_group().rank if model_parallel_is_initialized() else 0

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
        # Tasks early-promoted (first N layers done) whose remaining layers are
        # waited on-demand in wait_for_layer_load during the prefill forward.
        self._early_promoted_tasks: dict[ReqId, dict[str, Any]] = {}
        # Early-promoted tasks active for the current forward pass.
        self._active_promoted_tasks: dict[ReqId, dict[str, Any]] = {}

        if LOAD_KV_ASYNC_THRESHOLD < -1:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_THRESHOLD must be at least -1, "
                f"got {LOAD_KV_ASYNC_THRESHOLD}"
            )
        if ASYNC_LOAD_LAYER != -1 and not (1 <= ASYNC_LOAD_LAYER < self.num_layers):
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS must be -1 or in "
                f"[1, {self.num_layers}), got {ASYNC_LOAD_LAYER}"
            )
        if ASYNC_LOAD_LAYER != -1 and LOAD_KV_ASYNC_THRESHOLD == -1:
            raise ValueError(
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS requires "
                "KVSHRINK_VLLM_KV_ASYNC_LOAD_THRESHOLD >= 0"
            )

        if role == KVConnectorRole.SCHEDULER:
            self.kvstore: Optional[KVStore] = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(index) for index in range(self.num_layers)],
                tp_size=self.tp_size,
            )
        else:
            self.kvstore = None
            self._bind_cpu_affinity()
            self._bind_intel_accel()

    def _bind_cpu_affinity(self) -> None:
        if self.vllm_device == "cpu":
            return

        omp_bind = envs.VLLM_CPU_OMP_THREADS_BIND
        if not omp_bind or omp_bind in ("all", "auto"):
            raise ValueError(
                "VLLM_CPU_OMP_THREADS_BIND must assign CPUs to each worker"
            )

        worker_cpu_specs = omp_bind.split("|")
        if len(worker_cpu_specs) < self.tp_size:
            raise ValueError(
                f"VLLM_CPU_OMP_THREADS_BIND has {len(worker_cpu_specs)} entries, "
                f"but tensor parallel size is {self.tp_size}"
            )

        cpu_ids: set[int] = set()
        for part in worker_cpu_specs[self.rank].split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-", maxsplit=1))
                if start > end:
                    raise ValueError(f"Invalid CPU range: {part}")
                cpu_ids.update(range(start, end + 1))
            else:
                cpu_ids.add(int(part))

        if not cpu_ids:
            raise ValueError(f"No CPUs configured for rank {self.rank}")
        os.sched_setaffinity(0, cpu_ids)
        logger.info("Bound rank %d to CPUs %s", self.rank, sorted(cpu_ids))

    def _bind_intel_accel(self) -> None:
        for source, target in (
            ("KVSHRINK_QAT_DEVICES", "IAXL_QAT_DEVICES"),
            ("KVSHRINK_DSA_DEVICES", "IAXL_DSA_WQS"),
        ):
            spec = os.getenv(source)
            if not spec:
                continue
            devices = spec.split("|")
            if len(devices) <= self.rank:
                raise ValueError(
                    f"{source} has {len(devices)} entries, but rank is {self.rank}"
                )
            os.environ[target] = devices[self.rank]
            logger.info("Bound rank %d: %s=%s", self.rank, target, devices[self.rank])

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
        use_async = (
            num_new_tokens > 0
            and LOAD_KV_ASYNC_THRESHOLD >= 0
            and len(self._req_states) >= LOAD_KV_ASYNC_THRESHOLD
        )
        state.is_async = use_async

        logger.info(
            f"get_num_new_matched_tokens, req-{request.request_id}, "
            f"externally-cached tokens: {num_new_tokens}, "
            f"locally-cached tokens: {num_computed_tokens}, async={use_async}"
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

        first_kv_cache = next(iter(kv_caches.values()))
        block_dim = 0 if self.use_mla or first_kv_cache.shape[1] == 2 else 1
        self._last_layer_name = next(reversed(kv_caches))
        self._layer_names = list(kv_caches.keys())
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            block_dim=block_dim,
            kv_caches=kv_caches,
            rank=self.rank,
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

        # Submit asynchronous loads per request; they are polled across
        # scheduler steps in get_finished().
        for req_id, request in async_reqs:
            self._pending_load_tasks[req_id] = self._store().get(
                block_indices=request.block_ids,
                block_hashs=request.block_hashes,
                description=req_id,
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
            if ASYNC_LOAD_LAYER == -1:
                # Require all layers before marking the load finished.
                if self._store().get_wait(get_results=tasks, wait=False):
                    self._store().get_wait(get_results=tasks, wait=True)
                    del self._pending_load_tasks[req_id]
                    finished_recving.add(req_id)
            else:
                # Early promote once the first N layers are loaded; the remaining
                # layers are waited on-demand in wait_for_layer_load().
                first_n_layers = self._layer_names[:ASYNC_LOAD_LAYER]
                if self._store().get_wait(
                    get_results=tasks, layer_names=first_n_layers, wait=False
                ):
                    self._store().get_wait(
                        get_results=tasks, layer_names=first_n_layers, wait=True
                    )
                    del self._pending_load_tasks[req_id]
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
