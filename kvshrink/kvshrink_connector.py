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
    get_world_group,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
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
from .kv_cache_pages import bind_kv_caches

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


@dataclass
class ReqMeta:
    group_block_ids: tuple[tuple[int, ...], ...] = ()
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1

    def for_group(self, group_idx: int) -> tuple[list[int], list[str]]:
        """Pair group blocks with shared hashes, omitting null state slots."""
        pairs = [(block_id, block_hash) for block_id, block_hash in zip(
            self.group_block_ids[group_idx], self.block_hashes) if block_id != 0]
        if not pairs:
            return [], []
        block_ids, block_hashes = zip(*pairs)
        return list(block_ids), list(block_hashes)


@dataclass
class ReqState:
    num_computed_tokens: int = 0
    # Presence per namespace ("kv"/"mamba"), truncated at the first miss,
    # from one batched store query per namespace at request start.
    existence_cache: dict[str, list[bool]] = field(default_factory=dict)
    # Reference to the vLLM request's block_hashes list.
    block_hashes: list = field(default_factory=list)
    group_block_ids: list[list[int]] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        group_block_ids: tuple[tuple[int, ...], ...],
        block_hashes: list[str],
        is_async: bool = False,
        async_load_layers: int = -1,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            group_block_ids,
            [hash_str(h) for h in block_hashes],
            is_async,
            async_load_layers,
        )


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata


@dataclass(frozen=True)
class GroupInfo:
    group_idx: int
    kind: str  # "attention" | "mamba"
    layer_names: tuple[str, ...]
    spec: object = None


def hash_str(block_hash) -> str:
    return block_hash.hex() if isinstance(block_hash, bytes) else str(block_hash)


def find_longest_prefix(
    kv_flags: list[bool],
    mamba_flags: Optional[list[bool]],
    block_size: int,
    num_tokens: int,
) -> int:
    """Longest restorable prefix, in tokens."""
    limit = (num_tokens - 1) // block_size
    blocks = 0
    for exists in kv_flags[:limit]:
        if not exists:
            break
        blocks += 1
    if mamba_flags is not None:
        while blocks > 0 and not mamba_flags[blocks - 1]:
            blocks -= 1
    return blocks * block_size


def parse_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> tuple[list[GroupInfo], int, int, int]:
    """Parse vLLM KV cache groups, common block size, block count and page size.
    All groups must share the same block and page size."""
    groups: list[GroupInfo] = []
    sizes: set[int] = set()
    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        spec = g.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[g.layer_names[0]]
        kind = "mamba" if isinstance(spec, MambaSpec) else "attention"
        sizes.add(int(spec.block_size))
        groups.append(GroupInfo(g_idx, kind, tuple(g.layer_names), spec))
    if len(sizes) != 1:
        raise RuntimeError(
            f"kvshrink requires a common block size across groups, got {sorted(sizes)}"
        )
    return (groups, sizes.pop(), int(kv_cache_config.num_blocks),
            int(groups[0].spec.page_size_bytes))


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
        self.model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
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
        # Early-start layer count selected for each pending async request.
        self._pending_load_layers: dict[ReqId, int] = {}
        # Tasks early-promoted (first N layers done) whose remaining layers are
        # waited on-demand in wait_for_layer_load during the prefill forward.
        self._early_promoted_tasks: dict[ReqId, dict[str, Any]] = {}
        # Early-promoted tasks active for the current forward pass.
        self._active_promoted_tasks: dict[ReqId, dict[str, Any]] = {}

        (self.groups, self.block_size, self.num_blocks,
         self.page_bytes) = parse_kv_cache_config(kv_cache_config)
        self.has_mamba = any(g.kind == "mamba" for g in self.groups)
        self.mamba_layers = frozenset(
            ln for g in self.groups if g.kind == "mamba" for ln in g.layer_names)
        self.layer_group = {
            ln: g.group_idx for g in self.groups for ln in g.layer_names}

        # The configured layer counts are attention layers; mamba layers are
        # always waited for before the forward and are added by the config.
        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
            num_mamba_layers=len(self.mamba_layers),
        )

        if role == KVConnectorRole.SCHEDULER:
            self.kvstore: Optional[KVStore] = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(index) for index in range(self.num_layers)],
                tp_size=self.tp_size,
            )
        else:
            self.kvstore = None
            if not iaxl_envs.IAXL_RDMA_ENABLE:  # compression/DSA run on the daemon node
                self._bind_cpu_affinity()
                self._bind_intel_accel()

        logger.info(
            "kvshrink hybrid path enabled (%s role, tp=%d rank=%d, "
            "block_size=%d, groups=%s)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            self.tp_size, self.rank, self.block_size,
            [(g.group_idx, g.kind) for g in self.groups])

    def _bind_cpu_affinity(self) -> None:
        if self.vllm_device == "cpu":
            return
        bind_cpu_affinity(self.rank, self.tp_size, envs.VLLM_CPU_OMP_THREADS_BIND)

    def _bind_intel_accel(self) -> None:
        bind_intel_accel(self.rank)

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

        state = ReqState(
            num_computed_tokens=num_computed_tokens,
            block_hashes=request.block_hashes,
        )
        self._req_states[request.request_id] = state

        # One batched presence query per namespace, then a prefix scan.
        hashes = [hash_str(h) for h in state.block_hashes]
        kv_flags = self._store().has(hashes, truncate=False)
        existence = {"kv": kv_flags}
        mamba_flags = None
        if self.has_mamba:
            mamba_flags = self._store().has(
                hashes, label="mamba", truncate=False)
            existence["mamba"] = mamba_flags
        state.existence_cache = existence
        matched_tokens = find_longest_prefix(
            kv_flags, mamba_flags, self.block_size, request.num_tokens)
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

        block_ids = blocks.get_block_ids()
        state.group_block_ids = [list(ids) for ids in block_ids]
        if num_external_tokens == 0:
            return
        if num_external_tokens % self.block_size != 0:
            raise ValueError("External token count must be block aligned")

        load_start = state.num_computed_tokens // self.block_size
        # The scheduler guarantees the endpoint: local+external <= num_tokens
        # (its own assert) and the allocation covers local+external, so both
        # block_hashes and every attention group table are long enough.
        load_end = load_start + num_external_tokens // self.block_size
        if load_end <= load_start:
            return
        group_ids = [tuple(ids[load_start:load_end]) for ids in block_ids]
        for g_idx, group in enumerate(self.groups):
            if group.kind == "mamba":
                # Restore Mamba state directly into the execution slot (-1 - num_spec),
                # prepending 0 sentinels so hashes[-1] aligns with the target block ID.
                num_spec = getattr(group.spec, "num_speculative_blocks", 0)
                group_ids[g_idx] = tuple(
                    [0] * (load_end - load_start - 1)
                    + [block_ids[g_idx][-1 - num_spec]])
        state.num_computed_tokens += num_external_tokens
        self._reqs_to_load.add_request(
            request.request_id,
            tuple(group_ids),
            state.block_hashes[load_start:load_end],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def _add_request_to_save(
        self, req_id: ReqId, scheduled_tokens: int
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        start = state.num_computed_tokens // self.block_size
        # Prefill steps stay within the prompt, whose hashes cover every
        # full block; the decode overshoot case never reaches here (the
        # num_output_tokens gate filters it).
        end = min(
            (state.num_computed_tokens + scheduled_tokens) // self.block_size,
            len(state.block_hashes),
        )
        # Only save blocks the store is missing in every namespace. A block
        # present in one namespace but not another is still written for all
        # groups; the put is merely redundant for the namespace that has it.
        missing = [index for index in range(start, end)
                   if not self._exists_everywhere(state, index)]
        if not missing:
            return
        block_hashes = [state.block_hashes[index] for index in missing]
        block_ids = tuple(
            tuple(ids[index] for index in missing)
            for ids in state.group_block_ids
        )
        self._reqs_to_save.add_request(req_id, block_ids, block_hashes)

    def _exists_everywhere(self, state: ReqState, index: int) -> bool:
        """Whether block `index` is already present in every namespace."""
        if not state.existence_cache:
            return False
        for flags in state.existence_cache.values():
            if index >= len(flags) or not flags[index]:
                return False
        return True

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
        """SupportsHMA entry point (v0.23 calls this for hybrid models)."""
        return self.request_finished(request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        # A request's first schedule is always prefill. 1-token schedules
        # (full-hit-minus-one tail, single-token prompt) complete no full
        # block, so the save window is empty and nothing is issued.
        for request in scheduler_output.scheduled_new_reqs:
            self._add_request_to_save(
                request.req_id, scheduler_output.num_scheduled_tokens[request.req_id]
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                raise RuntimeError("Resuming from preemption is not supported")

            block_ids = cached_reqs.new_block_ids[index]
            state = self._req_states[req_id]
            state.num_computed_tokens = cached_reqs.num_computed_tokens[index]
            # num_output_tokens counts async-scheduling placeholders, which
            # are only added for decode steps -- 0 means still in prefill.
            # This filters MTP decode, which schedules 1 + num_spec > 1.
            if cached_reqs.num_output_tokens[index] != 0:
                continue
            if block_ids:
                for group_ids, ids in zip(state.group_block_ids, block_ids):
                    group_ids.extend(ids)
            self._add_request_to_save(
                req_id, scheduler_output.num_scheduled_tokens[req_id]
            )

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

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        from vllm.model_executor.models.utils import extract_layer_index

        # Exclude speculative draft layers while preserving registration order.
        kv_caches = {ln: cache for ln, cache in kv_caches.items()
                     if extract_layer_index(ln) < self.num_layers}
        # Order and bind the pages connector-side; the store receives tensors
        # whose dim 0 is the logical block and addresses them uniformly.
        bound, layout = bind_kv_caches(
            kv_caches, self.groups, self.num_blocks, self.page_bytes)
        self.mamba_layers = self.mamba_layers.intersection(bound)
        # `wait_for_layer_load` resets the per-step bookkeeping on the last
        # layer that is actually hooked: the last attention layer.
        self._last_layer_name = next(
            (ln for ln in reversed(list(layout.kinds))
             if layout.kinds[ln] != "mamba"), None)
        self._layer_names = list(bound)
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            kv_caches=bound,
            block_dim=0,
            rank=self.rank,
            tp_size=self.tp_size,
        )
        logger.info("Registered %d KV cache layers", len(bound))

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        # vLLM zeroes recycled attention blocks on the compute stream for hybrid
        # models (needs_kv_cache_zeroing == has_mamba_layers); our H2D copies run
        # on a private stream, so retire that zeroing before issuing any load.
        if self.has_mamba and self.vllm_device == "cuda":
            torch.cuda.current_stream().synchronize()

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
            if len(request.group_block_ids) != len(self.groups) or any(
                block_ids and len(block_ids) != len(request.block_hashes)
                for block_ids in request.group_block_ids
            ):
                raise ValueError(f"Mismatched block metadata for request {req_id}")
            if not request.block_hashes:
                continue
            if request.is_async:
                async_reqs.append((req_id, request))
            else:
                # Sync path only supports pure-attention models (single KV
                # group), so group 0 is the whole table.
                block_ids, block_hashes = request.for_group(0)
                sync_block_ids.extend(block_ids)
                sync_block_hashes.extend(block_hashes)

        # Submit synchronous (blocking) loads first as a single merged batch so
        # they are enqueued ahead of the asynchronous loads for this pass.
        self._current_get_tasks = None
        if sync_block_ids:
            self._current_get_tasks = self._store().get(
                block_indices=sync_block_ids,
                block_hashs=sync_block_hashes,
            )

        # Submit asynchronous loads per request; they are polled across
        # scheduler steps in get_finished(). Mamba groups go first: every mamba
        # layer must be resident before the forward (no per-layer hook), and
        # the attention hooks then wait their own group. Groups that share one
        # block table collapse into a single submission.
        for req_id, request in async_reqs:
            tasks: dict[str, Any] = {}
            for kind, label in (("mamba", "mamba"), ("attention", "kv")):
                batches: dict[tuple, list[str]] = {}
                for group in self.groups:
                    if group.kind != kind:
                        continue
                    block_ids, block_hashes = request.for_group(group.group_idx)
                    layer_names = [ln for ln in self._layer_names
                                   if ln in group.layer_names]
                    if not block_ids or not layer_names:
                        continue
                    key = (tuple(block_ids), tuple(block_hashes))
                    batches.setdefault(key, []).extend(layer_names)
                for (ids, hashes), layer_names in batches.items():
                    tasks.update(self._store().get(
                        block_indices=list(ids),
                        block_hashs=list(hashes),
                        layer_names=layer_names,
                        description=req_id,
                        **({"label": label} if self.has_mamba else {}),
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
        if self._connector_metadata is None:
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
                **({"label": "mamba" if layer_name in self.mamba_layers else "kv"}
                   if self.mamba_layers else {}),
            )
            self._current_put_tasks.setdefault(req_id, []).append(tasks)

    def wait_for_save(self) -> None:
        """Submit Mamba states after forward; these layers have no save hook."""
        if self._connector_metadata is None or not self.mamba_layers:
            return
        metadata = self._get_connector_metadata()
        if not metadata.reqs_to_save.requests:
            return
        for ln in self.mamba_layers:
            self.save_kv_layer(ln, None, None)

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
