# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math
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
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, generate_block_hashs, setup_root_logger

from .async_load_config import load_async_load_layer_config_from_env

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


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


class KVShrinkHybridConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker plan for the hybrid (GDN/mamba) path.

    ``requests`` are LOAD plans (executed before forward),
    ``save_requests`` are SAVE plans (executed after forward). Both are
    lists of ``hybrid_metadata.ReqMeta``; the worker only ever sees this
    object, so each plan is fully self-describing.
    """

    def __init__(self):
        super().__init__()
        self.requests: list = []
        self.save_requests: list = []


# Standalone metrics exporter for the hybrid path (dedicated HTTP port;
# vLLM's /metrics cannot surface our in-process store). Guarded import +
# no-op fallbacks: a metrics failure must NEVER affect inference.
try:
    from .hybrid_metrics_exporter import (  # noqa: E402
        start_metrics_server as _start_metrics_server,
        stop_metrics_server as _stop_metrics_server,
    )
except Exception:  # pragma: no cover - fail-open by design

    def _start_metrics_server(*a, **k):
        return None

    def _stop_metrics_server(*a, **k):
        pass


def _hybrid_save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")


class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    """KVShrink external KV cache connector.

    Two independent paths behind one vLLM connector class:

    - Pure attention models: the existing block-oriented KVStore path
      (async loads, early layer promotion, deferred free).
    - Hybrid models with GDN/mamba layers (``kv_cache_config.
      has_mamba_layers``): the hybrid stack in ``hybrid_*.py``
      (boundary-addressed snapshots, schema-4 manifest commits,
      synchronous fail-closed loads). Selected once in ``__init__``;
      every vLLM hook then dispatches on ``_hyb_sched``/``_hyb_worker``.

    ``SupportsHMA`` is mandatory in v0.23.0: the connector factory
    refuses any connector without it while the hybrid memory allocator
    is enabled (the default), and hybrid models require the allocator.
    For pure-attention models the extra base class changes nothing --
    ``request_finished_all_groups`` just forwards the single group's
    block ids to the existing ``request_finished``.
    """

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

        # Hybrid (GDN/mamba) models take the dedicated stack below; the
        # pure-attention branches stay exactly as they were.
        self._hyb_sched = None
        self._hyb_worker = None
        self._hyb_backend = None
        self._hyb_canon = None
        self._hyb_groups: list = []
        self._hyb_metrics_exporter = None

        if kv_cache_config is not None and kv_cache_config.has_mamba_layers:
            self._init_hybrid(vllm_config, role, kv_cache_config)
        elif role == KVConnectorRole.SCHEDULER:
            self.kvstore: Optional[KVStore] = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(index) for index in range(self.num_layers)],
                tp_size=self.tp_size,
            )
        else:
            self.kvstore = None
            self._bind_cpu_affinity()
            self._bind_intel_accel()

    ############################################################
    # Hybrid (GDN/mamba) path: construction
    ############################################################

    def _init_hybrid(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Build the hybrid stack for this role.

        Scheduler role gets the hit policy + plan builder over a
        READ-ONLY backend (presence checks only, no writer lease, no GPU
        pool). Worker role gets the canonical page views + the transfer
        engine over a WRITER backend holding this rank's single-writer
        lease. Both roles start the metrics exporter.
        """
        from .hybrid_backend import create_boundary_backend
        from .hybrid_canonical import Canonicalizer
        from .hybrid_config import compute_namespace, parse_kv_cache_config
        from .hybrid_metadata import SCHEMA_VERSION
        from .hybrid_scheduler import HybridRequestScheduler
        from .hybrid_worker import HybridWorker

        pc = vllm_config.parallel_config
        tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        if role == KVConnectorRole.WORKER:
            # parallel_config.rank, NOT get_world_group(): the connector
            # is constructed before distributed init in the worker
            # processes, so the world group would report rank 0 on every
            # TP rank and the ranks would overwrite each other's shards.
            rank = int(getattr(pc, "rank", 0) or 0)
        else:
            # Scheduler-side keys are always rank 0; each worker verifies
            # its own shard through its own backend.
            rank = 0
        self.kvstore = None
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        namespace = compute_namespace(
            model_id=str(getattr(model_config, "model", "model")),
            model_revision=str(getattr(model_config, "revision", None) or ""),
            tokenizer_revision=str(
                getattr(model_config, "tokenizer_revision", None) or ""),
            cache_dtype=str(getattr(cache_config, "cache_dtype", "auto")),
            kv_schema_version=SCHEMA_VERSION,
            tp_size=tp_size,
            pp_size=1,
        )
        groups, layer_infos, num_blocks = parse_kv_cache_config(
            kv_cache_config, hash_block_size=hash_block_size)

        # Fail-closed: speculative decoding widens the GDN state gather.
        # v0.23.0's mamba_get_block_table_tensor returns
        # block_table[start : start + 1 + num_speculative_blocks] and the
        # decode path reads all of those columns, but an external
        # snapshot only ever restores column 0 (the block holding this
        # step's last scheduled token). Serving a hit would let the
        # kernel read unrestored speculative slots.
        for g in kv_cache_config.kv_cache_groups:
            spec_blocks = int(
                getattr(g.kv_cache_spec, "num_speculative_blocks", 0) or 0)
            if spec_blocks:
                raise RuntimeError(
                    "kvshrink hybrid: speculative decoding is not "
                    f"supported (group has num_speculative_blocks="
                    f"{spec_blocks}); the external GDN snapshot only "
                    "restores the non-speculative state slot. Disable "
                    "speculative decoding or the KV connector.")
        self._hyb_groups = groups
        self._hyb_rank = rank
        self._hyb_tp_size = tp_size
        persist_dir = os.getenv("KVSHRINK_PERSIST_DIR") or None

        if role == KVConnectorRole.SCHEDULER:
            self._hyb_backend = create_boundary_backend(
                persist_dir=persist_dir, role="scheduler")
            self._hyb_backend.register_layout(
                groups, layer_infos, namespace, tp_size, rank)
            self._hyb_sched = HybridRequestScheduler(
                groups, self._hyb_backend, hash_block_size, namespace,
                tp_size, rank,
                prefix_caching_hash_algo=str(getattr(
                    cache_config, "prefix_caching_hash_algo", "sha256")))
        else:
            # writer_rank takes this rank's single-writer lease under the
            # persist root (fail-closed if a second writer is alive).
            self._hyb_backend = create_boundary_backend(
                persist_dir=persist_dir, writer_rank=rank, role="worker")
            self._hyb_backend.register_layout(
                groups, layer_infos, namespace, tp_size, rank)
            self._hyb_canon = Canonicalizer(layer_infos, num_blocks)
            self._hyb_worker = HybridWorker(
                groups, layer_infos, num_blocks, self._hyb_backend,
                self._hyb_canon, rank, tp_size)

        # Exporter lives in THIS process: the in-process metric store is
        # per-process and vLLM's /metrics cannot reach it.
        self._hyb_metrics_exporter = _start_metrics_server(rank=rank)
        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, %d layers, "
            "hash_block_size=%d, namespace=%s, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), len(layer_infos), hash_block_size, namespace,
            tp_size, rank)

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
        if self._hyb_sched is not None:
            return self._hyb_sched.get_num_new_matched_tokens(
                request, num_computed_tokens)

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
        if self._hyb_sched is not None:
            self._hyb_sched.update_state_after_alloc(
                request, blocks, num_external_tokens)
            return

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
        if self._hyb_sched is not None:
            # Hybrid saves are SYNCHRONOUS (wait_for_save persists pages
            # and commits manifests inside the step), so no in-flight job
            # can still reference these blocks and vLLM may free them
            # immediately. Returning True would promise a get_finished()
            # ack that never comes -- a deterministic block leak.
            # Committed boundaries are content-addressed and outlive the
            # request; they are never deleted here.
            self._hyb_sched.on_request_finished(request.request_id)
            return False, None

        # True = defer freeing to get_finished() (async load/save may still run).
        self._req_states.pop(request.request_id, None)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23.0 calls this instead of
        ``request_finished`` whenever the hybrid memory allocator is on,
        which is the default for every model).

        Both paths keep their own contract: hybrid frees immediately,
        pure attention keeps deferring to get_finished(). The pure path
        only ever has one KV cache group, so its single block list is
        forwarded unchanged.
        """
        if self._hyb_sched is not None:
            return self.request_finished(request, [])
        return self.request_finished(
            request, list(block_ids[0]) if block_ids else [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self._hyb_sched is not None:
            return self._hyb_build_connector_meta(scheduler_output)

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
    # Hybrid path: scheduler side
    ############################################################

    def _hyb_build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's hybrid plans.

        Order matters: for cached (running) requests the block-table
        sync (``on_cached_request``) MUST run before ``build_save_meta``,
        so the save plan already sees blocks allocated in the SAME pass.
        """
        meta = KVShrinkHybridConnectorMetadata()
        sched = self._hyb_sched
        debug = bool(os.getenv("KVSHRINK_DEBUG_LOG"))
        save_enabled = _hybrid_save_enabled()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            req_meta = sched.build_load_meta(
                new_req, num_sched.get(new_req.req_id, 0))
            if debug:
                logger.info(
                    "LOADMETA req=%s ops=%d computed_before_fwd=%d "
                    "num_scheduled_tokens=%s",
                    new_req.req_id, len(req_meta.group_ops),
                    new_req.num_computed_tokens,
                    num_sched.get(new_req.req_id))
                for op in req_meta.group_ops:
                    logger.info(
                        "LOADMETA  g%d kind=%s keys=%d gpu_ids=%d",
                        op.group_idx, self._hyb_groups[op.group_idx].kind,
                        len(op.keys), len(op.gpu_block_ids))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.requests.append(req_meta)
            if save_enabled:
                save_meta = sched.build_save_meta(
                    new_req.req_id, num_sched.get(new_req.req_id, 0))
                if save_meta.group_ops:
                    meta.save_requests.append(save_meta)

        # PREEMPTION-RESUMED requests ride scheduled_cached_reqs.
        # resumed_req_ids, NOT scheduled_new_reqs. Their external-hit
        # tokens were accepted this same pass, so without a load plan
        # here the worker would never restore the pages while the core
        # already skips recompute -- silent garbage output.
        cr = scheduler_output.scheduled_cached_reqs
        for req_id in (getattr(cr, "resumed_req_ids", None) or ()):
            req_meta = sched.build_resumed_load_meta(
                req_id, num_sched.get(req_id, 0))
            if req_meta is None:
                continue
            if debug:
                logger.info(
                    "LOADMETA(resumed) req=%s ops=%d",
                    req_id, len(req_meta.group_ops))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.requests.append(req_meta)

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings): sync their tables first,
        # then emit incremental saves.
        if save_enabled:
            resumed = getattr(cr, "resumed_req_ids", None) or set()
            new_bids = getattr(cr, "new_block_ids", None) or []
            ncts = getattr(cr, "num_computed_tokens", None) or []
            for i, req_id in enumerate(getattr(cr, "req_ids", []) or []):
                sched.on_cached_request(
                    req_id,
                    new_bids[i] if i < len(new_bids) else None,
                    req_id in resumed,
                    ncts[i] if i < len(ncts) else None)
                save_meta = sched.build_save_meta(
                    req_id, num_sched.get(req_id, 0))
                if save_meta.group_ops:
                    meta.save_requests.append(save_meta)

        if debug:
            logger.info(
                "build_connector_meta: %d load reqs, %d save reqs",
                len(meta.requests), len(meta.save_requests))
        return meta

    def lifecycle_stats(self) -> dict:
        """Lifecycle observability for the hybrid path (probe scripts).

        Also refreshes the contract gauges that document invariants the
        code guarantees by construction: the hybrid save path is
        synchronous, so pending store jobs, in-flight boundaries and
        deferred blocks are all 0 -- the series must still be present on
        /metrics for the probes to assert on. Metrics are fail-open.
        """
        stats: dict = {
            "hybrid": self._hyb_sched is not None or self._hyb_worker is not None,
            "role": "scheduler" if self._hyb_sched is not None else "worker",
            "pending_store_jobs": 0,
        }
        if self._hyb_sched is not None:
            stats.update(self._hyb_sched.lifecycle_stats())
        if self._hyb_backend is not None:
            try:
                stats.update(self._hyb_backend.orphan_stats())
            except Exception:  # pragma: no cover - observability only
                pass
        try:
            from .hybrid_metrics import set_gauge as _set_gauge

            _set_gauge("kvshrink_pending_store_jobs", value=0.0)
            _set_gauge("kvshrink_inflight_boundaries", value=0.0)
            _set_gauge("kvshrink_deferred_blocks", value=0.0)
            _set_gauge("kvshrink_cursor_rollbacks",
                       value=float(stats.get("cursor_rollbacks", 0)))
        except Exception:  # pragma: no cover - fail-open
            pass
        return stats

    ############################################################
    # Hybrid path: worker side
    ############################################################

    def _hyb_register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Bind canonical page views and the GDN piggyback map.

        The piggyback map needs the model EXECUTION order, which the
        ``kv_caches`` dict does not carry: v0.23.0 builds it group by
        group (``_kv_cache_spec_attn_group_iterator``), so mamba layers
        and attention layers arrive in separate runs. We recover the
        order the same way vLLM's own ``bind_kv_cache`` does -- from the
        layer index embedded in the layer name -- and fail closed if the
        names do not yield a unique order, since a wrong order would
        make a GDN layer wait on a transfer that has not been submitted
        yet (or never wait at all).
        """
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")
        from vllm.model_executor.models.utils import extract_layer_index

        missing = [ln for ln in self._hyb_worker._layer_infos
                   if ln not in kv_caches]
        if missing:
            raise RuntimeError(
                f"kvshrink hybrid: layers {sorted(missing)} are in the KV "
                "cache config but absent from kv_caches; refusing to start")

        indexed: dict[int, str] = {}
        for layer_name in kv_caches:
            try:
                idx = extract_layer_index(layer_name)
            except Exception as exc:
                raise RuntimeError(
                    "kvshrink hybrid: cannot derive the execution index of "
                    f"layer {layer_name!r} ({exc}); refusing to schedule "
                    "piggybacked GDN loads") from exc
            if idx in indexed:
                raise RuntimeError(
                    "kvshrink hybrid: layers "
                    f"{indexed[idx]!r} and {layer_name!r} share execution "
                    f"index {idx}; the execution order is ambiguous")
            indexed[idx] = layer_name
        order = [indexed[i] for i in sorted(indexed)]

        self._layer_names = order
        self._last_layer_name = order[-1]
        self._hyb_worker.register(kv_caches, order)

    def _hyb_metadata(self) -> KVShrinkHybridConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkHybridConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkHybridConnectorMetadata")
        return metadata

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._hyb_worker is not None:
            self._hyb_register_kv_caches(kv_caches)
            return

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
        if self._hyb_worker is not None:
            # Submits every load, then host-blocks only on the leading
            # GDN segment; all other layers are waited by their
            # piggyback hooks during forward.
            self._hyb_worker.start_load(self._hyb_metadata())
            return

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
            self._pending_load_layers[req_id] = request.async_load_layers

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._hyb_worker is not None:
            # This attention layer's pages + the GDN segment that runs
            # after it and before the next attention layer.
            self._hyb_worker.wait_layer_load(layer_name)
            return

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
        if self._hyb_worker is not None:
            if self._connector_metadata is None:
                return
            # Pipelined attention save: submit this layer's D2H+zip now
            # so it overlaps the remaining layers' compute. GDN groups
            # never reach this hook; they save in wait_for_save.
            self._hyb_worker.save_kv_layer(layer_name, self._hyb_metadata())
            return

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
        if self._hyb_worker is None:
            return

        hw = self._hyb_worker
        # A sticky load poison and any un-waited load must surface here,
        # before anything is persisted: entering the save path after the
        # forward read unrestored state would commit wrong data.
        hw.raise_load_poison()
        hw.loads_drained_check()
        if not hw.save_enabled():
            return
        metadata = self._hyb_metadata()
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("wait_for_save worker: save_requests=%d",
                        len(metadata.save_requests))
        pages, boundaries = hw.wait_save(metadata)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("chunk_save: %d pages, %d boundaries",
                        pages, boundaries)
        hw.debug_dump_state()

    def shutdown(self) -> None:
        """Release the hybrid backend (Record flush, writer lease) and
        the exporter. The pure-attention path owns no such resources."""
        try:
            if self._hyb_worker is not None:
                self._hyb_worker.shutdown()
            elif self._hyb_backend is not None:
                self._hyb_backend.close()
        finally:
            if self._hyb_metrics_exporter is not None:
                _stop_metrics_server()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if self._hyb_worker is not None:
            # Hybrid loads and saves both complete within the step, so
            # there is nothing to report -- but a sticky poison must
            # surface at every hook entry, never be swallowed by the
            # finish protocol.
            self._hyb_worker.raise_load_poison()
            return None, None
        if self._hyb_sched is not None:
            return None, None

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
