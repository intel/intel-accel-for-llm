# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import re
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
from .dsv4_patch import install_dsv4_attention_hook

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

# Which DSv4 compressor `state_cache` tensors to offload to the KV store.
# The C4A compressor (compress_ratio=4) uses an OVERLAPPING window: the first
# compressed token emitted after a loaded prefix reads partial state of the
# prefix's last 4 tokens. If that C4A state is not restored on a warm (DDR)
# load, the boundary token is computed from stale scratch -> GSM8K accuracy
# drop (measured 99% -> 97.5%). Storing the C4A state fixes it (port of the
# KVCacheClip fix "offload C4A compressor state to fix accuracy drop").
#   "c4a" (default): store only the C4A state group (block_size == 4).
#   "all"/"1"      : store all compressor states (C4A + C128A).
#   "0"/"off"      : store none (legacy; reproduces the accuracy drop).
# C4A vs C128A is told apart by block size: C4A state shape[1]==4, C128A==8.
_SAVE_COMPRESSOR_STATE = os.getenv("KVSHRINK_SAVE_COMPRESSOR_STATE", "c4a").lower()

# DSv4 C4A compressor-state windowing (offload-only optimization).
# The C4A compressor reads only a sliding window of the last
# (1 + overlap) * compress_ratio == (1 + 1) * 4 == 8 token-states to compress
# each next 4-token boundary; states older than that window are never re-read
# (their compressed output already lives in the main MLA KV cache). So when the
# offload connector stores a 256-token hash-block, it only needs the LAST few
# token-states of the block's C4A compressor-state group (block_size 4), not all
# 256. This stores/loads only the last KVSHRINK_C4A_STATE_WINDOW_TOKENS
# token-states per hash-block for C4A compressor-state groups, cutting that
# group's footprint by (block_size / window) (e.g. 256/8 = 32x). It is an
# offload-tier optimization only: vLLM's own paged state cache and prefix
# caching are untouched. Set to 0 to disable (store the full state, legacy).
_C4A_STATE_WINDOW_TOKENS = int(os.getenv("KVSHRINK_C4A_STATE_WINDOW_TOKENS", "8"))

ReqId = str


@dataclass
class ReqMeta:
    block_ids: list[int] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1
    # Per-group block_ids for HMA multi-group (hybrid) models like DSv4.
    # Empty for normal single-group models. Index: all_group_block_ids[g] = the
    # block_ids of HMA group g. Only consumed by the DSv4 save/load path.
    all_group_block_ids: tuple = ()


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
        all_group_block_ids: tuple = (),
    ) -> None:
        self.requests[req_id] = ReqMeta(
            block_ids,
            block_hashes,
            is_async,
            async_load_layers,
            all_group_block_ids,
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
        self.model_config = vllm_config.model_config
        self._kv_cache_config = kv_cache_config
        self.block_size = vllm_config.cache_config.block_size
        # In HMA (hybrid, e.g. DSv4) mode cache_config.block_size may be the
        # minimum block size across groups (e.g. 4 for the compressor state).
        # The connector hashes and counts tokens at the *max* group block size
        # (256, the MLA group), so override it here for consistent hashing.
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 1:
            hash_bs = max(
                g.kv_cache_spec.block_size
                for g in kv_cache_config.kv_cache_groups
            )
            if hash_bs != self.block_size:
                logger.info(
                    "[HMA] Overriding block_size %d -> %d (hash block size)",
                    self.block_size,
                    hash_bs,
                )
                self.block_size = hash_bs
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.use_mla = self.model_config.use_mla
        self.vllm_device = vllm_config.device_config.device_type
        self.rank = get_world_group().rank if model_parallel_is_initialized() else 0

        # === DSv4 (hybrid model) state ===
        # Config-level detection works on both scheduler and worker. The worker
        # confirms it via the actual KV-cache shapes in register_kv_caches()
        # (which is what flips self._dsv4_mode on).
        architectures = getattr(self.model_config, "architectures", None) or []
        model_type = getattr(
            getattr(self.model_config, "hf_config", None), "model_type", ""
        )
        self._is_dsv4 = (
            "DeepseekV4ForCausalLM" in architectures or model_type == "deepseek_v4"
        )
        self._dsv4_mode = False
        self._storable_caches: dict[str, torch.Tensor] = {}
        self._layer_to_group: dict[str, int] = {}
        self._layer_name_to_storable_keys: dict[str, list[str]] = {}
        self._dsv4_group_keys: dict[int, list[str]] = {}
        # HMA groups holding the C4A compressor state (block_size 4). For these
        # the connector may store only the trailing window of token-states per
        # hash-block (see _C4A_STATE_WINDOW_TOKENS).
        self._dsv4_c4a_state_groups: set[int] = set()
        self._dsv4_num_layers = 0
        self._dsv4_save_progress: dict[str, set] = {}
        self.num_blocks = 0
        # HMA: group_idx -> ratio = hash_block_size / group_block_size.
        # group 0 (MLA, 256) -> 1; SWA (64) -> 4. Empty for non-hybrid models.
        self._group_block_ratio: dict[int, int] = {}
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 1:
            for g_idx, group in enumerate(kv_cache_config.kv_cache_groups):
                gbs = group.kv_cache_spec.block_size
                self._group_block_ratio[g_idx] = self.block_size // gbs
            logger.info("[HMA] _group_block_ratio: %s", self._group_block_ratio)

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
        selected_layers = self._async_load_layer_config.select(
            len(self._req_states)
        )
        # A dynamic-map layer value of 0 selects synchronous loading. It is not
        # an async request that resumes before layer 0.
        use_async = num_new_tokens > 0 and selected_layers != 0
        # DSv4 loads per layer through the deepseek_v4 attention hook (patched);
        # the vLLM-level async-resume path is not used, so force sync here.
        if self._is_dsv4:
            use_async = False
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

        all_group_block_ids = blocks.get_block_ids()
        block_ids = all_group_block_ids[0]
        load_start = state.num_computed_tokens // self.block_size
        load_end = min(
            load_start + num_external_tokens // self.block_size,
            len(block_ids),
            len(state.block_hashes),
        )
        if load_end <= load_start:
            return

        # DSv4 HMA: slice each group's block_ids for the load range. A hash at
        # position i maps to group g's block_ids[i*ratio : (i+1)*ratio].
        load_all_gids: tuple = ()
        if len(all_group_block_ids) > 1 and self._group_block_ratio:
            load_all_gids = tuple(
                list(
                    all_group_block_ids[g_idx][
                        load_start * self._group_block_ratio.get(g_idx, 1):
                        load_end * self._group_block_ratio.get(g_idx, 1)
                    ]
                )
                for g_idx in range(len(all_group_block_ids))
            )

        self._reqs_to_load.add_request(
            request.request_id,
            list(block_ids[load_start:load_end]),
            state.block_hashes[load_start:load_end],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
            all_group_block_ids=load_all_gids,
        )

    def _add_request_to_save(
        self,
        req_id: ReqId,
        new_block_ids: list[int],
        all_group_block_ids: tuple = (),
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        start = state.num_seen_blocks
        end = start + len(new_block_ids)
        block_hashes = state.block_hashes[start:end]
        existence = state.existence_cache[start:end]
        # Indices (within new_block_ids) of blocks not yet cached.
        missing_idx = [i for i, exists in enumerate(existence) if not exists]

        # DSv4 HMA: the last hash block may be only partially allocated across
        # groups (HMA rounds up per group independently), so drop trailing hash
        # blocks not yet fully covered by every group; they complete in a later
        # chunked-prefill step or sit at the sequence tail where reuse is rare.
        if missing_idx and all_group_block_ids and self._group_block_ratio:
            max_full_idx = len(new_block_ids)
            for g_idx in range(len(all_group_block_ids)):
                ratio = self._group_block_ratio.get(g_idx, 1)
                if ratio > 1:
                    full = len(all_group_block_ids[g_idx]) // ratio
                    max_full_idx = min(max_full_idx, full)
            missing_idx = [i for i in missing_idx if i < max_full_idx]

        if missing_idx:
            save_block_ids = [new_block_ids[i] for i in missing_idx]
            save_hashes = [str(block_hashes[i]) for i in missing_idx]
            # DSv4 HMA: expand each group's block_ids for the saved hashes.
            save_all_gids: tuple = ()
            if all_group_block_ids and self._group_block_ratio:
                save_all_gids = tuple(
                    [
                        bid
                        for i in missing_idx
                        for bid in all_group_block_ids[g_idx][
                            i * self._group_block_ratio.get(g_idx, 1):
                            (i + 1) * self._group_block_ratio.get(g_idx, 1)
                        ]
                    ]
                    for g_idx in range(len(all_group_block_ids))
                )
            self._reqs_to_save.add_request(
                req_id,
                save_block_ids,
                save_hashes,
                all_group_block_ids=save_all_gids,
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

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # HMA path (hybrid models): per-group block_ids. Free logic is identical
        # to the single-group path.
        first = block_ids[0] if block_ids else []
        return self.request_finished(request, first)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        for request in scheduler_output.scheduled_new_reqs:
            if request.block_ids:
                self._add_request_to_save(
                    request.req_id,
                    request.block_ids[0],
                    all_group_block_ids=tuple(request.block_ids),
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                raise RuntimeError("Resuming from preemption is not supported")

            block_ids = cached_reqs.new_block_ids[index]
            is_prefill = scheduler_output.num_scheduled_tokens[req_id] > 1
            if block_ids and any(len(g) > 0 for g in block_ids) and is_prefill:
                self._add_request_to_save(
                    req_id,
                    block_ids[0],
                    all_group_block_ids=tuple(block_ids),
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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        # Group registered caches by shape to detect DSv4 (a hybrid MLA model
        # whose 167 KV tensors span several distinct shapes) vs a normal
        # single-shape model. Gated by the config-level DSv4 flag to avoid
        # mis-triggering on any other multi-shape MLA model.
        shapes_by_shape: dict[tuple, list[str]] = {}
        for name, tensor in kv_caches.items():
            shapes_by_shape.setdefault(tuple(tensor.shape), []).append(name)

        if self._is_dsv4 and len(shapes_by_shape) > 1:
            self._register_dsv4_kv_caches(kv_caches, shapes_by_shape)
            return

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

    def _register_dsv4_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        shapes_by_shape: dict[tuple, list[str]],
    ) -> None:
        self._dsv4_mode = True
        # Make DeepSeek-V4 attention drive the connector's per-layer save/load
        # (0.23.0 port of KVCacheClip's dsv4-layer-hook.patch).
        install_dsv4_attention_hook()
        first_kv_cache = next(iter(kv_caches.values()))
        self.num_blocks = first_kv_cache.shape[0]

        # Storable = the plain KV caches (MLA + SWA + indexer.k_cache) plus the
        # compressor state caches selected by _SAVE_COMPRESSOR_STATE. The C4A
        # compressor state MUST be stored (default "c4a") or a warm/DDR load
        # loses the overlapping-window boundary state -> GSM8K accuracy drop.
        def _keep_storable(name: str, tensor: torch.Tensor) -> bool:
            if "compressor.state_cache" not in name:
                return True
            if _SAVE_COMPRESSOR_STATE in ("all", "1", "true", "on"):
                return True
            if _SAVE_COMPRESSOR_STATE == "c4a":
                # C4A compressor state has block_size == 4; C128A has == 8.
                return tensor.ndim >= 2 and tensor.shape[1] == 4
            return False

        self._storable_caches = {
            name: tensor
            for name, tensor in kv_caches.items()
            if _keep_storable(name, tensor)
        }
        if not self._storable_caches:
            raise RuntimeError("DSv4: no storable KV caches after filtering")
        # [DIAG] Log every compressor-state cache and whether it is stored, to
        # confirm the C4A (shape[1]==4) vs C128A (==8) split on this build.
        _state_shapes: dict[str, int] = {}
        for _n, _t in kv_caches.items():
            if "compressor.state_cache" in _n:
                _k = f"{tuple(_t.shape)}|kept={_n in self._storable_caches}"
                _state_shapes[_k] = _state_shapes.get(_k, 0) + 1
        logger.info(
            "DSv4: KVSHRINK_SAVE_COMPRESSOR_STATE=%s -> %d storable "
            "(state caches kept: %d). compressor-state shapes: %s",
            _SAVE_COMPRESSOR_STATE,
            len(self._storable_caches),
            sum("compressor.state_cache" in n for n in self._storable_caches),
            _state_shapes,
        )

        # [DIAG] Report contiguity/strides of storable caches (chunk_dim=0).
        # HMA tensor-sharing can hand out strided views; the transfer layer
        # needs to know whether only dim 0 is strided (inner contiguous).
        _diag = []
        for _n, _t in self._storable_caches.items():
            if not _t.is_contiguous():
                _inner_contig = _t.is_contiguous() or (
                    _t.dim() >= 1 and _t[0].is_contiguous()
                )
                _diag.append(
                    (_n, list(_t.shape), tuple(_t.stride()), bool(_inner_contig))
                )
        logger.info(
            "DSv4: %d/%d storable caches NON-contiguous. "
            "(name, shape, stride, per_block_contiguous) examples: %s",
            len(_diag),
            len(self._storable_caches),
            _diag[:8],
        )

        # layer_name (== attention prefix "model.layers.X.attn") -> HMA group.
        if self._kv_cache_config is not None:
            for g_idx, group in enumerate(self._kv_cache_config.kv_cache_groups):
                for ln in group.layer_names:
                    self._layer_to_group[ln] = g_idx

        # Map the attention prefix (passed to the connector hooks as layer_name)
        # to its storable caches: "{prefix}" (MLA), "{prefix}.swa_cache",
        # "{prefix}.indexer.k_cache", "{prefix}.compressor.state_cache",
        # "{prefix}.indexer.compressor.state_cache".
        # Order matters: ".indexer.compressor.state_cache" must be tried before
        # ".compressor.state_cache" (the former ends with the latter).
        for key in self._storable_caches:
            prefix = key
            for suffix in (
                ".swa_cache",
                ".indexer.k_cache",
                ".indexer.compressor.state_cache",
                ".compressor.state_cache",
            ):
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    break
            self._layer_name_to_storable_keys.setdefault(prefix, []).append(key)

        for op in self._layer_name_to_storable_keys:
            assert (
                "state_cache" not in op
                and "swa_cache" not in op
                and "k_cache" not in op
            ), f"DSv4: storable key mapped to invalid layer prefix: {op}"

        self._dsv4_num_layers = len(self._layer_name_to_storable_keys)

        # Group storable keys by HMA group for ratio-coherent batched get().
        for key in self._storable_caches:
            gid = self._layer_to_group.get(key, 0)
            self._dsv4_group_keys.setdefault(gid, []).append(key)

        # Identify the C4A compressor-state groups: block_size == 4 (so ratio ==
        # block_size // 4) and every key is a compressor.state_cache. Only these
        # are eligible for trailing-window storage; nothing else has block_size
        # 4, and requiring state_cache keys keeps the check conservative.
        _c4a_ratio = self.block_size // 4 if self.block_size >= 4 else 0
        for gid, keys in self._dsv4_group_keys.items():
            if _c4a_ratio and self._group_block_ratio.get(gid, 1) == _c4a_ratio and all(
                "compressor.state_cache" in k for k in keys
            ):
                self._dsv4_c4a_state_groups.add(gid)
        if _C4A_STATE_WINDOW_TOKENS > 0 and self._dsv4_c4a_state_groups:
            logger.info(
                "DSv4: C4A state windowing ON: storing last %d token-states "
                "per hash-block for groups %s (ratio %d -> %d sub-blocks kept).",
                _C4A_STATE_WINDOW_TOKENS,
                sorted(self._dsv4_c4a_state_groups),
                _c4a_ratio,
                max(1, -(-_C4A_STATE_WINDOW_TOKENS // 4)),
            )

        # Worker KVStore in hybrid mode: heterogeneous shapes and no auto
        # put_finish (the connector flushes presence records via finish()).
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            block_dim=0,  # MLA layout: [num_blocks, ...]
            kv_caches=self._storable_caches,
            rank=self.rank,
            tp_size=self.tp_size,
            hybrid=True,
        )
        self._layer_names = list(self._storable_caches.keys())
        self._last_layer_name = self._layer_names[-1]
        logger.info(
            "DSv4: %d storable KV caches (dropped %d compressor states), "
            "%d op-layers, %d shapes, num_blocks=%d, groups=%s, ratios=%s",
            len(self._storable_caches),
            len(kv_caches) - len(self._storable_caches),
            self._dsv4_num_layers,
            len(shapes_by_shape),
            self.num_blocks,
            {g: len(k) for g, k in self._dsv4_group_keys.items()},
            self._group_block_ratio,
        )

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        if self._dsv4_mode:
            self._dsv4_start_load_kv(metadata)
            return

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
        if self._dsv4_mode:
            self._dsv4_wait_for_layer_load(layer_name)
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
        if self._dsv4_mode:
            self._dsv4_save_kv_layer(layer_name)
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
        if self._dsv4_mode:
            self._dsv4_wait_for_save()
            return
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if self._dsv4_mode:
            # DSv4 drains all put tasks synchronously in wait_for_save(), so a
            # finished request has no in-flight save left; report it as
            # finished_sending so the scheduler frees its blocks.
            self._deferred_finished_req_ids.update(finished_req_ids)
            completed = set(self._deferred_finished_req_ids)
            self._deferred_finished_req_ids.clear()
            return (completed or None), None

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

    ############################################################
    # DSv4 (hybrid model) worker helpers
    ############################################################

    def _dsv4_start_load_kv(
        self, metadata: KVShrinkConnectorMetadata
    ) -> None:
        # Issue one async H2D get() per HMA group. Merge all requests'
        # (block_index, chunk_label) per group so each storable tensor_key
        # appears in exactly one get() call (repeated keys would overwrite each
        # other's Task and free the buffers early).
        if not metadata.reqs_to_load.requests:
            self._current_get_tasks = None
            return

        merged: dict[int, tuple[list[int], list[str]]] = {}
        for request in metadata.reqs_to_load.requests.values():
            all_gids = request.all_group_block_ids
            block_hashes = request.block_hashes
            if not block_hashes:
                continue
            for g_idx in self._dsv4_group_keys:
                block_indices, chunk_labels = self._dsv4_group_slots(
                    g_idx, request, block_hashes, all_gids
                )
                acc = merged.setdefault(g_idx, ([], []))
                acc[0].extend(block_indices)
                acc[1].extend(chunk_labels)

        self._current_get_tasks = {}
        for g_idx, (block_indices, chunk_labels) in merged.items():
            if not block_indices:
                continue
            get_tasks = self.kvstore.get(
                block_indices=block_indices,
                block_hashs=chunk_labels,
                layer_names=self._dsv4_group_keys[g_idx],
                description=f"dsv4-load-g{g_idx}",
            )
            self._current_get_tasks.update(get_tasks)

    def _dsv4_group_slots(
        self,
        g_idx: int,
        request: ReqMeta,
        block_hashes: list[str],
        all_gids: tuple,
    ) -> tuple[list[int], list[str]]:
        # Map hashes to (block_indices, chunk_labels) for HMA group g_idx.
        # ratio==1 (MLA): 1 hash -> 1 GPU block, chunk label = plain hash.
        # ratio>1  (SWA): 1 hash -> ratio GPU blocks, labels "hash_r".
        # NOTE: the sub-block separator must NOT be ':' -- iaxl's kv_pool uses
        # ':' as its internal LABEL_SEP and validate_label_component() rejects
        # any chunk label containing ':' / '/' / '\\'. (KVCacheClip used "hash:r"
        # because its TensorZip did not reserve ':'.)
        ratio = self._group_block_ratio.get(g_idx, 1)
        if ratio == 1 or not all_gids:
            if all_gids:
                block_indices = list(all_gids[g_idx][: len(block_hashes)])
            else:
                block_indices = list(request.block_ids)
            chunk_labels = list(block_hashes)
        else:
            available = len(all_gids[g_idx])
            expected = len(block_hashes) * ratio
            if available < expected:
                # Partial trailing block: only the fully-covered hashes.
                n_full = available // ratio if ratio > 0 else len(block_hashes)
                hashes = block_hashes[:n_full]
            else:
                hashes = block_hashes
            gids = all_gids[g_idx]
            # C4A compressor-state windowing: for these groups keep only the
            # last `w` sub-blocks (== last w*group_block_size token-states) per
            # hash-block; the compressor never re-reads older states. Both the
            # save (put) and load (get) paths call this method, so the window is
            # applied symmetrically and each stored sub-block is restored to the
            # exact same relative slot (label "{hash}_{r}" encodes position r).
            if _C4A_STATE_WINDOW_TOKENS > 0 and g_idx in self._dsv4_c4a_state_groups:
                gbs = self.block_size // ratio if ratio else self.block_size
                w = max(1, min(ratio, -(-_C4A_STATE_WINDOW_TOKENS // gbs)))
                sub_range = range(ratio - w, ratio)
            else:
                sub_range = range(ratio)
            block_indices = []
            chunk_labels = []
            for i, h in enumerate(hashes):
                base = i * ratio
                for r in sub_range:
                    block_indices.append(gids[base + r])
                    chunk_labels.append(f"{h}_{r}")
        assert len(block_indices) == len(chunk_labels), (
            f"[DSv4] group-slot mismatch g={g_idx} ratio={ratio} "
            f"n_idx={len(block_indices)} n_lbl={len(chunk_labels)}"
        )
        return block_indices, chunk_labels

    def _dsv4_wait_for_layer_load(self, layer_name: str) -> None:
        # Per-layer async fence: wait only this layer's storable keys. Each key
        # is waited exactly once per forward pass (distinct keys per layer), and
        # _current_get_tasks is replaced every start_load_kv().
        if not self._current_get_tasks:
            return
        keys = self._layer_name_to_storable_keys.get(layer_name)
        if not keys:
            return
        self.kvstore.get_wait(
            get_results=self._current_get_tasks,
            layer_names=keys,
        )

    def _dsv4_save_kv_layer(self, layer_name: str) -> None:
        # Per-key async D2H put; presence is flushed later in wait_for_save().
        keys = self._layer_name_to_storable_keys.get(layer_name)
        if not keys:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            return
        if not metadata.reqs_to_save.requests:
            return
        for req_id, request in metadata.reqs_to_save.requests.items():
            all_gids = request.all_group_block_ids
            block_hashes = request.block_hashes
            if not block_hashes:
                continue
            for key in keys:
                g_idx = self._layer_to_group.get(key, 0)
                block_indices, chunk_labels = self._dsv4_group_slots(
                    g_idx, request, block_hashes, all_gids
                )
                if not block_indices:
                    continue
                new_tasks = self.kvstore.put(
                    block_indices=block_indices,
                    block_hashs=chunk_labels,
                    layer_names=[key],
                    description=f"dsv4-save-{layer_name}",
                )
                self._current_put_tasks.setdefault(req_id, []).append(new_tasks)
            # Track which op-layers have saved each block for the finish() flush.
            for h in block_hashes:
                self._dsv4_save_progress.setdefault(h, set()).add(layer_name)

    def _dsv4_wait_for_save(self) -> None:
        # Drain the async D2H puts issued this step, then flush a presence
        # record (plain hashes) for every block whose layers are all saved.
        for task_list in self._current_put_tasks.values():
            for tasks in task_list:
                self.kvstore.put_wait(tasks, wait=True)
        self._current_put_tasks.clear()

        completed = [
            h
            for h, layers in self._dsv4_save_progress.items()
            if len(layers) >= self._dsv4_num_layers
        ]
        for h in completed:
            del self._dsv4_save_progress[h]
        if completed:
            self.kvstore.finish(completed)

