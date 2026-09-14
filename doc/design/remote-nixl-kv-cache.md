# Remote (Cross-Node) KV Cache over NIXL -- Design

> Status: implemented in `iaxl/remote/` (control plane + TCP/NIXL data
> planes) and `iaxl/csrc/torch_ext/torch_ext.cpp` (two new native entry
> points). Wired into `kvshrink/kvshrink_connector.py` behind
> `KVSHRINK_REMOTE_CACHE_ENABLE`. Chinese version:
> [`remote-nixl-kv-cache.zh-CN.md`](remote-nixl-kv-cache.zh-CN.md). Usage:
> [`../usage/remote-nixl-kv-cache.md`](../usage/remote-nixl-kv-cache.md).

## 1. Background and goals

KVShrink caches GPU-produced KV blocks in a local, per-rank host-memory pool
(`iaxl.kvstore.KVStore`) to reduce TTFT on prefix hits. This feature adds a
second backend, `iaxl.remote.RemoteKVStore`, that moves that pool to a
**separate cache-only node**: a vLLM worker RDMA-transfers (NIXL/GDR) its KV
blocks to a remote daemon instead of compressing them locally; the daemon
compresses, pools and persists them, and RDMA-transfers them back on a hit.

This targets an earlier, now-superseded implementation of the same idea
(built against an older codebase, `KVCacheClip`/`kvclip`) with three concrete
changes:

1. **Reuse iaxl's real pool/compression/persistence code on the daemon**,
   instead of a separate, daemon-only Python reimplementation.
2. **Run the daemon in the same container image as the GPU node**
   (`vllm/vllm-openai:v0.23.0`), not a stripped-down CPU-only build.
3. **Avoid many-to-one contention**: every GPU worker rank talking to one
   daemon process serializes/contends on its control plane and codec gates.
   The daemon can now also run as **N processes, one per GPU rank**.

Everything else (block-level API shape, NIXL two-phase begin/commit/done
protocol, TCP fallback, multi-round staging budget) is carried over largely
unchanged, because it already worked.

## 2. Reuse scope: what "reuse iaxl's pool/compression/chunk logic" means

The local path is: `KVStore` -> `KVFlow` -> native `Context` (GPU transfer +
zip) -> native `Mem`/`Storage`/`Record` (DDR pool + disk persistence). The
`Context` class is **inherently GPU-bound**: `Context::create()` requires a
real CUDA/XPU tensor (`iaxl/csrc/torch_ext/context.h`), and
`iaxl.kvflow.KVFlow.put()`/`get()` assert `tensor.is_cuda or tensor.is_xpu`
(`iaxl/kvflow/flow.py`). A cache-only remote node has no GPU, so `KVFlow`
cannot be reused as-is.

Two things *can* be reused directly, because they were already
GPU-independent:

- **`iaxl.torch_ext.Mem` / `Storage` / `Record`** (`iaxl/csrc/include/
  kv_pool.h`): the grouped DDR pool (LRU, byte-budget), disk persistence
  (`chunks/` + `chunks.db` SQLite) and their Python bindings
  (`iaxl/csrc/torch_ext/torch_ext.cpp`) never touch a GPU. `Mem.put(keys,
  data)` / `Mem.get(keys)` / `Mem.has(keys)` already operate on plain byte
  blobs.
- **`kv_zip::kv_zip_compress_batch` / `kv_zip_decompress_batch`**
  (`iaxl/csrc/kv_zip/kv_zip.cpp`, declared in `iaxl/csrc/include/kv_zip.h`):
  the shared QAT/IAA/CPU compression task pool. Both functions assert their
  input tensors are **CPU-resident and contiguous**
  (`IAXL_CHECK(tensor.is_contiguous() && tensor.device().type()==CPU, ...)`)
  -- they have no GPU dependency at all. Previously they were only reachable
  through `Context::zip_to_mem()`/`unzip_from_mem()`, which fuse them with a
  GPU transfer step the remote daemon does not need (KV bytes already arrive
  in a CPU staging buffer via RDMA).

So the only native change needed is two small pybind11 free functions in
`torch_ext.cpp` that call these two pieces directly, without a `Context`:

```cpp
// iaxl/csrc/torch_ext/torch_ext.cpp
m.def("zip_compress_to_mem",
      [](Mem &mem, labels, cpu_tensors, compress) {
          kv_zip::kv_zip_compress_batch(cpu_tensors, out_bufs, out_sizes, orig_sizes, compress);
          mem.put(labels, std::move(out_bufs), out_sizes, orig_sizes);
      });
m.def("zip_decompress_from_mem",
      [](Mem &mem, labels, cpu_tensors) {
          auto results = mem.get(labels);
          kv_zip::kv_zip_decompress_batch(data_ptrs, cpu_tensors);
      });
```

These mirror the body of `Context::zip_to_mem`/`unzip_from_mem`
(`iaxl/csrc/torch_ext/zip.cpp`) minus the transfer/Context parts. The daemon
(`iaxl/remote/server.py`) calls them directly: **the same native pool, the
same compression backends (QAT/IAA/CPU), the same on-disk chunk format**
that a local, GPU-attached `KVStore` uses -- controlled by the exact same
`IAXL_KV_COMPRESSION` / `IAXL_QAT_ZIP_ENABLE` / `IAXL_QAT_DEVICES` /
`IAXL_QAT_ZIP_INSTANCES_PER_DEVICE` / ... environment variables. There is no
daemon-specific "codec" concept (no `codec=qat/zlib/none` choice as in the
superseded implementation): compression backend selection is whatever the
native library is configured to use, same as locally; `iaxl.remote.daemon`
only adds a `--no-compress` switch for raw-bandwidth benchmarking.

Because `zip_compress_to_mem`/`zip_decompress_from_mem` never touch CUDA/XPU,
the daemon can load the exact same `iaxl.torch_ext` extension that was built
for the GPU node and run it on a host with no GPU at all -- point 2 above
falls out of this for free (see section 8).

### 2.1 One native `Mem`/`Storage`/`Record` group per (model, tp_size, tp_rank)

The daemon creates one `RemoteCacheGroup` (its own native `Mem`, `Storage`,
`Record` triple) per `(model_name, tp_size, tp_rank)`, at
`{cache_dir}/{model}_tp{tp_size}_rank{tp_rank}/` -- the same
`chunks.db` + `chunks/` layout `iaxl.kvflow.KVFlow` uses locally at
`{persist_dir}/{model}_rank{rank}/` (tp_size is folded into the directory
name to disambiguate multiple TP configurations of one model sharing a
daemon). Native chunk keys use the identical
`{label}:{block_hash}:{tensor_key}` convention
(`iaxl/csrc/include/kv_pool.h::make_chunk_label`, `label="kv"` matching
`KVStore.LABEL`), so a block's shards across **every layer** live in one
native "group" (one LRU/persist unit), exactly like the local path.

One structural difference: locally, K and V of one layer are compressed
**together** as a single blob (`KVFlow.put()` calls `zip_to_mem` once per
layer with a `[2, ...]`-shaped CPU tensor already reassembled by the strided
D2H copy). The remote data plane needs K and V as **separate** RDMA
descriptors (they are not contiguous in GPU memory when `block_dim==1`), so
the daemon stores them as two sibling entries in the block's group:
`tensor_key = "{layer_id}"` (MLA / already-fused layouts) or
`"{layer_id}.k"` / `"{layer_id}.v"` (split layouts) --
`.` is used instead of `:` because `kv_pool.h::validate_label_component`
rejects `:` and `/` inside a single label component. This costs a small
amount of cross-K/V compression correlation but is otherwise fully
compatible with the native pool's group/LRU/persistence semantics.

## 3. Architecture

```
GPU node (vLLM)                                    Remote cache node(s)
+----------------------------+                     +----------------------------------+
| Scheduler process          |                      | iaxl.remote.daemon (1..N procs)   |
|  KVShrinkConnector(SCHED)  |   control (TCP)      |  +------------------------------+ |
|   -> RemoteKVStore(has-only)+--------------------->|  | control-plane server         | |
|                            |  has / capability     |  |  session/has/mark_ready       | |
+----------------------------+                      |  +------------------------------+ |
| Worker process (rank r)    |                      |  | RemoteCacheGroup per          | |
|  KVShrinkConnector(WORKER) |   control (TCP)      |  | (model, tp_size, tp_rank):    | |
|   -> RemoteKVStore(worker) +--------------------->|  |  native Mem/Storage/Record    | |
|      -> RemoteCacheClient  |  put/get begin/commit |  |  (LRU pool + chunks.db/dir)   | |
|      -> DataPlane(nixl/tcp)|                       |  +------------------------------+ |
|         (registers GPU HBM)|  data (RDMA/NIXL)     |        ^  zip_compress_to_mem /   |
|                            +=======================+========+  zip_decompress_from_mem |
+----------------------------+  RDMA WRITE/READ      +----------------------------------+
```

- **Scheduler-side `RemoteKVStore` (has-only)**: only queries `has()`; no GPU
  registration, no NIXL agent. Backs `get_num_new_matched_tokens()`.
- **Worker-side `RemoteKVStore`**: created in `register_kv_caches()`, builds
  a NIXL local agent over the real GPU KV tensors, opens a session, drives
  `put`/`get`/`_wait`.
- **Daemon**: control-plane TCP server + one `RemoteCacheGroup` per session +
  (optional) NIXL staging pool for the data plane.

## 4. Metadata and wire keys

Unchanged from the superseded implementation's design (still correct and
version/backend independent):

- A remote shard is identified by `(model_name, tp_size, tp_rank, block_hash,
  layer_id, tensor_key)`; `tensor_key` is `"k"`/`"v"` for non-MLA
  `[2, num_blocks, ...]` layouts (`block_dim==1`) or `"kv"` for MLA/fused
  layouts (`iaxl.remote.metadata.tensor_keys_for_layout`, mirrors
  `kvshrink_connector.register_kv_caches`'s own `block_dim` detection).
- `block_descriptors(shape, block_dim, elem_size, block_index)` computes the
  contiguous `(offset, length)` byte ranges of one block for a standard
  C-contiguous tensor -- used by both the NIXL data plane (to build transfer
  descriptors) and the TCP fallback (to slice bytes).
- On the wire, a shard key is the compact, session-relative
  `"{block_hash}|{layer_id}|{tensor_key}"` (no model/tp/rank prefix needed:
  a session already maps 1:1 to one `RemoteCacheGroup`).
- `iaxl.remote.metadata.full_chunk_label()` maps that to the native
  `Mem` key described in section 2.1.

## 5. Handshake and protocol

Same four-step handshake and message set as before
(`iaxl/remote/protocol.py`, `client.py`, `server.py`):

1. **Capability**: `RemoteCacheClient.connect()` retries until reachable
   (or fails fast if `KVSHRINK_REMOTE_FAIL_IF_UNREACHABLE=1`, default);
   exchanges protocol version.
2. **Session create**: worker/scheduler send `model_name/tp_size/tp_rank/
   num_layers/tensor_keys/dtype/block_size/shard_bytes`; daemon creates (or
   reuses) the `RemoteCacheGroup` and returns a `session_id` plus advertised
   NIXL staging capacity.
3. **NIXL handshake** (worker only, `transport=nixl`): exchange NIXL agent
   metadata; the worker registers its GPU KV tensors lazily on first
   put/get.
4. **Runtime `put`/`get`/`has`/`mark_ready`** -- see section 6.

Framing: 4-byte length prefix + JSON header (+ optional binary blob for the
TCP data plane). See `protocol.py` for the exact reasoning on why this stays
on TCP rather than RDMA SEND (section 9 below expands on it).

## 6. Runtime data flow

**SAVE (put)**: `save_kv_layer(layer)` -> `RemoteKVStore.put(block_ids,
block_hashes, [layer])` builds one `ShardRef` per (block, k/v) -> submitted
to a thread pool -> `DataPlane.put()`. NIXL: `begin` (daemon allocates
staging, returns descriptors) -> client RDMA `WRITE` GPU->staging -> `commit`
(daemon reads staging, calls `zip_compress_to_mem`, frees the slot). After
the last layer of a block, the worker calls `mark_ready()`.

**LOAD (get)**: scheduler's `has()` decides the hit prefix;
`start_load_kv()` -> `RemoteKVStore.get()` per layer -> `DataPlane.get()`.
NIXL: `begin` (daemon calls `zip_decompress_from_mem` into staging, or
zero-fills genuinely missing keys -- see below) -> client RDMA `READ`
staging->GPU -> `done` (daemon frees the slot, fire-and-forget).

A cache-miss subtlety: `zip_decompress_from_mem` aborts the **whole daemon
process** on a missing key (native `IAXL_CHECK` semantics, not a catchable
exception -- see `iaxl/csrc/include/iaxl_common.h`). `server.py` therefore
always calls `mem.has(keys)` first and only decompresses the present subset,
zero-filling anything missing (evicted between `has()` and `get()`, or a
genuine bug) instead of letting the daemon crash.

TCP fallback: shard bytes travel inline in the control-plane frame; the
daemon still calls `zip_compress_to_mem`/`zip_decompress_from_mem`
identically, just skipping the RDMA step.

## 7. Multiple RDMA ports/links

`KVSHRINK_REMOTE_NIXL_DEVICE` (client) / `NIXL_DEVICE` (daemon script) may be
a single device (`"mlx5_0:1"`) **or** a comma-separated list
(`"mlx5_0:1,mlx5_1:1"`). It is forwarded verbatim to `UCX_NET_DEVICES`; UCX
itself stripes one NIXL session's RDMA traffic across every listed device
("multi-rail"). This is preserved unchanged from the previous implementation
-- there is no separate per-device application-level abstraction to build or
maintain, since UCX already does the fan-out for one logical NIXL
connection.

## 8. Single-process vs. multi-process daemon

A single daemon process can serve every rank of a TP group, but every rank's
control-plane RPCs and codec calls then contend on that one process -- this
was measured to hurt performance under multi-GPU load (requests serialize on
the control socket dispatch loop and on the single-worker codec queue that
serializes every ``kv_zip_*`` call, see ``iaxl/remote/server.py::_CodecWorker``
for why the codec cannot be parallelised at this level). `iaxl.remote`
supports both:

- **Single process** (`--num-instances 1`, the default): one control port,
  one NIXL staging pool, shared by every rank.
  `KVSHRINK_REMOTE_QAT_DEVICES` (if set) is **flattened/unioned**: `"0|1|4|5"`
  and `"0,1,4,5"` both become `IAXL_QAT_DEVICES=0,1,4,5`, i.e. every listed
  device is used together by this one process (mirrors
  `iaxl.remote.daemon._resolve_qat_devices`, `num_instances<=1` branch).
- **Multi-process** (`--num-instances N --instance-id i`, or
  `tools/remote_daemon/run-daemon-multi.sh`): one process per GPU rank, each
  on its own control/NIXL port. Each instance gets **only its own slice** of
  `KVSHRINK_REMOTE_QAT_DEVICES` (same `"|"`-separated-per-rank convention as
  `KVSHRINK_QAT_DEVICES`/`KVSHRINK_DSA_DEVICES` in
  `kvshrink_connector._bind_intel_accel`): instance `i` gets
  `IAXL_QAT_DEVICES=<i-th entry>`, **not** the union.

This is the one place the two modes genuinely differ in behaviour, and it is
called out explicitly because getting it backwards either starves a
multi-process deployment of QAT devices (each instance only sees device 0)
or makes a single process fight itself over a device another rank also
claims exclusively.

To route GPU rank `r` to daemon instance `r` (avoiding the many-to-one
contention), `KVSHRINK_REMOTE_DAEMON_ADDR` and `KVSHRINK_REMOTE_NIXL_ADDR`
accept the same `"|"`-separated-per-rank convention
(`iaxl.remote.config._resolve_per_rank`): a plain `host:port` is shared by
every rank; a `"|"`-joined list picks entry `r` for rank `r`. The scheduler
role always resolves entry `0`, matching rank 0's readiness record being
used as the deployment-level hit proxy (`_h_has` in `server.py`).

## 9. Control-plane transport: TCP, not RDMA SEND

Considered and rejected: moving the control plane (capability, session
create, `has`, `mark_ready`, NIXL handshake, put-begin/commit,
get-begin/done) from TCP onto RDMA SEND/RECV.

- Control frames are small (a few hundred bytes to a few KB for a whole
  round's shard list) and few relative to the data volume NIXL WRITE/READ
  already moves for the same round.
- A local/RoCE TCP round trip costs low tens of microseconds; RDMA SEND
  might save a few microseconds of that. Either way it is negligible next to
  the actual RDMA transfer plus QAT/IAA (de)compress time of a round
  (typically hundreds of microseconds to low milliseconds, per the
  `RoundStats` daemon logging).
- RDMA SEND would need its own reliable-message channel (its own completion
  queue handling, sequencing, and agent bootstrap) layered next to the NIXL
  data plane's existing agent -- real implementation complexity for a
  measurement-noise-level win.
- The actual bottleneck observed (and the reason for point 3 in section 1)
  is control-plane/codec-gate **contention under many-to-one fan-in**, which
  the multi-process daemon (section 8) addresses directly; it is not
  control-plane **latency**, which RDMA SEND would target.

Decision: **keep the control plane on TCP** (`TCP_NODELAY` already set, one
socket per worker thread already avoids head-of-line blocking across
concurrent rounds -- see `iaxl/remote/transport/nixl_backend.py`).

## 10. Code layout

```
iaxl/remote/
  config.py          Client-side config (env KVSHRINK_REMOTE_* + kv_connector_extra_config)
  protocol.py         Control-plane frame format + message type constants
  metadata.py         RemoteKey, block_descriptors, native full_chunk_label()
  client.py           RemoteCacheClient (control-plane RPC)
  remote_kvstore.py   RemoteKVStore (KVStore-compatible facade)
  server.py           RemoteCacheDaemon: control-plane server + RemoteCacheGroup
                       (native Mem/Storage/Record) + zip_compress_to_mem /
                       zip_decompress_from_mem calls
  daemon.py           CLI entry point (single- and multi-process QAT device resolution)
  transport/
    base.py           DataPlane interface + ShardRef
    tcp_backend.py     TCP data plane (portable fallback)
    nixl_backend.py     NIXL data plane (production) + daemon-side staging pool

iaxl/csrc/torch_ext/torch_ext.cpp
  zip_compress_to_mem() / zip_decompress_from_mem()   new, GPU-independent bindings

kvshrink/kvshrink_connector.py
  _make_kvstore() / _remote_cache_config()             local-vs-remote KVStore factory

tools/remote_daemon/
  run-daemon.sh        single daemon process launcher (env-var configured)
  run-daemon-multi.sh   N-process launcher, one instance per GPU rank
```

## 11. Status and follow-ups

- Implemented: control plane, TCP and NIXL data planes, native
  compress/decompress + pool reuse, single/multi-process daemon, connector
  wiring, deployment scripts.
- Not yet load-tested against real QAT/RDMA hardware in this change (no such
  environment was available while writing it); build and validate with
  `start.sh` / `pip install -e .` before production use, and run the
  self-test in the usage doc first.
- Possible follow-ups (also open in the superseded implementation): session
  reconnection/fault tolerance, cross-rank strict readiness (today rank 0 is
  a hit proxy for the whole deployment), and secondary persistence of the
  remote pool to shared storage (DAOS or similar).
