# KVShrink tests

Two suites with different requirements:

| Suite | What it covers | Needs |
|---|---|---|
| `tests/unit` | Connector logic: hit policy, plan building, GDN slot selection, piggybacked loading, save pipelining, TP heal | vLLM + PyTorch importable |
| `tests/gpu` | End-to-end gates on a real model: cold/hot correctness, lifecycle, TP, eviction | GPU, model weights, built `iaxl` |

Nothing in either suite hard-codes a host, path, device or model: every
environment-specific value comes from an environment variable with a
default, so the same commands run unchanged on any machine.

---

## tests/unit

Pure logic. No GPU, no disk, no model, no network: storage backends and
transfer engines are faked, so the suite runs anywhere vLLM imports
(a few seconds).

```bash
python3 -m pytest tests/unit -q
```

Inside the project's dev container (`./start.sh`), or against any
environment with vLLM v0.23.0 installed:

```bash
docker run --rm -v "$PWD:/w" --entrypoint bash "$IAXL_BASE_DOCKER_IMAGE" \
    -c "cd /w && python3 -m pytest tests/unit -q"
```

A handful of tests exercise the connector facade itself, which imports
the compiled `iaxl` extension. They are skipped automatically when the
extension is not built and run once you have done `pip install -e .`
(the dev container does this on start).

`conftest.py` clears every `KVSHRINK_*` variable before each test, so a
developer's shell settings can never change the result.

### What the unit suite pins

| File | Contract |
|---|---|
| `test_hybrid_config.py` | `KVCacheConfig` -> groups/layers parsing; fail-closed on unknown specs, dtypes, missing layers, mixed page sizes |
| `test_hybrid_policy.py` | Longest-hit search: attention prefix scan, GDN right-to-left boundary scan, multi-group fixed point, alignment rules |
| `test_hybrid_canonical.py` | Canonical page views, including the split K/V attention layout |
| `test_hybrid_mamba_table.py` | GDN slot selection: the snapshot must land in `block_table[(computed + scheduled - 1) // block_size]`, the only slot v0.23.0's kernel reads; null / out-of-range / no-boundary cases all fail closed |
| `test_hybrid_piggyback.py` | GDN loads ride the preceding attention layer's hook; leading segment waits before forward; every GDN layer covered exactly once; un-waited layers, stale residue, TOCTOU changes and failed transfers all fail stop |
| `test_hybrid_pipelined_save.py` | Attention layers submit their save during forward, GDN groups after it; commits carry every layer's checksum; `KVSHRINK_SAVE_PIPELINED=0` falls back cleanly |
| `test_hybrid_lifecycle.py` | Preemption/resume: save-cursor rollback, resumed requests get load plans, fail-closed when credited tokens cannot be restored, immediate block free |
| `test_hybrid_tp_heal.py` | A boundary missing on any TP rank reads as MISS so the request recomputes and re-commits every rank |
| `test_hybrid_mamba_split.py` | vLLM's own block-aligned prefill split accepts external tokens (regression guard for the restriction that older versions had) |
| `test_hybrid_contract.py` | vLLM v0.23.0 connector API contract: metadata instance, 3-argument constructor, `SupportsHMA` |

---

## tests/gpu

End-to-end gates against a running engine. Each script prints `PASS` or
`FAIL` per check and exits non-zero on failure.

```bash
# hybrid (GDN) model only
MODEL=/path/to/Qwen3.5-4B TP_SIZE=2 tests/gpu/run_gates.sh

# hybrid + pure-attention regression in one pass
GATE_MODEL_HYBRID=/path/to/Qwen3.5-4B \
GATE_MODEL_ATTENTION=/path/to/Qwen3-14B \
TP_SIZE=2 tests/gpu/run_gates.sh
```

Individual gates run standalone too:

```bash
MODEL=/path/to/Qwen3.5-4B tests/gpu/probe_cold_hot.sh
```

### Configuration

The gates source `setvars.sh`, because the connector refuses to start
without the full runtime environment. Every variable there keeps a value
already present in the environment, so anything below can be overridden
on the command line. On a machine without the Intel accelerators, turn
them off:

```bash
IAXL_QAT_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=0 \
MODEL=/path/to/Qwen3.5-4B tests/gpu/run_gates.sh
```

All optional except the model; defaults shown.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` / `GATE_MODEL_HYBRID` | — | GDN model (e.g. Qwen3.5) for the hybrid gates |
| `GATE_MODEL_ATTENTION` | unset | Attention-only model (e.g. Qwen3) for the regression gate; skipped when unset |
| `TP_SIZE` | `1` | Tensor-parallel size; `2` also exercises the per-rank shard rules |
| `GATE_PORT` | `8000` | Port the gate's engine listens on |
| `GATE_CACHE_DIR` | `_data/gate-cache` | Scratch cache; wiped before and after each gate |
| `GATE_LOG_DIR` | `_data/gate-logs` | Engine logs and metric scrapes (kept for inspection) |
| `GATE_KEEP_CACHE` | `0` | `1` keeps the scratch cache for inspection |
| `GATE_REUSE_SERVER` | `0` | `1` runs the checks against an engine already listening on `GATE_PORT` instead of starting one |
| `GATE_SERVER_LOG` | unset | Engine log to read assertions from when reusing a server |
| `GATE_MAX_MODEL_LEN` | `8192` | `--max-model-len` for the gate engine |
| `GATE_GPU_UTIL` | `0.85` | `--gpu-memory-utilization` |
| `GATE_PROMPT_SEGMENTS` | `400` | Prompt length in segments; must be long enough to cross several GDN boundaries |
| `GATE_MAX_TOKENS` | `64` | Tokens generated per request |
| `GATE_BENEFIT_ROUNDS` | `5` | Interleaved rounds in the hit-benefit gate |
| `GATE_BENEFIT_SETTLE_S` | `3` | Seconds between the two arms; saving is synchronous within a pass, so without a gap the next request queues behind it and save cost is reported as restore cost |
| `GATE_STARTUP_TIMEOUT` | `600` | Seconds to wait for `/health` |
| `KVSHRINK_METRICS_PORT` | `18801` | Port scraped by the metrics gate |

### Gates

| Script | Question it answers |
|---|---|
| `probe_warm_reuse.sh` | Inside ONE engine process, is a saved boundary really reused? Drops vLLM's own prefix cache between two identical requests so the connector has to answer, then requires an external hit and byte-identical output. The fast development loop; says nothing about persistence |
| `probe_cold_hot.sh` | After a restart, does a run that RESTORES KV and GDN state produce byte-identical output to the run that computed it? Also asserts the hot run really hit the cache (otherwise identical output would prove nothing) and that no fail-closed guard fired |
| `probe_pure_attention.sh` | Does an attention-only model still take the original code path, with unchanged output and no hybrid initialisation? |
| `probe_hit_benefit.sh` | Is the feature worth its cost? Interleaves recompute and restore in one process and reports median TTFT for each. Fails if restoring is not faster |
| `probe_metrics.sh` | Are all documented metric series exposed, with non-zero transfer bytes and the synchronous-save contract gauges at zero? |

### Development loop

Engine startup dominates the iteration time, so the gates can run
against an engine that is already up:

```bash
# once, in one terminal (blocks)
MODEL=/path/to/Qwen3.5-4B tests/gpu/dev_server.sh

# then, repeatedly, in another
GATE_REUSE_SERVER=1 GATE_SERVER_LOG=_data/gate-logs/dev_server.log \
GATE_KEEP_CACHE=1 MODEL=/path/to/Qwen3.5-4B \
    tests/gpu/probe_warm_reuse.sh
```

`probe_cold_hot.sh` refuses `GATE_REUSE_SERVER=1`: the restart is the
whole point of that gate, and reusing a live engine would leave the
state in memory and prove nothing.

### Why the hybrid gates cannot disable prefix caching

For attention-only models it is common to test an offloader with
`--no-enable-prefix-caching`: block hashes are still computed whenever a
connector is registered (`v1/engine/core.py`), so the connector keeps
working while vLLM's own cache stops intercepting hits.

That does not carry over to hybrid models. With prefix caching off vLLM
rewrites `mamba_cache_mode` to `none`, sizes the mamba block at
`max_model_len` so a request holds a single block, and switches off
block-aligned chunk splitting (`need_mamba_block_aligned_split` requires
`align`). GDN state then has no addressable boundary: where a forward
stops depends on the token budget and the rest of the batch, so a
snapshot could not be keyed to a prefix length that a later request
would reproduce. The connector refuses to start in that configuration
rather than silently caching nothing.

`probe_warm_reuse.sh` gets the same isolation a different way: prefix
caching stays on, and `POST /reset_prefix_cache` drops only vLLM's copy
(`reset_external` defaults to false) so the next request must come to
the connector. It needs `VLLM_SERVER_DEV_MODE=1`, which the gate sets.

Cross-restart hits require reproducible block hashes; `setvars.sh` and
`tests/gpu/lib.sh` both pin `PYTHONHASHSEED` for this reason. Without it
vLLM randomizes its first block hash per process and every restart
misses.

How the hybrid path actually works, with sequence diagrams for the
load, save and lookup flows: [doc/design/kvshrink-hybrid.md](../doc/design/kvshrink-hybrid.md).
