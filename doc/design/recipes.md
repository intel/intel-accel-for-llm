# Recipes Design

## Purpose

Running KVShrink in a new scenario used to mean editing four files that everyone shares: `setvars.sh`, `start.sh`, `tools/auto_config.sh` and `examples/kvshrink-vllm-serve.sh`. Scenarios differ in many ways:

- **GPU or device**: a different GPU generation or vendor, XPU, or CPU-only inference, each with its own base image, runtime flags and build `DEVICE`.
- **Model**: model size and architecture drive `TP_SIZE`, `--max-model-len`, block size, KV cache budgets and the async KV load settings (`KVSHRINK_VLLM_KV_ASYNC_LOAD_*`, whose layer counts depend on the model's layer count and per-layer load time).
- **Xeon SKU and accelerators**: core count and NUMA layout, QAT versus IAA versus CPU compression, DSA availability, device-to-rank assignment and the in-tree versus out-of-tree QAT driver.
- **Deployment role**: an inference node, a storage-only daemon node, or a client of a remote pool, with different container names, mounts and network settings.

Those edits are hard to keep apart from the defaults, easy to lose on update, and impossible to switch between.

A **recipe** moves the scenario-specific part into its own directory under `examples/recipes/<name>/` and selects it with one variable, `IAXL_RECIPE`. The shared scripts stay generic and grow a small, fixed set of hook points; a recipe fills in the hooks it needs and leaves the rest alone. [examples/recipes/vllm-cpu](../../examples/recipes/vllm-cpu/README.md) is a fully commented recipe for the [vLLM CPU inference guide](../usage/vllm-cpu-inference.en.md) and is the starting point for new ones. [examples/recipes/qat-standalone](../../examples/recipes/qat-standalone/README.md) runs one rank per container in qatlib standalone mode and shows a recipe that replaces default `docker run` flags and forwards values the container cannot derive.

Design goals:

- **No change to the default behaviour.** With `IAXL_RECIPE` unset, every script runs exactly as before.
- **Command line wins.** A variable set in the environment before `./start.sh` beats the recipe, and the recipe beats the defaults in `setvars.sh`. Inside the container this holds for the variables forwarded through `ENV_VARS` (see [Container propagation](#container-propagation)).
- **One mechanism for host and container.** `IAXL_RECIPE` is forwarded into the container, so `kvshrink-vllm-serve.sh` and the benchmarks apply the same recipe there.
- **Hooks, not copies.** A recipe never replaces a shared script wholesale; it supplies fragments that the shared scripts source at defined points.

## Layout

```text
examples/recipes/<name>/
├── auto_config.sh    # optional: add or redefine shell functions
├── setvars.pre.sh    # optional: environment defaults
├── setvars.post.sh   # optional: fix-ups, ENV_VARS, DOCKER_RUN_ARGS
└── README.md         # recommended: what the scenario is and what the hooks assume
```

`IAXL_RECIPE` is either a name under `examples/recipes/` or a directory path. `setvars.sh` resolves it to the absolute `IAXL_RECIPE_DIR`, exports both, and prints `Recipe: <dir>`. A path is replaced by its absolute form, so a later script in another directory or in the container loads the same recipe. An unknown name is an error, so a typo cannot silently run with defaults.

## Load order

```mermaid
flowchart TD
    A["start.sh / kvshrink-vllm-serve.sh / benchmark"] --> B["source setvars.sh"]
    B --> C["resolve IAXL_RECIPE -> IAXL_RECIPE_DIR"]
    C --> D["source tools/auto_config.sh"]
    D --> D1["source $IAXL_RECIPE_DIR/auto_config.sh<br/>(add / redefine functions)"]
    D1 --> E["source $IAXL_RECIPE_DIR/setvars.pre.sh<br/>(env defaults)"]
    E --> F["setvars.sh defaults, auto-detection,<br/>validation, summary"]
    F --> G["default CONTAINER_NAME, ENV_VARS,<br/>DOCKER_RUN_ARGS"]
    G --> H["source $IAXL_RECIPE_DIR/setvars.post.sh<br/>(fix-ups, ENV_VARS+=, DOCKER_RUN_ARGS+=)"]
    H --> I["DOCKER_RUN_ARGS += --name, -e ENV_VARS"]
    I --> J["start.sh: docker run ${DOCKER_RUN_ARGS[@]} image"]
    I --> K["kvshrink-vllm-serve.sh: vllm serve ..."]
```

Each hook is sourced into the caller's shell, so plain variable assignments, `export`, bash arrays and function definitions all take effect directly.

A sourced file's status is that of its last command, so a hook ending in a guarded line such as `[[ -n "$FOO" ]] && export BAR=1` would fail whenever `FOO` is empty. Every hook therefore ends with `return 0`, and fails only through an explicit `return 1` after printing an `ERROR:` line. `setvars.sh` then names the failed hook and aborts, together with whatever sourced it.

## Hooks

### `auto_config.sh`: functions

`tools/auto_config.sh` is a library of detection and sizing functions. Its last step sources the recipe's `auto_config.sh`. Because bash keeps the most recent definition of a function, a recipe can:

- **add** helpers for its other hooks, e.g. the example's `rank_cpu_split`;
- **redefine** one of the functions listed below with the same arguments and output format, e.g. the example's `cpu_auto_detect` that skips the GPU probe. `setvars.sh` calls the redefined function wherever it used the original and still validates the output, so a bad override fails loudly;
- **wrap** a function by saving it under another name first: `eval "orig_$(declare -f cpu_auto_detect)"`.

These functions are meant to be redefined; their arguments and output format are a stable contract. Every function prints its result on stdout, diagnostics on stderr, and returns 1 on error. The other functions in `tools/auto_config.sh` are internal and may change.

| Function | Arguments | Output on stdout | Used by `setvars.sh` for | Checked by |
| --- | --- | --- | --- | --- |
| `cpu_auto_detect` | worker count | one CPU list per rank, separated by `\|`, e.g. `0-31\|64-95` | default `VLLM_CPU_OMP_THREADS_BIND` | `rank_cpu_counts`, `validate_omp_config` |
| `qat_auto_detect` | worker count | comma-separated QAT device indices per rank, separated by `\|`, e.g. `0,1\|2,3` | default `KVSHRINK_QAT_DEVICES` | each rank at startup (`bind_intel_accel` in `iaxl/utils/affinity.py`: one entry per rank), then the QAT driver when it opens the devices |
| `dsa_auto_detect` | worker count | DSA work-queue names per rank, separated by `\|`, e.g. `wq0.0\|wq2.0` | default `KVSHRINK_DSA_DEVICES` and `IAXL_DSA_WQS` (rank 0) | each rank at startup (`bind_intel_accel`), then the DSA library when it opens the queue |
| `ddr_pool_total_size_gb` | worker count | DDR pool budget in GB across all ranks, an integer | default `IAXL_DDR_POOL_TOTAL_SIZE_GB`; the per-rank size is the total divided by the worker count | none |
| `omp_thread_count` | QAT instances, CPU zip threads, IAA instances | thread count, an integer | default `IAXL_OMP_THREAD_NUM` when a zip backend is enabled; called after `rank_cpu_counts` has set `MIN_RANK_CPU_COUNT` | `validate_omp_config` |

Helper for hooks, not meant to be redefined:

| Function | Arguments | Output on stdout |
| --- | --- | --- |
| `iaxl_num_workers` | none | `TP_SIZE * DP_SIZE`, with the defaults of `setvars.sh` when unset; `setvars.sh` sets `NUM_WORKERS` from it |

The hook runs before any value is computed, so it must define functions only and not depend on `NUM_WORKERS` or other derived variables.

### `setvars.pre.sh`: environment defaults

Sourced before `setvars.sh` applies its own defaults, with all functions already available. Every setting in `setvars.sh` has the form `export VAR=${VAR:-default}`, so a recipe that exports `VAR` here changes the default without touching the precedence: environment > recipe > `setvars.sh`. The hook must follow the same `${VAR:-...}` convention for the command line to keep winning.

Typical content: `DEVICE`, docker images, `IAXL_LD_PRELOAD`, `IAXL_QATLIB`, `CONTAINER_NAME`, `MODEL`, `TP_SIZE`, feature switches, hard-coded `VLLM_CPU_OMP_THREADS_BIND` / `KVSHRINK_QAT_DEVICES` lists for fixed hardware, and the `kvshrink-vllm-serve.sh` parameters listed below. The hook may also call detection functions itself when it needs a value that `setvars.sh` would otherwise derive differently. It gets the worker count from `iaxl_num_workers` instead of repeating `TP_SIZE * DP_SIZE`. The example does this to split each rank's CPUs between vLLM and IAXL and to lower the default DDR pool budget by vLLM's KV cache. A default computed here still goes through the derivations and checks of `setvars.sh` (e.g. `rank_cpu_counts` and `validate_omp_config` for CPU lists, the per-rank DDR pool size from the total), so prefer this hook over `setvars.post.sh` for derived values.

### `setvars.post.sh`: fix-ups and docker

Sourced after `setvars.sh` has computed, validated and printed everything and has built the default docker settings. Three things live here:

| Variable | Default | Recipe use |
| --- | --- | --- |
| `CONTAINER_NAME` | `iaxl.vllm` | `docker --name`; also settable in `setvars.pre.sh` because the default is `${CONTAINER_NAME:-iaxl.vllm}` |
| `ENV_VARS` | proxy settings, `HOST_IP`, `MODEL`, `TP_SIZE`, `DEVICE`, `IAXL_QATLIB*`, `IAXL_VERSION`, `IAXL_LD_PRELOAD`, `IAXL_RECIPE` | Names whose host values start.sh forwards with `-e`; append the ones a user may set on the host command line and expects inside the container |
| `DOCKER_RUN_ARGS` | `--rm --privileged --pid host --net host -it -v /dev:/dev --entrypoint ""`, device flags for `DEVICE`, `HF_HOME`, `_data`, `/run/qat`, `MODEL` and repository mounts | Every `docker run` flag; append `--shm-size`, extra `-v`, `--cpuset-*`, or rebuild the array to drop a default |

`setvars.sh` appends `--name "$CONTAINER_NAME"` and `-e VAR=value` for each `ENV_VARS` entry *after* the hook, so edits to all three take effect, and `start.sh` passes nothing but `DOCKER_RUN_ARGS`, the image and the entry command to `docker run`.

Derived values (`IAXL_DDR_POOL_SIZE_GB`, `IAXL_OMP_THREAD_NUM`, device lists) are final after `setvars.sh` and can still be corrected here as a last resort, but such overrides bypass the summary and the checks (`validate_omp_config`), so echo and re-validate them yourself. A post-hook must be **idempotent**, because it runs again in the container. It computes from inputs only, never from a value that it or a forwarded variable may already have adjusted, and leaves values the user set on the command line alone.

### `kvshrink-vllm-serve.sh`: parameters

The serve script has no hook of its own and no settings in `setvars.sh`. It reads these variables, which a recipe's `setvars.pre.sh` or the environment can set:

| Variable | Default | `vllm serve` argument |
| --- | --- | --- |
| `MODEL`, `TP_SIZE`, `DP_SIZE` | from `setvars.sh` | model, `-tp`, `--data-parallel-size` (when > 1) |
| `PORT` | `8000` | `--port` |
| `MAX_MODEL_LEN` | `32765` | `--max-model-len` (the `vllm-cpu` recipe lowers it to `16384`) |
| `DEVICE` | from `setvars.sh` | `--gpu-memory-utilization 0.8`, except with `DEVICE=cpu`, where `VLLM_CPU_KVCACHE_SPACE` sizes the KV cache |
| `BLOCK_SIZE` | unset | `--block-size`, only when set |
| `VLLM_SERVE_EXTRA_ARGS` | unset | more options, appended to the defaults |
| `VLLM_SERVE_CMDLINE_ARGS` | unset | escape hatch: **replaces** `--trust-remote-code`, `--max-model-len`, `--port`, `--no-enable-prefix-caching` and `--gpu-memory-utilization`, so repeat the ones you want to keep |

Both `*_ARGS` variables are split on whitespace, so no value may contain a space. The `--kv-transfer-config` that loads the KVShrink connector is fixed, since it is the point of the script.

## Container propagation

`IAXL_RECIPE` is part of the default `ENV_VARS`, so the container shell started by `start.sh` has it, and `kvshrink-vllm-serve.sh` or a benchmark sourcing `setvars.sh` there resolves and applies the same recipe. For this to work the recipe directory must be visible in the container at the same path: directories under `examples/recipes/` are, because the repository is bind-mounted at its host path; a recipe outside the repository must add its own `-v` mount in `setvars.post.sh`.

Because the recipe runs again inside the container, forwarding recipe defaults through `ENV_VARS` is unnecessary. `ENV_VARS` carries **inputs** only: values a user may set on the host command line, such as a recipe's own knobs (the `vllm-cpu` recipe forwards `VLLM_CPU_KVCACHE_SPACE`, `MAX_MODEL_LEN` and `RECIPE_IAXL_CPUS_PER_RANK`). A value that `setvars.sh` or a hook derives must not be forwarded. The container derives it again, and a hook that adjusts it would then apply the adjustment twice. For example, forwarding `IAXL_DDR_POOL_TOTAL_SIZE_GB` from a host where the `vllm-cpu` recipe had already subtracted vLLM's KV cache would make the container subtract it again. The exception is a value the container cannot derive because it sees less of the host: the `qat-standalone` container sees only its GPU and QAT VFs, so the recipe forwards the CPUs, DDR pool share and ports the host picked, and its `setvars.pre.sh` keeps them as given. An unset name is forwarded as an empty string, which `${VAR:-default}` treats as unset, so forwarding an optional variable is harmless.

## Writing a recipe

1. Copy `examples/recipes/vllm-cpu` to `examples/recipes/<name>`, keep the hooks you need, and rewrite the README for the scenario.
2. Put plain defaults in `setvars.pre.sh` using `${VAR:-value}`, and defaults computed from inputs there too (with `iaxl_num_workers`); put only what needs values `setvars.sh` computed in `setvars.post.sh`, and keep it idempotent.
3. Redefine only the functions listed under [`auto_config.sh`](#auto_configsh-functions), and only when the original cannot be steered through its inputs; keep its output contract.
4. Avoid host-only commands in the hooks; they also run in the container. End every hook with `return 0`.
5. Dry-run with `IAXL_RECIPE=<name> bash -c 'source setvars.sh && printf "%q " "${DOCKER_RUN_ARGS[@]}"; echo'` and read the `Recipe:`, configuration and `OpenMP configuration` blocks before starting the container.
