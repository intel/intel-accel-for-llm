# Recipe `vllm-cpu`: vLLM CPU Inference

This directory is both a usable recipe and the reference example of a **recipe**: a set of per-scenario overrides for `setvars.sh`, `start.sh`, `tools/auto_config.sh` and `examples/kvshrink-vllm-serve.sh` that lives next to the code instead of being patched into it. It reproduces the setup from the [vLLM CPU inference guide](../../../doc/usage/vllm-cpu-inference.en.md) (CPU build, CPU base image, split CPU binding for vLLM and IAXL, QAT compression). Every hook contains a comment block that explains when it runs and what it may do, so copy this directory when you write a real recipe. The mechanism is described in [doc/design/recipes.md](../../../doc/design/recipes.md).

## Files

| File | Loaded by | Purpose |
| --- | --- | --- |
| [auto_config.sh](auto_config.sh) | `tools/auto_config.sh`, last | Add shell functions (`rank_cpu_split`) or replace existing ones (`cpu_auto_detect`) |
| [setvars.pre.sh](setvars.pre.sh) | `setvars.sh`, before its defaults | Environment defaults for the scenario: docker image, `DEVICE`, model, CPU binding, thread counts, DDR pool budget, `kvshrink-vllm-serve.sh` parameters |
| [setvars.post.sh](setvars.post.sh) | `setvars.sh`, after everything is computed | Forward the recipe's inputs into the container (`ENV_VARS`), add `docker run` flags (`DOCKER_RUN_ARGS`), adjust derived values as a last resort |

All three are optional; a recipe can consist of a single `setvars.pre.sh`.

## Run

```bash
# Host: build the CPU image, start the container (recipe name = directory under examples/recipes)
IAXL_RECIPE=vllm-cpu ./start.sh

# Container shell opened by start.sh (IAXL_RECIPE is forwarded)
./examples/kvshrink-vllm-serve.sh
```

Command-line variables still win over the recipe, because every hook uses `${VAR:-default}`. They reach the container when they are in `ENV_VARS`: `TP_SIZE` is by default, and `setvars.post.sh` adds the recipe's own inputs (`VLLM_CPU_KVCACHE_SPACE`, `MAX_MODEL_LEN`, `RECIPE_IAXL_CPUS_PER_RANK`):

```bash
IAXL_RECIPE=vllm-cpu TP_SIZE=1 VLLM_CPU_KVCACHE_SPACE=60 ./start.sh
```

`IAXL_RECIPE` also accepts a directory path, which can live outside the repository as long as the container can see it at the same path (add a `-v` mount in `setvars.post.sh`). `setvars.sh` turns a path into an absolute one before forwarding it.

## Write your own

1. `cp -r examples/recipes/vllm-cpu examples/recipes/<name>` and delete the hooks you do not need.
2. In `setvars.pre.sh`, keep only `export VAR=${VAR:-value}` lines (and the hard-coded CPU/QAT lists if the hardware is fixed), and end every hook with `return 0`.
3. In `auto_config.sh`, redefine only the functions [listed in the design](../../../doc/design/recipes.md#auto_configsh-functions), keeping their output format; setvars.sh validates the result.
4. In `setvars.post.sh`, prefer appending to `ENV_VARS` and `DOCKER_RUN_ARGS` over rebuilding them.
5. Check the result without starting anything: `IAXL_RECIPE=<name> bash -c 'source setvars.sh && printf "%q " "${DOCKER_RUN_ARGS[@]}"; echo'`.
