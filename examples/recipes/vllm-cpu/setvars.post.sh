# vllm-cpu recipe (reference example), hook 3 of 3: post-processing and docker run flags.
#
# When this file runs:
#   setvars.sh has computed, validated and printed every setting (CPU/QAT/DSA lists, pool
#   sizes, thread counts, ...) and built the default CONTAINER_NAME, ENV_VARS and
#   DOCKER_RUN_ARGS. Right after this file it appends `--name "$CONTAINER_NAME"` and one
#   `-e VAR=value` per name in ENV_VARS, and start.sh runs
#       docker run "${DOCKER_RUN_ARGS[@]}" "$IAXL_DEV_DOCKER_IMAGE" bash -c '...'
#
# Use it for what ./setvars.pre.sh cannot do because the values do not exist yet there:
#   (a) adjust a value setvars.sh derived from the hardware,
#   (b) forward more host variables into the container,
#   (c) add or replace docker run flags.
# Rules (also see ./setvars.pre.sh):
#   - Runs on the host and again in the container, so it must be idempotent: compute from inputs,
#     never from a value it or a forwarded variable may already have adjusted, and leave values
#     the user set on the command line alone.
#   - Values changed here bypass the derivations and checks of setvars.sh. Prefer computing a
#     default in ./setvars.pre.sh, which setvars.sh then builds on and checks as usual.
#   - End with `return 0`. `return 1`, after printing an ERROR, aborts setvars.sh.

# (a) Derived values: none here. The DDR pool budget that this recipe lowers by vLLM's KV cache is
# a default computed from inputs in ./setvars.pre.sh, so setvars.sh derives the per-rank size from it.

# (b) ENV_VARS: names whose host values start.sh passes with -e. The container re-runs setvars.sh
# and this recipe, so recipe defaults need no forwarding. Append only inputs a user sets on the
# host command line (FOO=1 IAXL_RECIPE=vllm-cpu ./start.sh) and expects to see inside, never a value
# setvars.sh or a hook derived, which would be derived twice; unset names are passed as empty,
# which `${VAR:-default}` treats as unset. This recipe forwards its own inputs, so
# `VLLM_CPU_KVCACHE_SPACE=60 IAXL_RECIPE=vllm-cpu ./start.sh` also applies in the container:
ENV_VARS+=(VLLM_CPU_KVCACHE_SPACE MAX_MODEL_LEN RECIPE_IAXL_CPUS_PER_RANK)

# (c) DOCKER_RUN_ARGS: extra docker run flags. vLLM's CPU backend uses /dev/shm for tensor-parallel
# communication, and the docker default of 64 MiB is too small for TP > 1.
DOCKER_RUN_ARGS+=(--shm-size 16g)
# Mount a dataset read-only:
# DOCKER_RUN_ARGS+=(-v /data/datasets:/datasets:ro)
# Pin the container to the CPUs and memory of the NUMA nodes in use:
# DOCKER_RUN_ARGS+=(--cpuset-cpus 0-127 --cpuset-mems 0-1)
# To drop a default flag, rebuild the array without it (e.g. run detached instead of -it):
# recipe_args=(); for a in "${DOCKER_RUN_ARGS[@]}"; do [[ "$a" == -it ]] || recipe_args+=("$a"); done
# DOCKER_RUN_ARGS=("${recipe_args[@]}" -d)
# CONTAINER_NAME can also be changed here; --name is appended after this hook.
return 0
