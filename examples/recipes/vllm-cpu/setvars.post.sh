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
# Values changed here are final but were already printed by setvars.sh, so echo them.
# Same rules as ./setvars.pre.sh: runs on the host and in the container; `return 1` aborts.

# (a) IAXL_DDR_POOL_SIZE_GB defaults to 80% of host RAM / NUM_WORKERS (capped at 80% of a NUMA node
# per rank). vLLM's own KV cache (VLLM_CPU_KVCACHE_SPACE GiB per rank) lives on the same node,
# so take it out of the compressed-pool budget.
recipe_pool_gb=$((IAXL_DDR_POOL_SIZE_GB - VLLM_CPU_KVCACHE_SPACE))
if ((recipe_pool_gb < 1)); then
    echo "ERROR: VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE GiB leaves no room for the IAXL DDR pool ($IAXL_DDR_POOL_SIZE_GB GB per rank)" >&2
    return 1
fi
export IAXL_DDR_POOL_SIZE_GB=$recipe_pool_gb
export IAXL_DDR_POOL_TOTAL_SIZE_GB=$((recipe_pool_gb * NUM_WORKERS))
unset recipe_pool_gb
printf '%s\n' \
    "Recipe adjustments:" \
    "  IAXL_DDR_POOL_SIZE_GB=$IAXL_DDR_POOL_SIZE_GB (minus VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE)" \
    "  IAXL_DDR_POOL_TOTAL_SIZE_GB=$IAXL_DDR_POOL_TOTAL_SIZE_GB"

# (b) ENV_VARS: names whose host values start.sh passes with -e. The container re-runs setvars.sh
# and this recipe, so recipe defaults need no forwarding. Append only a variable a user sets on the
# host command line (FOO=1 IAXL_RECIPE=vllm-cpu ./start.sh) and expects to see inside; unset names
# are passed as empty, which `${VAR:-default}` treats as unset. This recipe needs none:
# ENV_VARS+=(FOO)

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
