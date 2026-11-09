# vllm-cpu recipe (reference example), hook 2 of 3: environment defaults.
#
# Scenario: vLLM CPU inference with KVShrink compressing on Intel QAT, as described in
# doc/usage/vllm-cpu-inference.en.md. Copy this directory to examples/recipes/<name>, keep the
# hooks you need, and select it with IAXL_RECIPE=<name>.
#
# When this file runs:
#   setvars.sh resolved IAXL_RECIPE, sourced tools/auto_config.sh (plus ./auto_config.sh), and
#   sources this file BEFORE applying any of its own defaults. Everything set here therefore
#   acts as the new default of the matching `export VAR=${VAR:-...}` line in setvars.sh.
#
# Rules:
#   - Use `export VAR=${VAR:-value}` so `VAR=x ./start.sh` on the command line still wins.
#   - Available: TOP_DIR (repository root), IAXL_RECIPE_DIR (this directory) and every function
#     from tools/auto_config.sh, including the ones added or replaced by ./auto_config.sh.
#   - Nothing is computed yet (no NUM_WORKERS, MIN_RANK_CPU_COUNT, ...); derive what you need
#     from TP_SIZE / DP_SIZE as shown below, or move it to setvars.post.sh.
#   - The exit status of the last command is the hook's status: `return 1` aborts setvars.sh
#     (and start.sh / kvshrink-vllm-serve.sh), so end the file with a command that succeeds.
#   - This runs both on the host (./start.sh) and inside the container (kvshrink-vllm-serve.sh,
#     benchmarks), because IAXL_RECIPE is forwarded with -e. Keep it free of host-only tools.

# ---- Build / container (guide section 1) -----------------------------------
export DEVICE=${DEVICE:-cpu}                                               # Build IAXL for vLLM CPU inference
export IAXL_BASE_DOCKER_IMAGE=${IAXL_BASE_DOCKER_IMAGE:-vllm/vllm-openai-cpu:v0.29.0}
export IAXL_DEV_DOCKER_IMAGE=${IAXL_DEV_DOCKER_IMAGE:-vllm-iaxl-dev-cpu}  # Keep the CPU image apart from the GPU one
export IAXL_LD_PRELOAD=${IAXL_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so} # CPU image ships libiomp5 in its venv
export IAXL_QATLIB=${IAXL_QATLIB:-intree}                                  # In-tree QAT driver; unset for the out-of-tree QAT20 package
# export IAXL_QATLIB_VERSION=26.02.0                                       # Match the host qatmgr if it is not 26.02.0
CONTAINER_NAME=${CONTAINER_NAME:-iaxl.vllm-cpu}                            # docker --name; also settable in setvars.post.sh

# ---- vLLM (consumed by examples/kvshrink-vllm-serve.sh) --------------------
export MODEL=${MODEL:-Qwen/Qwen3-32B}
export TP_SIZE=${TP_SIZE:-2}
export DP_SIZE=${DP_SIZE:-1}
export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-40} # Live vLLM KV cache per rank in GiB (replaces --gpu-memory-utilization on CPU)
# All `vllm serve` options other than the model, -tp/--data-parallel-size and --block-size. Replaces
# the script's default list: no --gpu-memory-utilization on CPU, and a shorter --max-model-len so
# one request fits the CPU KV cache.
export VLLM_SERVE_CMDLINE_ARGS=${VLLM_SERVE_CMDLINE_ARGS:-"--trust-remote-code --max-model-len 16384 --port 8000 --no-enable-prefix-caching"}
# export BLOCK_SIZE=128                                     # --block-size; unset keeps vLLM's CPU default (128, 16 for MLA models)

# ---- CPU binding (guide section 2) -----------------------------------------
# vLLM and IAXL must not share cores: give IAXL the last RECIPE_IAXL_CPUS_PER_RANK CPUs of every
# rank and vLLM the rest. cpu_auto_detect is the CPU-only version from ./auto_config.sh and
# rank_cpu_split is the helper added there.
export RECIPE_IAXL_CPUS_PER_RANK=${RECIPE_IAXL_CPUS_PER_RANK:-2}
if [[ -z "${VLLM_CPU_OMP_THREADS_BIND:-}" ]]; then
    recipe_split=$(rank_cpu_split "$(cpu_auto_detect "$((TP_SIZE * DP_SIZE))")" "$RECIPE_IAXL_CPUS_PER_RANK") || return 1
    export VLLM_CPU_OMP_THREADS_BIND=${recipe_split%%$'\n'*}
    export KVSHRINK_CPU_OMP_THREADS_BIND=${KVSHRINK_CPU_OMP_THREADS_BIND:-${recipe_split##*$'\n'}}
    unset recipe_split
fi
# For a known host, hard-code the lists instead (node 0 = CPUs 0-63, node 1 = CPUs 64-127):
# export VLLM_CPU_OMP_THREADS_BIND="0-61|64-125"       # vLLM inference threads
# export KVSHRINK_CPU_OMP_THREADS_BIND="62-63|126-127" # IAXL threads
# export KVSHRINK_QAT_DEVICES="0,1|2,3"                # QAT devices on each rank's NUMA node (default: qat_auto_detect)

# ---- Threads (guide section 3) ---------------------------------------------
export IAXL_QAT_ZIP_ENABLE=${IAXL_QAT_ZIP_ENABLE:-1} # QAT does the compression ...
export IAXL_IAA_ZIP_ENABLE=${IAXL_IAA_ZIP_ENABLE:-0} # ... or IAA
export IAXL_CPU_ZIP_ENABLE=${IAXL_CPU_ZIP_ENABLE:-0} # No zlib workers: inference needs the cores
export IAXL_OMP_THREAD_NUM=${IAXL_OMP_THREAD_NUM:-2} # Threads driving the QAT/IAA instances; must fit in KVSHRINK_CPU_OMP_THREADS_BIND
# IAXL_USE_OMP defaults to 0 with DEVICE=cpu (setvars.sh), leaving OpenMP to vLLM.

# ---- Cache (guide section 4) -----------------------------------------------
# export IAXL_KV_DATA_SHUFFLE=0 # With IAXL_KV_LOSSY_TRUNC=0 too, the codec works in place on the vLLM blocks
# export IAXL_ZIP_SEGMENT_SIZE_KB=1024 # Needed when a KV block exceeds IAXL_ZIP_SRC_CAP / IAXL_ZIP_DST_CAP (4 MiB)
