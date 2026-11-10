# vllm-cpu recipe (reference example), hook 1 of 3: shell functions.
#
# When this file runs:
#   tools/auto_config.sh sources it as its last step, so every function it defines exists and
#   can be called from here, from ./setvars.pre.sh, ./setvars.post.sh and from setvars.sh itself.
#
# Two things are possible:
#   1. Add a function (rank_cpu_split below) for the other hooks to use.
#   2. Redefine a function of tools/auto_config.sh (cpu_auto_detect below). doc/design/recipes.md
#      lists the functions meant to be redefined and their contract. bash keeps the last
#      definition, so the copy here replaces the original everywhere setvars.sh calls it, with
#      the same name, arguments and output format. The original is gone after that; to wrap it
#      instead, copy it under another name first:
#          eval "orig_$(declare -f cpu_auto_detect)"   # defines orig_cpu_auto_detect
#          cpu_auto_detect() { ...; orig_cpu_auto_detect "$@"; }
#
# Keep functions free of side effects on the environment: they run in command substitutions,
# so print the result on stdout, diagnostics on stderr, and `return 1` on error.
# The file itself ends with `return 0`; a failing last command would abort setvars.sh.

# 1. New function.
# Split every rank entry of a '|'-separated CPU spec in two: the last `tail` CPUs of each rank go
# to the second output line, the rest to the first.
#   rank_cpu_split "0-63|64-127" 2   ->   0-61|64-125
#                                         62-63|126-127
rank_cpu_split() {
    local spec=$1 tail=$2
    local entry count
    local -a entries=() cpus=() heads=() tails=()

    if ! [[ "$tail" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: rank_cpu_split: CPU count must be a positive integer, got '$tail'" >&2
        return 1
    fi
    IFS='|' read -r -a entries <<<"$spec"
    for entry in "${entries[@]}"; do
        mapfile -t cpus < <(expand_cpu_list "$entry" | sort -n)
        count=${#cpus[@]}
        if ((tail >= count)); then
            echo "ERROR: rank_cpu_split: cannot take $tail of the $count CPUs in '$entry'" >&2
            return 1
        fi
        heads+=("$(compress_cpu_list "${cpus[@]:0:count-tail}")")
        tails+=("$(compress_cpu_list "${cpus[@]:count-tail}")")
    done
    (IFS='|'; printf '%s\n' "${heads[*]}" "${tails[*]}")
}

# 2. Replaced function.
# setvars.sh derives each rank's CPUs from the NUMA node of GPU `rank` and only falls back to an
# even split (with a warning) when nvidia-smi finds no GPU. CPU inference has no GPU to follow,
# so go straight to the even split: cpu_even_split hands out the physical cores (one logical
# CPU per core) in CPU-id order. With one NUMA node per rank and CPUs numbered node by node,
# that lands rank r on node r. With more nodes than ranks (e.g. sub-NUMA clustering), a rank
# spans several nodes while vLLM binds its memory to the node of its first CPU, which may be
# too small for the weights, the KV cache and the IAXL pools. Set VLLM_CPU_OMP_THREADS_BIND /
# KVSHRINK_CPU_OMP_THREADS_BIND explicitly in ./setvars.pre.sh on such hosts and on hosts that
# interleave CPU ids across nodes.
cpu_auto_detect() {
    local num_workers=${1:-${TP_SIZE:-}}
    cpu_even_split "$num_workers"
}

return 0
