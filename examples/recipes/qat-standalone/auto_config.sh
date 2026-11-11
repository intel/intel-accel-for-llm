# qat-standalone recipe, hook 1 of 3: shell functions.
#
# Sourced last by tools/auto_config.sh, on the host and again in the container. Adds the
# helpers that ./setvars.pre.sh uses to pick and place the container's one rank, and wraps
# omp_thread_count to cap the default IAXL_OMP_THREAD_NUM.

# Inside a container (start.sh), where only the container's own devices are visible.
in_container() {
    [[ -f /.dockerenv || -f /run/.containerenv ]]
}

# First of the candidates $2... without a running container named iaxl.vllm.<$1><candidate>, so
# starting the recipe again picks the next free GPU or NUMA node.
standalone_free_slot() {
    local kind=$1 candidate running
    shift

    running=$(docker ps --format '{{.Names}}' 2>/dev/null) || running=
    for candidate in "$@"; do
        grep -qxF "iaxl.vllm.$kind$candidate" <<<"$running" || {
            echo "$candidate"
            return 0
        }
    done
    echo "ERROR: every $([[ "$kind" == gpu ]] && echo GPU || echo "NUMA node with QAT") ($*) already runs a container;" \
        "select one with CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK or RECIPE_NUMA_NODE" >&2
    return 1
}

# QAT PFs bound to a supported driver, as PCI addresses; the same scan as qat_auto_detect.
qat_pfs() {
    local path driver

    for path in /sys/bus/pci/devices/*; do
        [[ -r "$path/class" && -L "$path/driver" ]] || continue
        [[ "$(<"$path/class")" == "0x0b4000" ]] || continue
        driver=$(basename "$(readlink -f "$path/driver")")
        [[ "$driver" != *vf* ]] || continue
        [[ "$driver" =~ (qat|4xxx|420xx|c6xx|c3xxx|200xx|dh895xcc) ]] || continue
        echo "${path##*/}"
    done | LC_ALL=C sort
}

# NUMA nodes with CPUs and at least one QAT PF, in ascending order.
qat_numa_nodes() {
    local pf node

    for pf in $(qat_pfs); do
        node=$(<"/sys/bus/pci/devices/$pf/numa_node")
        [[ -n "$(<"/sys/devices/system/node/node$node/cpulist")" ]] && echo "$node"
    done | sort -nu
}

# NUMA node of CPU $1.
cpu_numa_node() {
    local -a nodes=("/sys/devices/system/cpu/cpu$1"/node[0-9]*)

    if [[ ! "$1" =~ ^[0-9]+$ || ! -e "${nodes[0]}" ]]; then
        echo "ERROR: cannot find the NUMA node of CPU '$1'" >&2
        return 1
    fi
    echo "${nodes[0]##*node}"
}

# Memory of NUMA node $1 in GB.
numa_mem_gb() {
    local meminfo=/sys/devices/system/node/node$1/meminfo

    if [[ ! -r "$meminfo" ]]; then
        echo "ERROR: cannot read the memory of NUMA node $1" >&2
        return 1
    fi
    awk '/MemTotal:/ {printf "%d\n", $4 / 1048576}' "$meminfo"
}

# Size in GB of the weight files of model $1 (a directory, or a Hugging Face ID cached under
# hub directory $2); fails when they are not found locally.
model_weights_gb() {
    local model=$1 hub=$2 kb
    local -a dirs=("$model")

    [[ -d "$model" ]] || dirs=("$hub/models--${model//\//--}"/snapshots/*/)
    [[ -d "${dirs[-1]}" ]] || return 1
    kb=$(find -L "${dirs[-1]}" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -print0 |
        du -cLk --apparent-size --files0-from=- 2>/dev/null | awk 'END {print $1 + 0}')
    ((kb > 0)) || return 1
    echo $(((kb + 1048575) / 1048576))
}

# Splits CPU list $1 in two lines: the last $2 CPUs on the second, the rest on the first.
cpu_list_split() {
    local -a cpus
    local count

    mapfile -t cpus < <(expand_cpu_list "$1" | sort -n)
    count=${#cpus[@]}
    if ! [[ "$2" =~ ^[1-9][0-9]*$ ]] || (($2 >= count)); then
        echo "ERROR: cannot give $2 of the $count CPUs in '$1' to IAXL (RECIPE_IAXL_CPUS_PER_RANK)" >&2
        return 1
    fi
    compress_cpu_list "${cpus[@]:0:count-$2}"
    compress_cpu_list "${cpus[@]:count-$2}"
}

# One logical CPU per physical core on NUMA node $1, as a CPU list.
numa_physical_cpus() {
    local node_path=/sys/devices/system/node/node$1
    local cpu siblings
    local -a cpu_ids=()
    local -A seen_cores=()

    if [[ ! -r "$node_path/cpulist" ]]; then
        echo "ERROR: cannot read CPU list for NUMA node $1" >&2
        return 1
    fi
    if [[ -z "$(<"$node_path/cpulist")" ]]; then
        echo "ERROR: NUMA node $1 has no CPUs" >&2
        return 1
    fi
    while read -r cpu; do
        siblings=$(<"/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list")
        [[ -n "${seen_cores[$siblings]:-}" ]] && continue
        seen_cores[$siblings]=1
        cpu_ids+=("$cpu")
    done < <(expand_cpu_list "$(<"$node_path/cpulist")")
    compress_cpu_list "${cpu_ids[@]}"
}

# Host side: prints "<vf> <iommu group>" for one VF of each QAT PF on NUMA node $1: VF $2 counted
# from the last (one slot per container on the node), as far as possible from the first VFs a
# host qatmgr hands out.
qat_standalone_vfs() {
    local numa=$1
    local slot=${2:-0}
    local pf vf num_vfs found=0

    for pf in $(qat_pfs); do
        [[ "$(<"/sys/bus/pci/devices/$pf/numa_node")" == "$numa" ]] || continue
        num_vfs=$(<"/sys/bus/pci/devices/$pf/sriov_numvfs")
        vf=/sys/bus/pci/devices/$pf/virtfn$((num_vfs - 1 - slot))
        if ((slot >= num_vfs)) || [[ "$(basename "$(readlink -f "$vf/driver")")" != vfio-pci ]]; then
            echo "ERROR: QAT PF $pf has no VF $((num_vfs - 1 - slot)) bound to vfio-pci (run qat_init or start the qat service)" >&2
            return 1
        fi
        vf=$(readlink -f "$vf")
        echo "${vf##*/} $(basename "$(readlink -f "$vf/iommu_group")")"
        found=1
    done
    if ((found == 0)); then
        echo "ERROR: no QAT PF found on NUMA node $numa" >&2
        return 1
    fi
}

# Container side: prints "<vf> <iommu group>" for each VF that start.sh passed in.
qat_container_vfs() {
    local dev
    local -a members

    for dev in /dev/vfio/[0-9]*; do
        [[ -e "$dev" ]] || continue
        members=("/sys/kernel/iommu_groups/${dev##*/}/devices"/*)
        echo "${members[0]##*/} ${dev##*/}"
    done
    if [[ ! -e "${dev:-}" ]]; then
        echo "ERROR: no QAT VF was passed into this container (/dev/vfio); restart it with IAXL_RECIPE=qat-standalone ./start.sh" >&2
        return 1
    fi
}

# The rank gets every instance of its QAT VFs (16 with four VFs). Each thread can poll several,
# and in kvstore_benchmark 16 instances ran fastest on 8 threads: more slowed the host<->GPU
# copies. So cap the default thread count at 8 and at the CPUs IAXL has: its own
# KVSHRINK_CPU_OMP_THREADS_BIND entry when that is apart from vLLM's, else the rank's CPUs minus
# the reserved ones. setvars.sh calls this for the default IAXL_OMP_THREAD_NUM only, after
# rank_cpu_counts has set MIN_RANK_CPU_COUNT and after IAXL_RESERVED_CPU_NUM, and still checks
# the result.
eval "recipe_orig_$(declare -f omp_thread_count)"
omp_thread_count() {
    local threads max

    threads=$(recipe_orig_omp_thread_count "$@") || return 1
    if [[ -n "${KVSHRINK_CPU_OMP_THREADS_BIND:-}" && "$KVSHRINK_CPU_OMP_THREADS_BIND" != "${VLLM_CPU_OMP_THREADS_BIND:-}" ]]; then
        max=$(expand_cpu_list "${KVSHRINK_CPU_OMP_THREADS_BIND%%|*}" | wc -l)
    else
        max=$((${MIN_RANK_CPU_COUNT:-0} - ${IAXL_RESERVED_CPU_NUM:-0}))
    fi
    ((max <= 8)) || max=8
    ((max < 1 || threads <= max)) || threads=$max
    echo "$threads"
}

return 0
