#!/usr/bin/env bash

validate_tp_size() {
    local tp_size=$1
    if ! [[ "$tp_size" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TP_SIZE must be a positive integer, got '$tp_size'" >&2
        return 1
    fi
}

gpu_rows() {
    local tp_size=$1
    local gpu_output

    validate_tp_size "$tp_size" || return 1
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required to detect GPU NUMA nodes" >&2
        return 1
    fi
    if ! gpu_output=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits); then
        echo "ERROR: failed to query GPU PCI information with nvidia-smi" >&2
        return 1
    fi

    local -a gpu_rows=()
    mapfile -t gpu_rows <<<"$gpu_output"
    if ((${#gpu_rows[@]} < tp_size)); then
        echo "ERROR: TP_SIZE=$tp_size requires $tp_size GPUs, but only ${#gpu_rows[@]} were found" >&2
        return 1
    fi
    printf '%s\n' "${gpu_rows[@]}"
}

gpu_info() {
    local row=$1
    local gpu_index pci_bus pci_address numa

    IFS=, read -r gpu_index pci_bus <<<"$row"
    gpu_index=${gpu_index//[[:space:]]/}
    pci_bus=${pci_bus//[[:space:]]/}
    pci_address=$(sed -E 's/^[[:xdigit:]]{4}([[:xdigit:]]{4}:)/\1/' <<<"$pci_bus" | tr '[:upper:]' '[:lower:]')
    if [[ ! -r "/sys/bus/pci/devices/$pci_address/numa_node" ]]; then
        echo "ERROR: cannot read NUMA node for GPU $gpu_index at PCI $pci_bus" >&2
        return 1
    fi
    numa=$(<"/sys/bus/pci/devices/$pci_address/numa_node")
    if ((numa < 0)); then
        echo "ERROR: GPU $gpu_index at PCI $pci_bus has no NUMA node" >&2
        return 1
    fi
    printf '%s %s %s\n' "$gpu_index" "$pci_bus" "$numa"
}

expand_cpu_list() {
    local cpu_list=$1
    local part start end cpu

    for part in ${cpu_list//,/ }; do
        if [[ "$part" == *-* ]]; then
            IFS=- read -r start end <<<"$part"
            for ((cpu = start; cpu <= end; cpu++)); do
                echo "$cpu"
            done
        else
            echo "$part"
        fi
    done
}

compress_cpu_list() {
    printf '%s\n' "$@" | sort -n | awk '
        NR == 1 { start = previous = $1; next }
        $1 == previous + 1 { previous = $1; next }
        {
            printf "%s%s", separator, start == previous ? start : start "-" previous
            separator = ","
            start = previous = $1
        }
        END {
            if (NR > 0)
                printf "%s%s\n", separator, start == previous ? start : start "-" previous
        }
    '
}

cpu_auto_detect() {
    local tp_size=${1:-${TP_SIZE:-}}
    local gpu_output rank gpu_index pci_bus numa node_path cpu topology siblings
    local ranks local_rank cores_per_rank extra_cores core_count core_start offset cpu_group result
    local -a gpu_rows=() gpu_numas=() gpu_indices=() gpu_buses=() core_groups=()
    local -a cpu_ids=() rank_cpu_groups=()
    local -A ranks_per_numa=() next_rank_on_numa=() seen_cores=()

    gpu_output=$(gpu_rows "$tp_size") || return 1
    mapfile -t gpu_rows <<<"$gpu_output"

    for ((rank = 0; rank < tp_size; rank++)); do
        read -r gpu_index pci_bus numa < <(gpu_info "${gpu_rows[$rank]}") || return 1
        gpu_numas+=("$numa")
        gpu_indices+=("$gpu_index")
        gpu_buses+=("$pci_bus")
        ranks_per_numa[$numa]=$((${ranks_per_numa[$numa]:-0} + 1))
    done

    for ((rank = 0; rank < tp_size; rank++)); do
        numa=${gpu_numas[$rank]}
        node_path=/sys/devices/system/node/node$numa
        if [[ ! -r "$node_path/cpulist" ]]; then
            echo "ERROR: cannot read CPU list for NUMA node $numa" >&2
            return 1
        fi

        seen_cores=()
        core_groups=()
        while read -r cpu; do
            topology=/sys/devices/system/cpu/cpu$cpu/topology
            [[ -r "$topology/thread_siblings_list" ]] || continue
            siblings=$(<"$topology/thread_siblings_list")
            [[ -n "${seen_cores[$siblings]:-}" ]] && continue
            seen_cores[$siblings]=1
            core_groups+=("$cpu")
        done < <(expand_cpu_list "$(<"$node_path/cpulist")")

        ranks=${ranks_per_numa[$numa]}
        if ((${#core_groups[@]} < ranks)); then
            echo "ERROR: NUMA node $numa has ${#core_groups[@]} physical cores for $ranks GPU ranks" >&2
            return 1
        fi
        local_rank=${next_rank_on_numa[$numa]:-0}
        next_rank_on_numa[$numa]=$((local_rank + 1))
        cores_per_rank=$((${#core_groups[@]} / ranks))
        extra_cores=$((${#core_groups[@]} % ranks))
        core_count=$cores_per_rank
        ((local_rank < extra_cores)) && core_count=$((core_count + 1))
        core_start=$((local_rank * cores_per_rank + (local_rank < extra_cores ? local_rank : extra_cores)))

        cpu_ids=()
        for ((offset = 0; offset < core_count; offset++)); do
            cpu_ids+=("${core_groups[$((core_start + offset))]}")
        done
        cpu_group=$(compress_cpu_list "${cpu_ids[@]}")
        rank_cpu_groups+=("$cpu_group")
        echo "KVShrink CPU rank $rank: GPU=${gpu_indices[$rank]} PCI=${gpu_buses[$rank]} NUMA=$numa CPUs=$cpu_group" >&2
    done

    result=$(IFS='|'; echo "${rank_cpu_groups[*]}")
    echo "$result"
}

qat_auto_detect() {
    local tp_size=${1:-${TP_SIZE:-}}
    local path driver devices_per_rank extra_devices next_device rank count offset pci numa group result
    local -a qat_devices=() indices=() pci_devices=() rank_groups=()

    validate_tp_size "$tp_size" || return 1
    for path in /sys/bus/pci/devices/*; do
        [[ -r "$path/class" && -L "$path/driver" ]] || continue
        [[ "$(<"$path/class")" == "0x0b4000" ]] || continue
        driver=$(basename "$(readlink -f "$path/driver")")
        [[ "$driver" != *vf* ]] || continue
        [[ "$driver" =~ (qat|4xxx|420xx|c6xx|c3xxx|200xx|dh895xcc) ]] || continue
        qat_devices+=("${path##*/}")
    done

    if ((${#qat_devices[@]} == 0)); then
        echo "ERROR: no QAT devices bound to a supported PF driver were found" >&2
        return 1
    fi
    mapfile -t qat_devices < <(printf '%s\n' "${qat_devices[@]}" | LC_ALL=C sort)
    if ((${#qat_devices[@]} < tp_size)); then
        echo "ERROR: TP_SIZE=$tp_size requires at least $tp_size QAT devices, but only ${#qat_devices[@]} were found" >&2
        return 1
    fi

    devices_per_rank=$((${#qat_devices[@]} / tp_size))
    extra_devices=$((${#qat_devices[@]} % tp_size))
    next_device=0
    for ((rank = 0; rank < tp_size; rank++)); do
        count=$devices_per_rank
        ((rank < extra_devices)) && count=$((count + 1))
        indices=()
        pci_devices=()
        for ((offset = 0; offset < count; offset++)); do
            indices+=("$next_device")
            pci=${qat_devices[$next_device]}
            numa=$(<"/sys/bus/pci/devices/$pci/numa_node")
            pci_devices+=("$pci(numa=$numa)")
            next_device=$((next_device + 1))
        done
        group=$(IFS=,; echo "${indices[*]}")
        rank_groups+=("$group")
        echo "KVShrink QAT rank $rank: indices=$group devices=${pci_devices[*]}" >&2
    done

    result=$(IFS='|'; echo "${rank_groups[*]}")
    echo "$result"
}

dsa_auto_detect() {
    local tp_size=${1:-${TP_SIZE:-}}
    local gpu_output rank gpu_index pci_bus gpu_numa dsa_path dsa_name dsa_numa dsa_id wq_path
    local selected_dsa selected_wq result
    local -a gpu_rows=() dsa_paths=() used_dsa=() rank_wqs=() candidate_wqs=()

    gpu_output=$(gpu_rows "$tp_size") || return 1
    mapfile -t gpu_rows <<<"$gpu_output"
    mapfile -t dsa_paths < <(
        for dsa_path in /sys/bus/dsa/devices/dsa[0-9]*; do
            [[ -d "$dsa_path" ]] && echo "$dsa_path"
        done | sort -V
    )
    if ((${#dsa_paths[@]} < tp_size)); then
        echo "ERROR: TP_SIZE=$tp_size requires at least $tp_size DSA devices, but only ${#dsa_paths[@]} were found" >&2
        return 1
    fi

    for ((rank = 0; rank < tp_size; rank++)); do
        read -r gpu_index pci_bus gpu_numa < <(gpu_info "${gpu_rows[$rank]}") || return 1
        echo "KVShrink GPU rank $rank: index=$gpu_index pci=$pci_bus numa=$gpu_numa" >&2

        selected_dsa=""
        selected_wq=""
        for dsa_path in "${dsa_paths[@]}"; do
            dsa_name=${dsa_path##*/}
            [[ " ${used_dsa[*]} " == *" $dsa_name "* ]] && continue
            dsa_numa=$(<"$dsa_path/numa_node")
            [[ "$dsa_numa" == "$gpu_numa" ]] || continue

            dsa_id=${dsa_name#dsa}
            mapfile -t candidate_wqs < <(
                for wq_path in /sys/bus/dsa/devices/wq"$dsa_id".*; do
                    [[ -e "$wq_path" ]] || continue
                    [[ "$(<"$wq_path/type")" == "user" ]] || continue
                    [[ "$(<"$wq_path/state")" == "enabled" ]] || continue
                    basename "$wq_path"
                done | sort -V
            )
            ((${#candidate_wqs[@]} > 0)) || continue
            selected_dsa=$dsa_name
            selected_wq=${candidate_wqs[0]}
            break
        done

        if [[ -z "$selected_dsa" ]]; then
            echo "ERROR: no unused DSA with enabled user WQs is available on NUMA $gpu_numa for GPU $gpu_index" >&2
            return 1
        fi
        used_dsa+=("$selected_dsa")
        rank_wqs+=("$selected_wq")
        echo "KVShrink GPU rank $rank: DSA=$selected_dsa WQ=$selected_wq" >&2
    done

    result=$(IFS='|'; echo "${rank_wqs[*]}")
    echo "$result"
}

rank_cpu_counts() {
    local affinity=$1
    local tp_size=$2
    local rank cpu_count
    local -a rank_cpu_specs=()

    validate_tp_size "$tp_size" || return 1
    IFS='|' read -r -a rank_cpu_specs <<<"$affinity"
    if ((${#rank_cpu_specs[@]} < tp_size)); then
        echo "ERROR: VLLM_CPU_OMP_THREADS_BIND has ${#rank_cpu_specs[@]} entries for TP_SIZE=$tp_size" >&2
        return 1
    fi

    RANK_CPU_COUNTS=()
    MIN_RANK_CPU_COUNT=0
    for ((rank = 0; rank < tp_size; rank++)); do
        cpu_count=$(awk -F, '
            {
                count = 0
                for (i = 1; i <= NF; i++) {
                    gsub(/[[:space:]]/, "", $i)
                    if ($i ~ /^[0-9]+$/) {
                        count++
                    } else if ($i ~ /^[0-9]+-[0-9]+$/) {
                        split($i, range, "-")
                        if (range[1] > range[2])
                            exit 1
                        count += range[2] - range[1] + 1
                    } else {
                        exit 1
                    }
                }
                print count
            }
        ' <<<"${rank_cpu_specs[$rank]}") || {
            echo "ERROR: invalid CPU affinity for rank $rank: ${rank_cpu_specs[$rank]}" >&2
            return 1
        }
        RANK_CPU_COUNTS+=("$cpu_count")
        if ((MIN_RANK_CPU_COUNT == 0 || cpu_count < MIN_RANK_CPU_COUNT)); then
            MIN_RANK_CPU_COUNT=$cpu_count
        fi
    done
}

qat_thread_count() {
    local devices=$1
    local instances_per_device=$2
    local -a device_indices=()

    if ! [[ "$instances_per_device" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: IAXL_QAT_ZIP_INSTANCES_PER_DEVICE must be a positive integer" >&2
        return 1
    fi
    IFS=, read -r -a device_indices <<<"$devices"
    if ((${#device_indices[@]} == 0)); then
        echo "ERROR: IAXL_QAT_DEVICES must contain at least one device index" >&2
        return 1
    fi
    echo "$((${#device_indices[@]} * instances_per_device))"
}

cpu_zip_thread_count() {
    local min_rank_cpu_count=$1
    local qat_threads=$2
    local reserved_cpus=$3

    if ! [[ "$min_rank_cpu_count" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: minimum rank CPU count must be a positive integer" >&2
        return 1
    fi
    if ! [[ "$qat_threads" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_QAT_INSTANCE_NUM must be a non-negative integer" >&2
        return 1
    fi
    if ! [[ "$reserved_cpus" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_RESERVED_CPU_NUM must be a non-negative integer" >&2
        return 1
    fi
    echo "$((min_rank_cpu_count - qat_threads - reserved_cpus))"
}

omp_thread_count() {
    local qat_threads=$1
    local cpu_zip_threads=$2

    if ! [[ "$qat_threads" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_QAT_INSTANCE_NUM must be a non-negative integer" >&2
        return 1
    fi
    if ! [[ "$cpu_zip_threads" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_CPU_ZIP_THREADS must be a non-negative integer" >&2
        return 1
    fi
    echo "$((qat_threads + cpu_zip_threads))"
}

validate_omp_config() {
    local min_rank_cpu_count=$1
    local rank_cpu_counts

    if ! [[ "$IAXL_QAT_INSTANCE_NUM" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_QAT_INSTANCE_NUM must be a non-negative integer" >&2
        return 1
    fi
    if ! [[ "$IAXL_RESERVED_CPU_NUM" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_RESERVED_CPU_NUM must be a non-negative integer" >&2
        return 1
    fi
    if ! [[ "$IAXL_CPU_ZIP_THREADS" =~ ^[0-9]+$ ]]; then
        echo "ERROR: IAXL_CPU_ZIP_THREADS must be a non-negative integer" >&2
        return 1
    fi
    if ((IAXL_QAT_INSTANCE_NUM + IAXL_CPU_ZIP_THREADS == 0)); then
        echo "ERROR: at least one QAT or CPU zip worker must be enabled" >&2
        return 1
    fi
    if ! [[ "$IAXL_OMP_THREAD_NUM" =~ ^[1-9][0-9]*$ ]] ||
        ((IAXL_OMP_THREAD_NUM != IAXL_QAT_INSTANCE_NUM + IAXL_CPU_ZIP_THREADS)); then
        echo "ERROR: IAXL_OMP_THREAD_NUM must equal IAXL_QAT_INSTANCE_NUM + IAXL_CPU_ZIP_THREADS" >&2
        return 1
    fi
    if ((IAXL_OMP_THREAD_NUM + IAXL_RESERVED_CPU_NUM > min_rank_cpu_count)); then
        echo "ERROR: QAT ($IAXL_QAT_INSTANCE_NUM) + CPU zip ($IAXL_CPU_ZIP_THREADS) + reserved " \
            "($IAXL_RESERVED_CPU_NUM) exceeds the smallest rank CPU allocation ($min_rank_cpu_count)" >&2
        return 1
    fi
    if [[ "$OMP_NUM_THREADS" != "$IAXL_OMP_THREAD_NUM" ||
        "$OMP_THREAD_LIMIT" != "$IAXL_OMP_THREAD_NUM" ]] ||
        ! [[ "$OMP_MAX_ACTIVE_LEVELS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: OpenMP environment does not match IAXL_OMP_THREAD_NUM" >&2
        return 1
    fi

    rank_cpu_counts=$(IFS=,; echo "${RANK_CPU_COUNTS[*]}")
    printf '%s\n' \
        "OpenMP configuration:" \
        "  rank CPU counts=$rank_cpu_counts" \
        "  IAXL_QAT_INSTANCE_NUM=$IAXL_QAT_INSTANCE_NUM" \
        "  IAXL_CPU_ZIP_THREADS=$IAXL_CPU_ZIP_THREADS" \
        "  IAXL_RESERVED_CPU_NUM=$IAXL_RESERVED_CPU_NUM" \
        "  IAXL_OMP_THREAD_NUM=$IAXL_OMP_THREAD_NUM"
}
