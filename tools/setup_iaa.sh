#!/usr/bin/env bash

set -euo pipefail

SYSFS=${IAA_SYSFS_ROOT:-/sys/bus/dsa/devices}
DRY_RUN=0

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
    "Usage: $0 [--dry-run]" \
        "" \
    "Automatically configures all detected IAA devices and engines." \
    "Each device gets one shared QPL queue; no configuration files needed." \
        "Requires bash, jq and accel-config. Run as root to apply changes." \
    "All active IAA devices are disabled before reconfiguration." \
        "Stop IAA users first. DSA devices are not changed."
}

for ARG in "$@"; do
    case "$ARG" in
        --help|-h) usage; exit 0 ;;
        --dry-run) DRY_RUN=1 ;;
        *) die "unknown argument: $ARG (use --help)" ;;
    esac
done

shopt -s nullglob
command -v jq >/dev/null || die "jq is not installed or not on PATH"
if ((!DRY_RUN)); then
    [[ $(id -u) == 0 ]] || die "run as root (or use --dry-run); this script does not invoke sudo"
    command -v accel-config >/dev/null || die "accel-config is not installed or not on PATH"
fi

DEVS=()
for SYS in "$SYSFS"/iax[0-9]*; do
    [[ -d "$SYS" && ${SYS##*/} =~ ^iax[0-9]+$ ]] || continue
    DEVS+=("${SYS##*/}")
done
((${#DEVS[@]})) || die "no IAA device found in $SYSFS"
mapfile -t DEVS < <(printf '%s\n' "${DEVS[@]}" | sort -V)

number() {
    local value
    value=$(<"$1/$2")
    [[ "$value" =~ ^[0-9]+$ ]] || die "$1: invalid $2: $value"
    printf '%s\n' "$value"
}

children() {
    local device=$1 kind=$2 path
    local device_id=${device#iax}
    for path in "$SYSFS/$device/$kind$device_id."*; do
        [[ -d "$path" && ${path##*.} =~ ^[0-9]+$ ]] || continue
        printf '%s\n' "${path##*/}"
    done
}

CONFIG=$(mktemp --suffix=.conf)
trap 'rm -f -- "$CONFIG"' EXIT

emit_device() {
    local device=$1 name=$2 total=$3 maxxfer=$4 engines=$5
    jq -n --arg dev "$device" --arg id "${device#iax}" --arg name "$name" \
        --argjson total "$total" --argjson maxxfer "$maxxfer" \
        --argjson engines "$engines" '
        ([128, $total] | min) as $size |
        {
            dev: $dev,
            groups: [{
                dev: ("group" + $id + ".0"),
                grouped_workqueues: [{
                    dev: ("wq" + $id + ".0"),
                    mode: "shared",
                    size: $size,
                    group_id: 0,
                    priority: 10,
                    block_on_fault: 1,
                    max_transfer_size: ([2147483648, $maxxfer] | min),
                    driver_name: "user",
                    type: "user",
                    name: $name,
                    threshold: $size
                }],
                grouped_engines: $engines
            }]
        }'
}

INDEX=0
for DEV in "${DEVS[@]}"; do
    INDEX=$((INDEX + 1))
    TOTAL=$(number "$SYSFS/$DEV" max_work_queues_size)
    MAXXFER=$(number "$SYSFS/$DEV" max_transfer_size)
    ENGINES=$(children "$DEV" engine | sort -V | jq -Rsc \
        'split("\n") | map(select(length > 0) | {dev: ., group_id: 0})')
    emit_device "$DEV" "app$INDEX" "$TOTAL" "$MAXXFER" "$ENGINES"
done >"$CONFIG"
GENERATED=$(jq -s . "$CONFIG")
printf '%s\n' "$GENERATED" >"$CONFIG"

jq -e '
    def nonempty_array: type == "array" and length > 0;
    def integer: type == "number" and . == floor;
    nonempty_array and
    all(.[];
        (.dev | type == "string" and test("^iax[0-9]+$")) and
        (.groups | nonempty_array) and
        all(.groups[];
            (.dev | type == "string" and test("^group[0-9]+\\.[0-9]+$")) and
            (.grouped_engines | nonempty_array) and
            (.grouped_workqueues | nonempty_array) and
            all(.grouped_engines[];
                (.dev | type == "string" and test("^engine[0-9]+\\.[0-9]+$")) and
                (.group_id | integer and . >= 0)) and
            all(.grouped_workqueues[];
                (.dev | type == "string" and test("^wq[0-9]+\\.[0-9]+$")) and
                (.group_id | integer and . >= 0) and
                (.size | integer and . > 0) and
                (.max_transfer_size | integer and . > 0))
        )
    )' "$CONFIG" >/dev/null || die "configuration must contain valid IAA devices, groups, engines and work queues"

declare -A SELECTED=()
declare -A RESOURCES=()
mapfile -t TARGETS < <(jq -r '.[].dev' "$CONFIG")
for DEV in "${TARGETS[@]}"; do
    [[ -d "$SYSFS/$DEV" && ! ${SELECTED[$DEV]+present} ]] || \
        die "missing, non-IAA, or duplicate device: $DEV"
    SELECTED[$DEV]=1
    ID=${DEV#iax}
    TOTAL=$(number "$SYSFS/$DEV" max_work_queues_size)
    MAXXFER=$(number "$SYSFS/$DEV" max_transfer_size)
    TOTAL_SIZE=0
    while IFS=$'\t' read -r KIND RESOURCE GROUP_ID SIZE XFER; do
        [[ "$RESOURCE" == "$KIND$ID."* && -d "$SYSFS/$DEV/$RESOURCE" ]] || \
            die "$DEV: unavailable $KIND $RESOURCE"
        [[ ! ${RESOURCES[$RESOURCE]+present} ]] || die "$DEV: duplicate resource $RESOURCE"
        RESOURCES[$RESOURCE]=1
        if [[ "$KIND" == group ]]; then
            GROUP=${RESOURCE##*.}
        else
            [[ "$GROUP_ID" == "$GROUP" ]] || die "$DEV: incorrect group_id for $RESOURCE"
        fi
        if [[ "$KIND" == wq ]]; then
            ((XFER <= MAXXFER)) || die "$DEV: unsupported transfer size for $RESOURCE"
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
        fi
    done < <(jq -r --arg dev "$DEV" '
        .[] | select(.dev == $dev) | .groups[] |
        (["group", .dev, 0, 0, 0],
         (.grouped_engines[] | ["engine", .dev, .group_id, 0, 0]),
         (.grouped_workqueues[] | ["wq", .dev, .group_id, .size, .max_transfer_size])) |
        @tsv' "$CONFIG")
    ((TOTAL_SIZE <= TOTAL)) || die "$DEV: configuration exceeds available queue capacity"
done

echo "IAA hardware:"
for DEV in "${DEVS[@]}"; do
    NODE=$(<"$SYSFS/$DEV/numa_node")
    echo "  numa $NODE: $DEV"
done
echo "Selected: ${TARGETS[*]}"
echo "All active IAA devices will be disabled; DSA devices are untouched."

run() {
    printf '+ accel-config'
    printf ' %q' "$@"
    printf '\n'
    if ((!DRY_RUN)); then
        accel-config "$@" || die "accel-config $* failed"
    fi
}

if ((DRY_RUN)); then
    jq . "$CONFIG"
fi
for DEV in "${DEVS[@]}"; do
    while IFS= read -r WQ; do
        if [[ $(<"$SYSFS/$DEV/$WQ/state") == enabled ]]; then
            run disable-wq "$DEV/$WQ"
        fi
    done < <(children "$DEV" wq)
    if [[ $(<"$SYSFS/$DEV/state") == enabled ]]; then
        run disable-device "$DEV"
    fi
done

for DEV in "${TARGETS[@]}"; do
    while IFS= read -r WQ; do
        run config-wq "$DEV/$WQ" --group-id=-1 --wq-size=0
    done < <(children "$DEV" wq)
    while IFS= read -r ENGINE; do
        run config-engine "$DEV/$ENGINE" --group-id=-1
    done < <(children "$DEV" engine)
done

run load-config -c "$CONFIG"
for DEV in "${TARGETS[@]}"; do
    run enable-device "$DEV"
    while IFS= read -r WQ; do
        run enable-wq "$DEV/$WQ"
    done < <(jq -r --arg dev "$DEV" \
        '.[] | select(.dev == $dev) | .groups[].grouped_workqueues[].dev' "$CONFIG")
done

if ((DRY_RUN)); then
    echo "Dry run complete. Would configure ${#TARGETS[@]} IAA device(s)."
else
    echo "Done. Configured ${#TARGETS[@]} IAA device(s)."
fi