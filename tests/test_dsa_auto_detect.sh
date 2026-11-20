#!/usr/bin/env bash
# Run from the repository root: bash tests/test_dsa_auto_detect.sh
set -euo pipefail
source tools/auto_config.sh

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
export IAXL_DSA_SYSFS_ROOT="$workdir/sys"
root="$IAXL_DSA_SYSFS_ROOT/bus/dsa/devices"
mkdir -p "$root"

# Override only GPU discovery, leaving DSA discovery untouched.
gpu_rows() {
    local count=$1 i
    for ((i=0; i<count; i++)); do printf '%s\n' "$i,mock"; done
}
gpu_info() {
    local index=${1%%,*}
    # Four GPUs on each NUMA node for eight-rank tests.
    echo "$index mock $((index / 4))"
}

add_dsa() {
    local id=$1 numa=$2
    mkdir -p "$root/dsa$id"
    printf '%s\n' "$numa" > "$root/dsa$id/numa_node"
}
add_wq() {
    local id=$1 queue=$2 state=${3:-enabled} mode=${4:-dedicated} type=${5:-user}
    local path="$root/wq$id.$queue"
    mkdir -p "$path"
    printf '%s\n' "$state" > "$path/state"
    printf '%s\n' "$mode" > "$path/mode"
    printf '%s\n' "$type" > "$path/type"
}

for id in 0 2 4 6; do
    if ((id < 4)); then numa=0; else numa=1; fi
    add_dsa "$id" "$numa"
    add_wq "$id" 0
    add_wq "$id" 1
done

expected='wq0.0|wq2.0|wq0.1|wq2.1|wq4.0|wq6.0|wq4.1|wq6.1'
actual=$(dsa_auto_detect 8 2>"$workdir/success.log")
[[ "$actual" == "$expected" ]] || { echo "FAIL TP=8: $actual"; exit 1; }
echo 'PASS TP=8 balances ranks across NUMA-local DSAs'

actual=$(dsa_auto_detect 4 2>"$workdir/tp4.log")
[[ "$actual" == 'wq0.0|wq2.0|wq0.1|wq2.1' ]] || { echo "FAIL TP=4: $actual"; exit 1; }
echo 'PASS TP=4 balances ranks across DSAs'

# Regression: with two physical DSAs on NUMA 0, TP=2 must use both
# before allocating a second work queue from either device.
actual=$(dsa_auto_detect 2 2>"$workdir/tp2.log")
[[ "$actual" == 'wq0.0|wq2.0' ]] || { echo "FAIL TP=2: $actual"; exit 1; }
echo 'PASS TP=2 spreads work across physical DSAs'

# Insufficient total queues: one WQ disabled.
printf '%s\n' disabled >"$root/wq6.1/state"
if dsa_auto_detect 8 >"$workdir/out" 2>"$workdir/err"; then
    echo 'FAIL: missing eighth WQ unexpectedly accepted'; exit 1
fi
grep -q 'found 7 across 4 DSA devices' "$workdir/err"
echo 'PASS insufficient total enabled queues gives actionable error'
printf '%s\n' enabled >"$root/wq6.1/state"

# Shared WQ is unsupported by the dedicated-queue transfer path.
printf '%s\n' shared >"$root/wq6.1/mode"
if dsa_auto_detect 8 >"$workdir/out" 2>"$workdir/err"; then
    echo 'FAIL: shared WQ unexpectedly accepted'; exit 1
fi
echo 'PASS shared WQ is excluded'
printf '%s\n' dedicated >"$root/wq6.1/mode"

# Eight eligible queues globally, but only three on NUMA 0.
printf '%s\n' 1 >"$root/dsa2/numa_node"
if dsa_auto_detect 8 >"$workdir/out" 2>"$workdir/err"; then
    echo 'FAIL: cross-NUMA fallback unexpectedly accepted'; exit 1
fi
grep -q 'NUMA 0' "$workdir/err"
echo 'PASS cross-NUMA allocation rejected'

echo 'ALL DSA AUTO-DETECTION TESTS PASSED'
