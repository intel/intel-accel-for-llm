#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: this script must be run as root (use sudo)." >&2
    exit 1
fi

CONFS=()
for f in /etc/4xxx_dev[0-7].conf; do
    [[ -f "$f" ]] && CONFS+=("$f")
done

if [[ ${#CONFS[@]} -eq 0 ]]; then
    echo "ERROR: no /etc/4xxx_dev[0-7].conf found." >&2
    echo "       Run this on the host, not inside a container, and make sure QAT is installed." >&2
    exit 1
fi

IOMMU=0
if grep -qE "iommu=(on|pt)" /proc/cmdline; then
    IOMMU=1
    echo "IOMMU enabled: adding SVMEnabled/ATEnabled to [GENERAL]"
else
    echo "IOMMU disabled: removing SVMEnabled/ATEnabled from [GENERAL]"
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

SSL_NUM_PROCESSES=8
SSL_NUM_DC_INSTANCES=8
echo "Setting [SSL] NumProcesses = $SSL_NUM_PROCESSES, NumberDcInstances = $SSL_NUM_DC_INSTANCES"

for CONF in "${CONFS[@]}"; do
    # The Dc instance blocks are regenerated wholesale, so blank lines and the
    # comments that belong to them are buffered in `pending` until we know the
    # following line is kept.
    awk -v iommu="$IOMMU" -v procs="$SSL_NUM_PROCESSES" -v dc="$SSL_NUM_DC_INSTANCES" '
        function flush() { if (pending != "") { printf "%s", pending; pending = "" } }
        function hold(line) { pending = pending line "\n" }
        function emit_dc(   i, base) {
            base = cy + 1
            for (i = 0; i < dc; i++) {
                print ""
                print "# Data Compression - User instance #" i
                print "Dc" i "Name = \"Dc" i "\""
                print "Dc" i "IsPolled = 1"
                print "# List of core affinities"
                print "Dc" i "CoreAffinity = " (base + i)
            }
            emitted = 1
        }
        /^[[:space:]]*\[/ {
            if (section == "SSL" && !emitted) { pending = ""; emit_dc() }
            flush()
            section = $0; gsub(/[][[:space:]]/, "", section)
        }
        /^[[:space:]]*(SVMEnabled|ATEnabled)[[:space:]]*=/ { next }
        section == "SSL" && /^[[:space:]]*NumberCyInstances[[:space:]]*=/ { cy = $NF + 0 }
        section == "SSL" && /^[[:space:]]*NumProcesses[[:space:]]*=/ {
            flush(); print "NumProcesses = " procs; next
        }
        section == "SSL" && /^[[:space:]]*NumberDcInstances[[:space:]]*=/ {
            flush(); print "NumberDcInstances = " dc; next
        }
        section == "SSL" && /^[[:space:]]*Dc[0-9]+[A-Za-z]*[[:space:]]*=/ { pending = ""; next }
        section == "SSL" && /^[[:space:]]*#[[:space:]]*Data Compression - User instance/ { pending = ""; next }
        section == "SSL" && /^[[:space:]]*#[[:space:]]*List of core affinities[[:space:]]*$/ { hold($0); next }
        section == "SSL" && /^[[:space:]]*$/ { hold($0); next }
        { flush(); print }
        iommu == 1 && /^[[:space:]]*\[GENERAL\][[:space:]]*$/ {
            print "SVMEnabled = 1"
            print "ATEnabled = 1"
        }
        END {
            if (section == "SSL" && !emitted) { pending = ""; emit_dc() }
            flush()
        }
    ' "$CONF" >"$TMP"

    if cmp -s "$TMP" "$CONF"; then
        echo "  $CONF: already up to date"
        continue
    fi

    cp -a "$CONF" "$CONF.bak.$(date +%s)"
    cat "$TMP" >"$CONF"
    echo "  $CONF: updated"
done

for cmd in adf_ctl modprobe modinfo; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' not found in PATH." >&2
        echo "       Install the Intel QAT driver package (it provides adf_ctl) and run this on the host." >&2
        exit 1
    fi
done

PF_MODULES=()
for m in qat_4xxx; do
    modinfo "$m" >/dev/null 2>&1 && PF_MODULES+=("$m")
done
if [[ ${#PF_MODULES[@]} -eq 0 ]]; then
    echo "ERROR: no QAT PF driver module (qat_4xxx/qat_420xx) is installed." >&2
    exit 1
fi
if ! modinfo usdm_drv >/dev/null 2>&1; then
    echo "ERROR: the usdm_drv module is not installed; it ships with the Intel QAT package." >&2
    exit 1
fi

echo "Reloading QAT drivers: ${PF_MODULES[*]} usdm_drv intel_qat"
adf_ctl down
if ! modprobe -r "${PF_MODULES[@]}" usdm_drv intel_qat; then
    echo "ERROR: failed to unload the QAT modules; stop every process still using QAT and retry." >&2
    exit 1
fi
sleep 1
modprobe -a "${PF_MODULES[@]}"
sleep 1
modprobe usdm_drv
sleep 1
adf_ctl up

echo "Done. QAT drivers reloaded with the updated configuration."
