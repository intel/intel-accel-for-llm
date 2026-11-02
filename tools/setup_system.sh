#!/usr/bin/env bash
#
# /proc/sys is read-only in unprivileged containers; the host setting applies there.
[[ -w /proc/sys/kernel/numa_balancing ]] && sysctl -w kernel.numa_balancing=0 || true
