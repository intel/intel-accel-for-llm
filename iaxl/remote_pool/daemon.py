#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""remote_pool daemon launcher. Mirrors the vLLM-side process topology: one rank
process per TP rank (KVStore with RDMA data plane, listening on port+1+rank) plus
one scheduler process (has-only KVStore, listening on `port`).

Requires IAXL_RDMA_ENABLE=1 (the RDMA backend is always built into `iaxl.torch_ext`).

Run:  python3 -m iaxl.remote_pool.daemon [--nic-ips IP[,IP...]] [--port PORT] [--tp-size N]
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time

logger = logging.getLogger(__name__)


def serve_rank(rank: int, tp_size: int, ips: list[str], port: int):
    from iaxl import setup_root_logger
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel
    from . import rpc
    from .nixl_impl import configure_ucx_env, rdma_xfer_cpp

    setup_root_logger()
    bind_cpu_affinity(rank, tp_size, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)
    configure_ucx_env(ips, rank=rank)
    xfer = rdma_xfer_cpp(f"daemon{rank}", port)
    logger.info("daemon rank %d listening on %s:%d", rank, ips[0], port)
    rpc.serve(xfer, rpc.KVStoreService(xfer, "worker", rank=rank, tp_size=tp_size))


def serve_scheduler(tp_size: int, ips: list[str], port: int):
    from iaxl import setup_root_logger
    from . import rpc
    from .nixl_impl import rdma_xfer

    setup_root_logger()
    xfer = rdma_xfer("daemon_sched", ips, listen_port=port)
    logger.info("daemon scheduler listening on %s:%d", ips[0], port)
    rpc.serve(xfer, rpc.KVStoreService(xfer, "controller", tp_size=tp_size))


def main():
    from iaxl.envs import envs
    from .nixl_impl import parse_nic_ips
    from .rpc import rank_port

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nic-ips", default=envs.IAXL_RDMA_DAEMON_NIC_IPS,
                    help="comma list of local RDMA NIC IPs (a single IP is allowed); rank r uses entry r %% len, "
                         "entry 0 is also the control-plane IP (IAXL_RDMA_DAEMON_NIC_IPS)")
    ap.add_argument("--port", type=int, default=envs.IAXL_RDMA_DAEMON_PORT)
    ap.add_argument("--tp-size", type=int, default=envs.IAXL_RDMA_TP_SIZE)
    args = ap.parse_args()
    try:
        ips = parse_nic_ips(args.nic_ips, "--nic-ips or IAXL_RDMA_DAEMON_NIC_IPS")
    except ValueError as e:
        ap.error(str(e))
    if not envs.IAXL_RDMA_ENABLE:
        ap.error("IAXL_RDMA_ENABLE=1 is required on the daemon")

    ctx = multiprocessing.get_context("spawn")  # UCX / torch state must not be forked
    procs = [ctx.Process(target=serve_rank, args=(r, args.tp_size, ips, rank_port(args.port, r)),
                         name=f"daemon{r}") for r in range(args.tp_size)]
    procs.append(ctx.Process(target=serve_scheduler, args=(args.tp_size, ips, args.port), name="daemon_sched"))
    for p in procs:
        p.start()

    def forward(signum, _frame):
        for p in procs:
            if p.is_alive():
                p.terminate()

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    while all(p.is_alive() for p in procs):
        time.sleep(0.2)
    exited = [p for p in procs if not p.is_alive()]
    print(f"[daemon] {exited[0].name} exited (code {exited[0].exitcode}); shutting down", file=sys.stderr)
    forward(None, None)
    for p in procs:
        p.join()
    sys.exit(exited[0].exitcode or 0)


if __name__ == "__main__":
    main()
