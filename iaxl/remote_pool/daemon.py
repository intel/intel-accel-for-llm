#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""remote_pool daemon launcher. Mirrors the vLLM-side process topology: one rank
process per TP rank (KVStore with RDMA data plane, listening on port+1+rank) plus
one scheduler process (has-only KVStore, listening on `port`).

Requires IAXL_RDMA_ENABLE=1 (the RDMA backend is always built into `iaxl.torch_ext`).

Run:  python3 -m iaxl.remote_pool.daemon [--ip IP] [--port PORT] [--tp-size N]
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time

logger = logging.getLogger(__name__)


def serve_rank(rank: int, tp_size: int, ip: str, port: int):
    from iaxl import setup_root_logger
    from iaxl.envs import envs
    from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel
    from . import rpc
    from .nixl_impl import configure_ucx_env, rdma_xfer_cpp

    setup_root_logger()
    bind_cpu_affinity(rank, tp_size, os.getenv("VLLM_CPU_OMP_THREADS_BIND"))
    bind_intel_accel(rank)
    configure_ucx_env(ip, rank=rank, ips=envs.IAXL_RDMA_DAEMON_NIC_IPS or None)
    xfer = rdma_xfer_cpp(f"daemon{rank}", port)
    logger.info("daemon rank %d listening on %s:%d", rank, ip, port)
    rpc.serve(xfer, rpc.KVStoreService(xfer, "worker", rank=rank, tp_size=tp_size))


def serve_scheduler(tp_size: int, ip: str, port: int):
    from iaxl import setup_root_logger
    from . import rpc
    from .nixl_impl import rdma_xfer

    setup_root_logger()
    xfer = rdma_xfer("daemon_sched", listen_port=port, local_ip=ip)
    logger.info("daemon scheduler listening on %s:%d", ip, port)
    rpc.serve(xfer, rpc.KVStoreService(xfer, "controller", tp_size=tp_size))


def main():
    from iaxl.envs import envs
    from .rpc import rank_port

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=envs.IAXL_RDMA_DAEMON_IP or None, help="RDMA NIC IP (IAXL_RDMA_DAEMON_IP)")
    ap.add_argument("--port", type=int, default=envs.IAXL_RDMA_DAEMON_PORT)
    ap.add_argument("--tp-size", type=int, default=envs.IAXL_RDMA_TP_SIZE)
    args = ap.parse_args()
    if not args.ip:
        ap.error("--ip or IAXL_RDMA_DAEMON_IP is required")
    if not envs.IAXL_RDMA_ENABLE:
        ap.error("IAXL_RDMA_ENABLE=1 is required on the daemon")

    ctx = multiprocessing.get_context("spawn")  # UCX / torch state must not be forked
    procs = [ctx.Process(target=serve_rank, args=(r, args.tp_size, args.ip, rank_port(args.port, r)),
                         name=f"daemon{r}") for r in range(args.tp_size)]
    procs.append(ctx.Process(target=serve_scheduler, args=(args.tp_size, args.ip, args.port), name="daemon_sched"))
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
