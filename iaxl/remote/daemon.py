# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for the remote cache daemon.

Example (single-host TCP smoke test, no compression):

    python -m iaxl.remote.daemon --control-host 0.0.0.0 --control-port 19000 \
        --pool-size-gb 8 --no-compress

Example (production, NIXL + native QAT/IAA/CPU compression, single process
serving every rank of one TP group):

    python -m iaxl.remote.daemon --control-host 0.0.0.0 --control-port 19000 \
        --nixl-host 10.10.10.20 --nixl-port 19100 \
        --pool-size-gb 256 --staging-slots 256 --staging-slot-mb 8

Example (multi-process daemon, one instance per GPU rank -- see the design
doc's "why multi-process daemon" section for when this is worth it):

    for i in 0 1 2 3; do
        KVSHRINK_REMOTE_QAT_DEVICES="0,1|2,3|4,5|6,7" \
        python -m iaxl.remote.daemon --instance-id "$i" --num-instances 4 \
            --control-port "$((19000 + i))" --nixl-port "$((19100 + i))" \
            --pool-size-gb 64 --staging-slots 256 --staging-slot-mb 8 &
    done

Compression itself is never selected here (no ``--codec`` flag): the daemon
always calls the same native zip entry points the local, GPU-attached
KVStore uses (``iaxl.torch_ext.zip_compress_to_mem`` /
``zip_decompress_from_mem``), so backend selection (QAT / IAA / CPU) and
tuning are controlled by the same ``IAXL_*_ZIP_ENABLE`` /
``IAXL_KV_COMPRESSION`` / ``IAXL_QAT_ZIP_INSTANCES_PER_DEVICE`` environment
variables as the local path. ``--no-compress`` only disables compression for
raw-bandwidth benchmarking (equivalent to the old ``codec=none``).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal

from .server import RemoteCacheDaemon

logger = logging.getLogger(__name__)


def _resolve_qat_devices(instance_id: "int | None", num_instances: int) -> None:
    """Point the native QAT backend at the right device(s) for this process.

    ``KVSHRINK_REMOTE_QAT_DEVICES`` follows the same ``|``-separated,
    per-rank convention as ``KVSHRINK_QAT_DEVICES``/``KVSHRINK_DSA_DEVICES``
    in ``kvshrink_connector._bind_intel_accel``. The native library reads
    ``IAXL_QAT_DEVICES`` (comma-separated device indices) lazily on first use.

    - Single-process daemon (``num_instances<=1``): every listed device is
      used *together* by this one process -- flatten ``"0|1|4|5"`` (or
      ``"0,1,4,5"``) into ``IAXL_QAT_DEVICES=0,1,4,5``.
    - Multi-process daemon (``num_instances>1``, one instance per GPU rank):
      each instance uses only *its own* slice, exactly like a worker rank --
      instance ``i`` gets ``IAXL_QAT_DEVICES=<i-th entry>``. This is the
      behavioural difference the deployment requirement calls out explicitly:
      union-of-devices for one process, one-slice-per-process otherwise.

    An explicit ``IAXL_QAT_DEVICES`` always wins and is left untouched.
    """
    if os.environ.get("IAXL_QAT_DEVICES"):
        return
    spec = os.environ.get("KVSHRINK_REMOTE_QAT_DEVICES")
    if not spec:
        return

    if num_instances <= 1:
        seen: list[str] = []
        for tok in spec.replace("|", ",").split(","):
            tok = tok.strip()
            if tok and tok not in seen:
                seen.append(tok)
        if seen:
            os.environ["IAXL_QAT_DEVICES"] = ",".join(seen)
            logger.info("single-process daemon: KVSHRINK_REMOTE_QAT_DEVICES=%s -> "
                        "IAXL_QAT_DEVICES=%s (union of all listed devices)",
                        spec, os.environ["IAXL_QAT_DEVICES"])
        return

    if instance_id is None:
        raise ValueError("--instance-id is required when --num-instances > 1")
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) == 1:
        # Not actually a per-instance list -- every instance shares it.
        os.environ["IAXL_QAT_DEVICES"] = parts[0]
    else:
        if instance_id >= len(parts):
            raise ValueError(
                f"KVSHRINK_REMOTE_QAT_DEVICES has {len(parts)} '|'-separated "
                f"entries, but --instance-id={instance_id}"
            )
        os.environ["IAXL_QAT_DEVICES"] = parts[instance_id]
    logger.info("multi-process daemon instance %d/%d: IAXL_QAT_DEVICES=%s",
                instance_id, num_instances, os.environ["IAXL_QAT_DEVICES"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="iaxl remote (NIXL/RDMA) KV cache daemon")
    p.add_argument("--control-host", default="0.0.0.0")
    p.add_argument("--control-port", type=int, default=19000)
    p.add_argument("--nixl-host", default="")
    p.add_argument("--nixl-port", type=int, default=19100)
    p.add_argument("--pool-size-gb", type=float, default=8.0,
                   help="Per (model, tp_size, tp_rank) group pool budget in GiB.")
    p.add_argument("--cache-dir", default=os.environ.get(
        "KVSHRINK_REMOTE_CACHE_DIR", "_data/kvcache/remote"),
        help="Base directory for per-group chunks.db + chunks/ (same layout "
             "as the local KVStore's persist_dir).")
    p.add_argument("--compress", dest="compress", action="store_true", default=True)
    p.add_argument("--no-compress", dest="compress", action="store_false",
                   help="Store shards uncompressed (raw-bandwidth benchmarking).")
    p.add_argument("--staging-slots", type=int, default=0,
                   help="Number of NIXL staging slots (0 disables NIXL).")
    p.add_argument("--staging-slot-mb", type=float, default=4.0,
                   help="Size of each NIXL staging slot in MiB.")
    p.add_argument("--device", default="cpu",
                   help="Device for NIXL staging buffers (cpu, cuda:0, ...). "
                        "The zip pipeline runs on CPU and needs no GPU.")
    p.add_argument("--instance-id", type=int, default=None,
                   help="This process's index when running a multi-process "
                        "daemon (one instance per GPU rank); required if "
                        "--num-instances > 1.")
    p.add_argument("--num-instances", type=int, default=1,
                   help="Total number of daemon processes in this deployment "
                        "(>1 selects the per-instance QAT device slice; see "
                        "_resolve_qat_devices).")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.compress:
        _resolve_qat_devices(args.instance_id, args.num_instances)

    daemon = RemoteCacheDaemon(
        host=args.control_host,
        port=args.control_port,
        pool_bytes=int(args.pool_size_gb * 1024 ** 3),
        cache_dir=args.cache_dir,
        compress=args.compress,
        nixl_host=args.nixl_host,
        nixl_port=args.nixl_port,
        staging_slot_bytes=int(args.staging_slot_mb * 1024 ** 2),
        staging_slots=args.staging_slots,
        device=args.device,
    )

    def _sigterm(_signo, _frame):
        daemon.shutdown()

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)
    daemon.serve_forever()


if __name__ == "__main__":
    main()
