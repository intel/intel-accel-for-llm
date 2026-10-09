"""NIXL wrapper. `rdma_xfer` owns one NIXL agent (UCX backend pinned to the RDMA
NIC selected from the process's NIC IP list) and exposes memory registration, peer metadata
exchange, notifications and one-shot RDMA writes. It knows nothing about KV
caches or RPC (see rpc.py). `rdma_xfer_cpp` offers the same surface on top of
the C++ agent embedded in `iaxl.torch_ext` (daemon rank processes)."""

import json
import logging
import os
import subprocess
import time

DEFAULT_PORT = 5555
LOCAL_AGENT = "NIXL_INIT_AGENT"


def netdev_of_ip(ip: str) -> str:
    out = subprocess.check_output(["ip", "-j", "-4", "addr"], text=True)
    for link in json.loads(out):
        if any(a.get("local") == ip for a in link.get("addr_info", [])):
            return link["ifname"]
    raise RuntimeError(f"no network interface owns {ip}")


def parse_nic_ips(ips: str | None, name: str) -> list[str]:
    """Split a required comma-separated NIC IP list (a single IP is allowed); `name` labels errors."""
    lst = [s.strip() for s in (ips or "").split(",")]
    if not all(lst):
        raise ValueError(f"{name} is required: a comma-separated list of local RDMA NIC IPs "
                         f"without empty entries (got {ips!r})")
    return lst


def configure_ucx_env(ips: list[str], rank: int | None = None) -> str:
    """Pin UCX to the RDMA NIC owning this process's IP. Must run before `import nixl`.

    Rank r uses ips[r % len]; the scheduler (rank=None) also uses ips[0]. The IP is
    resolved on this host, so interface names may differ across nodes."""
    ip = ips[0] if rank is None else ips[rank % len(ips)]
    nic = netdev_of_ip(ip)
    ib_dir = f"/sys/class/net/{nic}/device/infiniband"
    if not os.path.isdir(ib_dir):
        raise RuntimeError(f"{nic} ({ip}) is not an RDMA-capable NIC")
    ibdev = sorted(os.listdir(ib_dir))[0]
    who = "scheduler" if rank is None else f"rank {rank}"
    log = logging.getLogger(__name__)
    prev = os.environ.get("UCX_NET_DEVICES")
    if prev and prev != f"{ibdev}:1":
        log.warning("%s: overriding UCX_NET_DEVICES=%s -> %s:1", who, prev, ibdev)
    os.environ["UCX_NET_DEVICES"] = f"{ibdev}:1"
    # One NIC per process; UCX multi-rail rendezvous has crashed here before.
    os.environ["UCX_MAX_RNDV_RAILS"] = "1"
    os.environ.setdefault("UCX_TLS", "rc,cuda_copy,cuda_ipc")
    log.info("%s: RDMA NIC %s (%s, %s)", who, ibdev, nic, ip)
    return ibdev


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


class rdma_xfer:
    """One NIXL agent (Python bindings). `listen_port` is the metadata listener
    port; None lets the OS pick one (connect-only agent)."""

    def __init__(self, name: str, ips: list[str], listen_port: int | None = None,
                 rank: int | None = None):
        self.ibdev = configure_ucx_env(ips, rank=rank)  # must precede `import nixl`
        from nixl._api import nixl_agent, nixl_agent_config

        # The listen (comm) thread also drives fetch_remote_metadata /
        # send_local_metadata; without it those calls are silently dropped.
        cfg = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            backends=["UCX"],
            listen_port=listen_port or 0,
        )
        self.name = name
        self.agent = nixl_agent(name, cfg)
        self._handles = []

    # -- peers ---------------------------------------------------------------
    def connect(self, peer: str, ip: str, port: int, timeout_s: float | None = None):
        """Exchange metadata with a listening peer. Register local memory first:
        rkeys of already-registered buffers ride along with our metadata."""
        self.agent.fetch_remote_metadata(peer, ip, port)
        self.agent.send_local_metadata(ip, port)
        self.wait_peer(peer, timeout_s)

    def wait_peer(self, peer: str, timeout_s: float | None = None):
        t0 = time.perf_counter()
        while not self.agent.check_remote_metadata(peer):
            if timeout_s is not None and time.perf_counter() - t0 > timeout_s:
                raise TimeoutError(f"no metadata from {peer}")
            time.sleep(1e-3)

    def disconnect(self, peer: str):
        self.agent.remove_remote_agent(peer)

    # -- memory --------------------------------------------------------------
    def register_memory(self, tensor):
        h = self.agent.register_memory(tensor)
        self._handles.append(h)
        return h

    def deregister_memory(self, handle):
        self.agent.deregister_memory(handle)

    # -- transfers -----------------------------------------------------------
    def write(self, peer: str, local_addr: int, remote_addr: int, nbytes: int, notif: bytes = b"",
              timeout_s: float = 60.0):
        """RDMA WRITE `nbytes` from a registered local DRAM buffer into the peer's
        registered DRAM buffer, delivering `notif` once the data has landed."""
        a = self.agent
        h = a.initialize_xfer(
            "WRITE",
            a.get_xfer_descs([(local_addr, nbytes, 0)], "DRAM"),
            a.get_xfer_descs([(remote_addr, nbytes, 0)], "DRAM"),
            peer,
            notif,
        )
        try:
            state = a.transfer(h)
            t0 = time.perf_counter()
            while state != "DONE":
                if state == "ERR":
                    raise RuntimeError("NIXL write failed")
                if time.perf_counter() - t0 > timeout_s:
                    raise TimeoutError("NIXL write timed out")
                state = a.check_xfer_state(h)
        finally:
            a.release_xfer_handle(h)

    # -- notifications -------------------------------------------------------
    def send_notif(self, peer: str, payload: bytes):
        self.agent.send_notif(peer, payload)

    def iter_notifs(self):
        """Yield (peer_name, payload_bytes) for every pending notification."""
        for peer, msgs in self.agent.get_new_notifs().items():
            for m in msgs:
                yield _s(peer), m


class rdma_xfer_cpp:
    """Notification/registration surface over the agent owned by iaxl.torch_ext
    (always built in). The data plane runs inside the extension."""

    def __init__(self, name: str, listen_port: int):
        from iaxl import torch_ext

        self.name = name
        self._ext = torch_ext
        torch_ext.rdma_init(name, listen_port)

    def wait_peer(self, peer: str, timeout_s: float | None = None):
        self._ext.rdma_wait_peer(peer, timeout_s if timeout_s is not None else 3600.0)

    def disconnect(self, peer: str):
        self._ext.rdma_remove_peer(peer)

    def register_memory(self, tensor):
        self._ext.rdma_register_mem(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        return tensor

    def send_notif(self, peer: str, payload: bytes):
        self._ext.rdma_send_notif(peer, payload)

    def iter_notifs(self):
        yield from self._ext.rdma_get_notifs()
