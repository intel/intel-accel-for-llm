#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw_topo.py - Print GPU / NIC information and the CPU NUMA / PCIe topology; analyse
bandwidth sharing and P2P paths.

Depends only on Linux sysfs (lspci / nvidia-smi are optional, used for names) and runs as
a normal user. When run as root it also reads the PCIe ACS configuration to tell whether
P2P between devices under the same switch is redirected to the CPU.

Usage:
  python3 hw_topo.py                        # console report + HTTP server; open the printed URL in a browser
  python3 hw_topo.py --port 9000            # choose the HTTP port (default 8080)
  python3 hw_topo.py --no-http              # console report only, exit immediately

Output:
  1. System overview: CPU / NUMA / IOMMU / related kernel parameters
  2. GPUs (NVIDIA / Intel / AMD): index (matches nvidia-smi), PCI address, NUMA, local CPUs, link, BAR1, parent switch
  3. NICs: PCI address, netdev / RDMA devices and state, NUMA, link, parent switch
  4. Accelerators: Intel QAT / DSA / IAA (on-die or add-in card), NUMA, driver, WQ / VF state
  5. Topology tree: NUMA → on-die accelerators, Root Port → PCIe Switch → GPU/NIC, with every link and
     uplink oversubscription
  6. GPU↔GPU / GPU↔NIC affinity matrices (PIX/PXB/PHB/NODE/SYS, same meaning as nvidia-smi topo -m)

Notes:
  * Bandwidth is the theoretical per-direction value (encoding overhead removed);
    oversubscription is computed from each device's maximum link capability.
  * Idle GPUs drop to Gen1 to save power: a link whose speed (but not width) is below max is
    reported as "idle-downclocked", not a fault; re-check under load. "⚠degraded" means the
    link width is below max (check slot / riser / BIOS).
  * DSA / IAA tags are the idxd device names (dsa0, iax1 → /dev/dsa/wqN.M); QAT index follows
    sorted PCI order, the same as tools/auto_config.sh.
"""
import argparse
import datetime
import html
import os
import re
import socket
import subprocess
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, field

SYSFS = "/sys/bus/pci/devices"
BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
HB_RE = re.compile(r"^pci[0-9a-f]{4}:[0-9a-f]{2}$")
GEN_OF = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
GPU_VENDORS = {"0x10de", "0x8086", "0x1002"}  # NVIDIA, Intel, AMD
MLX_VENDOR = "0x15b3"
INTEL = "0x8086"
ACCEL_KINDS = ("qat", "dsa", "iaa")
# PF device IDs: QAT dh895xcc/c3xxx/c6xx/200xx/d15xx/4xxx/420xx; DSA/IAA SPR, GNR-D, DMR
QAT_IDS = {"0x0435", "0x19e2", "0x37c8", "0x18ee", "0x6f54", "0x4940", "0x4942", "0x4944", "0x4946"}
QAT_DRV_RE = re.compile(r"qat|4xxx|420xx|c6xx|c3xxx|200xx|dh895xcc|d15xx")  # as in tools/auto_config.sh
DSA_IDS = {"0x0b25", "0x11fb", "0x1216"}
IAA_IDS = {"0x0cfe", "0x1212", "0x1217"}
IDXD_BUS = "/sys/bus/dsa/devices"
ACS_BITS = ["SrcValid", "TransBlk", "P2pReqRedir", "P2pCmpltRedir", "UpstreamFwd",
            "EgressCtrl", "DirectTrans"]
ACS_REDIRECT_MASK = 0b11100  # P2pReqRedir | P2pCmpltRedir | UpstreamFwd
AFF_DESC = {
    "PIX": "same PCIe switch, no CPU involved",
    "PXB": "multiple PCIe switches, no CPU involved",
    "PHB": "same host bridge, via CPU root complex",
    "NODE": "same NUMA node, different host bridge, via CPU",
    "SYS": "different NUMA nodes, via UPI",
}


# ----------------------------------------------------------------- sysfs helpers
def rd(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def parse_int(s, default=0):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def parse_speed(s):
    m = re.match(r"\s*([\d.]+)\s*GT/s", s or "")
    return float(m.group(1)) if m else 0.0


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def bw_per_dir(gts, width):
    """Theoretical PCIe bandwidth per direction, GB/s"""
    if gts <= 0 or width <= 0:
        return 0.0
    if gts <= 5.0:
        eff = 8 / 10
    elif gts <= 32.0:
        eff = 128 / 130
    else:
        eff = 242 / 256
    return gts * width * eff / 8


@dataclass
class Link:
    speed: float
    width: int
    max_speed: float
    max_width: int

    @property
    def bw(self):
        return bw_per_dir(self.speed, self.width)

    @property
    def max_bw(self):
        return bw_per_dir(self.max_speed, self.max_width)

    @property
    def degraded(self):
        """Width below max: likely a slot / riser / BIOS problem"""
        return bool(self.width and self.width < self.max_width)

    @property
    def downclocked(self):
        """Speed below max at full width: normal for idle GPUs saving power"""
        return bool(self.speed and self.speed < self.max_speed) and not self.degraded

    @staticmethod
    def _fmt(speed, width):
        if speed <= 0 or width <= 0:
            return "n/a"
        return f"Gen{GEN_OF.get(speed, '?')}x{width}"

    def cur(self):
        return self._fmt(self.speed, self.width)

    def mx(self):
        return self._fmt(self.max_speed, self.max_width)

    def text(self):
        """'Gen5x16 63.0 GB/s', 'Gen1x16 (max Gen5x16 63.0 GB/s) idle-downclocked'
        or 'Gen5x8 (max Gen5x16 63.0 GB/s) ⚠degraded'"""
        if self.degraded or self.downclocked:
            note = "⚠degraded" if self.degraded else "idle-downclocked"
            return f"{self.cur()} (max {self.mx()} {self.max_bw:.1f} GB/s) {note}"
        return f"{self.cur()} {self.max_bw:.1f} GB/s"


def link_of(bdf):
    d = f"{SYSFS}/{bdf}"
    return Link(parse_speed(rd(d + "/current_link_speed")),
                parse_int(rd(d + "/current_link_width")),
                parse_speed(rd(d + "/max_link_speed")),
                parse_int(rd(d + "/max_link_width")))


def chain_of(bdf):
    """BDF chain from the Root Port down to this device"""
    real = os.path.realpath(f"{SYSFS}/{bdf}")
    return [p for p in real.split("/") if BDF_RE.match(p)]


def host_bridge_of(bdf):
    real = os.path.realpath(f"{SYSFS}/{bdf}")
    return next((p for p in real.split("/") if HB_RE.match(p)), "?")


def children_of(bdf):
    real = os.path.realpath(f"{SYSFS}/{bdf}")
    try:
        return sorted(x for x in os.listdir(real)
                      if BDF_RE.match(x) and os.path.isdir(f"{real}/{x}"))
    except OSError:
        return []


def is_bridge(bdf):
    return rd(f"{SYSFS}/{bdf}/class").startswith("0x0604")


def numa_of(bdf):
    return parse_int(rd(f"{SYSFS}/{bdf}/numa_node"), -1)


def lspci_names():
    names = {}
    for line in run(["lspci", "-D", "-nn"]).splitlines():
        bdf, _, rest = line.partition(" ")
        rest = re.sub(r"\s*\(rev [0-9a-f]+\)", "", rest)
        names[bdf] = rest.partition(": ")[2] or rest   # strip the leading class description
    return names


def dev_name(bdf, names, limit=60):
    s = names.get(bdf) or \
        f"[{rd(f'{SYSFS}/{bdf}/vendor')[2:]}:{rd(f'{SYSFS}/{bdf}/device')[2:]}]"  # no lspci
    return s if len(s) <= limit else s[:limit - 3] + "..."


def short_name(s, limit=40):
    """Strip a trailing [vendor:device], then truncate"""
    s = re.sub(r"\s*\[[0-9a-f]{4}:[0-9a-f]{4}\]$", "", s) or s
    return s if len(s) <= limit else s[:limit - 3] + "..."


def largest_bar(bdf):
    """Largest of the six standard BARs (BAR1 on GPUs), in bytes"""
    best = 0
    for i, line in enumerate(rd(f"{SYSFS}/{bdf}/resource").splitlines()):
        if i >= 6:
            break
        try:
            start, end = (int(x, 16) for x in line.split()[:2])
        except ValueError:
            continue
        if end > start:
            best = max(best, end - start + 1)
    return best


def acs_ctl(bdf):
    """ACS Control register; None = unreadable (needs root), -1 = device has no ACS capability"""
    try:
        with open(f"{SYSFS}/{bdf}/config", "rb") as f:
            cfg = f.read()
    except OSError:
        return None
    if len(cfg) < 0x104:
        return None
    off = 0x100
    for _ in range(64):
        hdr = int.from_bytes(cfg[off:off + 4], "little")
        if hdr == 0:
            break
        if hdr & 0xffff == 0x000d:
            return int.from_bytes(cfg[off + 6:off + 8], "little") if off + 8 <= len(cfg) else None
        off = (hdr >> 20) & 0xffc
        if off < 0x100:
            break
    return -1


def fmt_size(nbytes):
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.0f} GiB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MiB"
    return f"{nbytes} B"


# ----------------------------------------------------------------- device discovery
@dataclass
class Dev:
    bdf: str
    kind: str                     # gpu / nic / qat / dsa / iaa
    idx: int = -1
    name: str = ""
    chain: list = field(default_factory=list)
    numa: int = -1
    cpus: str = ""
    link: Link = None
    info: dict = field(default_factory=dict)

    @property
    def tag(self):
        return self.info.get("tag") or f"{self.kind.upper()}{self.idx}"

    @property
    def ondie(self):
        """Root Complex integrated endpoint (no Root Port above it): no PCIe link or switch"""
        return len(self.chain) <= 1


def nic_state(bdf):
    """Netdev / RDMA port state summary, e.g. 'ens1f0 up 100G; mlx5_0/p1 ACTIVE 400G NDR'"""
    parts = []
    for ifn in sorted(os.listdir(f"{SYSFS}/{bdf}/net")) if os.path.isdir(f"{SYSFS}/{bdf}/net") else []:
        state = rd(f"/sys/class/net/{ifn}/operstate", "?")
        mbps = parse_int(rd(f"/sys/class/net/{ifn}/speed"), -1)
        spd = f" {mbps // 1000}G" if mbps >= 1000 else (f" {mbps}M" if mbps > 0 else "")
        parts.append(f"{ifn} {state}{spd}")
    ibdir = f"{SYSFS}/{bdf}/infiniband"
    for ib in sorted(os.listdir(ibdir)) if os.path.isdir(ibdir) else []:
        pdir = f"{ibdir}/{ib}/ports"
        for port in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
            state = rd(f"{pdir}/{port}/state", "?").split(":")[-1].strip()
            rate = rd(f"{pdir}/{port}/rate", "")
            m = re.match(r"(\d+)\s*Gb/sec\s*\((.*)\)", rate)
            rate = f" {m.group(1)}G {m.group(2).split()[-1]}" if m else ""
            parts.append(f"{ib}/p{port} {state}{rate}")
    return "; ".join(parts) or "-"


def idxd_devices():
    """{PCI BDF: (idxd name, state)} from /sys/bus/dsa, e.g. 'dsa0', 'enabled, WQ 4/16 enabled (4 user), 4 engines'"""
    out = {}
    for n in sorted(os.listdir(IDXD_BUS)) if os.path.isdir(IDXD_BUS) else []:
        if not re.match(r"(dsa|iax)\d+$", n):
            continue
        d = f"{IDXD_BUS}/{n}"
        bdf = os.path.basename(os.path.realpath(d + "/.."))
        wqs = [x for x in os.listdir(d) if re.match(r"wq\d+\.\d+$", x)]
        enabled = [w for w in wqs if rd(f"{d}/{w}/state") == "enabled"]
        user = sum(rd(f"{d}/{w}/type") == "user" for w in enabled)
        engines = sum(x.startswith("engine") for x in os.listdir(d))
        out[bdf] = (n, f"{rd(d + '/state', '?')}, WQ {len(enabled)}/{len(wqs)} enabled ({user} user), "
                       f"{engines} engines")
    return out


def qat_state(bdf):
    """'up, sym;dc, VFs 0/16' (qat/state and cfg_services exist only with the in-tree driver)"""
    d = f"{SYSFS}/{bdf}"
    parts = [s for s in (rd(d + "/qat/state"), rd(d + "/qat/cfg_services")) if s]
    total_vfs = parse_int(rd(d + "/sriov_totalvfs"))
    if total_vfs:
        parts.append(f"VFs {parse_int(rd(d + '/sriov_numvfs'))}/{total_vfs}")
    return ", ".join(parts) or "-"


def discover(names):
    gpus, nics, accels = [], [], []
    idxd = idxd_devices()
    for bdf in sorted(os.listdir(SYSFS)):
        d = f"{SYSFS}/{bdf}"
        cls, vendor, device = rd(d + "/class"), rd(d + "/vendor").lower(), rd(d + "/device").lower()
        driver = os.path.basename(os.readlink(d + "/driver")) if os.path.islink(d + "/driver") else "-"
        idxd_name = idxd.get(bdf, ("",))[0]
        if cls.startswith("0x03") and vendor in GPU_VENDORS:
            kind = "gpu"
        elif cls.startswith("0x02") or cls.startswith("0x0c06"):
            kind = "nic"
        elif cls.startswith("0x0b40") and vendor == INTEL and not os.path.islink(d + "/physfn") \
                and (device in QAT_IDS or QAT_DRV_RE.search(driver)):
            kind = "qat"
        elif cls.startswith("0x0880") and vendor == INTEL and (idxd_name or device in DSA_IDS | IAA_IDS):
            kind = "dsa" if idxd_name.startswith("dsa") or (not idxd_name and device in DSA_IDS) else "iaa"
        else:
            continue
        dev = Dev(bdf, kind, name=dev_name(bdf, names), chain=chain_of(bdf), numa=numa_of(bdf),
                  cpus=rd(d + "/local_cpulist"), link=link_of(bdf))
        dev.info["ids"] = f"{vendor[2:]}:{device[2:]}"
        dev.info["driver"] = driver
        if kind == "gpu":
            dev.info["bar1"] = largest_bar(bdf)
            gpus.append(dev)
        elif kind == "nic":
            dev.info["mlx"] = vendor == MLX_VENDOR
            dev.info["state"] = nic_state(bdf)
            nics.append(dev)
        else:
            family = f" {driver}" if kind == "qat" and driver != "-" else ""
            dev.name = f"Intel {kind.upper()}{family} [{dev.info['ids']}]"   # lspci has no name for these
            if kind == "qat":
                dev.info["state"] = qat_state(bdf)
            else:
                dev.info["tag"], dev.info["state"] = idxd.get(bdf, ("", "idxd driver not bound"))
            accels.append(dev)
    for i, g in enumerate(gpus):
        g.idx = i
    for i, n in enumerate(nics):
        n.idx = i
    accels.sort(key=lambda a: (ACCEL_KINDS.index(a.kind), a.bdf))
    for kind in ACCEL_KINDS:
        for i, a in enumerate(a for a in accels if a.kind == kind):
            a.idx = i
    return gpus, nics, accels


def enrich_nvidia_smi(gpus):
    """Fill in GPU index / name / memory from nvidia-smi (index matches nvidia-smi)"""
    out = run(["nvidia-smi", "--query-gpu=index,pci.bus_id,name,memory.total",
               "--format=csv,noheader"])
    info = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        dom, bus, rest = parts[1].split(":")
        info[f"{int(dom, 16):04x}:{bus.lower()}:{rest.lower()}"] = parts
    if not info or not all(g.bdf in info for g in gpus):
        return
    for g in gpus:
        idx, _, name, mem = info[g.bdf]
        g.idx, g.name = int(idx), f"{name} [{g.info['ids']}]"
        m = re.match(r"(\d+)\s*MiB", mem)
        g.info["mem"] = f"{int(m.group(1)) / 1024:.0f} GiB" if m else mem


# ----------------------------------------------------------------- topology analysis
class Topo:
    def __init__(self, gpus, nics, accels, names):
        self.gpus, self.nics, self.accels, self.names = gpus, nics, accels, names
        self.ondie = [a for a in accels if a.ondie]
        self.eps = gpus + nics + [a for a in accels if not a.ondie]   # add-in cards join the PCIe tree
        self.by_bdf = {d.bdf: d for d in self.eps + self.ondie}
        self._role = {}
        roots = {d.chain[0] for d in self.eps if d.chain}
        self.roots = sorted(roots, key=lambda r: (numa_of(r), r))
        self.acs_readable = acs_ctl(self.roots[0]) is not None if self.roots else False

    def role(self, bdf):
        """RootPort / SW-Up / SW-Down / EP"""
        if bdf not in self._role:
            chain = chain_of(bdf)
            if len(chain) <= 1:
                r = "RootPort"
            elif not is_bridge(bdf):
                r = "EP"
            else:
                r = "SW-Down" if self.role(chain[-2]) == "SW-Up" else "SW-Up"
            self._role[bdf] = r
        return self._role[bdf]

    def switch_of(self, dev):
        """Nearest upstream switch (upstream-port BDF); None when attached directly to a Root Port"""
        return next((b for b in reversed(dev.chain[:-1]) if self.role(b) == "SW-Up"), None)

    def eps_under(self, bdf):
        return [d for d in self.eps if bdf in d.chain[:-1]]

    def down_bw(self, eps):
        """Sum of max link bandwidth, de-duplicating multi-function devices"""
        seen = {}
        for d in eps:
            seen.setdefault(d.bdf[:-2], d.link.max_bw)
        return sum(seen.values())

    def acs_ports(self, dev):
        """Downstream ports on dev's path that have P2P redirect enabled"""
        out = []
        for b in dev.chain[:-1]:
            if self.role(b) == "SW-Down":
                ctl = acs_ctl(b)
                if ctl is not None and ctl > 0 and ctl & ACS_REDIRECT_MASK:
                    out.append(b)
        return out

    def affinity(self, a, b):
        """Return (class, reason); classes as in nvidia-smi topo -m"""
        if a.bdf == b.bdf:
            return "X", ""
        if a.numa != b.numa and a.numa >= 0 and b.numa >= 0:
            return "SYS", f"NUMA {a.numa} ↔ NUMA {b.numa}"
        k = 0
        while k < min(len(a.chain), len(b.chain)) and a.chain[k] == b.chain[k]:
            k += 1
        if k == 0:
            ha, hb = host_bridge_of(a.bdf), host_bridge_of(b.bdf)
            if ha == hb:
                return "PHB", f"via {ha}"
            return "NODE", f"{ha} ↔ {hb}"
        anc = a.chain[k - 1]
        if self.role(anc) == "RootPort":
            return "PHB", f"via RootPort {anc}"
        hops = (self.role(anc) == "SW-Up") + \
            sum(self.role(x) == "SW-Up" for x in a.chain[k:-1] + b.chain[k:-1])
        if hops <= 1:
            return "PIX", f"Switch {anc}"
        return "PXB", f"{hops} switch levels, joined at {anc}"

    def groups(self, devs):
        """Group by nearest switch (or Root Port): [(key, [dev...])]"""
        out = {}
        for d in devs:
            out.setdefault(self.switch_of(d) or d.chain[0], []).append(d)
        return sorted(out.items(), key=lambda kv: (kv[1][0].numa, kv[0]))

    def location(self, dev):
        """'on-die' / 'SW 36:00.0' / 'RP d5:02.0'"""
        if dev.ondie:
            return "on-die"
        sw = self.switch_of(dev)
        return f"SW {sw[5:]}" if sw else f"RP {dev.chain[0][5:]}"


def accel_summary(accels):
    """'QAT×4 DSA×4 IAA×4' or '-'"""
    counts = [(k, sum(a.kind == k for a in accels)) for k in ACCEL_KINDS]
    return " ".join(f"{k.upper()}×{c}" for k, c in counts if c) or "-"


def system_info(topo):
    cpu, sockets = "", set()
    for line in rd("/proc/cpuinfo").splitlines():
        if line.startswith("model name") and not cpu:
            cpu = line.split(":", 1)[1].strip()
        elif line.startswith("physical id"):
            sockets.add(line.split(":", 1)[1].strip())
    nodes = []
    base = "/sys/devices/system/node"
    for n in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if not re.match(r"node\d+$", n):
            continue
        nid = int(n[4:])
        mem_kb = next((parse_int(l.split()[-2]) for l in rd(f"{base}/{n}/meminfo").splitlines()
                       if "MemTotal" in l), 0)
        nodes.append(dict(id=nid, cpus=rd(f"{base}/{n}/cpulist"), mem_gb=mem_kb / 1048576,
                          gpus=[g for g in topo.gpus if g.numa == nid],
                          nics=[x for x in topo.nics if x.numa == nid],
                          accels=[a for a in topo.accels if a.numa == nid]))
    groups = "/sys/kernel/iommu_groups"
    tokens = [t for t in rd("/proc/cmdline").split() if "iommu" in t or "acs" in t.lower()]
    return dict(host=socket.gethostname(), kernel=os.uname().release, cpu=cpu or "?",
                sockets=len(sockets) or 1, nodes=nodes,
                iommu_groups=len(os.listdir(groups)) if os.path.isdir(groups) else 0,
                cmdline=" ".join(tokens) or "-",
                unplaced=[d for d in topo.eps + topo.ondie if d.numa < 0])


# ----------------------------------------------------------------- text report
def dw(s):
    """Terminal display width (CJK characters take two columns)"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def table(headers, rows):
    rows = [[str(x) for x in r] for r in rows]
    widths = [max(dw(x) for x in col) for col in zip(headers, *rows)]

    def line(cells):
        return " " + "  ".join(c + " " * (w - dw(c)) for c, w in zip(cells, widths))

    return [line(headers), " " + "  ".join("-" * w for w in widths)] + [line(r) for r in rows]


def ep_label(topo, bdf):
    d = topo.by_bdf.get(bdf)
    if d is None:
        return f"other {bdf}  {short_name(dev_name(bdf, topo.names))}"
    s = f"{d.tag} {bdf}  {short_name(d.name)}"
    if d.kind != "gpu":
        s += f"  [{d.info['state']}]"
    return s


def ondie_lines(accels, last):
    """One tree line per accelerator kind: '├─ on-die QAT ×4: QAT0 01:00.0, QAT1 06:00.0, ...'"""
    kinds = [(k, [a for a in accels if a.kind == k]) for k in ACCEL_KINDS]
    kinds = [(k, devs) for k, devs in kinds if devs]
    out = []
    for i, (k, devs) in enumerate(kinds):
        end = last and i == len(kinds) - 1
        t = f"on-die {k.upper()} ×{len(devs)}: " + ", ".join(f"{d.tag} {d.bdf[5:]}" for d in devs)
        out.extend(textwrap.wrap(t, 100, initial_indent="└─ " if end else "├─ ",
                                 subsequent_indent="   " if end else "│  "))
    return out


def acs_text(topo, port):
    if not topo.acs_readable:
        return ""
    ctl = acs_ctl(port)
    if ctl is None or ctl < 0:
        return ""
    if ctl & ACS_REDIRECT_MASK:
        return "  [ACS ⚠P2P-redirect]"
    return "  [ACS on]" if ctl else "  [ACS off]"


def tree_lines(topo, root, last):
    lines = []

    def rec(bdf, prefix, last):
        role = topo.role(bdf)
        conn = "└─ " if last else "├─ "
        cp = prefix + ("   " if last else "│  ")
        kids = children_of(bdf)
        if role == "RootPort":
            lines.append(f"{prefix}{conn}RootPort {bdf}  [{host_bridge_of(bdf)}]")
            for i, k in enumerate(kids):
                rec(k, cp, i == len(kids) - 1)
        elif role == "SW-Up":
            lk = link_of(bdf)
            lines.append(f"{prefix}{conn}PCIe Switch {bdf}  "
                         f"{short_name(dev_name(bdf, topo.names), 44)}  ── uplink {lk.text()}")
            eps = topo.eps_under(bdf)
            if eps and lk.max_bw:
                down = topo.down_bw(eps)
                ratio = down / lk.max_bw
                note = "  (shared under concurrent traffic)" if ratio > 1.05 else ""
                lines.append(f"{cp}│  downstream {', '.join(d.tag for d in eps)} total {down:.1f} GB/s"
                             f" → uplink oversubscribed {ratio:.1f}:1{note}")
            used, empty, others = [], [], []
            for p in kids:
                pk = children_of(p)
                if not pk:
                    empty.append(p)
                elif any(c in topo.by_bdf or is_bridge(c) for c in pk):
                    used.append(p)
                else:
                    others.extend(pk)
            tail = []
            if others:
                tail.append("other ×%d: %s" % (len(others), ", ".join(
                    f"{o[5:]} ({short_name(dev_name(o, topo.names), 36)})" for o in others)))
            if empty:
                tail.append(f"empty ports ×{len(empty)}: {', '.join(e[5:] for e in empty)}")
            for i, p in enumerate(used):
                rec(p, cp, i == len(used) - 1 and not tail)
            for i, t in enumerate(tail):
                end = i == len(tail) - 1
                lines.extend(textwrap.wrap(t, 100, initial_indent=cp + ("└─ " if end else "├─ "),
                                           subsequent_indent=cp + ("   " if end else "│  ")))
        elif role == "SW-Down":
            if len(kids) == 1 and topo.role(kids[0]) == "EP":
                lines.append(f"{prefix}{conn}port {bdf[5:]}{acs_text(topo, bdf)} ── "
                             f"{link_of(kids[0]).text()} ── {ep_label(topo, kids[0])}")
            else:
                lines.append(f"{prefix}{conn}port {bdf[5:]}{acs_text(topo, bdf)}")
                for i, k in enumerate(kids):
                    rec(k, cp, i == len(kids) - 1)
        else:
            lines.append(f"{prefix}{conn}{ep_label(topo, bdf)} ── {link_of(bdf).text()}")

    rec(root, "", last)
    return lines


def matrix_lines(rows, cols, topo):
    w = max(6, max(len(c.tag) for c in cols) + 2)
    out = [" " * 8 + "".join(f"{c.tag:>{w}}" for c in cols)]
    for r in rows:
        out.append(f" {r.tag:<7}" + "".join(f"{topo.affinity(r, c)[0]:>{w}}" for c in cols))
    return out


def text_report(topo, sysinfo):
    L = []
    bar = "=" * 96

    def section(t):
        L.extend(["", bar, f" {t}", bar])

    section("System overview")
    L.append(f" Host {sysinfo['host']}   Kernel {sysinfo['kernel']}   "
             f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    L.append(f" CPU  {sysinfo['cpu']}  × {sysinfo['sockets']} socket, "
             f"{len(sysinfo['nodes'])} NUMA nodes")
    for n in sysinfo["nodes"]:
        L.append(f" NUMA {n['id']}: CPU {n['cpus']:<20} Memory {n['mem_gb']:.0f} GB   "
                 f"GPU: {', '.join(g.tag for g in n['gpus']) or '-'}   "
                 f"NIC: {', '.join(x.tag for x in n['nics']) or '-'}   "
                 f"Accel: {accel_summary(n['accels'])}")
    if sysinfo["unplaced"]:
        L.append(f" NUMA unknown: {', '.join(d.tag for d in sysinfo['unplaced'])}")
    iommu = f"on ({sysinfo['iommu_groups']} groups)" if sysinfo["iommu_groups"] else "off"
    acs = "readable" if topo.acs_readable else "needs root to read"
    L.append(f" IOMMU {iommu}   Kernel args [{sysinfo['cmdline']}]   ACS config {acs}")

    section("GPUs (index matches nvidia-smi; Link = current/max; see the HTML page for full details)")
    rows = []
    for g in topo.gpus:
        sw = topo.switch_of(g)
        rows.append([g.tag, g.bdf, g.name[:34], g.numa, g.cpus,
                     f"{g.link.cur()}/{g.link.mx()}", fmt_size(g.info["bar1"]),
                     g.info.get("mem", "-"), f"SW {sw[5:]}" if sw else f"RP {g.chain[0][5:]}"])
    L += table(["GPU", "PCI addr", "Name [vendor:dev]", "NUMA", "Local CPUs", "Link", "BAR1",
                "Memory", "Upstream"], rows) if rows else [" (no GPU found)"]

    section("NICs")
    rows = []
    for x in topo.nics:
        sw = topo.switch_of(x)
        rows.append([x.tag + ("*" if x.info["mlx"] else ""), x.bdf, x.name[:40],
                     x.info["state"], x.numa, f"{x.link.cur()}/{x.link.mx()}",
                     f"SW {sw[5:]}" if sw else f"RP {x.chain[0][5:]}"])
    L += table(["NIC", "PCI addr", "Name [vendor:dev]", "Netdev/RDMA state", "NUMA", "Link", "Upstream"],
               rows) if rows else [" (no NIC found)"]
    if any(x.info["mlx"] for x in topo.nics):
        L.append(" * = Mellanox/NVIDIA NIC (supports GPUDirect RDMA)")

    section("Accelerators (Intel QAT / DSA / IAA; on-die = Root Complex integrated, no PCIe link; "
            "DSA/IAA tags = idxd names)")
    rows = [[a.tag, a.bdf, a.name, a.numa, a.info["driver"], topo.location(a), a.info["state"]]
            for a in topo.accels]
    L += table(["Accel", "PCI addr", "Name [vendor:dev]", "NUMA", "Driver", "Location", "State"],
               rows) if rows else [" (no QAT / DSA / IAA found)"]

    section("Topology tree (NUMA → Root Port → PCIe Switch → device; bandwidth = theoretical per direction)")
    for n in sysinfo["nodes"] + [dict(id=-1, cpus="?")]:
        roots = [r for r in topo.roots if numa_of(r) == n["id"]]
        ondie = [a for a in topo.ondie if a.numa == n["id"]]
        if not roots and not ondie:
            continue
        L.append(f" NUMA {n['id'] if n['id'] >= 0 else 'unknown'}  CPU {n['cpus']}")
        L += [" " + s for s in ondie_lines(ondie, last=not roots)]
        for i, r in enumerate(roots):
            L += [" " + s for s in tree_lines(topo, r, i == len(roots) - 1)]
        L.append("")

    section("GPU ↔ GPU affinity matrix (same as nvidia-smi topo -m)")
    if len(topo.gpus) > 1:
        L += matrix_lines(topo.gpus, topo.gpus, topo)
    else:
        L.append(" (fewer than 2 GPUs)")
    L.append("")
    for k, v in AFF_DESC.items():
        L.append(f"   {k:<5} {v}")

    if topo.gpus and topo.nics:
        section("GPU ↔ NIC affinity matrix (prefer PIX/PXB for GPUDirect RDMA)")
        L += matrix_lines(topo.gpus, topo.nics, topo)
        L.append("")
        for x in topo.nics:
            order = list(AFF_DESC)
            cls = min((topo.affinity(g, x)[0] for g in topo.gpus), key=order.index)
            same = [g.tag for g in topo.gpus if topo.affinity(g, x)[0] == cls]
            L.append(f"   {x.tag} ({x.name[:32]}): nearest GPU {', '.join(same)} [{cls}]")

    section("Bandwidth sharing and P2P path summary")
    L.append(" [Shared uplink] devices under the same switch share one link to the CPU:")
    for key, members in topo.groups(topo.eps):
        if topo.role(key) != "SW-Up":
            L.append(f"   {', '.join(d.tag for d in members)}: directly on RootPort {key}, dedicated uplink")
            continue
        up = link_of(key)
        down = topo.down_bw(members)
        ratio = down / up.max_bw if up.max_bw else 0
        flag = "  ⚠ oversubscribed" if ratio > 1.05 else ""
        L.append(f"   Switch {key} (NUMA {numa_of(key)}): {', '.join(d.tag for d in members)}"
                 f"  uplink {up.mx()} {up.max_bw:.1f} GB/s, downstream {down:.1f} GB/s"
                 f" → {ratio:.1f}:1{flag}")
    if len(topo.gpus) > 1:
        L.append(" [GPU P2P]")
        ggroups = topo.groups(topo.gpus)
        names = {key: chr(ord("A") + i) for i, (key, _) in enumerate(ggroups)}
        for key, members in ggroups:
            where = f"Switch {key}" if topo.role(key) == "SW-Up" else f"RootPort {key}"
            L.append(f"   Group {names[key]} = {{{', '.join(g.tag for g in members)}}}"
                     f"  @ {where}, NUMA {members[0].numa}")
        L.append("   Intra-group P2P stays inside the switch (PIX): no uplink or CPU involved;")
        L.append("   same NUMA, different group (NODE/PHB): forwarded by the CPU root complex, bounded by uplink and CPU;")
        L.append("   cross-NUMA (SYS): also crosses UPI, lowest bandwidth and highest latency.")
    redirect = {p for g in topo.gpus for p in topo.acs_ports(g)}
    if redirect:
        bits = {n for p in redirect for i, n in enumerate(ACS_BITS) if acs_ctl(p) >> i & 1}
        L.append(f" [ACS] ⚠ P2P redirect ({'+'.join(n for n in ACS_BITS if n in bits)}) enabled on downstream ports:")
        L.append(f"       {', '.join(sorted(p[5:] for p in redirect))}")
        L.append("       same-switch GPU P2P is routed through the CPU/IOMMU; disable with tools/disable-acs.sh")
    elif topo.acs_readable:
        L.append(" [ACS] no P2P redirect on GPU downstream ports; same-switch P2P goes direct")
    else:
        L.append(" [ACS] run as root to check whether downstream-port ACS redirects P2P to the CPU (affects same-switch P2P)")
    idle = [d.tag for d in topo.eps if d.link.downclocked]
    if idle:
        L.append(f" [Link] idle-downclocked (speed below max, full width): {', '.join(idle)}"
                 " (normal for idle GPUs saving power; re-check under load)")
    degraded = [d.tag for d in topo.eps if d.link.degraded]
    if degraded:
        L.append(f" [Link] ⚠ degraded (width below max): {', '.join(degraded)}"
                 " (check slot / riser / BIOS PCIe settings)")
    if topo.ondie:
        L.append(" [Accel] on-die QAT / DSA / IAA have no PCIe link; use the ones on the GPU's NUMA node"
                 " (tools/auto_config.sh does this):")
        for n in sysinfo["nodes"]:
            acc = [a for a in n["accels"] if a.ondie]
            if acc:
                by_kind = "; ".join(", ".join(a.tag for a in acc if a.kind == k)
                                    for k in ACCEL_KINDS if any(a.kind == k for a in acc))
                L.append(f"   NUMA {n['id']} (GPU: {', '.join(g.tag for g in n['gpus']) or '-'}): {by_kind}")
    return L


# ----------------------------------------------------------------- HTML report
CSS = """
body{font-family:system-ui,"Segoe UI",Helvetica,Arial,sans-serif;margin:20px;color:#222;background:#fafafa}
h1{font-size:20px;margin-bottom:4px} .sub{color:#666;font-size:13px}
h2{font-size:16px;margin-top:30px;border-bottom:2px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;font-size:13px;margin:8px 0;background:#fff}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;white-space:nowrap} th{background:#eee}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.numa{border:2px solid #607d8b;border-radius:8px;margin:14px 0;padding:10px;background:#fff}
.numa-hdr{font-weight:600;color:#37474f;margin-bottom:8px}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.rp{border:1px dashed #90a4ae;border-radius:6px;padding:8px;background:#f5f7f8}
.hdr{font-size:12px;font-weight:600;margin-bottom:4px} .hdr small{font-weight:400;color:#666}
.link{font-size:11px;color:#555;padding-left:8px;border-left:3px solid #999;margin:2px 0 4px 10px}
.link.warn{color:#c62828;border-left-color:#c62828}
.link.idle{color:#ef6c00;border-left-color:#ef6c00}
.sw{border:2px solid #f9a825;border-radius:6px;padding:8px;background:#fff8e1}
.over{font-size:11px;color:#444;margin-bottom:6px} .over.warn{color:#c62828;font-weight:600}
.ports{display:flex;flex-wrap:wrap;gap:8px}
.port{border:1px solid #ddd;border-radius:4px;padding:6px;background:#fff;min-width:150px}
.port-hdr{font-size:11px;color:#777}
.ep{border-radius:4px;padding:6px 8px;font-size:12px;margin-top:4px;line-height:1.35}
.ep b{font-size:13px}
.gpu{background:#c8e6c9;border:1px solid #2e7d32} .nic{background:#bbdefb;border:1px solid #1565c0}
.mlx{background:#90caf9} .other{background:#eeeeee;border:1px solid #9e9e9e;color:#555}
.qat{background:#e1bee7;border:1px solid #6a1b9a} .dsa{background:#ffe0b2;border:1px solid #e65100}
.iaa{background:#f8bbd0;border:1px solid #ad1457}
.ondie{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:10px} .ondie .ep{margin-top:0}
.badge{display:inline-block;padding:0 6px;border-radius:10px;font-size:11px;color:#fff;background:#c62828;margin-left:4px}
td.m{text-align:center;font-weight:600}
.m-X{background:#e0e0e0} .m-PIX{background:#a5d6a7} .m-PXB{background:#c5e1a5}
.m-PHB{background:#fff59d} .m-NODE{background:#ffcc80} .m-SYS{background:#ef9a9a}
.legend span{display:inline-block;padding:2px 8px;margin:2px 6px 2px 0;border-radius:3px;font-size:12px}
pre{background:#f4f4f4;padding:10px;font-size:12px;overflow:auto;line-height:1.3}
ul{font-size:13px}
"""


def h(s):
    return html.escape(str(s))


def html_link(lk):
    cls = "link warn" if lk.degraded else "link idle" if lk.downclocked else "link"
    return f'<div class="{cls}">{h(lk.text())}</div>'


def html_ep(topo, bdf):
    d = topo.by_bdf.get(bdf)
    if d is None:
        return f'<div class="ep other">{h(bdf)}<br>{h(dev_name(bdf, topo.names, 40))}</div>'
    cls = "ep " + d.kind + (" mlx" if d.info.get("mlx") else "")
    body = f"<b>{h(d.tag)}</b> <span class='mono'>{h(d.bdf)}</span><br>{h(d.name)}"
    if d.kind == "gpu":
        body += f"<br>BAR1 {h(fmt_size(d.info['bar1']))}" + \
            (f" · Mem {h(d.info['mem'])}" if d.info.get("mem") else "")
    else:
        body += f"<br>{h(d.info['state'])}"
    acs = topo.acs_ports(d)
    if acs:
        body += '<span class="badge">ACS redirect</span>'
    return f'<div class="{cls}">{body}</div>'


def html_node(topo, bdf):
    role = topo.role(bdf)
    kids = children_of(bdf)
    if role == "RootPort":
        inner = "".join(html_node(topo, k) for k in kids)
        return (f'<div class="rp"><div class="hdr">Root Port <span class="mono">{h(bdf)}</span>'
                f' <small>{h(host_bridge_of(bdf))}</small></div>{inner}</div>')
    if role == "SW-Up":
        lk = link_of(bdf)
        eps = topo.eps_under(bdf)
        over = ""
        if eps and lk.max_bw:
            down = topo.down_bw(eps)
            ratio = down / lk.max_bw
            over = (f'<div class="over{" warn" if ratio > 1.05 else ""}">uplink {h(lk.mx())} '
                    f'{lk.max_bw:.1f} GB/s · downstream {h(", ".join(d.tag for d in eps))} total '
                    f'{down:.1f} GB/s → oversubscription {ratio:.1f}:1</div>')
        ports, empty, others = "", [], []
        for p in kids:
            pk = children_of(p)
            if not pk:
                empty.append(p)
            elif any(c in topo.by_bdf or is_bridge(c) for c in pk):
                inner = "".join(html_node(topo, c) if is_bridge(c) else html_link(link_of(c)) +
                                html_ep(topo, c) for c in pk)
                ports += (f'<div class="port"><div class="port-hdr">port {h(p[5:])}'
                          f'{h(acs_text(topo, p))}</div>{inner}</div>')
            else:
                others.extend(pk)
        if others:
            ports += '<div class="port"><div class="port-hdr">Other devices</div>' + "".join(
                html_ep(topo, o) for o in others) + "</div>"
        if empty:
            ports += (f'<div class="port"><div class="port-hdr">Empty ports ×{len(empty)}</div>'
                      f'<div class="ep other">{h(", ".join(e[5:] for e in empty))}</div></div>')
        return (f'{html_link(lk)}<div class="sw"><div class="hdr">PCIe Switch '
                f'<span class="mono">{h(bdf)}</span> <small>{h(dev_name(bdf, topo.names))}'
                f'</small></div>{over}<div class="ports">{ports}</div></div>')
    if role == "SW-Down":
        inner = "".join(html_node(topo, k) for k in kids)
        return (f'<div class="port"><div class="port-hdr">port {h(bdf[5:])}'
                f'{h(acs_text(topo, bdf))}</div>{inner}</div>')
    return html_link(link_of(bdf)) + html_ep(topo, bdf)


def html_table(headers, rows):
    s = "<table><tr>" + "".join(f"<th>{h(x)}</th>" for x in headers) + "</tr>"
    for r in rows:
        s += "<tr>" + "".join(f"<td>{h(x)}</td>" for x in r) + "</tr>"
    return s + "</table>"


def html_matrix(rows, cols, topo):
    s = "<table><tr><th></th>" + "".join(f"<th>{h(c.tag)}</th>" for c in cols) + "</tr>"
    for r in rows:
        s += f"<tr><th>{h(r.tag)}</th>"
        for c in cols:
            cls, why = topo.affinity(r, c)
            s += f'<td class="m m-{cls}" title="{h(why)}">{cls}</td>'
        s += "</tr>"
    return s + "</table>"


def html_report(topo, sysinfo, text):
    P = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Topology {h(sysinfo['host'])}"
         f"</title><style>{CSS}</style></head><body>"]
    P.append(f"<h1>CPU / GPU / NIC / PCIe topology — {h(sysinfo['host'])}</h1>")
    P.append(f"<div class='sub'>{h(sysinfo['cpu'])} × {sysinfo['sockets']} socket · "
             f"Kernel {h(sysinfo['kernel'])} · {datetime.datetime.now():%Y-%m-%d %H:%M} · "
             f"IOMMU {'on (%d groups)' % sysinfo['iommu_groups'] if sysinfo['iommu_groups'] else 'off'}"
             f" · Kernel args [{h(sysinfo['cmdline'])}]</div>")

    P.append("<h2>Topology</h2><div class='legend'><span class='gpu'>GPU</span>"
             "<span class='nic'>NIC</span><span class='nic mlx'>Mellanox NIC</span>"
             "<span class='qat'>QAT</span><span class='dsa'>DSA</span><span class='iaa'>IAA</span>"
             "<span class='sw'>PCIe Switch</span><span class='rp'>Root Port</span>"
             "<span class='other'>Other</span> &nbsp; link labels are theoretical per-direction "
             "bandwidth; orange = idle-downclocked, red = degraded (width below max)</div>")
    for n in sysinfo["nodes"] + [dict(id=-1)]:
        roots = [r for r in topo.roots if numa_of(r) == n["id"]]
        ondie = [a for a in topo.ondie if a.numa == n["id"]]
        if not roots and not ondie:
            continue
        hdr = (f"NUMA {n['id']} · CPU {h(n['cpus'])} · Memory {n['mem_gb']:.0f} GB" if n["id"] >= 0
               else "NUMA unknown")
        P.append(f"<div class='numa'><div class='numa-hdr'>{hdr}</div>")
        if ondie:
            P.append("<div class='ondie'><span class='hdr'>On-die accelerators (no PCIe link)</span>" +
                     "".join(html_ep(topo, a.bdf) for a in ondie) + "</div>")
        P.append("<div class='row'>")
        P += [html_node(topo, r) for r in roots]
        P.append("</div></div>")

    P.append("<h2>GPUs</h2>")
    rows = []
    for g in topo.gpus:
        sw = topo.switch_of(g)
        rows.append([g.tag, g.bdf, g.name, g.numa, g.cpus,
                     f"{g.link.cur()} / {g.link.mx()}", f"{g.link.max_bw:.1f}",
                     fmt_size(g.info["bar1"]), g.info.get("mem", "-"),
                     sw or "direct", g.chain[0], g.info["driver"]])
    P.append(html_table(["GPU", "PCI addr", "Name [vendor:dev]", "NUMA", "Local CPUs",
                         "Link cur / max", "GB/s", "BAR1", "Memory", "Switch", "RootPort", "Driver"],
                        rows) if rows else "<p>No GPU found</p>")

    P.append("<h2>NICs</h2>")
    rows = []
    for x in topo.nics:
        sw = topo.switch_of(x)
        rows.append([x.tag, x.bdf, x.name + (" (Mellanox)" if x.info["mlx"] else ""),
                     x.info["state"], x.numa, f"{x.link.cur()} / {x.link.mx()}",
                     f"{x.link.max_bw:.1f}", sw or "direct", x.chain[0], x.info["driver"]])
    P.append(html_table(["NIC", "PCI addr", "Name [vendor:dev]", "Netdev/RDMA state", "NUMA",
                         "Link cur / max", "GB/s", "Switch", "RootPort", "Driver"], rows)
             if rows else "<p>No NIC found</p>")

    P.append("<h2>Accelerators (Intel QAT / DSA / IAA)</h2>")
    rows = [[a.tag, a.bdf, a.name, a.numa, a.info["driver"], topo.location(a), a.info["state"]]
            for a in topo.accels]
    P.append(html_table(["Accel", "PCI addr", "Name [vendor:dev]", "NUMA", "Driver", "Location", "State"],
                        rows) if rows else "<p>No QAT / DSA / IAA found</p>")

    P.append("<h2>GPU ↔ GPU affinity matrix</h2>")
    if len(topo.gpus) > 1:
        P.append(html_matrix(topo.gpus, topo.gpus, topo))
    P.append("<div class='legend'>" + "".join(
        f"<span class='m-{k}'>{k}: {h(v)}</span>" for k, v in AFF_DESC.items()) + "</div>")
    if topo.gpus and topo.nics:
        P.append("<h2>GPU ↔ NIC affinity matrix (prefer PIX/PXB for GPUDirect RDMA)</h2>")
        P.append(html_matrix(topo.gpus, topo.nics, topo))

    P.append("<h2>Bandwidth sharing and P2P path summary</h2><ul>")
    start = text.index(next(s for s in text if s.startswith(" [Shared uplink]")))
    for line in text[start:]:
        if line.strip():
            P.append(f"<li{' style=font-weight:600' if line.startswith(' [') else ''}>"
                     f"{h(line.strip())}</li>")
    P.append("</ul>")
    P.append("<details><summary>Full text report</summary><pre>" + h("\n".join(text)) +
             "</pre></details></body></html>")
    return "\n".join(P)


# ----------------------------------------------------------------- http
def host_ip():
    """Outward-facing IP of this host (UDP connect sends nothing); falls back to the hostname"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def serve_http(page, port):
    import http.server
    data = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    try:
        srv = http.server.ThreadingHTTPServer(("", port), Handler)
    except OSError as e:
        sys.exit(f"Cannot listen on port {port}: {e.strerror or e} (try another --port)")
    print(f"\nHTTP server started, open in a browser:  http://{host_ip()}:{port}/"
          f"   (local: http://localhost:{port}/)   Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        srv.server_close()


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="CPU/GPU/NIC/PCIe topology report")
    ap.add_argument("--no-http", action="store_true", help="do not start the HTTP server for the HTML report")
    ap.add_argument("--port", type=int, default=8080, help="HTTP server port (default 8080)")
    ap.add_argument("--quiet", action="store_true", help="do not print the report to the console")
    args = ap.parse_args()

    if not os.path.isdir(SYSFS):
        sys.exit("/sys/bus/pci/devices not found; this tool requires Linux.")
    names = lspci_names()
    gpus, nics, accels = discover(names)
    if not gpus and not nics and not accels:
        sys.exit("No GPU, NIC or accelerator found (sysfs may be incomplete inside a VM/container).")
    enrich_nvidia_smi(gpus)
    topo = Topo(gpus, nics, accels, names)
    sysinfo = system_info(topo)

    text = text_report(topo, sysinfo)
    if not args.quiet:
        print("\n".join(text))
    if not args.no_http:
        serve_http(html_report(topo, sysinfo, text), args.port)


if __name__ == "__main__":
    main()
