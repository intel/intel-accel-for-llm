#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw_topo.py - 打印 GPU / NIC 信息与 CPU NUMA / PCIe 拓扑, 分析带宽共享与 P2P 路径。

只依赖 Linux sysfs(可选 lspci / nvidia-smi 用于显示名称), 普通用户即可运行;
以 root 运行时额外读取 PCIe ACS 配置, 判断同 Switch 下的 P2P 是否被重定向到 CPU。

用法:
  python3 hw_topo.py                        # 控制台输出
  python3 hw_topo.py --html topo.html       # 同时保存为网页
  python3 hw_topo.py --gpu-vendor 0x10de,0x1002

输出内容:
  1. 系统概览: CPU / NUMA / IOMMU / 相关内核参数
  2. GPU 列表: 编号(与 nvidia-smi 一致), PCI 地址, NUMA, 亲和 CPU, 链路, BAR1, 所属 Switch
  3. NIC 列表: PCI 地址, 网口 / RDMA 设备及状态, NUMA, 链路, 所属 Switch
  4. 拓扑树: NUMA → Root Port → PCIe Switch → GPU/NIC, 标注每段链路与上行超额订阅
  5. GPU↔GPU / GPU↔NIC 亲和矩阵 (PIX/PXB/PHB/NODE/SYS, 含义同 nvidia-smi topo -m)

说明:
  * 带宽为每方向理论值(扣除编码开销); 超额订阅按各设备的最大链路能力计算。
  * GPU 空闲时链路会降到 Gen1 省电, 标注为 ⚠降速 不一定是故障, 有负载时复查。
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
MLX_VENDOR = "0x15b3"
ACS_BITS = ["SrcValid", "TransBlk", "P2pReqRedir", "P2pCmpltRedir", "UpstreamFwd",
            "EgressCtrl", "DirectTrans"]
ACS_REDIRECT_MASK = 0b11100  # P2pReqRedir | P2pCmpltRedir | UpstreamFwd
AFF_DESC = {
    "PIX": "同一 PCIe Switch, 不经 CPU",
    "PXB": "多级 PCIe Switch, 不经 CPU",
    "PHB": "同一 Host Bridge, 经 CPU Root Complex",
    "NODE": "同 NUMA 跨 Host Bridge, 经 CPU",
    "SYS": "跨 NUMA, 经 UPI",
}


# ----------------------------------------------------------------- sysfs 工具
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
    """PCIe 单方向理论带宽 GB/s"""
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
        return (self.speed and self.speed < self.max_speed) or \
               (self.width and self.width < self.max_width)

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
        """'Gen5x16 63.0 GB/s' 或 'Gen1x16 (max Gen5x16 63.0 GB/s) ⚠降速'"""
        if self.degraded:
            return f"{self.cur()} (max {self.mx()} {self.max_bw:.1f} GB/s) ⚠降速"
        return f"{self.cur()} {self.max_bw:.1f} GB/s"


def link_of(bdf):
    d = f"{SYSFS}/{bdf}"
    return Link(parse_speed(rd(d + "/current_link_speed")),
                parse_int(rd(d + "/current_link_width")),
                parse_speed(rd(d + "/max_link_speed")),
                parse_int(rd(d + "/max_link_width")))


def chain_of(bdf):
    """从 Root Port 到该设备的 BDF 链"""
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
        names[bdf] = rest.partition(": ")[2] or rest   # 去掉前面的 class 描述
    return names


def dev_name(bdf, names, limit=60):
    s = names.get(bdf) or \
        f"[{rd(f'{SYSFS}/{bdf}/vendor')[2:]}:{rd(f'{SYSFS}/{bdf}/device')[2:]}]"  # 无 lspci 时
    return s if len(s) <= limit else s[:limit - 3] + "..."


def short_name(s, limit=40):
    """去掉末尾的 [vendor:device], 再截断"""
    s = re.sub(r"\s*\[[0-9a-f]{4}:[0-9a-f]{4}\]$", "", s) or s
    return s if len(s) <= limit else s[:limit - 3] + "..."


def largest_bar(bdf):
    """六个标准 BAR 中最大的一个(GPU 即 BAR1), 字节数"""
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
    """ACS Control 寄存器; None=无法读取(需 root), -1=设备没有 ACS 能力"""
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


# ----------------------------------------------------------------- 设备发现
@dataclass
class Dev:
    bdf: str
    kind: str                     # gpu / nic
    idx: int = -1
    name: str = ""
    chain: list = field(default_factory=list)
    numa: int = -1
    cpus: str = ""
    link: Link = None
    info: dict = field(default_factory=dict)

    @property
    def tag(self):
        return f"{self.kind.upper()}{self.idx}"


def nic_state(bdf):
    """网口/RDMA 端口状态摘要, 如 'ens1f0 up 100G; mlx5_0 ACTIVE 400G NDR'"""
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


def discover(gpu_vendors, names):
    gpus, nics = [], []
    for bdf in sorted(os.listdir(SYSFS)):
        d = f"{SYSFS}/{bdf}"
        cls, vendor = rd(d + "/class"), rd(d + "/vendor").lower()
        if cls.startswith("0x03") and vendor in gpu_vendors:
            kind = "gpu"
        elif cls.startswith("0x02") or cls.startswith("0x0c06"):
            kind = "nic"
        else:
            continue
        dev = Dev(bdf, kind, name=dev_name(bdf, names), chain=chain_of(bdf), numa=numa_of(bdf),
                  cpus=rd(d + "/local_cpulist"), link=link_of(bdf))
        dev.info["ids"] = f"{vendor[2:]}:{rd(d + '/device')[2:]}"
        dev.info["driver"] = os.path.basename(os.readlink(d + "/driver")) \
            if os.path.islink(d + "/driver") else "-"
        if kind == "gpu":
            dev.info["bar1"] = largest_bar(bdf)
        else:
            dev.info["mlx"] = vendor == MLX_VENDOR
            dev.info["state"] = nic_state(bdf)
        (gpus if kind == "gpu" else nics).append(dev)
    for i, g in enumerate(gpus):
        g.idx = i
    for i, n in enumerate(nics):
        n.idx = i
    return gpus, nics


def enrich_nvidia_smi(gpus):
    """用 nvidia-smi 的编号/名称/显存补全 GPU 信息(编号与 nvidia-smi 一致)"""
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


# ----------------------------------------------------------------- 拓扑分析
class Topo:
    def __init__(self, gpus, nics, names):
        self.gpus, self.nics, self.names = gpus, nics, names
        self.eps = gpus + nics
        self.by_bdf = {d.bdf: d for d in self.eps}
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
        """最近的上游 Switch(上行端口 BDF), 直连 Root Port 时为 None"""
        return next((b for b in reversed(dev.chain[:-1]) if self.role(b) == "SW-Up"), None)

    def eps_under(self, bdf):
        return [d for d in self.eps if bdf in d.chain[:-1]]

    def down_bw(self, eps):
        """去重多功能设备后的最大链路带宽合计"""
        seen = {}
        for d in eps:
            seen.setdefault(d.bdf[:-2], d.link.max_bw)
        return sum(seen.values())

    def acs_ports(self, dev):
        """dev 路径上开启了 P2P 重定向的下行端口"""
        out = []
        for b in dev.chain[:-1]:
            if self.role(b) == "SW-Down":
                ctl = acs_ctl(b)
                if ctl is not None and ctl > 0 and ctl & ACS_REDIRECT_MASK:
                    out.append(b)
        return out

    def affinity(self, a, b):
        """返回 (类别, 说明), 类别同 nvidia-smi topo -m"""
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
                return "PHB", f"经 {ha}"
            return "NODE", f"{ha} ↔ {hb}"
        anc = a.chain[k - 1]
        if self.role(anc) == "RootPort":
            return "PHB", f"经 RootPort {anc}"
        hops = (self.role(anc) == "SW-Up") + \
            sum(self.role(x) == "SW-Up" for x in a.chain[k:-1] + b.chain[k:-1])
        if hops <= 1:
            return "PIX", f"Switch {anc}"
        return "PXB", f"{hops} 级 Switch, 汇聚于 {anc}"

    def groups(self, devs):
        """按最近的 Switch(或 Root Port) 分组: [(key, [dev...])]"""
        out = {}
        for d in devs:
            out.setdefault(self.switch_of(d) or d.chain[0], []).append(d)
        return sorted(out.items(), key=lambda kv: (kv[1][0].numa, kv[0]))


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
                          nics=[x for x in topo.nics if x.numa == nid]))
    groups = "/sys/kernel/iommu_groups"
    tokens = [t for t in rd("/proc/cmdline").split() if "iommu" in t or "acs" in t.lower()]
    return dict(host=socket.gethostname(), kernel=os.uname().release, cpu=cpu or "?",
                sockets=len(sockets) or 1, nodes=nodes,
                iommu_groups=len(os.listdir(groups)) if os.path.isdir(groups) else 0,
                cmdline=" ".join(tokens) or "-",
                unplaced=[d for d in topo.eps if d.numa < 0])


# ----------------------------------------------------------------- 文本报告
def dw(s):
    """终端显示宽度(中文等占两格)"""
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
        return f"其他 {bdf}  {short_name(dev_name(bdf, topo.names))}"
    s = f"{d.tag} {bdf}  {short_name(d.name)}"
    if d.kind == "nic":
        s += f"  [{d.info['state']}]"
    return s


def acs_text(topo, port):
    if not topo.acs_readable:
        return ""
    ctl = acs_ctl(port)
    if ctl is None or ctl < 0:
        return ""
    if ctl & ACS_REDIRECT_MASK:
        return "  [ACS ⚠P2P重定向]"
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
                         f"{short_name(dev_name(bdf, topo.names), 44)}  ── 上行 {lk.text()}")
            eps = topo.eps_under(bdf)
            if eps and lk.max_bw:
                down = topo.down_bw(eps)
                ratio = down / lk.max_bw
                note = "  (同时传输时各卡分摊上行带宽)" if ratio > 1.05 else ""
                lines.append(f"{cp}│  下行 {', '.join(d.tag for d in eps)} 合计 {down:.1f} GB/s"
                             f" → 上行超额订阅 {ratio:.1f}:1{note}")
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
                tail.append("其他 ×%d: %s" % (len(others), ", ".join(
                    f"{o[5:]} ({short_name(dev_name(o, topo.names), 36)})" for o in others)))
            if empty:
                tail.append(f"空端口 ×{len(empty)}: {', '.join(e[5:] for e in empty)}")
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

    section("系统概览")
    L.append(f" 主机 {sysinfo['host']}   内核 {sysinfo['kernel']}   "
             f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    L.append(f" CPU  {sysinfo['cpu']}  × {sysinfo['sockets']} socket, "
             f"{len(sysinfo['nodes'])} NUMA 节点")
    for n in sysinfo["nodes"]:
        L.append(f" NUMA {n['id']}: CPU {n['cpus']:<20} 内存 {n['mem_gb']:.0f} GB   "
                 f"GPU: {', '.join(g.tag for g in n['gpus']) or '-'}   "
                 f"NIC: {', '.join(x.tag for x in n['nics']) or '-'}")
    if sysinfo["unplaced"]:
        L.append(f" NUMA 未知: {', '.join(d.tag for d in sysinfo['unplaced'])}")
    iommu = f"开启 ({sysinfo['iommu_groups']} groups)" if sysinfo["iommu_groups"] else "关闭"
    acs = "root 可读" if topo.acs_readable else "需 root 才能读取"
    L.append(f" IOMMU {iommu}   内核参数 [{sysinfo['cmdline']}]   ACS 配置 {acs}")

    section("GPU 列表 (编号与 nvidia-smi 一致; 链路 = 当前/最大; 完整信息见 --html)")
    rows = []
    for g in topo.gpus:
        sw = topo.switch_of(g)
        rows.append([g.tag, g.bdf, g.name[:34], g.numa, g.cpus,
                     f"{g.link.cur()}/{g.link.mx()}", fmt_size(g.info["bar1"]),
                     g.info.get("mem", "-"), f"SW {sw[5:]}" if sw else f"RP {g.chain[0][5:]}"])
    L += table(["GPU", "PCI 地址", "名称 [vendor:dev]", "NUMA", "亲和 CPU", "链路", "BAR1",
                "显存", "上游"], rows) if rows else [" (未发现 GPU)"]

    section("NIC 列表")
    rows = []
    for x in topo.nics:
        sw = topo.switch_of(x)
        rows.append([x.tag + ("*" if x.info["mlx"] else ""), x.bdf, x.name[:40],
                     x.info["state"], x.numa, f"{x.link.cur()}/{x.link.mx()}",
                     f"SW {sw[5:]}" if sw else f"RP {x.chain[0][5:]}"])
    L += table(["NIC", "PCI 地址", "名称 [vendor:dev]", "网口/RDMA 状态", "NUMA", "链路", "上游"],
               rows) if rows else [" (未发现 NIC)"]
    if any(x.info["mlx"] for x in topo.nics):
        L.append(" * = Mellanox/NVIDIA 网卡 (支持 GPUDirect RDMA)")

    section("拓扑树 (NUMA → Root Port → PCIe Switch → 设备; 带宽为每方向理论值)")
    for n in sysinfo["nodes"] + [dict(id=-1, cpus="?")]:
        roots = [r for r in topo.roots if numa_of(r) == n["id"]]
        if not roots:
            continue
        L.append(f" NUMA {n['id'] if n['id'] >= 0 else '未知'}  CPU {n['cpus']}")
        for i, r in enumerate(roots):
            L += [" " + s for s in tree_lines(topo, r, i == len(roots) - 1)]
        L.append("")

    section("GPU ↔ GPU 亲和矩阵 (同 nvidia-smi topo -m)")
    if len(topo.gpus) > 1:
        L += matrix_lines(topo.gpus, topo.gpus, topo)
    else:
        L.append(" (GPU 少于 2 个)")
    L.append("")
    for k, v in AFF_DESC.items():
        L.append(f"   {k:<5} {v}")

    if topo.gpus and topo.nics:
        section("GPU ↔ NIC 亲和矩阵 (GPUDirect RDMA 首选 PIX/PXB)")
        L += matrix_lines(topo.gpus, topo.nics, topo)
        L.append("")
        for x in topo.nics:
            order = list(AFF_DESC)
            cls = min((topo.affinity(g, x)[0] for g in topo.gpus), key=order.index)
            same = [g.tag for g in topo.gpus if topo.affinity(g, x)[0] == cls]
            L.append(f"   {x.tag} ({x.name[:32]}): 最近 GPU {', '.join(same)} [{cls}]")

    section("带宽共享与 P2P 路径小结")
    L.append(" [共享上行] 同一 Switch 下的设备共用一条到 CPU 的链路:")
    for key, members in topo.groups(topo.eps):
        if topo.role(key) != "SW-Up":
            L.append(f"   {', '.join(d.tag for d in members)}: 直连 RootPort {key}, 独占上行")
            continue
        up = link_of(key)
        down = topo.down_bw(members)
        ratio = down / up.max_bw if up.max_bw else 0
        flag = "  ⚠ 超额订阅" if ratio > 1.05 else ""
        L.append(f"   Switch {key} (NUMA {numa_of(key)}): {', '.join(d.tag for d in members)}"
                 f"  上行 {up.mx()} {up.max_bw:.1f} GB/s, 下行合计 {down:.1f} GB/s"
                 f" → {ratio:.1f}:1{flag}")
    if len(topo.gpus) > 1:
        L.append(" [GPU P2P]")
        ggroups = topo.groups(topo.gpus)
        names = {key: chr(ord("A") + i) for i, (key, _) in enumerate(ggroups)}
        for key, members in ggroups:
            where = f"Switch {key}" if topo.role(key) == "SW-Up" else f"RootPort {key}"
            L.append(f"   组 {names[key]} = {{{', '.join(g.tag for g in members)}}}"
                     f"  @ {where}, NUMA {members[0].numa}")
        L.append("   组内 P2P 走 Switch 内部 (PIX), 不占用上行链路也不经过 CPU;")
        L.append("   同 NUMA 不同组 (NODE/PHB) 经 CPU Root Complex 转发, 受上行链路与 CPU 限制;")
        L.append("   跨 NUMA (SYS) 还要跨 UPI, 带宽最低、延迟最高。")
    redirect = {p for g in topo.gpus for p in topo.acs_ports(g)}
    if redirect:
        bits = {n for p in redirect for i, n in enumerate(ACS_BITS) if acs_ctl(p) >> i & 1}
        L.append(f" [ACS] ⚠ 下行端口 {', '.join(sorted(p[5:] for p in redirect))} 开启了 P2P 重定向"
                 f" ({'+'.join(n for n in ACS_BITS if n in bits)}):")
        L.append("       同 Switch 的 GPU P2P 也会绕行 CPU/IOMMU, 可用 tools/disable-acs.sh 关闭")
    elif topo.acs_readable:
        L.append(" [ACS] GPU 所在下行端口均未开启 P2P 重定向, 同 Switch P2P 可直通")
    else:
        L.append(" [ACS] 以 root 运行可检查下行端口 ACS 是否把 P2P 重定向到 CPU (影响同 Switch P2P)")
    degraded = [d.tag for d in topo.eps if d.link.degraded]
    if degraded:
        L.append(f" [链路] ⚠ 当前降速: {', '.join(degraded)} (GPU 空闲省电属正常, 有负载时复查)")
    return L


# ----------------------------------------------------------------- HTML 报告
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
.sw{border:2px solid #f9a825;border-radius:6px;padding:8px;background:#fff8e1}
.over{font-size:11px;color:#444;margin-bottom:6px} .over.warn{color:#c62828;font-weight:600}
.ports{display:flex;flex-wrap:wrap;gap:8px}
.port{border:1px solid #ddd;border-radius:4px;padding:6px;background:#fff;min-width:150px}
.port-hdr{font-size:11px;color:#777}
.ep{border-radius:4px;padding:6px 8px;font-size:12px;margin-top:4px;line-height:1.35}
.ep b{font-size:13px}
.gpu{background:#c8e6c9;border:1px solid #2e7d32} .nic{background:#bbdefb;border:1px solid #1565c0}
.mlx{background:#90caf9} .other{background:#eeeeee;border:1px solid #9e9e9e;color:#555}
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
    cls = "link warn" if lk.degraded else "link"
    return f'<div class="{cls}">{h(lk.text())}</div>'


def html_ep(topo, bdf):
    d = topo.by_bdf.get(bdf)
    if d is None:
        return f'<div class="ep other">{h(bdf)}<br>{h(dev_name(bdf, topo.names, 40))}</div>'
    cls = "ep " + d.kind + (" mlx" if d.info.get("mlx") else "")
    body = f"<b>{h(d.tag)}</b> <span class='mono'>{h(d.bdf)}</span><br>{h(d.name)}"
    if d.kind == "gpu":
        body += f"<br>BAR1 {h(fmt_size(d.info['bar1']))}" + \
            (f" · 显存 {h(d.info['mem'])}" if d.info.get("mem") else "")
    else:
        body += f"<br>{h(d.info['state'])}"
    acs = topo.acs_ports(d)
    if acs:
        body += '<span class="badge">ACS 重定向</span>'
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
            over = (f'<div class="over{" warn" if ratio > 1.05 else ""}">上行 {h(lk.mx())} '
                    f'{lk.max_bw:.1f} GB/s · 下行 {h(", ".join(d.tag for d in eps))} 合计 '
                    f'{down:.1f} GB/s → 超额订阅 {ratio:.1f}:1</div>')
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
            ports += '<div class="port"><div class="port-hdr">其他设备</div>' + "".join(
                html_ep(topo, o) for o in others) + "</div>"
        if empty:
            ports += (f'<div class="port"><div class="port-hdr">空端口 ×{len(empty)}</div>'
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
    P = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>拓扑 {h(sysinfo['host'])}"
         f"</title><style>{CSS}</style></head><body>"]
    P.append(f"<h1>CPU / GPU / NIC / PCIe 拓扑 — {h(sysinfo['host'])}</h1>")
    P.append(f"<div class='sub'>{h(sysinfo['cpu'])} × {sysinfo['sockets']} socket · "
             f"内核 {h(sysinfo['kernel'])} · {datetime.datetime.now():%Y-%m-%d %H:%M} · "
             f"IOMMU {'开启 (%d groups)' % sysinfo['iommu_groups'] if sysinfo['iommu_groups'] else '关闭'}"
             f" · 内核参数 [{h(sysinfo['cmdline'])}]</div>")

    P.append("<h2>拓扑图</h2><div class='legend'><span class='gpu'>GPU</span>"
             "<span class='nic'>NIC</span><span class='nic mlx'>Mellanox NIC</span>"
             "<span class='sw'>PCIe Switch</span><span class='rp'>Root Port</span>"
             "<span class='other'>其他</span> &nbsp; 链路标注为每方向理论带宽, 红色 = 当前降速</div>")
    for n in sysinfo["nodes"]:
        roots = [r for r in topo.roots if numa_of(r) == n["id"]]
        if not roots:
            continue
        P.append(f"<div class='numa'><div class='numa-hdr'>NUMA {n['id']} · CPU {h(n['cpus'])} · "
                 f"内存 {n['mem_gb']:.0f} GB</div><div class='row'>")
        P += [html_node(topo, r) for r in roots]
        P.append("</div></div>")
    roots = [r for r in topo.roots if numa_of(r) < 0]
    if roots:
        P.append("<div class='numa'><div class='numa-hdr'>NUMA 未知</div><div class='row'>")
        P += [html_node(topo, r) for r in roots]
        P.append("</div></div>")

    P.append("<h2>GPU 列表</h2>")
    rows = []
    for g in topo.gpus:
        sw = topo.switch_of(g)
        rows.append([g.tag, g.bdf, g.name, g.numa, g.cpus,
                     f"{g.link.cur()} / {g.link.mx()}", f"{g.link.max_bw:.1f}",
                     fmt_size(g.info["bar1"]), g.info.get("mem", "-"),
                     sw or "直连", g.chain[0], g.info["driver"]])
    P.append(html_table(["GPU", "PCI 地址", "名称 [vendor:dev]", "NUMA", "亲和 CPU",
                         "链路 当前/最大", "GB/s", "BAR1", "显存", "Switch", "RootPort", "驱动"],
                        rows) if rows else "<p>未发现 GPU</p>")

    P.append("<h2>NIC 列表</h2>")
    rows = []
    for x in topo.nics:
        sw = topo.switch_of(x)
        rows.append([x.tag, x.bdf, x.name + (" (Mellanox)" if x.info["mlx"] else ""),
                     x.info["state"], x.numa, f"{x.link.cur()} / {x.link.mx()}",
                     f"{x.link.max_bw:.1f}", sw or "直连", x.chain[0], x.info["driver"]])
    P.append(html_table(["NIC", "PCI 地址", "名称 [vendor:dev]", "网口/RDMA 状态", "NUMA",
                         "链路 当前/最大", "GB/s", "Switch", "RootPort", "驱动"], rows)
             if rows else "<p>未发现 NIC</p>")

    P.append("<h2>GPU ↔ GPU 亲和矩阵</h2>")
    if len(topo.gpus) > 1:
        P.append(html_matrix(topo.gpus, topo.gpus, topo))
    P.append("<div class='legend'>" + "".join(
        f"<span class='m-{k}'>{k}: {h(v)}</span>" for k, v in AFF_DESC.items()) + "</div>")
    if topo.gpus and topo.nics:
        P.append("<h2>GPU ↔ NIC 亲和矩阵 (GPUDirect RDMA 首选 PIX/PXB)</h2>")
        P.append(html_matrix(topo.gpus, topo.nics, topo))

    P.append("<h2>带宽共享与 P2P 路径小结</h2><ul>")
    start = text.index(next(s for s in text if s.startswith(" [共享上行]")))
    for line in text[start:]:
        if line.strip():
            P.append(f"<li{' style=font-weight:600' if line.startswith(' [') else ''}>"
                     f"{h(line.strip())}</li>")
    P.append("</ul>")
    P.append("<details><summary>完整文本报告</summary><pre>" + h("\n".join(text)) +
             "</pre></details></body></html>")
    return "\n".join(P)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="CPU/GPU/NIC/PCIe 拓扑报告")
    ap.add_argument("--gpu-vendor", default="0x10de",
                    help="GPU 厂商 ID, 逗号分隔, 默认 NVIDIA 0x10de (AMD 0x1002)")
    ap.add_argument("--html", metavar="FILE", help="另存为 HTML 网页")
    ap.add_argument("--quiet", action="store_true", help="不打印到控制台(配合 --html)")
    args = ap.parse_args()

    if not os.path.isdir(SYSFS):
        sys.exit("找不到 /sys/bus/pci/devices, 需要在 Linux 上运行。")
    names = lspci_names()
    gpus, nics = discover({v.strip().lower() for v in args.gpu_vendor.split(",")}, names)
    if not gpus and not nics:
        sys.exit("未发现 GPU 或 NIC (虚拟机/容器中 sysfs 可能不完整)。")
    enrich_nvidia_smi(gpus)
    topo = Topo(gpus, nics, names)
    sysinfo = system_info(topo)

    text = text_report(topo, sysinfo)
    if not args.quiet:
        print("\n".join(text))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html_report(topo, sysinfo, text))
        print(f"\n已保存网页: {args.html}")


if __name__ == "__main__":
    main()
