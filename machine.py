"""
Machine class definition

本文件定义了离子阱（Trap）- 传输段（Segment）- 交叉点（Junction）的机器模型，并提供：
1) 构建机器拓扑（graph）接口：add_trap / add_segment / add_junction
2) 调度器需要的时间模型：
   - gate_time：两比特门时间（按 gate_type）
   - split_time / merge_time：分裂/合并时间与 split-swap 统计
   - move_time：按物理距离与速度计算的移动时间（Table 1: 2 μm/us）
   - junction_cross_time：穿越 junction 的额外时间
3) 论文复现所需的“物理量参数化”（论文没给的外提为可调参数）：
   - segment_length_um：segment 物理长度（um）
   - move_speed_um_per_us：移动速度（um/us）
   - inter_ion_spacing_um：阱内离子间距（um）
   - alpha_bg：背景 Bi 模型强度（供 Analyzer 用）
4) Large-scale 阶段的架构元数据：
   - qccd / zone role
   - optical/fiber 互连登记
   - 模块查询辅助接口

设计原则：
- 完全保留现有 small-scale 代码路径的接口与行为。
- large-scale 仅新增元数据与查询接口，不改变 small 的时间模型。
- 未被当前仓库使用、但将来容易误用的接口不保留“半成品”实现；
  只保留明确可用、行为可预期的方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


# =========================
#   基本结构：Trap / Segment / Junction
# =========================
class Trap:
    """离子阱：容纳一条离子链（chain）。orientation 记录每条 segment 在阱的左右侧。"""

    def __init__(self, idx: int, capacity: int):
        self.id = int(idx)
        self.capacity = int(capacity)
        self.ions: List[int] = []  # 旧逻辑保留，便于调试/展示
        self.orientation: Dict[int, str] = {}

        # ---- QCCD / EML metadata ----
        self.qccd_id: int = 0
        self.zone_type: str = "storage"   # storage / operation / optical
        self.zone_level: int = 0           # smaller means higher priority in mapper
        self.can_execute_local_2q: bool = False
        self.can_execute_remote_2q: bool = False

    def show(self) -> str:
        return "T" + str(self.id)

    def __repr__(self) -> str:
        return f"Trap(id={self.id}, cap={self.capacity}, qccd={self.qccd_id}, zone={self.zone_type})"


class Segment:
    """传输段：连接两个节点（Trap 或 Junction）的边对象，带物理长度（um）。"""

    def __init__(self, idx: int, capacity: int, length_um: float):
        self.id = int(idx)
        self.capacity = int(capacity)
        self.length = float(length_um)
        self.ions: List[int] = []  # 旧逻辑保留

    def __repr__(self) -> str:
        return f"Segment(id={self.id}, cap={self.capacity}, len={self.length})"


class Junction:
    """交叉点：用于多段 segment 的连接（图的节点）。"""

    def __init__(self, idx: int):
        self.id = int(idx)
        self.objs: List[object] = []  # 旧逻辑保留

    def show(self) -> str:
        return "J" + str(self.id)

    def __repr__(self) -> str:
        return f"Junction(id={self.id})"


@dataclass
class FiberLink:
    """Large-scale EML-QCCD 的模块间光纤互连描述。"""

    src_qccd: int
    dst_qccd: int
    src_trap_id: int
    dst_trap_id: int
    latency_us: float = 200.0
    fidelity: float = 0.99


# =========================
#   参数容器：MachineParams
# =========================
class MachineParams:
    """
    参数容器（建议固定“论文明确给出”的默认值；论文没给的做显式 knob）

    论文明确给出的（MUSS-TI Table 1）：
      - split_merge_time (us) = 80
      - ion_swap_time (us)    = 40
      - move_speed_um_per_us  = 2.0 (um/us)
      - junction{2,3,4}_cross_time (us) = 5
      - shuttle_time (us) = 5 （旧字段，兼容回退）

    论文没明确钉死 / 实现需要外提的 knob：
      - segment_length_um (um)        ：每条 segment 默认长度（影响 move_time 与 move heating）
      - inter_ion_spacing_um (um)     ：阱内离子间距（影响 Duan/Trout/PM 的 gate_time 距离项）
      - alpha_bg                      ：背景 Bi 模型强度（Analyzer 用）
      - gate_type / swap_type         ：门/交换实现类型
      - max_qubits_per_qccd           ：EML-QCCD 单模块上限（large-scale 阶段默认 32）
      - num_optical_zones             ：每模块 optical zone 数（第一阶段默认 1）
      - qccd_fiber_latency_us         ：模块间 fiber gate 时间（后续 large scheduler / analyzer 用）
      - qccd_fiber_fidelity           ：模块间 fiber gate fidelity（后续 analyzer 用）

    swap_type 支持：
      - "PaperSwapDirect"：论文对准版，链内 direct internal swap，一次即可，不按 hop 数累加时间
      - "GateSwap"       ：旧逻辑，swap 时间 = 3 * gate_time(...)
      - "IonSwap"        ：旧逻辑，swap 通过 split/move/merge 物理实现
    """

    def __init__(self):
        # ===== MUSS-TI Table 1 (paper-fixed defaults) =====
        self.split_merge_time = 80
        self.shuttle_time = 5
        self.ion_swap_time = 40

        self.junction2_cross_time = 5
        self.junction3_cross_time = 5
        self.junction4_cross_time = 5

        self.move_speed_um_per_us = 2.0  # 2 μm/us

        # ===== Not explicitly fixed by paper / implementation knobs =====
        self.segment_length_um = 80.0
        self.inter_ion_spacing_um = 1.0
        self.alpha_bg = 0.0

        # ===== gate/swap type flags =====
        self.gate_type = "PM"                  # "FM"/"PM"/"Duan"/"Trout"
        self.swap_type = "PaperSwapDirect"     # "PaperSwapDirect"/"GateSwap"/"IonSwap"
        self.enable_partition = False
        self.partition_strategy = "none"
        self.architecture_scale = "small"

        # ===== Large-scale knobs =====
        self.max_qubits_per_qccd = 32
        self.num_optical_zones = 1
        self.qccd_fiber_latency_us = 200.0
        self.qccd_fiber_fidelity = 0.99


# =========================
#   Machine：机器模型主体
# =========================
class Machine:
    """
    Machine 用 networkx.Graph() 存拓扑：
    - 节点：Trap 或 Junction
    - 边：Segment（作为 edge 属性 "seg" 挂在图上）

    额外维护：
    - traps / segments / junctions 列表（方便遍历）
    - segments_by_id：避免 seg_id != index 时访问错误
    - dist_cache：trap-to-trap 最短路径 hop 数缓存
    - qccd_graph：EML 模块级互连图（large-scale）
    - fiber_links：EML 光纤互连注册表
    """

    def __init__(self, mparams: Optional[MachineParams] = None):
        self.mparams = mparams if mparams is not None else MachineParams()

        self.graph = nx.Graph()
        self.traps: List[Trap] = []
        self.segments: List[Segment] = []
        self.junctions: List[Junction] = []

        self.segments_by_id: Dict[int, Segment] = {}
        self.dist_cache: Dict[Tuple[int, int], int] = {}

        # ---- large-scale metadata ----
        self.qccd_graph = nx.Graph()
        self.fiber_links: List[FiberLink] = []

    # -------------------------
    # Build graph API
    # -------------------------
    def add_trap(self, idx: int, capacity: int) -> Trap:
        """添加一个 Trap 节点。"""
        new_trap = Trap(idx, capacity)
        self.traps.append(new_trap)
        self.graph.add_node(new_trap)
        return new_trap

    def add_junction(self, idx: int) -> Junction:
        """添加一个 Junction 节点。"""
        new_junct = Junction(idx)
        self.junctions.append(new_junct)
        self.graph.add_node(new_junct)
        return new_junct

    def add_segment(self, idx: int, obj1, obj2, orientation: str = "L") -> Segment:
        """
        添加一条 Segment 边，并把 Segment 对象挂到 graph edge 的属性 "seg" 上。

        参数：
        - idx：segment id
        - obj1 / obj2：图的两个端点（Trap 或 Junction）
        - orientation：仅对 Trap 有意义。记录该 segment 在 trap 的左/右侧。
        """
        seg_len = float(getattr(self.mparams, "segment_length_um", 10.0))
        new_seg = Segment(idx, 16, seg_len)

        self.segments.append(new_seg)
        self.segments_by_id[new_seg.id] = new_seg

        if isinstance(obj1, Trap):
            obj1.orientation[new_seg.id] = orientation
        if isinstance(obj2, Trap):
            # 保留旧 API：orientation 只记录 trap 侧，junction-junction 边不记录
            # 若调用者传入 trap 作为 obj2，也允许记录同一个 orientation。
            obj2.orientation[new_seg.id] = orientation

        self.graph.add_edge(obj1, obj2, seg=new_seg)
        self.dist_cache.clear()
        return new_seg

    # -------------------------
    # QCCD / zone metadata
    # -------------------------
    def set_trap_role(self, trap_id: int, qccd_id: int, zone_type: str, zone_level: int) -> Trap:
        """为 trap 打上 EML-QCCD 角色元数据。"""
        tr = self.get_trap(trap_id)
        tr.qccd_id = int(qccd_id)
        tr.zone_type = str(zone_type)
        tr.zone_level = int(zone_level)

        if tr.zone_type == "storage":
            tr.can_execute_local_2q = False
            tr.can_execute_remote_2q = False
        elif tr.zone_type == "operation":
            tr.can_execute_local_2q = True
            tr.can_execute_remote_2q = False
        elif tr.zone_type == "optical":
            tr.can_execute_local_2q = True
            tr.can_execute_remote_2q = True
        else:
            raise ValueError(f"Unsupported zone_type: {zone_type}")

        self.qccd_graph.add_node(tr.qccd_id)
        return tr

    def get_trap(self, trap_id: int) -> Trap:
        for tr in self.traps:
            if tr.id == trap_id:
                return tr
        raise KeyError(f"trap {trap_id} not found")

    def traps_in_qccd(self, qccd_id: int) -> List[Trap]:
        return [tr for tr in self.traps if getattr(tr, "qccd_id", 0) == qccd_id]

    def qccd_ids(self) -> List[int]:
        return sorted({getattr(tr, "qccd_id", 0) for tr in self.traps})

    def traps_by_zone(self, zone_type: str) -> List[Trap]:
        return [tr for tr in self.traps if getattr(tr, "zone_type", None) == zone_type]

    def qccd_traps_by_zone(self, qccd_id: int, zone_type: str) -> List[Trap]:
        return [
            tr for tr in self.traps
            if getattr(tr, "qccd_id", 0) == qccd_id and getattr(tr, "zone_type", None) == zone_type
        ]

    def get_qccd_storage_traps(self, qccd_id: int) -> List[Trap]:
        return self.qccd_traps_by_zone(qccd_id, "storage")

    def get_qccd_operation_traps(self, qccd_id: int) -> List[Trap]:
        return self.qccd_traps_by_zone(qccd_id, "operation")

    def get_qccd_optical_traps(self, qccd_id: int) -> List[Trap]:
        return self.qccd_traps_by_zone(qccd_id, "optical")

    def add_fiber_link(
        self,
        src_qccd: int,
        dst_qccd: int,
        src_trap_id: int,
        dst_trap_id: int,
        latency_us: Optional[float] = None,
        fidelity: Optional[float] = None,
    ) -> FiberLink:
        """登记模块级光纤互连。第一阶段只登记，不改变片上 graph。"""
        link = FiberLink(
            src_qccd=int(src_qccd),
            dst_qccd=int(dst_qccd),
            src_trap_id=int(src_trap_id),
            dst_trap_id=int(dst_trap_id),
            latency_us=float(
                self.mparams.qccd_fiber_latency_us if latency_us is None else latency_us
            ),
            fidelity=float(
                self.mparams.qccd_fiber_fidelity if fidelity is None else fidelity
            ),
        )
        self.fiber_links.append(link)
        self.qccd_graph.add_edge(link.src_qccd, link.dst_qccd)
        return link

    def fiber_neighbors(self, qccd_id: int) -> List[int]:
        if qccd_id not in self.qccd_graph:
            return []
        return sorted(int(x) for x in self.qccd_graph.neighbors(qccd_id))

    def get_fiber_link(self, qccd_a: int, qccd_b: int) -> Optional[FiberLink]:
        for link in self.fiber_links:
            if (link.src_qccd, link.dst_qccd) == (qccd_a, qccd_b):
                return link
            if (link.src_qccd, link.dst_qccd) == (qccd_b, qccd_a):
                return link
        return None

    def get_fiber_links_between(self, qccd_a: int, qccd_b: int) -> List[FiberLink]:
        """返回两个 QCCD 之间登记的全部 fiber link。"""
        out: List[FiberLink] = []
        for link in self.fiber_links:
            if (link.src_qccd, link.dst_qccd) == (qccd_a, qccd_b):
                out.append(link)
            elif (link.src_qccd, link.dst_qccd) == (qccd_b, qccd_a):
                out.append(link)
        return out

    def add_comm_capacity(self, val: int) -> None:
        """给所有 trap 增加容量（旧接口保留）。"""
        for t in self.traps:
            t.capacity += val

    def print_machine_stats(self) -> None:
        """打印基本机器信息；旧接口保留，并补充 large-scale 元数据。"""
        print(f"#Traps={len(self.traps)} #Junctions={len(self.junctions)} #Segments={len(self.segments)}")
        if self.traps:
            print(f"TrapCapacity(default)={self.traps[0].capacity}")
        if self.fiber_links:
            print(f"#FiberLinks={len(self.fiber_links)} #QCCDs={len(self.qccd_ids())}")

    def get_segment_length_um(self, seg_id: int) -> float:
        """
        给 Analyzer/调度器调用：返回 segment 的物理长度（um）
        优先用 segments_by_id；否则回退假设 seg_id==index；再回退默认值。
        """
        if seg_id in self.segments_by_id:
            return float(self.segments_by_id[seg_id].length)
        try:
            return float(self.segments[seg_id].length)
        except Exception:
            return float(getattr(self.mparams, "segment_length_um", 10.0))

    # -------------------------
    # Gate / Trap 操作时间
    # -------------------------
    def gate_time(self, sys_state, trap_id: int, ion1: int, ion2: int) -> int:
        """
        计算两比特门时间（us）。

        对齐论文评估：
        - gate_type == "FM" 时直接返回固定 40us（MUSS-TI Table 1）
        其它 gate_type 保留原经验公式（与离子距离相关）。
        """
        assert ion1 != ion2
        mp = self.mparams

        p1 = sys_state.trap_ions[trap_id].index(ion1)
        p2 = sys_state.trap_ions[trap_id].index(ion2)

        d_const = float(getattr(mp, "inter_ion_spacing_um", 1.0))
        ion_dist = abs(p1 - p2) * d_const

        gate_type = getattr(mp, "gate_type", "FM")

        if gate_type == "Duan":
            t = -22 + 100 * ion_dist
        elif gate_type == "Trout":
            t = 10 + 38 * ion_dist
        elif gate_type == "FM":
            return 40
        elif gate_type == "PM":
            t = 160 + 5 * ion_dist
        else:
            raise AssertionError(f"Unsupported gate_type: {gate_type}")

        t = max(t, 1)
        return int(t)

    def split_time(self, sys_state, trap_id: int, seg_id: int, ion1: int):
        """
        计算 split_time，并给出 split swap 的统计信息（供 schedule/event 记录）。

        返回：
          (split_estimate,
           split_swap_count,
           split_swap_hops,
           i1, i2,
           ion_swap_hops)

        重要修正：
        - PaperSwapDirect 语义下：
          不是 hop-by-hop 相邻交换，而是“若目标离子不在链端，则与链端离子直接交换一次”。
        """
        t = self.traps[trap_id]
        split_estimate = 0
        split_swap_count = 0
        ion_swap_hops = 0
        split_swap_hops = 0
        i1 = 0
        i2 = 0

        if t.orientation[seg_id] == "L":
            ion2 = sys_state.trap_ions[trap_id][0]
        else:
            ion2 = sys_state.trap_ions[trap_id][-1]

        if ion1 == ion2:
            split_estimate = int(self.mparams.split_merge_time)
            split_swap_count = 0
            split_swap_hops = 0
        else:
            mp = self.mparams
            swap_type = getattr(mp, "swap_type", "PaperSwapDirect")

            p1 = sys_state.trap_ions[trap_id].index(ion1)
            p2 = sys_state.trap_ions[trap_id].index(ion2)
            num_hops = abs(p1 - p2)
            split_swap_hops = num_hops

            if swap_type == "PaperSwapDirect":
                split_estimate = int(self.mparams.split_merge_time) + int(self.mparams.ion_swap_time)
                split_swap_count = 1
                i1 = ion1
                i2 = ion2

            elif swap_type == "GateSwap":
                swap_est = 3 * self.gate_time(sys_state, trap_id, ion1, ion2)
                split_estimate = int(swap_est + self.mparams.split_merge_time)
                split_swap_count = 1
                i1 = ion1
                i2 = ion2

            elif swap_type == "IonSwap":
                swap_est = num_hops * self.mparams.split_merge_time
                swap_est += (num_hops - 1) * self.mparams.split_merge_time
                swap_est += self.mparams.ion_swap_time * num_hops
                split_estimate = int(swap_est)
                ion_swap_hops = num_hops

            else:
                raise AssertionError(f"Unsupported swap_type: {swap_type}")

        return int(split_estimate), split_swap_count, split_swap_hops, i1, i2, ion_swap_hops

    def merge_time(self, trap_id: int) -> int:
        """合并时间（us），论文 Table 1：与 split 同为 split_merge_time。"""
        return int(self.mparams.split_merge_time)

    # -------------------------
    # Move / Junction 时间
    # -------------------------
    def move_time(self, seg1_id: int, seg2_id: int) -> int:
        """
        按论文 Table 1 的速度模型计算 Move 时间：
            speed = move_speed_um_per_us（默认 2.0）
            time_us = distance_um / speed
        """
        speed = getattr(self.mparams, "move_speed_um_per_us", None)
        if speed is None:
            return int(getattr(self.mparams, "shuttle_time", 5))

        dist_um = self.get_segment_length_um(seg2_id)
        try:
            t_us = float(dist_um) / float(speed)
        except Exception:
            t_us = float(getattr(self.mparams, "shuttle_time", 5))

        return int(round(max(t_us, 1.0)))

    def junction_cross_time(self, junct: Junction) -> int:
        """junction crossing 的额外时间，按 junction 度数选择参数。"""
        deg = self.graph.degree(junct)
        if deg == 2:
            return int(self.mparams.junction2_cross_time)
        if deg == 3:
            return int(self.mparams.junction3_cross_time)
        if deg == 4:
            return int(self.mparams.junction4_cross_time)
        raise AssertionError(f"Unsupported junction degree: {deg}")

    # -------------------------
    # 其它接口（保留原有功能）
    # -------------------------
    def single_qubit_gate_time(self, gate_type) -> int:
        """1Q gate 时间（旧实现保留，固定常数）。"""
        return 5

    def precompute_distances(self) -> None:
        """预计算 trap-to-trap 的最短路径长度（按 graph hop 数）。"""
        self.dist_cache = {}
        id_map = {t.id: t for t in self.traps}

        try:
            all_paths = dict(nx.all_pairs_shortest_path_length(self.graph))
        except Exception as exc:
            print("Warning: Failed to compute distances:", exc)
            all_paths = {}

        for id1, t1 in id_map.items():
            for id2, t2 in id_map.items():
                if t1 == t2:
                    self.dist_cache[(id1, id2)] = 0
                elif t1 in all_paths and t2 in all_paths.get(t1, {}):
                    self.dist_cache[(id1, id2)] = all_paths[t1][t2]
                else:
                    self.dist_cache[(id1, id2)] = 1000

    def trap_distance(self, trap1_id: int, trap2_id: int) -> int:
        """兼容旧代码的查询接口：返回预计算的 trap-to-trap hop 距离。"""
        if (trap1_id, trap2_id) in self.dist_cache:
            return self.dist_cache[(trap1_id, trap2_id)]

        t1 = None
        t2 = None
        for tr in self.traps:
            if tr.id == trap1_id:
                t1 = tr
            if tr.id == trap2_id:
                t2 = tr

        if t1 is None or t2 is None:
            return 1000

        try:
            d = nx.shortest_path_length(self.graph, t1, t2)
            self.dist_cache[(trap1_id, trap2_id)] = d
            return d
        except Exception:
            return 1000
