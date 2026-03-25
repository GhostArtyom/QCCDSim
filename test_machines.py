# test_machines.py
# -*- coding: utf-8 -*-

"""
测试用离子阱机器拓扑集合（Test Machines）

本文件职责：
1) 保留并兼容现有 small-scale 工厂函数；
2) 为第一阶段 large-scale 搭建新增机器：
   - G3x4 / G4x5：大规模 baseline QCCD grid proxy
   - EML：由多个 2x2 improved-QCCD module 组成的系统
3) 所有工厂函数统一显式接收 mparams，避免参数漂移。

设计原则：
- 不改坏现有 small path：test_trap_2x2 / test_trap_2x3 / linear / hex 等全部保留。
- large path 新增独立工厂函数，不覆盖 small 的拓扑。
- 第一阶段只负责“架构搭起来”，因此 EML 的 fiber 只做模块级登记，
  不把 fiber 伪装成片上可 shuttle 的 graph edge。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from machine import Machine, MachineParams


# ---------------------------------------------------------------------------
# 架构模式判断
# ---------------------------------------------------------------------------
def _is_large_arch(mparams) -> bool:
    return str(getattr(mparams, "architecture_scale", "small")).lower() == "large"


def _is_small_arch(mparams) -> bool:
    return not _is_large_arch(mparams)


# ---------------------------------------------------------------------------
# 小规模角色打标（保留现状，不影响 small）
# ---------------------------------------------------------------------------
def _tag_qccd_roles_2x2(m: Machine) -> Machine:
    if _is_small_arch(m.mparams):
        return m

    # Trap ids: 0/2 storage, 1 operation, 3 optical.
    roles = {
        0: (0, "storage", 2),
        1: (0, "operation", 1),
        2: (0, "storage", 2),
        3: (0, "optical", 0),
    }
    for tid, (qid, zt, zl) in roles.items():
        m.set_trap_role(tid, qid, zt, zl)
    return m


def _tag_qccd_roles_2x3(m: Machine) -> Machine:
    if _is_small_arch(m.mparams):
        return m

    roles = {
        0: (0, "storage", 2),
        1: (0, "operation", 1),
        2: (0, "storage", 2),
        3: (0, "storage", 2),
        4: (0, "optical", 0),
        5: (0, "storage", 2),
    }
    for tid, (qid, zt, zl) in roles.items():
        m.set_trap_role(tid, qid, zt, zl)
    return m


# ---------------------------------------------------------------------------
# 现有 small 拓扑（完整保留）
# ---------------------------------------------------------------------------
def test_trap_2x3(capacity, mparams):
    """
    2x3 grid with row-major trap numbering.

    Layout (row-major):
        top row:    T0  T1  T2
        bottom row: T3  T4  T5

    Junction backbone:
        J0 -- J1 -- J2

    Local branches:
        T0, T3 connect to J0
        T1, T4 connect to J1
        T2, T5 connect to J2

    Orientation convention:
        top-row trap branch uses "R"
        bottom-row trap branch uses "L"
    """
    m = Machine(mparams)

    t = [m.add_trap(i, capacity) for i in range(6)]
    j = [m.add_junction(i) for i in range(3)]

    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")

    m.add_segment(3, t[3], j[0], "L")
    m.add_segment(4, t[4], j[1], "L")
    m.add_segment(5, t[5], j[2], "L")

    m.add_segment(6, j[0], j[1])
    m.add_segment(7, j[1], j[2])

    return _tag_qccd_roles_2x3(m) if _is_large_arch(mparams) else m



def test_trap_2x2(capacity, mparams):
    """
    2x2 网格：4 个 Trap + 2 个 Junction + 1 条 junction 主干
    拓扑结构：
        T0 --(seg0)-- J0 --(seg4)-- J1 --(seg2)-- T1
        T3 --(seg1)-- J0            J1 --(seg3)-- T2
    """
    m = Machine(mparams)

    t = [m.add_trap(i, capacity) for i in range(4)]
    j = [m.add_junction(i) for i in range(2)]

    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[3], j[0], "L")
    m.add_segment(2, t[1], j[1], "R")
    m.add_segment(3, t[2], j[1], "L")
    m.add_segment(4, j[0], j[1])

    return _tag_qccd_roles_2x2(m) if _is_large_arch(mparams) else m


# ---------------------------------------------------------------------------
# 线性 / 六边形等旧拓扑（保留）
# ---------------------------------------------------------------------------
def make_linear_machine(zones, capacity, mparams):
    """
    线性拓扑：zones 个 trap 串成一条链，每对相邻 trap 通过一个 junction 连接。
    """
    m = Machine(mparams)
    traps = [m.add_trap(i, capacity) for i in range(zones)]
    junctions = [m.add_junction(i) for i in range(zones - 1)]

    for i in range(zones - 1):
        m.add_segment(2 * i, traps[i], junctions[i], "R")
        m.add_segment(2 * i + 1, traps[i + 1], junctions[i], "L")

    return m



def make_single_hexagon_machine(capacity, mparams):
    """单个六边形：6 trap + 6 junction。"""
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(6)]
    j = [m.add_junction(i) for i in range(6)]

    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")
    m.add_segment(3, t[3], j[3], "R")
    m.add_segment(4, t[4], j[4], "R")
    m.add_segment(5, t[5], j[5], "R")

    m.add_segment(6, t[0], j[5], "L")
    m.add_segment(7, t[1], j[0], "L")
    m.add_segment(8, t[2], j[1], "L")
    m.add_segment(9, t[3], j[2], "L")
    m.add_segment(10, t[4], j[3], "L")
    m.add_segment(11, t[5], j[4], "L")

    return m


# ---------------------------------------------------------------------------
# 其它旧拓扑（保留）
# ---------------------------------------------------------------------------
def mktrap4x2(capacity, mparams):
    m = Machine(mparams)
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    j0 = m.add_junction(0)
    j1 = m.add_junction(1)

    m.add_segment(0, t0, j0, "R")
    m.add_segment(1, t1, j0, "R")
    m.add_segment(2, t2, j1, "R")
    m.add_segment(3, t3, j1, "R")
    m.add_segment(4, j0, j1)
    return m



def mktrap_4star(capacity, mparams):
    m = Machine(mparams)
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    j0 = m.add_junction(0)

    m.add_segment(0, t0, j0, "R")
    m.add_segment(1, t1, j0, "R")
    m.add_segment(2, t2, j0, "R")
    m.add_segment(3, t3, j0, "R")
    return m



def mktrap6x3(capacity, mparams):
    m = Machine(mparams)
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    t4 = m.add_trap(4, capacity)
    t5 = m.add_trap(5, capacity)

    j0 = m.add_junction(0)
    j1 = m.add_junction(1)
    j2 = m.add_junction(2)

    m.add_segment(0, t0, j0, "R")
    m.add_segment(1, t1, j0, "R")
    m.add_segment(2, t2, j1, "R")
    m.add_segment(3, t3, j1, "R")
    m.add_segment(4, t4, j2, "R")
    m.add_segment(5, t5, j2, "R")

    m.add_segment(6, j0, j1)
    m.add_segment(7, j1, j2)
    return m



def mktrap8x4(capacity, mparams):
    m = Machine(mparams)
    t0 = m.add_trap(0, capacity)
    t1 = m.add_trap(1, capacity)
    t2 = m.add_trap(2, capacity)
    t3 = m.add_trap(3, capacity)
    t4 = m.add_trap(4, capacity)
    t5 = m.add_trap(5, capacity)
    t6 = m.add_trap(6, capacity)
    t7 = m.add_trap(7, capacity)

    j0 = m.add_junction(0)
    j1 = m.add_junction(1)
    j2 = m.add_junction(2)
    j3 = m.add_junction(3)

    m.add_segment(0, t0, j0, "R")
    m.add_segment(1, t1, j0, "R")
    m.add_segment(2, t2, j1, "R")
    m.add_segment(3, t3, j1, "R")
    m.add_segment(4, t4, j2, "R")
    m.add_segment(5, t5, j2, "R")
    m.add_segment(6, t6, j3, "R")
    m.add_segment(7, t7, j3, "R")

    m.add_segment(8, j0, j1)
    m.add_segment(9, j1, j2)
    m.add_segment(10, j2, j3)
    return m



def make_3x3_grid(capacity, mparams):
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(9)]
    j = [m.add_junction(i) for i in range(6)]

    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")
    m.add_segment(3, t[3], j[3], "R")
    m.add_segment(4, t[4], j[4], "R")
    m.add_segment(5, t[5], j[5], "R")
    m.add_segment(6, t[3], j[0], "L")
    m.add_segment(7, t[4], j[1], "L")
    m.add_segment(8, t[5], j[2], "L")
    m.add_segment(9, t[6], j[3], "L")
    m.add_segment(10, t[7], j[4], "L")
    m.add_segment(11, t[8], j[5], "L")
    m.add_segment(12, j[0], j[1])
    m.add_segment(13, j[1], j[2])
    m.add_segment(14, j[3], j[4])
    m.add_segment(15, j[4], j[5])

    return m



def make_9trap(capacity, mparams):
    m = Machine(mparams)

    t = [m.add_trap(i, capacity) for i in range(9)]
    j = [m.add_junction(i) for i in range(9)]

    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")
    m.add_segment(3, t[3], j[2], "L")
    m.add_segment(4, t[4], j[5], "R")
    m.add_segment(5, t[5], j[8], "R")
    m.add_segment(6, t[6], j[8], "L")
    m.add_segment(7, t[7], j[7], "R")
    m.add_segment(8, t[8], j[6], "R")

    m.add_segment(9, j[0], j[1])
    m.add_segment(10, j[0], j[3])
    m.add_segment(11, j[3], j[6])
    m.add_segment(12, j[3], j[4])
    m.add_segment(13, j[6], j[7])
    m.add_segment(14, j[1], j[4])
    m.add_segment(15, j[1], j[2])
    m.add_segment(16, j[4], j[7])
    m.add_segment(17, j[4], j[5])
    m.add_segment(18, j[7], j[8])
    m.add_segment(19, j[2], j[5])
    m.add_segment(20, j[5], j[8])

    return m


# ---------------------------------------------------------------------------
# 第一阶段新增：large-scale baseline QCCD grid proxy
# ---------------------------------------------------------------------------
def _build_column_backbone_grid(rows: int, cols: int, capacity: int, mparams) -> Machine:
    """
    构造 rows x cols 的 column-backbone grid proxy。

    说明：
    - 该结构与现有 2x3 的设计风格保持一致：每一列有一个 backbone junction，列间 junction 串联。
    - 同一列的多行 trap 都挂在该列 junction 上。
    - 这是第一阶段用于“把 large 架构搭起来”的稳定片上图模型，
      后续若需要更细粒度的 cell-level 物理布局，可在不改 run.py 接口的前提下替换本函数。
    """
    m = Machine(mparams)

    traps = [m.add_trap(i, capacity) for i in range(rows * cols)]
    junctions = [m.add_junction(i) for i in range(cols)]

    seg_id = 0
    for r in range(rows):
        for c in range(cols):
            tid = r * cols + c
            orientation = "R" if (r % 2 == 0) else "L"
            m.add_segment(seg_id, traps[tid], junctions[c], orientation)
            seg_id += 1

    for c in range(cols - 1):
        m.add_segment(seg_id, junctions[c], junctions[c + 1])
        seg_id += 1

    return m



def _assign_large_grid_roles_row_major(m: Machine, rows: int, cols: int) -> Machine:
    """
    为 large baseline grid 分配 zone role。

    策略：
    - 每一行的中间位置优先作为 operation / optical；
    - 其余为 storage；
    - 所有 trap 默认属于同一个 qccd_id=0（baseline grid 不是模块化 EML）。
    """
    if _is_small_arch(m.mparams):
        return m

    center = cols // 2
    optical_col = min(center + 1, cols - 1)
    operation_col = center

    for r in range(rows):
        for c in range(cols):
            tid = r * cols + c
            if c == optical_col:
                zone_type, zone_level = "optical", 0
            elif c == operation_col:
                zone_type, zone_level = "operation", 1
            else:
                zone_type, zone_level = "storage", 2
            m.set_trap_role(tid, qccd_id=0, zone_type=zone_type, zone_level=zone_level)
    return m



def make_qccd_grid_3x4(capacity, mparams):
    """Large baseline: 3x4 QCCD grid proxy。"""
    m = _build_column_backbone_grid(3, 4, capacity, mparams)
    return _assign_large_grid_roles_row_major(m, 3, 4)



def make_qccd_grid_4x5(capacity, mparams):
    """Large baseline: 4x5 QCCD grid proxy。"""
    m = _build_column_backbone_grid(4, 5, capacity, mparams)
    return _assign_large_grid_roles_row_major(m, 4, 5)


# ---------------------------------------------------------------------------
# 第一阶段新增：EML-QCCD system builder
# ---------------------------------------------------------------------------
def _grid_dims_for_modules(num_modules: int) -> Tuple[int, int]:
    """将模块数排成尽量接近方形的网格。"""
    cols = int(math.ceil(math.sqrt(num_modules)))
    rows = int(math.ceil(num_modules / cols))
    return rows, cols



def _module_row_col(module_id: int, rows: int, cols: int) -> Tuple[int, int]:
    return module_id // cols, module_id % cols



def _build_eml_module(
    m: Machine,
    module_id: int,
    trap_capacity: int,
    trap_start: int,
    junction_start: int,
    seg_start: int,
    num_optical_zones: int,
) -> Tuple[Dict[str, List[int]], int, int, int]:
    """
    在已有 Machine 上追加一个 2x2 improved-QCCD module。

    角色约定：
      - 默认 4 个 trap：2 storage + 1 operation + 1 optical
      - 若 num_optical_zones == 2：将一个 storage trap 升格为第二个 optical
    """
    t = [m.add_trap(trap_start + i, trap_capacity) for i in range(4)]
    j = [m.add_junction(junction_start + i) for i in range(2)]

    # 与现有 2x2 small 拓扑保持一致，降低兼容风险。
    m.add_segment(seg_start + 0, t[0], j[0], "R")
    m.add_segment(seg_start + 1, t[3], j[0], "L")
    m.add_segment(seg_start + 2, t[1], j[1], "R")
    m.add_segment(seg_start + 3, t[2], j[1], "L")
    m.add_segment(seg_start + 4, j[0], j[1])

    roles = {
        t[0].id: (module_id, "storage", 2),
        t[1].id: (module_id, "operation", 1),
        t[2].id: (module_id, "storage", 2),
        t[3].id: (module_id, "optical", 0),
    }
    if num_optical_zones >= 2:
        roles[t[2].id] = (module_id, "optical", 0)

    for tid, (qid, zt, zl) in roles.items():
        m.set_trap_role(tid, qid, zt, zl)

    role_map = {
        "storage": [tid for tid, (_, zt, _) in roles.items() if zt == "storage"],
        "operation": [tid for tid, (_, zt, _) in roles.items() if zt == "operation"],
        "optical": [tid for tid, (_, zt, _) in roles.items() if zt == "optical"],
    }
    return role_map, trap_start + 4, junction_start + 2, seg_start + 5



def make_eml_qccd_system(
    num_qubits: int,
    trap_capacity: int,
    mparams,
    max_qubits_per_qccd: int | None = None,
    num_optical_zones: int | None = None,
):
    """
    构造第一阶段 large-scale EML-QCCD 系统。

    规则：
    - 每个模块默认最多承载 32 qubits（可由 mparams / 参数覆盖）；
    - 每个模块使用一个 2x2 improved-QCCD tile；
    - 模块按近似方阵排列；
    - 相邻模块之间登记 optical-to-optical fiber links；
    - fiber link 只登记在 qccd_graph / fiber_links 中，不加入片上 graph。
    """
    m = Machine(mparams)

    max_qubits = int(
        max_qubits_per_qccd if max_qubits_per_qccd is not None
        else getattr(mparams, "max_qubits_per_qccd", 32)
    )
    optical_zones = int(
        num_optical_zones if num_optical_zones is not None
        else getattr(mparams, "num_optical_zones", 1)
    )

    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive for EML construction")
    if max_qubits <= 0:
        raise ValueError("max_qubits_per_qccd must be positive")
    if optical_zones not in (1, 2):
        raise ValueError("num_optical_zones must be 1 or 2 in phase-1 builder")

    num_modules = int(math.ceil(num_qubits / max_qubits))
    rows, cols = _grid_dims_for_modules(num_modules)

    next_trap_id = 0
    next_junction_id = 0
    next_segment_id = 0
    module_roles: Dict[int, Dict[str, List[int]]] = {}

    for module_id in range(num_modules):
        roles, next_trap_id, next_junction_id, next_segment_id = _build_eml_module(
            m=m,
            module_id=module_id,
            trap_capacity=trap_capacity,
            trap_start=next_trap_id,
            junction_start=next_junction_id,
            seg_start=next_segment_id,
            num_optical_zones=optical_zones,
        )
        module_roles[module_id] = roles

    # 为所有模块对显式登记 fiber 互连。
    #
    # 论文中的 EML-QCCD 通过 optical fibers 在不同 QCCD 间执行两比特门。
    # 为了让 large-scale 调度严格依赖 machine 中“已登记的链路”，这里对任意
    # 两个模块之间的 optical zones 都建立显式链路；当每个模块存在多个 optical
    # zone 时，登记全部 optical trap 组合，供 gate-level target pair 选择。
    for src_module in range(num_modules):
        src_opts = list(module_roles[src_module]["optical"])
        for dst_module in range(src_module + 1, num_modules):
            dst_opts = list(module_roles[dst_module]["optical"])
            for src_optical in src_opts:
                for dst_optical in dst_opts:
                    m.add_fiber_link(src_module, dst_module, src_optical, dst_optical)

    return m


# ---------------------------------------------------------------------------
# run.py 侧统一调用的入口：第一阶段新增 machine_type 支持
# ---------------------------------------------------------------------------
def build_machine_by_type(machine_type: str, capacity: int, mparams, *, num_qubits: int | None = None):
    """
    根据 machine_type 统一构造机器。

    保证：
    - 旧类型全部兼容；
    - 新类型 G3x4 / G4x5 / EML 可直接在 run.py 中接入；
    - EML 需要 num_qubits 以动态确定模块数。
    """
    mtype = machine_type.upper()

    if mtype == "G2X3" or mtype == "G2x3":
        return test_trap_2x3(capacity, mparams)
    if mtype == "G2X2" or mtype == "G2x2":
        return test_trap_2x2(capacity, mparams)
    if mtype == "L6":
        return make_linear_machine(6, capacity, mparams)
    if mtype == "H6":
        return make_single_hexagon_machine(capacity, mparams)
    if mtype == "G3X4" or mtype == "G3x4":
        return make_qccd_grid_3x4(capacity, mparams)
    if mtype == "G4X5" or mtype == "G4x5":
        return make_qccd_grid_4x5(capacity, mparams)
    if mtype in {"EML", "EML-QCCD", "EML_QCCD", "EML1Z"}:
        if num_qubits is None:
            raise ValueError("EML machine construction requires num_qubits")
        return make_eml_qccd_system(num_qubits, capacity, mparams, num_optical_zones=1)
    if mtype in {"EML2Z", "EML-QCCD-2Z", "EML_QCCD_2Z"}:
        if num_qubits is None:
            raise ValueError("EML2Z machine construction requires num_qubits")
        return make_eml_qccd_system(num_qubits, capacity, mparams, num_optical_zones=2)

    raise ValueError(f"Unsupported machine type '{machine_type}'")
