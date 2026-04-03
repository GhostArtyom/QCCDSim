"""
Large-scale machine topologies

迁移自项目根目录 test_machines.py 中的大规模拓扑函数
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from src.machine.core import Machine


def _is_large_arch(mparams) -> bool:
    return str(getattr(mparams, "architecture_scale", "small")).lower() == "large"


def _is_small_arch(mparams) -> bool:
    return not _is_large_arch(mparams)


# ---------------------------------------------------------------------------
# Large baseline QCCD grid proxy
# ---------------------------------------------------------------------------
def _build_column_backbone_grid(rows: int, cols: int, capacity: int, mparams) -> Machine:
    """
    构造 rows x cols 的 column-backbone grid proxy。

    说明：
    - 该结构与现有 2x3 的设计风格保持一致：每一列有一个 backbone junction，列间 junction 串联。
    - 同一列的多行 trap 都挂在该列 junction 上。
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
# EML-QCCD system builder
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

    # 与现有 2x2 small 拓扑保持一致
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

    # 为所有模块对显式登记 fiber 互连
    for src_module in range(num_modules):
        src_opts = list(module_roles[src_module]["optical"])
        for dst_module in range(src_module + 1, num_modules):
            dst_opts = list(module_roles[dst_module]["optical"])
            for src_optical in src_opts:
                for dst_optical in dst_opts:
                    m.add_fiber_link(src_module, dst_module, src_optical, dst_optical)

    return m


# ---------------------------------------------------------------------------
# 统一入口
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
        from src.machine.topologies.small import test_trap_2x3
        return test_trap_2x3(capacity, mparams)
    if mtype == "G2X2" or mtype == "G2x2":
        from src.machine.topologies.small import test_trap_2x2
        return test_trap_2x2(capacity, mparams)
    if mtype == "L6":
        from src.machine.topologies.small import make_linear_machine
        return make_linear_machine(6, capacity, mparams)
    if mtype == "H6":
        from src.machine.topologies.small import make_single_hexagon_machine
        return make_single_hexagon_machine(capacity, mparams)
    if mtype in {"G3X4", "G3x4"}:
        return make_qccd_grid_3x4(capacity, mparams)
    if mtype in {"G4X5", "G4x5"}:
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
