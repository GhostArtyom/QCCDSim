"""
Small-scale machine topologies

迁移自项目根目录 test_machines.py 中的小规模拓扑函数
"""

from __future__ import annotations

from src.machine.core import Machine


def _is_small_arch(mparams) -> bool:
    return str(getattr(mparams, "architecture_scale", "small")).lower() == "small"


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

    return m


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

    return m


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
