# test_machines.py
# -*- coding: utf-8 -*-

"""
测试用离子阱机器拓扑集合（Test Machines）

本文件的核心职责：
1) 根据给定的容量 capacity（每个 trap 初始容量）和机器参数 mparams（论文参数/可调参数容器）
2) 构造不同拓扑的 Machine 图结构：Trap / Junction 作为节点，Segment 作为边
3) 返回可用于调度器/Analyzer 的 Machine 对象

重要修复 & 设计原则：
- 【修复】test_trap_2x3() 中 segment_id 重复的问题（原来 6 被用两次），会导致 segments_by_id 覆盖，
         从而影响 move_time / Analyzer 的 segment length 查询，造成实验数据不可信。
- 【必须】所有 Machine 构造都显式传入 mparams，确保 run.py 中设置的参数（速度、长度、alpha_bg 等）真正生效。
- 【统一】所有工厂函数统一签名为 (capacity, mparams)。
"""

from machine import Machine, MachineParams


def _is_large_arch(mparams):
    return str(getattr(mparams, "architecture_scale", "small")).lower() == "large"


def _is_small_arch(mparams):
    return not _is_large_arch(mparams)


# ─────────────────────────────────────────────────────────────
# ISCA / MUSS-TI 论文常用测试拓扑
# ─────────────────────────────────────────────────────────────


def _tag_qccd_roles_2x2(m):
    if _is_small_arch(m.mparams):
        return m
    """Assign paper-aligned zone roles for a 2x2 logical QCCD tile.
    Trap ids: 0/2 storage, 1 operation, 3 optical.
    """
    roles = {
        0: (0, "storage", 0),
        1: (0, "operation", 1),
        2: (0, "storage", 0),
        3: (0, "optical", 2),
    }
    for tid, (qid, zt, zl) in roles.items():
        m.set_trap_role(tid, qid, zt, zl)
    return m


def _tag_qccd_roles_2x3(m):
    if _is_small_arch(m.mparams):
        return m
    """Assign paper-aligned zone roles for a 2x3 small-grid proxy.
    Uses one operation zone, one optical zone, and four storage zones.
    """
    roles = {
        0: (0, "storage", 0),
        1: (0, "operation", 1),
        2: (0, "storage", 0),
        3: (0, "storage", 0),
        4: (0, "optical", 2),
        5: (0, "storage", 0),
    }
    for tid, (qid, zt, zl) in roles.items():
        m.set_trap_role(tid, qid, zt, zl)
    return m


def test_trap_2x3(capacity, mparams):
    """
    2x3 网格：6 个 Trap + 3 个 Junction
    直觉理解：两列三行（或两行三列）trap，通过中间 3 个 junction 串起来
    """
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(6)]
    j = [m.add_junction(i) for i in range(3)]

    # Trap -> Junction 的“支线段”
    # orientation 对 Trap 有意义：表示该 segment 在 Trap 的左(L)/右(R)侧，用于 split 时选链端
    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")

    m.add_segment(3, t[3], j[2], "L")
    m.add_segment(4, t[4], j[1], "L")
    m.add_segment(5, t[5], j[0], "L")

    # Junction 之间的“主干段”
    # 【修复】原文件把 6 用了两次，这里改为 6 和 7
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

    # 4 个 trap
    t = [m.add_trap(i, capacity) for i in range(4)]
    # 2 个 junction
    j = [m.add_junction(i) for i in range(2)]

    # Row 0
    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[3], j[0], "L")

    # Row 1
    m.add_segment(2, t[1], j[1], "R")
    m.add_segment(3, t[2], j[1], "L")

    # Spine (J0 <-> J1)
    m.add_segment(4, j[0], j[1])

    return _tag_qccd_roles_2x2(m) if _is_large_arch(mparams) else m


def make_linear_machine(zones, capacity, mparams):
    """
    线性拓扑：zones 个 trap 串成一条链，每对相邻 trap 通过一个 junction 连接
    每个“相邻 trap 对”对应两个 trap-branch seg：t_i->j_i 与 t_{i+1}->j_i
    """
    m = Machine(mparams)
    traps = [m.add_trap(i, capacity) for i in range(zones)]
    junctions = [m.add_junction(i) for i in range(zones - 1)]

    for i in range(zones - 1):
        # trap_i ---- seg(2i) ---- junction_i ---- seg(2i+1) ---- trap_{i+1}
        m.add_segment(2 * i, traps[i], junctions[i], "R")
        m.add_segment(2 * i + 1, traps[i + 1], junctions[i], "L")

    return m


def make_single_hexagon_machine(capacity, mparams):
    """
    单个六边形：6 trap + 6 junction
    每个 trap 连接自己“右侧”的 junction（seg0..5），同时再连接一个“左侧”的 junction（seg6..11）
    直觉：形成环状结构
    """
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(6)]
    j = [m.add_junction(i) for i in range(6)]

    # trap -> junction (R)
    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")
    m.add_segment(3, t[3], j[3], "R")
    m.add_segment(4, t[4], j[4], "R")
    m.add_segment(5, t[5], j[5], "R")

    # trap -> junction (L) 形成环
    m.add_segment(6, t[0], j[5], "L")
    m.add_segment(7, t[1], j[0], "L")
    m.add_segment(8, t[2], j[1], "L")
    m.add_segment(9, t[3], j[2], "L")
    m.add_segment(10, t[4], j[3], "L")
    m.add_segment(11, t[5], j[4], "L")

    return m


# ─────────────────────────────────────────────────────────────
# 其它拓扑（原文件保留，统一改为显式 mparams）
# ─────────────────────────────────────────────────────────────

def mktrap4x2(capacity, mparams):
    """
    4 traps + 2 junctions：
      (t0,t1) 连到 j0
      (t2,t3) 连到 j1
      j0 <-> j1 主干
    """
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
    """
    4-star：一个中心 junction j0，4 个 trap 全连到它
    """
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
    """
    6 traps + 3 junctions，分三组，每组两个 trap 共用一个 junction，再把 junction 串起来
    """
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
    """
    8 traps + 4 junctions，四组，每组两个 trap -> 一个 junction
    junctions 串成一条线
    """
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
    """
    3x3 trap 网格的一个实现（原文件保留连接方式）
    9 traps + 6 junctions + 若干 segments
    """
    m = Machine(mparams)
    t = [m.add_trap(i, capacity) for i in range(9)]
    j = [m.add_junction(i) for i in range(6)]

    # 上排 3 traps -> 上排 3 junctions
    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")

    # 中排 3 traps -> 下排 3 junctions（注意：这是原文件的连接方式）
    m.add_segment(3, t[3], j[3], "R")
    m.add_segment(4, t[4], j[4], "R")
    m.add_segment(5, t[5], j[5], "R")

    # 中排 traps 也连到上排 junctions（形成网格竖向连接）
    m.add_segment(6, t[3], j[0], "L")
    m.add_segment(7, t[4], j[1], "L")
    m.add_segment(8, t[5], j[2], "L")

    # 下排 traps -> 下排 junctions
    m.add_segment(9, t[6], j[3], "L")
    m.add_segment(10, t[7], j[4], "L")
    m.add_segment(11, t[8], j[5], "L")

    # junction 内部横向连接（两排）
    m.add_segment(12, j[0], j[1])
    m.add_segment(13, j[1], j[2])
    m.add_segment(14, j[3], j[4])
    m.add_segment(15, j[4], j[5])

    return m


def make_9trap(capacity, mparams):
    """
    9 traps + 9 junctions 的较复杂网格/路由拓扑（原文件保留连接关系）

    【关键修复】
    原文件使用旧构造：Machine(alpha=..., inter_ion_dist=..., split_factor=..., move_factor=...)
    这在你新版 Machine(mparams) 下不兼容。
    现在统一改为：m = Machine(mparams)

    如果你还想保留 split_factor/move_factor 作为“论文未定义可调参数”，
    建议把它们挂到 mparams 上（例如 mparams.split_factor / mparams.move_factor），
    然后由 Analyzer 或调度器读取使用。
    """
    m = Machine(mparams)

    t = [m.add_trap(i, capacity) for i in range(9)]
    j = [m.add_junction(i) for i in range(9)]

    # 原始连边保持不动（只补 orientation，避免默认 L 带来不可控偏差）
    m.add_segment(0, t[0], j[0], "R")
    m.add_segment(1, t[1], j[1], "R")
    m.add_segment(2, t[2], j[2], "R")

    m.add_segment(3, t[3], j[2], "L")
    m.add_segment(4, t[4], j[5], "R")
    m.add_segment(5, t[5], j[8], "R")

    m.add_segment(6, t[6], j[8], "L")
    m.add_segment(7, t[7], j[7], "R")
    m.add_segment(8, t[8], j[6], "R")

    # junction-junction edges（主干与网格连通）
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
