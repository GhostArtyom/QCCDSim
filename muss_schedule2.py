# muss_schedule2.py
# ============================================================
# MUSS Scheduler (V2) —— 论文复现版（兼容修复版）
#
# 设计目标：
# 1) 核心调度逻辑严格保留（frontier + rebalance + LRU + shuttle 链）
# 2) 增强可观测性（shuttle_trace / schedule_events）
# 3) 兼容修复后的 analyzer.py：
#    - [FIX-A1] 给 Split / Move / Merge 事件补充 shuttle_id，
#               使 analyzer 的 aggregate shuttle fidelity 模式可用
# 4) 兼容修复后的 machine.py：
#    - [FIX-M1] 不再对链内 swap 语义做任何额外假设，完全信任 machine.split_time() 返回值
# 5) 保留对时间对齐有帮助的两个修复：
#    - [FIX-1] junction 的时间不应额外 +junction_cross_time（通常折进 move 的距离/速度）
#    - [FIX-2] fire_shuttle 估时时不能把 junction.id 当 segment.id 传给 move_time
#
# 注意：
# - “论文中明确给定的参数”（split_merge_time=80, gate(FM)=40, move_speed=2um/us 等）不在本文件修改
# - MOVE 的物理距离模型由 machine.move_time() 决定（你可用 segment_length_um 等 knob 对齐 1760us / 2320us）
# ============================================================

import networkx as nx
import numpy as np
import collections

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction
from rebalance import *


class MUSSSchedule:
    """
    输入:
      1) ir: gate dependency DAG (networkx DiGraph)
      2) gate_info: gate -> involved qubits（可能是 list，也可能是 dict{qubits/type/...}）
      3) M: machine object
      4) init_map: 初始映射 trap_id -> [ion_ids...]（链顺序）
      5) 串行开关：SerialTrapOps / SerialCommunication / GlobalSerialLock
    """

    def __init__(self, ir, gate_info, M, init_map, serial_trap_ops, serial_comm, global_serial_lock):
        self.ir = ir
        self.gate_info = gate_info
        self.machine = M
        self.init_map = init_map

        # 串行控制开关
        self.SerialTrapOps = serial_trap_ops          # 阱内操作串行
        self.SerialCommunication = serial_comm        # 通信（Split/Move/Merge）串行
        self.GlobalSerialLock = global_serial_lock    # 全系统全串行

        self.schedule = Schedule(M)
        self.router = BasicRoute(M)
        self.gate_finish_times = {}

        # Scheduling statistics
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # ============ 可观测性（调度正确性验证用） ============
        # 论文意义的一次“穿梭/搬运”（一次 fire_shuttle 调用）计数
        self.shuttle_counter = 0

        # shuttle 日志（全过程）
        self.shuttle_log = []

        # 当前 shuttle 上下文（用于 split/move/merge 写入日志）
        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None

        # 是否在运行时实时打印 trace（默认 False，避免刷屏）
        self.enable_runtime_trace_print = False
        # =====================================================

        # -------- 初始化系统状态 MachineState --------
        trap_ions = {}
        seg_ions = {}
        for i in M.traps:
            trap_ions[i.id] = init_map[i.id][:] if init_map.get(i.id, None) else []
        for i in M.segments:
            seg_ions[i.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)

        # 预计算 trap-to-trap 最短路（给 shuttling_direction / lookahead 使用）
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # === MUSS Strict Requirement: LRU Tracking ===
        # 记录每个 ion 最近一次被使用的时间（防 rebalance 随机搬走“刚要用”的离子）
        all_ions = set()
        for t_ions in trap_ions.values():
            all_ions.update(t_ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}

        # rebalance 时保护“当前 gate 涉及的离子”不被驱逐
        self.protected_ions = set()

        # 预计算一个静态拓扑序，用于 tie-breaking（FCFS）
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except Exception:
            self.static_topo_list = list(self.ir.nodes)
        self.static_topo_order = {g: i for i, g in enumerate(self.static_topo_list)}

        # 兼容旧代码：self.gates 给 lookahead 用
        self.gates = self.static_topo_list

    # ==========================================================
    # 观测/导出接口
    # ==========================================================
    def dump_shuttle_trace(self, max_lines=None):
        """
        输出所有 shuttle 的全过程记录：
          shuttle_id / ion / src->dst / route / split/move/merge 时间段
        """
        lines = []
        for rec in self.shuttle_log:
            sid = rec.get("shuttle_id")
            ion = rec.get("ion")
            src = rec.get("src_trap")
            dst = rec.get("dst_trap")
            route_txt = rec.get("route_text", "")
            lines.append(f"[SHUTTLE {sid}] ion={ion}  T{src} -> T{dst}  route={route_txt}")

            steps = rec.get("steps", [])
            for st in steps:
                et = st["etype"]
                stt = st["t_start"]
                edt = st["t_end"]
                desc = st["desc"]
                lines.append(f"    - {et:<5}  ({stt} -> {edt})  {desc}")

        if max_lines is not None:
            lines = lines[:max_lines]
        return "\n".join(lines)

    def dump_schedule_events(self):
        """直接打印 Schedule.events（粗粒度），便于和 analyzer replay 对照。"""
        self.schedule.print_events()

    def _trace_print(self, s):
        if self.enable_runtime_trace_print:
            print(s)

    def _trace_add_step(self, etype, t_start, t_end, desc):
        """
        在当前 shuttle 上下文里追加一个 step。
        只有在 shuttle 过程中才会记录（_current_shuttle_id != None）
        """
        if self._current_shuttle_id is None:
            return
        sid = self._current_shuttle_id
        if sid < 0 or sid >= len(self.shuttle_log):
            return
        self.shuttle_log[sid]["steps"].append(
            {"etype": etype, "t_start": int(t_start), "t_end": int(t_end), "desc": desc}
        )

    def _annotate_last_event_with_shuttle_id(self):
        """
        [FIX-A1]
        给刚刚写入 schedule 的最后一个事件补充 shuttle_id。
        这样 analyzer 在 aggregate 模式下才能把
        Split / Move / Merge 聚合成同一次 shuttle。
        """
        if self._current_shuttle_id is None:
            return
        if not hasattr(self.schedule, "events"):
            return
        if not self.schedule.events:
            return
        try:
            self.schedule.events[-1][4]["shuttle_id"] = self._current_shuttle_id
        except Exception:
            pass

    # ==========================================================
    # Ready time / ion location 推断
    # ==========================================================
    def gate_ready_time(self, gate):
        """根据依赖边，找到 gate 最早可执行时间（所有前驱 gate 结束）。"""
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def ion_ready_info(self, ion_id):
        """
        返回 (该 ion 最近一次操作完成时间, 当前所在 trap_id)。
        并做一致性检查：schedule 推断的位置必须与 sys_state 一致。
        """
        s = self.schedule
        this_ion_ops = s.filter_by_ion(s.events, ion_id)
        this_ion_last_op_time = 0
        this_ion_trap = None

        if len(this_ion_ops):
            # 最后一次必须是 Gate 或 Merge（因为 Split/Move 结束后应 Merge 回 trap 才能 gate）
            assert (this_ion_ops[-1][1] == Schedule.Gate) or (this_ion_ops[-1][1] == Schedule.Merge)
            this_ion_last_op_time = this_ion_ops[-1][3]
            this_ion_trap = this_ion_ops[-1][4]["trap"]
        else:
            # 没有历史事件：从 init_map 里找
            did_not_find = True
            for trap_id in self.init_map.keys():
                if ion_id in self.init_map[trap_id]:
                    this_ion_trap = trap_id
                    did_not_find = False
                    break
            if did_not_find:
                print("Did not find:", ion_id)
            assert did_not_find is False

        # 强一致性检查：schedule 推断位置 vs sys_state
        if this_ion_trap != self.sys_state.find_trap_id_by_ion(ion_id):
            print(ion_id, this_ion_trap, self.sys_state.find_trap_id_by_ion(ion_id))
            self.sys_state.print_state()
            raise AssertionError("ion location mismatch between schedule-inferred and sys_state")

        return this_ion_last_op_time, this_ion_trap

    # ==========================================================
    # 基础操作：Split / Move / Merge / Gate
    # ==========================================================
    def add_split_op(self, clk, src_trap, dest_seg, ion):
        """
        在 src_trap 上把 ion split 到 dest_seg。
        会考虑：
          - Trap 串行（SerialTrapOps）
          - Comm 串行（SerialCommunication）
          - 全局串行（GlobalSerialLock）
        """
        m = self.machine

        # 1) 决定 split_start
        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(src_trap.id)
            split_start = max(clk, last_event_time_on_trap)
        else:
            split_start = clk

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            split_start = max(split_start, last_comm_time)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            split_start = max(split_start, last_event_time_in_system)

        # 2) 计算 split 时间 + swap 信息
        # [FIX-M1] 完全信任 machine.split_time() 的返回，不额外解释 swap 语义
        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops = \
            m.split_time(self.sys_state, src_trap.id, dest_seg.id, ion)

        self.split_swap_counter += split_swap_count
        split_end = split_start + split_duration

        # 3) 写入 schedule
        self.schedule.add_split_or_merge(
            split_start, split_end, [ion],
            src_trap.id, dest_seg.id,
            Schedule.Split,
            split_swap_count, split_swap_hops, i1, i2, ion_swap_hops
        )

        # [FIX-A1] 给该事件补上 shuttle_id
        self._annotate_last_event_with_shuttle_id()

        # 4) trace
        self._trace_add_step(
            "SPLIT", split_start, split_end,
            f"ion {ion}: T{src_trap.id} -> Seg{dest_seg.id} "
            f"(swap_cnt={split_swap_count}, swap_hops={split_swap_hops}, ion_hops={ion_swap_hops}, i1={i1}, i2={i2})"
        )
        self._trace_print(f"[TRACE] SPLIT ion={ion} T{src_trap.id} -> Seg{dest_seg.id} ({split_start}->{split_end})")

        return split_end

    def add_merge_op(self, clk, dest_trap, src_seg, ion):
        """
        把 ion 从 src_seg merge 回 dest_trap。
        """
        m = self.machine

        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(dest_trap.id)
            merge_start = max(clk, last_event_time_on_trap)
        else:
            merge_start = clk

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            merge_start = max(merge_start, last_comm_time)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            merge_start = max(merge_start, last_event_time_in_system)

        merge_end = merge_start + m.merge_time(dest_trap.id)

        self.schedule.add_split_or_merge(
            merge_start, merge_end, [ion],
            dest_trap.id, src_seg.id,
            Schedule.Merge,
            0, 0, 0, 0, 0
        )

        # [FIX-A1] 给该事件补上 shuttle_id
        self._annotate_last_event_with_shuttle_id()

        self._trace_add_step("MERGE", merge_start, merge_end, f"ion {ion}: Seg{src_seg.id} -> T{dest_trap.id}")
        self._trace_print(f"[TRACE] MERGE ion={ion} Seg{src_seg.id} -> T{dest_trap.id} ({merge_start}->{merge_end})")

        return merge_end

    def add_move_op(self, clk, src_seg, dest_seg, junct, ion):
        """
        segment->segment 的 move（通过某个 junction）。
        论文贴合修复：
          [FIX-1] 不再额外叠加 junction_cross_time（junction 时间折进 move 的距离/速度模型里）
        """
        m = self.machine
        move_start = clk

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            move_start = max(move_start, last_event_time_in_system)

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            move_start = max(move_start, last_comm_time)

        # [FIX-1] 这里去掉 + m.junction_cross_time(junct)
        move_end = move_start + m.move_time(src_seg.id, dest_seg.id)

        # junction 交通冲突（同一 junction 同时过车），这是调度冲突约束，不是物理额外时间
        move_start, move_end = self.schedule.junction_traffic_crossing(src_seg, dest_seg, junct, move_start, move_end)

        self.schedule.add_move(move_start, move_end, [ion], src_seg.id, dest_seg.id)

        # [FIX-A1] 给该事件补上 shuttle_id
        self._annotate_last_event_with_shuttle_id()

        self._trace_add_step(
            "MOVE", move_start, move_end,
            f"ion {ion}: Seg{src_seg.id} -> Seg{dest_seg.id} via J{junct.id}"
        )
        self._trace_print(f"[TRACE] MOVE ion={ion} Seg{src_seg.id}->{dest_seg.id} via J{junct.id} ({move_start}->{move_end})")

        return move_end

    def add_gate_op(self, clk, trap_id, gate, ion1, ion2):
        """
        在 trap_id 上执行 2Q gate。
        """
        fire_time = clk
        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(trap_id)
            fire_time = max(clk, last_event_time_on_trap)

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            fire_time = max(fire_time, last_event_time_in_system)

        gate_duration = self.machine.gate_time(self.sys_state, trap_id, ion1, ion2)
        self.schedule.add_gate(fire_time, fire_time + gate_duration, [ion1, ion2], trap_id)
        self.gate_finish_times[gate] = fire_time + gate_duration
        return fire_time + gate_duration

    # ==========================================================
    # Lookahead：决定“谁去找谁”（论文 strict + 未来门预测）
    # ==========================================================
    def get_future_score(self, ion, current_gate_idx, target_trap_id):
        """
        未来代价预测：看接下来若干个 gate 中 ion 的伙伴在哪，
        用 trap-to-trap hop 距离做一个折扣累计。
        """
        score = 0
        gamma = 1.0
        weight = 1.0
        lookahead_depth = 8
        found_count = 0

        total_gates = len(self.gates)
        start_idx = current_gate_idx + 1 if current_gate_idx is not None else 0

        for i in range(start_idx, total_gates):
            if found_count >= lookahead_depth:
                break

            next_gate = self.gates[i]
            if next_gate in self.gate_info:
                gate_data = self.gate_info[next_gate]
                q_list = gate_data if isinstance(gate_data, list) else gate_data["qubits"]

                if ion in q_list:
                    # 1Q gate 不产生跨阱需求
                    if len(q_list) == 1:
                        continue

                    partner = q_list[1] if q_list[0] == ion else q_list[0]
                    _, partner_trap = self.ion_ready_info(partner)

                    dist = self.machine.dist_cache.get((target_trap_id, partner_trap), 10)
                    score += weight * dist
                    weight *= gamma
                    found_count += 1

        return score

    def shuttling_direction(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        """
        决定哪个离子搬到对方阱：
        同时考虑：
          - 当前搬运代价（trap-hop 距离）
          - 未来代价（lookahead）
          - capacity 约束
        """
        m = self.machine
        ALPHA = 1

        if current_gate_idx is None:
            return ion1_trap, ion2_trap

        # Option A: ion1 去 ion2（在 Trap2 相遇）
        cost_current_move1 = m.dist_cache.get((ion1_trap, ion2_trap), 100)
        future_score_1 = self.get_future_score(ion1, current_gate_idx, ion2_trap)
        future_score_2 = self.get_future_score(ion2, current_gate_idx, ion2_trap)
        total_score_meet_at_t2 = cost_current_move1 + ALPHA * (future_score_1 + future_score_2)

        # Option B: ion2 去 ion1（在 Trap1 相遇）
        cost_current_move2 = m.dist_cache.get((ion2_trap, ion1_trap), 100)
        future_score_1_at_t1 = self.get_future_score(ion1, current_gate_idx, ion1_trap)
        future_score_2_at_t1 = self.get_future_score(ion2, current_gate_idx, ion1_trap)
        total_score_meet_at_t1 = cost_current_move2 + ALPHA * (future_score_1_at_t1 + future_score_2_at_t1)

        # capacity 约束：谁那边满了就不能往那边搬
        ss = self.sys_state
        cap1 = m.traps[ion1_trap].capacity - len(ss.trap_ions[ion1_trap])
        cap2 = m.traps[ion2_trap].capacity - len(ss.trap_ions[ion2_trap])

        if cap1 <= 0 and cap2 > 0:
            return ion1_trap, ion2_trap
        if cap2 <= 0 and cap1 > 0:
            return ion2_trap, ion1_trap

        # 选择 total_score 更低的一侧
        if total_score_meet_at_t2 < total_score_meet_at_t1:
            return ion1_trap, ion2_trap
        else:
            return ion2_trap, ion1_trap

    # ==========================================================
    # Shuttle：Split / Move / Merge 链（论文意义的“跨区搬运”）
    # ==========================================================
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        """
        执行一次跨区搬运（论文意义 shuttle）：
          1) route 为空则 router.find_route
          2) 根据路径估计时间 -> identify_start_time
          3) 插入 split/move/merge
          4) sys_state 更新 trap 的离子顺序（保持原逻辑）
        """
        m = self.machine

        # route: Trap/Junction 节点序列
        rpath = route if len(route) else self.router.find_route(src_trap, dest_trap)

        # ============ shuttle 计数 + trace record ============
        shuttle_id = self.shuttle_counter
        self.shuttle_counter += 1

        self._current_shuttle_id = shuttle_id
        self._current_shuttle_route = rpath
        self._current_shuttle_ion = ion
        self._current_shuttle_src = src_trap
        self._current_shuttle_dst = dest_trap

        # route 文本化
        route_txt = []
        for node in rpath:
            if isinstance(node, Trap):
                route_txt.append(f"T{node.id}")
            elif isinstance(node, Junction):
                route_txt.append(f"J{node.id}")
            else:
                route_txt.append(str(node))

        # 统一把 src/dst 存成 trap_id（避免存对象导致打印/对齐混乱）
        src_id = src_trap.id if isinstance(src_trap, Trap) else int(src_trap)
        dst_id = dest_trap.id if isinstance(dest_trap, Trap) else int(dest_trap)

        self.shuttle_log.append(
            {
                "shuttle_id": shuttle_id,
                "ion": ion,
                "src_trap": src_id,
                "dst_trap": dst_id,
                "route_text": "->".join(route_txt),
                "steps": [],
            }
        )
        self._trace_print(f"[TRACE] === SHUTTLE {shuttle_id} START ion={ion} T{src_id}->{dst_id} route={route_txt} ===")
        # =====================================================

        # --------- 估算总搬运用时（仅用于找 earliest feasible start）---------
        # [FIX-2] 不能用 junction.id 当 segment.id 传入 move_time！
        # 正确做法：对 rpath 的每一条 edge 取出对应 segment，然后按 segment hop 估时。
        t_est = 0
        for i in range(len(rpath) - 1):
            u = rpath[i]
            v = rpath[i + 1]
            seg = self.machine.graph[u][v]["seg"]

            if isinstance(u, Trap) and isinstance(v, Junction):
                # Trap->Junction：Split
                t_est += m.mparams.split_merge_time
            elif isinstance(u, Junction) and isinstance(v, Junction):
                # Junction->Junction：Move（按 segment hop 估时）
                # move_time 接口是 (seg1_id, seg2_id)，但当前实现只看 seg2_id 的长度
                t_est += m.move_time(seg.id, seg.id)
            elif isinstance(u, Junction) and isinstance(v, Trap):
                # Junction->Trap：Merge
                t_est += m.merge_time(v.id)

        # identify_start_time：检查路径上相关 segment 的冲突，找最早可开始的 clk
        clk = self.schedule.identify_start_time(rpath, gate_fire_time, t_est)
        clk = self._add_shuttle_ops(rpath, ion, clk)

        # shuttle 结束
        self._trace_print(f"[TRACE] === SHUTTLE {shuttle_id} END at t={clk} ===")

        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None

        return clk

    def _add_shuttle_ops(self, spath, ion, clk):
        """
        保留原逻辑：
          - 找出路径中的 Trap 位置
          - 每段 Trap->...->Trap 做 partial_shuttle
          - 更新 sys_state：从源 trap 删除 ion，并按 orientation 插入目标 trap（保持链方向一致）
        """
        trap_pos = []
        for i in range(len(spath)):
            if type(spath[i]) == Trap:
                trap_pos.append(i)

        for i in range(len(trap_pos) - 1):
            idx0 = trap_pos[i]
            idx1 = trap_pos[i + 1] + 1

            clk = self._add_partial_shuttle_ops(spath[idx0:idx1], ion, clk)

            # sys_state: 从源 trap 删除 ion
            self.sys_state.trap_ions[spath[trap_pos[i]].id].remove(ion)

            # sys_state: 插入到目标 trap，保持方向一致（原逻辑）
            last_junct = spath[trap_pos[i + 1] - 1]
            dest_trap = spath[trap_pos[i + 1]]
            last_seg = self.machine.graph[last_junct][dest_trap]["seg"]
            orient = dest_trap.orientation[last_seg.id]
            if orient == "R":
                self.sys_state.trap_ions[dest_trap.id].append(ion)
            else:
                self.sys_state.trap_ions[dest_trap.id].insert(0, ion)

        return clk

    def _add_partial_shuttle_ops(self, spath, ion, clk):
        """
        partial path 必须是 Trap ... Trap（中间是 Junctions）
        """
        assert len([item for item in spath if type(item) == Trap]) == 2

        seg_list = []
        for i in range(len(spath) - 1):
            u = spath[i]
            v = spath[i + 1]
            seg_list.append(self.machine.graph[u][v]["seg"])

        # split：Trap -> 第一段 segment
        clk = self.add_split_op(clk, spath[0], seg_list[0], ion)

        # move：跨 junction 链（segment -> segment）
        for i in range(len(seg_list) - 1):
            u = seg_list[i]
            v = seg_list[i + 1]
            junct = spath[1 + i]  # spath 的 junction 节点
            clk = self.add_move_op(clk, u, v, junct, ion)

        # merge：最后一段 segment -> Trap
        clk = self.add_merge_op(clk, spath[-1], seg_list[-1], ion)
        return clk

    # ==========================================================
    # MUSS Strict：冲突处理（rebalance）+ LRU eviction
    # ==========================================================
    def rebalance_traps(self, focus_traps, fire_time):
        """
        当两个 focus trap 都满（或都被 block）时，触发 rebalance 清空阻塞。
        """
        m = self.machine
        ss = self.sys_state
        t1 = focus_traps[0]
        t2 = focus_traps[1]
        excess_cap1 = m.traps[t1].capacity - len(ss.trap_ions[t1])
        excess_cap2 = m.traps[t2].capacity - len(ss.trap_ions[t2])
        need_rebalance = False

        ftr = FreeTrapRoute(m, ss)
        status12, route12 = ftr.find_route(t1, t2)
        status21, route21 = ftr.find_route(t2, t1)

        if excess_cap1 == 0 and excess_cap2 == 0:
            need_rebalance = True
        else:
            if status12 == 1 and status21 == 1:
                need_rebalance = True

        if need_rebalance:
            finish_time = self.do_rebalance_traps(fire_time)
            return 1, finish_time
        else:
            return 0, fire_time

    def do_rebalance_traps(self, fire_time):
        """
        通过 RebalanceTraps 计算 flow，然后按 DFS tree 逐条搬运，
        搬运离子选择使用 LRU（最久没用的先搬）。
        """
        self.count_rebalance += 1
        rebal = RebalanceTraps(self.machine, self.sys_state)
        flow_dict = rebal.clear_all_blocks()

        shuttle_graph = nx.DiGraph()
        used_flow = {}
        for i in flow_dict:
            for j in flow_dict[i]:
                if flow_dict[i][j] != 0:
                    shuttle_graph.add_edge(i, j, weight=flow_dict[i][j])
                    used_flow[(i, j)] = 0

        fin_time = fire_time
        for node in shuttle_graph.nodes():
            if shuttle_graph.in_degree(node) == 0 and type(node) == Trap:
                updated_graph = shuttle_graph.copy()
                for edge in used_flow:
                    if used_flow[edge] == updated_graph[edge[0]][edge[1]]["weight"]:
                        updated_graph.remove_edge(edge[0], edge[1])

                T = nx.dfs_tree(updated_graph, source=node)
                shuttle_route = []
                for tnode in T:
                    if T.out_degree(tnode) == 0 and tnode != node:
                        try:
                            shuttle_route = nx.shortest_path(T, node, tnode)
                            break
                        except Exception:
                            continue

                if not shuttle_route:
                    continue

                for i in range(len(shuttle_route) - 1):
                    e0 = shuttle_route[i]
                    e1 = shuttle_route[i + 1]
                    if (e0, e1) in used_flow:
                        used_flow[(e0, e1)] += 1
                    elif (e1, e0) in used_flow:
                        used_flow[(e1, e0)] += 1

                # === LRU eviction：选择最久未使用的离子搬走 ===
                candidates = self.sys_state.trap_ions[node.id]
                valid_candidates = [ion for ion in candidates if ion not in self.protected_ions]
                if not valid_candidates:
                    moving_ion = candidates[0]
                else:
                    moving_ion = min(valid_candidates, key=lambda ion: self.ion_last_used.get(ion, -1))

                ion_time, _ = self.ion_ready_info(moving_ion)
                fire_time = max(fire_time, ion_time)

                fin_time_new = self.fire_shuttle(node.id, tnode.id, moving_ion, fire_time, route=shuttle_route)
                fin_time = max(fin_time, fin_time_new)

        return fin_time


    def _gate_payload(self, gate):
        data = self.gate_info.get(gate, None)
        if data is None:
            return None, [], "unknown"
        if isinstance(data, dict):
            return data, list(data.get("qubits", [])), data.get("type", "unknown")
        return {"qubits": list(data), "type": "cx" if len(data) == 2 else "u"}, list(data), "cx" if len(data) == 2 else "u"

    def _candidate_meeting_traps(self, trap_a, trap_b):
        if not getattr(self.machine.mparams, "enable_partition", False):
            return [trap_a, trap_b]
        try:
            ta = self.machine.get_trap(trap_a)
            tb = self.machine.get_trap(trap_b)
        except Exception:
            return [trap_a, trap_b]
        candidates = []
        seen = set()
        # same-QCCD case: prefer operation zone, then optical, then storage
        qccds = []
        if ta.qccd_id == tb.qccd_id:
            qccds = [ta.qccd_id]
        else:
            qccds = [ta.qccd_id, tb.qccd_id]
        order = {"operation": 0, "optical": 1, "storage": 2}
        for qid in qccds:
            traps = sorted(self.machine.traps_in_qccd(qid), key=lambda tr: (order.get(getattr(tr, "zone_type", "storage"), 9), self.machine.trap_distance(tr.id, trap_a) + self.machine.trap_distance(tr.id, trap_b), tr.id))
            for tr in traps:
                if tr.id not in seen:
                    candidates.append(tr.id)
                    seen.add(tr.id)
        for tid in [trap_a, trap_b]:
            if tid not in seen:
                candidates.append(tid)
        return candidates

    def _choose_partition_target(self, ion1_trap, ion2_trap, ion1, ion2, current_gate_idx):
        candidates = self._candidate_meeting_traps(ion1_trap, ion2_trap)
        best = None
        best_score = None
        for target in candidates:
            free_slots = self.machine.get_trap(target).capacity - len(self.sys_state.trap_ions[target])
            if target not in (ion1_trap, ion2_trap) and free_slots < 2:
                continue
            if target == ion1_trap:
                moving = ion2
                score = self.machine.trap_distance(ion2_trap, target) + self.get_future_score(ion1, current_gate_idx, target) + self.get_future_score(ion2, current_gate_idx, target)
            elif target == ion2_trap:
                moving = ion1
                score = self.machine.trap_distance(ion1_trap, target) + self.get_future_score(ion1, current_gate_idx, target) + self.get_future_score(ion2, current_gate_idx, target)
            else:
                # pick the cheaper ion to move into dedicated operation/optical zone
                cost1 = self.machine.trap_distance(ion1_trap, target)
                cost2 = self.machine.trap_distance(ion2_trap, target)
                if cost1 <= cost2:
                    moving = ion1
                    score = cost1 + self.get_future_score(ion1, current_gate_idx, target) + self.get_future_score(ion2, current_gate_idx, target)
                else:
                    moving = ion2
                    score = cost2 + self.get_future_score(ion1, current_gate_idx, target) + self.get_future_score(ion2, current_gate_idx, target)
            if best_score is None or score < best_score:
                best = (moving, target)
                best_score = score
        return best

    # ==========================================================
    # Gate scheduling：按 frontier 逐个 gate 执行（MUSS strict）
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        调度一个 gate：
          - 1Q gate：原地执行并进入 schedule / analyzer
          - 2Q gate：
              * 同阱：原地执行
              * 异阱：先检查是否需要 rebalance，否则决定方向并 fire_shuttle，再 gate
        """
        gate_data, qubits, gate_type = self._gate_payload(gate)
        if gate_data is None:
            self.gate_finish_times[gate] = self.gate_ready_time(gate)
            return

        self.protected_ions = set(qubits)
        finish_time = 0

        if len(qubits) == 1:
            ion1 = qubits[0]
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            fire_time = max(ready, ion1_time, specified_time)
            duration = self.machine.single_qubit_gate_time(gate_type)
            zone_type = getattr(self.machine.get_trap(ion1_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
            self.schedule.add_gate(fire_time, fire_time + duration, [ion1], ion1_trap, gate_type=gate_type, zone_type=zone_type, gate_id=gate)
            self.gate_finish_times[gate] = fire_time + duration
            finish_time = fire_time + duration
            self.ion_last_used[ion1] = finish_time

        elif len(qubits) == 2:
            ion1, ion2 = qubits[0], qubits[1]
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            ion2_time, ion2_trap = self.ion_ready_info(ion2)
            fire_time = max(ready, ion1_time, ion2_time, specified_time)

            if ion1_trap == ion2_trap:
                gate_duration = self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
                zone_type = getattr(self.machine.get_trap(ion1_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
                self.schedule.add_gate(fire_time, fire_time + gate_duration, [ion1, ion2], ion1_trap, gate_type=gate_type, zone_type=zone_type, gate_id=gate)
                self.gate_finish_times[gate] = fire_time + gate_duration
                finish_time = fire_time + gate_duration
            else:
                rebal_flag, new_fin_time = self.rebalance_traps(focus_traps=[ion1_trap, ion2_trap], fire_time=fire_time)
                if not rebal_flag:
                    meet_choice = self._choose_partition_target(ion1_trap, ion2_trap, ion1, ion2, gate_idx)
                    if meet_choice is not None:
                        moving_ion, dest_trap = meet_choice
                        source_trap = ion1_trap if moving_ion == ion1 else ion2_trap
                    else:
                        source_trap, dest_trap = self.shuttling_direction(ion1_trap, ion2_trap, ion1, ion2, gate_idx)
                        moving_ion = ion1 if source_trap == ion1_trap else ion2
                    clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, fire_time)
                    gate_duration = self.machine.gate_time(self.sys_state, dest_trap, ion1, ion2)
                    zone_type = getattr(self.machine.get_trap(dest_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
                    self.schedule.add_gate(clk, clk + gate_duration, [ion1, ion2], dest_trap, gate_type=gate_type, zone_type=zone_type, gate_id=gate)
                    self.gate_finish_times[gate] = clk + gate_duration
                    finish_time = clk + gate_duration
                else:
                    self.protected_ions = set()
                    self.schedule_gate(gate, specified_time=new_fin_time, gate_idx=gate_idx)
                    return

            self.ion_last_used[ion1] = finish_time
            self.ion_last_used[ion2] = finish_time

        else:
            self.gate_finish_times[gate] = self.gate_ready_time(gate)

        self.protected_ions = set()

    def is_executable_local(self, gate):
        """Helper：gate 是否无需移动即可执行（两比特在同一 trap）。"""
        _, qubits, _ = self._gate_payload(gate)
        if len(qubits) < 2:
            return True
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # ==========================================================
    # 主循环：Frontier scheduling（MUSS strict）
    # ==========================================================
    def run(self):
        """
        MUSS strict 的 frontier 调度：
          - 维护 ready_gates（入度为 0）
          - 优先执行 local gate（无需搬运）
          - tie-breaking 按 static topo order（FCFS）
        """
        in_degree = {n: self.ir.in_degree(n) for n in self.ir.nodes}
        ready_gates = [n for n in self.ir.nodes if in_degree[n] == 0]

        processed_count = 0
        total_gates = len(self.ir.nodes)

        while processed_count < total_gates:
            if not ready_gates:
                break  # cycle or parse error

            local_candidates = []
            remote_candidates = []

            for g in ready_gates:
                if self.is_executable_local(g):
                    local_candidates.append(g)
                else:
                    remote_candidates.append(g)

            # tie-breaking：FCFS by static topo order
            if local_candidates:
                best_gate = min(local_candidates, key=lambda x: self.static_topo_order.get(x, float("inf")))
            else:
                best_gate = min(remote_candidates, key=lambda x: self.static_topo_order.get(x, float("inf")))

            gate_idx = self.static_topo_order.get(best_gate, 0)
            self.schedule_gate(best_gate, gate_idx=gate_idx)

            ready_gates.remove(best_gate)
            processed_count += 1

            for successor in self.ir.successors(best_gate):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_gates.append(successor)
