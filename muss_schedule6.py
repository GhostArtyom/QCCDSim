# muss_schedule6_strict.py
# ============================================================
# MUSS Scheduler (V6 Strict) —— 论文 Section 3.2 严格小规模模式
#
# 适用范围：
# 1) 本文件只对应论文 Section 3.2 的 strict small-scale mode。
# 2) 目标是尽量贴近论文 3.2 的主干语义：frontier、prioritize executable gates、
#    first-come-first-served、multi-level scheduling、LRU conflict handling。
# 3) 本文件故意不实现论文 3.3 的 SWAP insertion / look-ahead，也不声称覆盖论文
#    全文的完整 MUSS-TI 流程；跨 QCCD / optical 相关扩展仍不在本文件范围内。
#
# 设计原则：
# 1) 目标 trap（论文语义中的目标执行 zone）先按“available + closest-in-level + distance”
#    独立选择，不把 route legality 混入目标选择评分。
# 2) 在目标 trap 选定后，再单独检查 routing 合法性；若当前目标不可达，则按论文式流程
#    顺延到下一个候选目标，而不是在评分阶段提前掺入路由偏好。
# 3) conflict handling 只围绕“已选定的目标执行 trap”展开，采用局部 LRU 驱逐；
#    不使用全局 flow rebalance，也不对 source trap 做额外工程化疏通。
# 4) 保留 shuttle_id、1Q 延后插入、与现有 analyzer 兼容的事件语义；除与论文对齐
#    直接相关之处外，未修改的既有实现全部保留。
# 5) 针对论文图 4 中“链边搬运前需要 SWAP 重排链内顺序”的表述，
#    本版仅在调度器层将该动作显式化为可观测的重排计划与事件注释；
#    不新增 Schedule 事件类型，因此不需要联动修改 analyzer / schedule.py。
#
# 说明：
# - 论文第 3.2 节强调：当目标 zone 已满时，采用 multi-level scheduling 选择新的安置 zone，
#   并用 LRU 驱逐长期未使用离子，而不是做全局网络流式 rebalance。
# - 因此本版在 V6 基础上进一步收紧：去掉 endpoint_bias，并把“target selection”与
#   “route legality check”严格拆成两步，使流程更贴近论文图 3 / 图 4。
# ============================================================

import networkx as nx
import numpy as np
import collections

from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction

# 注意：
# 这里故意不再导入 rebalance.py。
# 这是本文件最关键的“干净消融”改动之一。
# from rebalance import *


class MUSSSchedule:
    """
    输入:
      1) ir: gate dependency DAG (networkx DiGraph)
      2) gate_info: gate -> involved qubits（可能是 list，也可能是 dict{qubits/type/...}）
      3) M: machine object
      4) init_map: 初始映射 trap_id -> [ion_ids...]（链顺序）
      5) 串行开关：SerialTrapOps / SerialCommunication / GlobalSerialLock
    """

    def __init__(
        self,
        ir_or_parse,
        gate_info_or_machine,
        M_or_init_map,
        init_map_or_serial_trap_ops,
        serial_trap_ops=None,
        serial_comm=None,
        global_serial_lock=None,
    ):
        """
        新接口（论文复现主接口）:
            MUSSSchedule(parse_obj, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)

        兼容旧接口:
            MUSSSchedule(ir, gate_info, machine, init_map, serial_trap_ops, serial_comm, global_serial_lock)
        """
        if hasattr(ir_or_parse, "all_gate_map") and hasattr(ir_or_parse, "gate_graph"):
            parse_obj = ir_or_parse
            self.parse_obj = parse_obj
            self.full_ir = parse_obj.gate_graph
            self.ir = getattr(parse_obj, "twoq_gate_graph", parse_obj.gate_graph)
            self.gate_info = parse_obj.all_gate_map
            self.machine = gate_info_or_machine
            self.init_map = M_or_init_map
            self.SerialTrapOps = init_map_or_serial_trap_ops
            self.SerialCommunication = serial_trap_ops
            self.GlobalSerialLock = global_serial_lock
        else:
            self.parse_obj = None
            self.full_ir = ir_or_parse
            self.ir = ir_or_parse
            self.gate_info = gate_info_or_machine
            self.machine = M_or_init_map
            self.init_map = init_map_or_serial_trap_ops
            self.SerialTrapOps = serial_trap_ops
            self.SerialCommunication = serial_comm
            self.GlobalSerialLock = global_serial_lock

        self.architecture_scale = str(
            getattr(self.machine.mparams, "architecture_scale", "small")
        ).lower()
        self.is_small_mode = self.architecture_scale == "small"
        self.is_large_mode = not self.is_small_mode

        self.schedule = Schedule(self.machine)

        # 主路径使用 capacity-aware router；BasicRoute 仅保留作调试兜底
        self.basic_router = BasicRoute(self.machine)
        self.router = None  # 在 sys_state 初始化之后绑定为 FreeTrapRoute
        self.gate_finish_times = {}

        # 调度统计信息
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # ============ 可观测性（调度正确性验证用） ============
        self.shuttle_counter = 0
        self.shuttle_log = []
        # 显式记录“链边搬运前的链内 SWAP 重排”
        # 说明：为了保持与现有 Schedule / Analyzer 兼容，这里不新增新的事件类型，
        # 而是把该动作显式规划、显式记入 trace / metadata / log。
        self.chain_reorder_log = []

        self._current_shuttle_id = None
        self._current_shuttle_route = None
        self._current_shuttle_ion = None
        self._current_shuttle_src = None
        self._current_shuttle_dst = None

        self.enable_runtime_trace_print = False
        # =====================================================

        # -------- 初始化系统状态 MachineState --------
        trap_ions = {}
        seg_ions = {}
        for i in self.machine.traps:
            trap_ions[i.id] = self.init_map[i.id][:] if self.init_map.get(i.id, None) else []
        for i in self.machine.segments:
            seg_ions[i.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)

        # 绑定 capacity-aware router
        self.router = FreeTrapRoute(self.machine, self.sys_state)

        # 预计算 trap-to-trap 最短路（给 endpoint 目标选择与局部驱逐排序使用）
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # === MUSS Strict Requirement: LRU Tracking ===
        all_ions = set()
        for t_ions in trap_ions.values():
            all_ions.update(t_ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}

        # =====================================================
        # 方案二：short-term sticky meeting trap（仅用于小规模 3.2 严格版的轻量 tie-break）
        #
        # 设计目标：
        #   - 不改变论文 3.2 的主流程：frontier / FCFS / target selection / local LRU conflict handling；
        #   - 只在“available + level”同级候选内部，增加一个非常轻量的“最近工作区复用”偏好，
        #     以缓解 G2x3 上中心 qubit / 当前工作集在多个 meeting trap 之间来回震荡的问题；
        #   - 不引入 3.3 look-ahead，不引入全局优化器，不引入 rebalance。
        #
        # 语义：
        #   - 当某个 ion 刚在某个 remote meeting trap 上完成一次 2Q gate，
        #     且后续很短的一段时间内它仍然活跃时，
        #     在多个同优先级候选 target trap 中，优先继续复用这个最近工作 trap。
        #
        # 注意：
        #   - 这里只是 tie-break，不替代论文的 available / closest-in-level / distance；
        #   - 为避免“黏住”过久，只保留一个很短的时间窗口。
        # =====================================================
        self.recent_remote_trap_by_ion = {}
        self.recent_remote_time_by_ion = {}
        self.sticky_trap_horizon_us = int(
            getattr(self.machine.mparams, "sticky_trap_horizon_us", 600)
        )

        # 保护“当前 gate 涉及的离子”不被驱逐
        # 注意：在本无-rebalance版本中，这个集合仍保留，
        # 因为它属于原有调度语义的一部分，也便于后续一致性扩展。
        self.protected_ions = set()

        # 2Q scheduling order (paper-faithful MUSS)
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except Exception:
            self.static_topo_list = list(self.ir.nodes)
        self.static_topo_order = {g: i for i, g in enumerate(self.static_topo_list)}
        self.gates = self.static_topo_list

        # Full-program order kept for delayed 1Q scheduling / timing replay
        try:
            self.full_topo_list = list(nx.topological_sort(self.full_ir))
        except Exception:
            self.full_topo_list = list(self.full_ir.nodes)
        self.full_topo_order = {g: i for i, g in enumerate(self.full_topo_list)}

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

    def _estimate_chain_reorder_duration(self, split_duration):
        """
        估计“链内 SWAP 重排”在 split 总时长中所占的窗口。

        这里不改机器层时间模型，只做调度层显式化：
          - split_duration 仍完全由 machine.split_time(...) 决定；
          - 本函数只把其中“除去一次基础 split/merge 动作后的剩余部分”
            解释为链内重排成本，用于 trace / metadata / log。

        这样可以做到：
          1) 不改总时序；
          2) 不要求 analyzer / schedule.py / machine.py 联动修改；
          3) 让论文图 4 中的“先重排、再从链边搬运”在调度器层变成显式动作。
        """
        base_split = int(getattr(self.machine.mparams, "split_merge_time", 0))
        return max(0, int(split_duration) - base_split)

    def _plan_chain_edge_reorder(self, src_trap, dest_seg, ion):
        """
        显式构造“链边搬运前的链内重排计划”。

        说明：
        - 论文图 4 说明 shuttle 只能从链边发生；若目标离子不在链边，
          需要先通过 SWAP 重排链顺序。
        - 现有机器层已经通过 machine.split_time(...) 计算了这部分代价与统计量；
          本函数不改变物理时间模型，只把它提升为调度器层的显式计划对象。

        返回字段：
          {
            "required": 是否需要链内重排,
            "split_duration": split 总时长,
            "reorder_duration": 其中可解释为链内重排的时间窗口,
            "split_only_duration": 纯 split 剩余时间窗口,
            "swap_count": 链内交换次数,
            "swap_hops": 链内交换 hop 数,
            "ion_swap_hops": 目标离子因重排跨越的 hop 数,
            "edge_ion_left": split_time 返回的辅助离子标识 i1,
            "edge_ion_right": split_time 返回的辅助离子标识 i2,
          }
        """
        m = self.machine
        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops =             m.split_time(self.sys_state, src_trap.id, dest_seg.id, ion)
        reorder_duration = self._estimate_chain_reorder_duration(split_duration)
        split_only_duration = max(0, int(split_duration) - int(reorder_duration))
        return {
            "required": bool(split_swap_count > 0 or ion_swap_hops > 0 or reorder_duration > 0),
            "split_duration": int(split_duration),
            "reorder_duration": int(reorder_duration),
            "split_only_duration": int(split_only_duration),
            "swap_count": int(split_swap_count),
            "swap_hops": int(split_swap_hops),
            "ion_swap_hops": int(ion_swap_hops),
            "edge_ion_left": i1,
            "edge_ion_right": i2,
        }

    def _log_chain_reorder(self, split_start, split_end, src_trap, dest_seg, ion, plan):
        """
        在调度器层显式记录一次链内重排。

        这里不新增 Schedule 事件类型；
        但会把该动作：
          1) 记入 chain_reorder_log；
          2) 以 CHAIN_REORDER step 的形式写入 shuttle trace；
          3) 作为 metadata 挂到紧随其后的 Split 事件上。
        """
        if not plan.get("required", False):
            return

        reorder_start = int(split_start)
        reorder_end = min(int(split_end), int(split_start) + int(plan.get("reorder_duration", 0)))

        rec = {
            "shuttle_id": self._current_shuttle_id,
            "ion": ion,
            "src_trap": src_trap.id,
            "dest_seg": dest_seg.id,
            "t_start": reorder_start,
            "t_end": reorder_end,
            "swap_count": int(plan.get("swap_count", 0)),
            "swap_hops": int(plan.get("swap_hops", 0)),
            "ion_swap_hops": int(plan.get("ion_swap_hops", 0)),
            "edge_ion_left": plan.get("edge_ion_left", None),
            "edge_ion_right": plan.get("edge_ion_right", None),
        }
        self.chain_reorder_log.append(rec)

        self._trace_add_step(
            "CHAIN_REORDER",
            reorder_start,
            reorder_end,
            (
                f"ion {ion}: 在 T{src_trap.id} 内为链边搬运做显式重排 "
                f"(swap_cnt={plan.get('swap_count', 0)}, "
                f"swap_hops={plan.get('swap_hops', 0)}, "
                f"ion_hops={plan.get('ion_swap_hops', 0)}, "
                f"edge_hint=({plan.get('edge_ion_left', None)}, {plan.get('edge_ion_right', None)}))"
            ),
        )
        self._trace_print(
            f"[TRACE] CHAIN_REORDER ion={ion} T{src_trap.id} "
            f"({reorder_start}->{reorder_end}) swaps={plan.get('swap_count', 0)}"
        )

    def _annotate_last_event_with_chain_reorder(self, plan):
        """
        把“链边搬运前链内重排”的显式计划挂到刚写入的 Split 事件 metadata 上。

        这样做的目的：
          - schedule.py / analyzer.py 无需修改；
          - 调度层仍然可以把图 4 的重排行为显式暴露出来；
          - 后续若需要单独分析，可直接从事件 metadata 提取。
        """
        if not hasattr(self.schedule, "events"):
            return
        if not self.schedule.events:
            return
        try:
            meta = self.schedule.events[-1][4]
            meta["chain_reorder_required"] = bool(plan.get("required", False))
            meta["chain_reorder_duration"] = int(plan.get("reorder_duration", 0))
            meta["chain_reorder_swap_count"] = int(plan.get("swap_count", 0))
            meta["chain_reorder_swap_hops"] = int(plan.get("swap_hops", 0))
            meta["chain_reorder_ion_swap_hops"] = int(plan.get("ion_swap_hops", 0))
            meta["chain_reorder_edge_ion_left"] = plan.get("edge_ion_left", None)
            meta["chain_reorder_edge_ion_right"] = plan.get("edge_ion_right", None)
        except Exception:
            pass

    # ==========================================================
    # Ready time / ion location 推断
    # ==========================================================
    def gate_ready_time(self, gate):
        """根据 2Q-only DAG 依赖边，找到 gate 最早可执行时间（给 MUSS 2Q 调度使用）。"""
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def gate_ready_time_full(self, gate):
        """根据完整 DAG 依赖边，找到 gate 最早可执行时间（给延后插入的 1Q gate 使用）。"""
        ready_time = 0
        for in_edge in self.full_ir.in_edges(gate):
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
    # 容量 / 路由辅助
    # ==========================================================
    def _trap_has_free_slot(self, trap_id, incoming=1):
        cur = len(self.sys_state.trap_ions[trap_id])
        cap = self.machine.traps[trap_id].capacity
        return (cur + incoming) <= cap

    def _find_route_or_none(self, source_trap, dest_trap):
        """
        统一包装 FreeTrapRoute：
          - 返回合法路径则给出 route
          - 若被 block，则返回 None
        """
        status, route = self.router.find_route(source_trap, dest_trap)
        if status == 0:
            return route
        return None

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

        self._annotate_last_event_with_shuttle_id()

        self._trace_add_step("MERGE", merge_start, merge_end, f"ion {ion}: Seg{src_seg.id} -> T{dest_trap.id}")
        self._trace_print(f"[TRACE] MERGE ion={ion} Seg{src_seg.id} -> T{dest_trap.id} ({merge_start}->{merge_end})")

        return merge_end

    def add_move_op(self, clk, src_seg, dest_seg, junct, ion):
        """
        segment->segment 的 move（通过某个 junction）。
        论文贴合修复：
          junction 的阻塞由 junction_traffic_crossing 处理；
          不额外叠加 junction_cross_time，避免重复记时。
        """
        m = self.machine
        move_start = clk

        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            move_start = max(move_start, last_event_time_in_system)

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            move_start = max(move_start, last_comm_time)

        move_end = move_start + m.move_time(src_seg.id, dest_seg.id)

        # junction 交通冲突（同一 junction 同时过车），这是调度冲突约束，不是物理额外时间
        move_start, move_end = self.schedule.junction_traffic_crossing(src_seg, dest_seg, junct, move_start, move_end)

        self.schedule.add_move(move_start, move_end, [ion], src_seg.id, dest_seg.id)
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
    # 小规模严格版目标选择：不使用 look-ahead
    # ==========================================================
    def _current_move_cost(self, source_trap, dest_trap):
        """
        小规模严格版只使用当前搬运代价，不看未来门。

        这里直接复用机器预计算的 trap-to-trap 距离；
        若缓存里没有，则退化为一个较大的默认值。
        """
        if source_trap == dest_trap:
            return 0
        return self.machine.dist_cache.get((source_trap, dest_trap), 100)

    # ==========================================================
    # Shuttle：Split / Move / Merge 链（论文意义的“跨区搬运”）
    # ==========================================================
    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        """
        执行一次跨区搬运（论文意义 shuttle）：
          1) route 为空则使用 capacity-aware route
          2) 根据路径估计时间 -> identify_start_time
          3) 插入 split/move/merge
          4) sys_state 更新 trap 的离子顺序（保持原逻辑）
        """
        m = self.machine

        if len(route):
            rpath = route
        else:
            rpath = self._find_route_or_none(src_trap, dest_trap)
            if rpath is None:
                raise RuntimeError(f"No legal route found for shuttle: T{src_trap} -> T{dest_trap}")

        # ============ shuttle 计数 + trace record ============
        shuttle_id = self.shuttle_counter
        self.shuttle_counter += 1

        self._current_shuttle_id = shuttle_id
        self._current_shuttle_route = rpath
        self._current_shuttle_ion = ion
        self._current_shuttle_src = src_trap
        self._current_shuttle_dst = dest_trap

        route_txt = []
        for node in rpath:
            if isinstance(node, Trap):
                route_txt.append(f"T{node.id}")
            elif isinstance(node, Junction):
                route_txt.append(f"J{node.id}")
            else:
                route_txt.append(str(node))

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

        # --------- 估算总搬运用时（仅用于找 earliest feasible start）---------
        t_est = 0
        for i in range(len(rpath) - 1):
            u = rpath[i]
            v = rpath[i + 1]
            seg = self.machine.graph[u][v]["seg"]

            if isinstance(u, Trap) and isinstance(v, Junction):
                t_est += m.mparams.split_merge_time
            elif isinstance(u, Junction) and isinstance(v, Junction):
                t_est += m.move_time(seg.id, seg.id)
            elif isinstance(u, Junction) and isinstance(v, Trap):
                t_est += m.merge_time(v.id)

        clk = self.schedule.identify_start_time(rpath, gate_fire_time, t_est)
        clk = self._add_shuttle_ops(rpath, ion, clk)

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

            self.sys_state.trap_ions[spath[trap_pos[i]].id].remove(ion)

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

        clk = self.add_split_op(clk, spath[0], seg_list[0], ion)

        for i in range(len(seg_list) - 1):
            u = seg_list[i]
            v = seg_list[i + 1]
            junct = spath[1 + i]
            clk = self.add_move_op(clk, u, v, junct, ion)

        clk = self.add_merge_op(clk, spath[-1], seg_list[-1], ion)
        return clk

    # ==========================================================
    # MUSS Strict：冲突处理（rebalance）接口占位
    # ==========================================================
    def rebalance_traps(self, focus_traps, fire_time):
        """
        V6 中不再使用全局 rebalance。

        保留该接口仅为了兼容旧调用链与外部统计脚本。
        真正的论文式 conflict handling 已下沉到：
          _prepare_meeting_trap() / _ensure_space_on_trap()
        也就是“先选目标执行 zone，再做局部 LRU 驱逐”。
        """
        return 0, fire_time

    def do_rebalance_traps(self, fire_time):
        """
        兼容占位接口。V6 不走全局 rebalance 路径。
        """
        return fire_time

    def _gate_payload(self, gate):
        data = self.gate_info.get(gate, None)
        if data is None:
            return None, [], "unknown"
        if isinstance(data, dict):
            return data, list(data.get("qubits", [])), data.get("type", "unknown")
        return {"qubits": list(data), "type": "cx" if len(data) == 2 else "u"}, list(data), "cx" if len(data) == 2 else "u"

    def _trap_can_execute_twoq(self, trap_id):
        """
        判断某个 trap 是否允许执行 2Q gate。

        兼容策略：
          1) 若 trap 显式给出了 can_execute_2q / supports_twoq，则严格使用。
          2) 否则按论文小规模复现的保守口径，默认所有 trap 都可执行本地 2Q gate。
        """
        if not hasattr(self.machine, "get_trap"):
            return True
        try:
            trap = self.machine.get_trap(trap_id)
        except Exception:
            return True

        if hasattr(trap, "can_execute_2q"):
            return bool(getattr(trap, "can_execute_2q"))
        if hasattr(trap, "supports_twoq"):
            return bool(getattr(trap, "supports_twoq"))
        return True

    def _preferred_sticky_traps(self, ion1, ion2, fire_time):
        """
        返回当前 gate 可优先复用的“最近工作 trap”集合。

        规则：
          1) 只看最近一次 remote 2Q gate 留下的 meeting trap；
          2) 只保留时间上仍在短窗口内的记录；
          3) 若两个参与离子都带有最近工作 trap，则都纳入候选集合。

        这不是论文外的新目标函数，只是小规模 3.2 严格版中的轻量 tie-break 信息。
        """
        preferred = set()
        horizon = int(getattr(self, "sticky_trap_horizon_us", 0))

        for ion in [ion1, ion2]:
            if ion not in self.recent_remote_trap_by_ion:
                continue
            last_t = self.recent_remote_time_by_ion.get(ion, None)
            if last_t is None:
                continue
            if horizon > 0 and (int(fire_time) - int(last_t)) > horizon:
                continue
            preferred.add(self.recent_remote_trap_by_ion[ion])

        return preferred

    def _record_recent_remote_meeting(self, ion1, ion2, trap_id, finish_time):
        """
        记录一次 remote 2Q gate 刚刚使用过的 meeting trap。

        这里只在“原先两颗离子不在同一 trap，需要 remote 会合”的路径中调用；
        对完全本地的 2Q gate 不更新该状态，避免把普通 local trap 误当成工作区锚点。
        """
        self.recent_remote_trap_by_ion[ion1] = trap_id
        self.recent_remote_trap_by_ion[ion2] = trap_id
        self.recent_remote_time_by_ion[ion1] = int(finish_time)
        self.recent_remote_time_by_ion[ion2] = int(finish_time)

    def _candidate_meeting_traps(self, ion1_trap, ion2_trap, ion1, ion2, fire_time):
        """
        返回论文 3.2 语义下的“目标执行 trap 候选列表”。

        这里严格只做“目标选择”本身，不掺入 route legality。
        也就是说，本函数仍按论文文字里明确出现的偏好排序：

          1) available：当前目标 trap 是否能直接容纳本门所需离子数
          2) closest in level：目标 trap 的 level 是否更接近当前两颗离子所在层级
          3) （轻量 sticky tie-break）若本 gate 的参与离子最近刚在某个 meeting trap 上完成 remote gate，
             则在同级候选中优先继续复用这个最近工作 trap
          4) distance：当前两颗离子搬运到目标 trap 的总图距离是否更小
          5) trap id：仅作稳定 tie-break，避免非确定性

        注意：
        - 这里显式去掉 endpoint_bias。论文没有“同分优先 endpoint”的表述，
          因此严格版不再保留这条工程偏好。
        - sticky 偏好只是一层轻量 tie-break，不替代论文主排序项；
          它的作用是缓解 G2x3 上中心工作区在多个 meeting trap 间震荡。
        - route 是否可达不在这里判断；论文流程更接近“先选目标，再做 routing”。
        """
        src_level_ref = max(self._trap_level(ion1_trap), self._trap_level(ion2_trap))
        preferred_sticky_traps = self._preferred_sticky_traps(ion1, ion2, fire_time)
        candidates = []

        for trap in self.machine.traps:
            tid = trap.id
            if not self._trap_can_execute_twoq(tid):
                continue

            # 当前 2Q gate 最终需要在目标 trap 内同时容纳两颗参与门的离子。
            # 因此容量不足 2 的 trap 不可能成为目标执行位置。
            if getattr(trap, "capacity", 0) < 2:
                continue

            incoming_needed = 0
            if ion1_trap != tid:
                incoming_needed += 1
            if ion2_trap != tid:
                incoming_needed += 1

            available_penalty = 0 if self._trap_has_free_slot(tid, incoming=incoming_needed) else 1
            level_gap = abs(self._trap_level(tid) - src_level_ref)
            dist_sum = (
                self.machine.dist_cache.get((ion1_trap, tid), 10 ** 6)
                + self.machine.dist_cache.get((ion2_trap, tid), 10 ** 6)
            )
            sticky_penalty = 0 if tid in preferred_sticky_traps else 1

            # 关键点：
            #   - available / level 仍然是论文主导项；
            #   - sticky 只在同级候选中发挥“最近工作区复用”的 tie-break 作用；
            #   - 再之后才比较当前 distance。
            candidates.append((available_penalty, level_gap, sticky_penalty, dist_sum, tid))

        candidates.sort()
        return [x[-1] for x in candidates]

    def _build_move_plan_for_target(self, ion1_trap, ion2_trap, ion1, ion2, target_trap):
        """
        为选定 target trap 构造严格、确定性的搬运计划。

        返回字段：
          plan: [(moving_ion, source_trap, target_trap), ...]
          incoming_needed: target 当前还需接收多少个离子
          total_move_cost: 当前总代价（不含未来门）

        若 target 为某个 endpoint，则只需搬运另一个离子。
        若 target 为第三方 trap，则两颗离子都要搬进去。

        为避免实现歧义，双离子搬运的顺序固定为：
          1) 按离子当前 ready time 更早者优先
          2) 若相同，则按当前搬运距离更长者优先
          3) 若仍相同，则按 ion id 更小者优先
        """
        plan = []
        total_move_cost = 0
        incoming_needed = 0

        if ion1_trap != target_trap:
            total_move_cost += self._current_move_cost(ion1_trap, target_trap)
            incoming_needed += 1
            plan.append((ion1, ion1_trap, target_trap))

        if ion2_trap != target_trap:
            total_move_cost += self._current_move_cost(ion2_trap, target_trap)
            incoming_needed += 1
            plan.append((ion2, ion2_trap, target_trap))

        if len(plan) <= 1:
            return {
                "plan": plan,
                "incoming_needed": incoming_needed,
                "total_move_cost": total_move_cost,
            }

        ordered = []
        for moving_ion, src, dst in plan:
            ready_t, _ = self.ion_ready_info(moving_ion)
            move_cost = self._current_move_cost(src, dst)
            ordered.append((ready_t, -move_cost, moving_ion, (moving_ion, src, dst)))
        ordered.sort()
        plan = [x[-1] for x in ordered]

        return {
            "plan": plan,
            "incoming_needed": incoming_needed,
            "total_move_cost": total_move_cost,
        }

    def _routes_exist_for_target(self, move_plan):
        """
        检查在当前系统状态下，给定 move_plan 所需的每一条搬运路径是否至少存在一条合法 route。

        注意：
        - 该函数只做“路径合法性验证”，不参与目标评分。
        - 这里不做额外 source/destination 疏通；论文严格版的 conflict handling
          只允许围绕“已选中的目标执行 zone”展开。
        """
        for _, source_trap, dest_trap in move_plan:
            if source_trap == dest_trap:
                continue
            route = self._find_route_or_none(source_trap, dest_trap)
            if route is None:
                return False
        return True

    def _choose_partition_target(self, ion1_trap, ion2_trap, ion1, ion2, fire_time, current_gate_idx):
        """
        严格版目标选择：只做“论文 3.2 里的目标执行 trap 选择”。

        本函数不检查 route legality，只返回按论文偏好排序后的候选目标及其搬运计划。
        这样可以把流程明确拆成两步：
          第一步：按 available + closest-in-level + sticky working-area tie-break + current distance 选择 target
          第二步：对选中的 target 再单独检查 routing 是否可行

        返回：
          [
            {
              "target_trap": ...,
              "plan": [(moving_ion, source_trap, target_trap), ...],
              "incoming_needed": ...,
              "total_move_cost": ...,
            },
            ...
          ]
        """
        ordered_choices = []
        src_level_ref = max(self._trap_level(ion1_trap), self._trap_level(ion2_trap))

        for target in self._candidate_meeting_traps(
            ion1_trap, ion2_trap, ion1, ion2, current_gate_idx
        ):
            plan_info = self._build_move_plan_for_target(
                ion1_trap, ion2_trap, ion1, ion2, target
            )
            move_plan = plan_info["plan"]
            incoming_needed = plan_info["incoming_needed"]
            total_move_cost = plan_info["total_move_cost"]

            available_penalty = 0 if self._trap_has_free_slot(target, incoming=incoming_needed) else 1
            level_gap = abs(self._trap_level(target) - src_level_ref)

            ordered_choices.append(
                (
                    available_penalty,
                    level_gap,
                    total_move_cost,
                    target,
                    {
                        "target_trap": target,
                        "plan": move_plan,
                        "incoming_needed": incoming_needed,
                        "total_move_cost": total_move_cost,
                    },
                )
            )

        ordered_choices.sort()
        return [item[-1] for item in ordered_choices]

    def _select_reachable_target(self, ordered_choices):
        """
        在“已按论文规则排好序”的目标候选中，顺序检查 route legality。

        这是对论文图 3 / 图 4 更贴合的实现：
          1) 先选目标
          2) 再做 routing
          3) 若当前目标不可达，则顺延到下一个候选目标

        注意：
        - 这里不对 source trap 做额外 relief，也不把 route 可达性反向掺入目标评分。
        - route legality 的职责仅仅是“验证当前已选目标是否可执行”。
        """
        for choice in ordered_choices:
            if self._routes_exist_for_target(choice["plan"]):
                return choice
        return None

    def _trap_zone_type(self, trap_id):
        """
        获取 trap 的 zone_type。

        优先使用机器对象里已有的显式标注；
        若小规模机器没有提供该字段，则退化为 storage。
        """
        if not hasattr(self.machine, "get_trap"):
            return "storage"
        try:
            trap = self.machine.get_trap(trap_id)
        except Exception:
            return "storage"

        zone_type = getattr(trap, "zone_type", None)
        if zone_type is not None:
            return str(zone_type)

        # 兼容少量旧机器对象可能只暴露了 level / zone_level 的情况
        lvl = getattr(trap, "zone_level", getattr(trap, "level", None))
        if lvl is None:
            return "storage"
        if int(lvl) >= 2:
            return "optical"
        if int(lvl) >= 1:
            return "operation"
        return "storage"

    def _zone_level(self, zone_type):
        """
        论文中的 multi-level 语义：
          storage = level 0
          operation = level 1
          optical = level 2
        未知类型默认按 storage 处理。
        """
        return {"storage": 0, "operation": 1, "optical": 2}.get(str(zone_type), 0)

    def _trap_level(self, trap_id):
        return self._zone_level(self._trap_zone_type(trap_id))

    def _candidate_traps_for_eviction(self, src_trap):
        """
        论文的 conflict handling：
        当目标 zone 满时，把其中一个“长期未使用”的离子迁移到其它更合适的 zone。

        这里的排序原则是：
          1) 优先迁往不高于当前 trap level 的 zone（符合“从高层逐步回落到低层”的论文描述）
          2) 在 level 合法时优先 level 更近
          3) 再优先图距离更近
          4) 最后按 trap id 稳定打破平局
        """
        src_level = self._trap_level(src_trap)
        cand = []
        for trap in self.machine.traps:
            tid = trap.id
            if tid == src_trap:
                continue
            if not self._trap_has_free_slot(tid, incoming=1):
                continue
            dst_level = self._trap_level(tid)
            downward_penalty = 0 if dst_level <= src_level else 1000
            level_gap = abs(src_level - dst_level)
            graph_dist = self.machine.dist_cache.get((src_trap, tid), 10 ** 6)
            cand.append((downward_penalty, level_gap, graph_dist, tid))
        cand.sort()
        return [x[-1] for x in cand]

    def _select_lru_victim(self, trap_id, forbidden_ions=None):
        """
        从 trap 中选择一个可驱逐离子：
          - 不允许驱逐当前 gate 涉及离子
          - 使用论文明确提到的 LRU 策略
        """
        if forbidden_ions is None:
            forbidden_ions = set()
        ions = list(self.sys_state.trap_ions[trap_id])
        candidates = [ion for ion in ions if ion not in forbidden_ions]
        if not candidates:
            return None
        return min(candidates, key=lambda ion: self.ion_last_used.get(ion, -1))

    def _ensure_space_on_trap(self, trap_id, fire_time, required_incoming=1, forbidden_ions=None):
        """
        局部冲突处理核心：
        若 target trap 当前无法再接收 required_incoming 个离子，
        则严格按论文 3.2 的思路，仅围绕该 target trap 做 LRU 驱逐。

        这里与旧版最关键的区别有两点：
          1) 只处理“已选中的目标执行 trap”，不对 source trap 或其它局部区域做额外疏通。
          2) 需要释放的空位数由 required_incoming 显式给出，
             因而支持第三方 target trap 同时接收两颗离子的情形。

        返回：
          (success, new_time)
        """
        if forbidden_ions is None:
            forbidden_ions = set()

        cur_time = fire_time
        guard = 0
        while not self._trap_has_free_slot(trap_id, incoming=required_incoming):
            guard += 1
            if guard > len(self.machine.traps) + 8:
                return False, cur_time

            victim = self._select_lru_victim(trap_id, forbidden_ions=forbidden_ions)
            if victim is None:
                return False, cur_time

            moved = False
            victim_ready, victim_trap = self.ion_ready_info(victim)
            cur_time = max(cur_time, victim_ready)

            for dst_trap in self._candidate_traps_for_eviction(victim_trap):
                route = self._find_route_or_none(victim_trap, dst_trap)
                if route is None:
                    continue
                cur_time = self.fire_shuttle(victim_trap, dst_trap, victim, cur_time, route=route)
                moved = True
                break

            if not moved:
                return False, cur_time

        return True, cur_time

    def _prepare_meeting_trap(self, ion1_trap, ion2_trap, ion1, ion2, fire_time, gate_idx):
        """
        为当前 2Q gate 选择执行 trap，并在必要时做严格的论文式 conflict handling。

        与上一版不同的是，这里把流程明确拆成三段：
          1) 先按论文 3.2 的 available / level / distance 规则产生目标候选序列
          2) 再逐个检查这些候选的 route legality，选出当前真正可达的目标
          3) 仅围绕这个已选定目标做局部 LRU conflict handling

        返回：
          (success, chosen_target_info, ready_time)

        其中 chosen_target_info 形如：
          {
            "target_trap": ...,
            "plan": [(moving_ion, source_trap, target_trap), ...],
            "incoming_needed": ...,
            "total_move_cost": ...,
          }
        """
        cur_time = fire_time

        ordered_choices = self._choose_partition_target(
            ion1_trap, ion2_trap, ion1, ion2, fire_time, gate_idx
        )
        if not ordered_choices:
            return False, None, cur_time

        choice = self._select_reachable_target(ordered_choices)
        if choice is None:
            return False, None, cur_time

        ok, cur_time = self._ensure_space_on_trap(
            choice["target_trap"],
            cur_time,
            required_incoming=choice["incoming_needed"],
            forbidden_ions={ion1, ion2},
        )
        if not ok:
            return False, None, cur_time

        # 目标 trap 腾出空间后，再次确认 route 仍然合法。
        # 这一步不是为了参与目标评分，而是为了保证局部驱逐没有破坏当前搬运路径。
        if not self._routes_exist_for_target(choice["plan"]):
            return False, None, cur_time

        return True, choice, cur_time

    # ==========================================================
    # Gate scheduling：按 frontier 逐个 gate 执行（MUSS strict）
    # ==========================================================
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        Paper-faithful MUSS gate scheduling:
          - 1Q gates do NOT participate in the MUSS frontier loop.
          - This method is for 2Q gates only.
          - 1Q gates are inserted later in _schedule_delayed_one_qubit_gates().
        """
        gate_data, qubits, gate_type = self._gate_payload(gate)
        if gate_data is None:
            self.gate_finish_times[gate] = self.gate_ready_time(gate)
            return

        if len(qubits) != 2:
            self.gate_finish_times[gate] = self.gate_ready_time(gate)
            return

        ion1, ion2 = qubits[0], qubits[1]
        self.protected_ions = {ion1, ion2}
        ready = self.gate_ready_time(gate)
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        fire_time = max(ready, ion1_time, ion2_time, specified_time)

        if self.is_large_mode:
            t1_obj = self.machine.get_trap(ion1_trap)
            t2_obj = self.machine.get_trap(ion2_trap)
            if getattr(t1_obj, "qccd_id", 0) != getattr(t2_obj, "qccd_id", 0):
                raise NotImplementedError(
                    "Large-scale cross-QCCD optical/fiber + inter-module swap-insert path "
                    "is explicitly reserved for the next stage and is not enabled in this file yet."
                )

        finish_time = 0
        if ion1_trap == ion2_trap:
            gate_duration = self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
            zone_type = getattr(self.machine.get_trap(ion1_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
            self.schedule.add_gate(
                fire_time, fire_time + gate_duration, [ion1, ion2], ion1_trap,
                gate_type=gate_type, zone_type=zone_type, gate_id=gate
            )
            self.gate_finish_times[gate] = fire_time + gate_duration
            finish_time = fire_time + gate_duration
        else:
            # V6.2-small：严格按小规模 3.2 口径推进：先选 gate / 再选目标 trap / 再做局部 conflict handling / 最后执行 gate。
            ok, choice, prep_time = self._prepare_meeting_trap(
                ion1_trap, ion2_trap, ion1, ion2, fire_time, gate_idx
            )
            if not ok:
                self.protected_ions = set()
                raise RuntimeError(
                    "Scheduling deadlock under strict paper-faithful target-zone conflict handling. "
                    f"gate={gate}, ion1={ion1}, ion2={ion2}, "
                    f"ion1_trap={ion1_trap}, ion2_trap={ion2_trap}, fire_time={fire_time}"
                )

            dest_trap = choice["target_trap"]
            clk = prep_time

            for moving_ion, source_trap, _ in choice["plan"]:
                if source_trap == dest_trap:
                    continue
                route = self._find_route_or_none(source_trap, dest_trap)
                if route is None:
                    self.protected_ions = set()
                    raise RuntimeError(
                        "Scheduling deadlock: selected target zone is fixed, but a legal route does not exist. "
                        f"gate={gate}, moving_ion={moving_ion}, source_trap={source_trap}, "
                        f"dest_trap={dest_trap}, fire_time={clk}"
                    )
                clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, clk, route=route)

            dest_ions = self.sys_state.trap_ions[dest_trap]
            if ion1 not in dest_ions or ion2 not in dest_ions:
                raise RuntimeError(
                    f"2Q gate cannot execute on trap {dest_trap}: ions not co-located. "
                    f"gate={gate}, ion1={ion1}, ion2={ion2}, trap_ions={dest_ions}, "
                    f"plan={choice['plan']}"
                )

            gate_duration = self.machine.gate_time(self.sys_state, dest_trap, ion1, ion2)
            zone_type = getattr(self.machine.get_trap(dest_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
            self.schedule.add_gate(
                clk, clk + gate_duration, [ion1, ion2], dest_trap,
                gate_type=gate_type, zone_type=zone_type, gate_id=gate
            )
            self.gate_finish_times[gate] = clk + gate_duration
            finish_time = clk + gate_duration

            # 方案二：记录最近一次 remote meeting trap。
            # 这只影响后续同优先级 target 的轻量 tie-break，不改变论文 3.2 的主流程。
            self._record_recent_remote_meeting(ion1, ion2, dest_trap, finish_time)

        self.ion_last_used[ion1] = finish_time
        self.ion_last_used[ion2] = finish_time
        self.protected_ions = set()

    def add_one_qubit_gate(self, gate):
        """
        Insert a 1Q gate after all 2Q scheduling has established ion trajectories.
        1Q gates affect final time/fidelity, but never participate in the MUSS frontier.
        """
        gate_data, qubits, gate_type = self._gate_payload(gate)
        if gate_data is None or len(qubits) != 1:
            return

        ion = qubits[0]
        ready = self.gate_ready_time_full(gate)
        ion_time, ion_trap = self.ion_ready_info(ion)
        fire_time = max(ready, ion_time)

        duration = self.machine.single_qubit_gate_time(gate_type)
        zone_type = getattr(self.machine.get_trap(ion_trap), "zone_type", None) if hasattr(self.machine, "get_trap") else None
        self.schedule.add_gate(
            fire_time, fire_time + duration, [ion], ion_trap,
            gate_type=gate_type, zone_type=zone_type, gate_id=gate
        )
        self.gate_finish_times[gate] = fire_time + duration
        self.ion_last_used[ion] = fire_time + duration

    def _schedule_delayed_one_qubit_gates(self):
        if self.full_ir is None:
            return
        for g in self.full_topo_list:
            gate_data, qubits, _ = self._gate_payload(g)
            if gate_data is None:
                continue
            if len(qubits) == 1:
                self.add_one_qubit_gate(g)

    def is_executable_local(self, gate):
        """2Q helper: whether the gate can execute locally without a shuttle."""
        _, qubits, _ = self._gate_payload(gate)
        if len(qubits) != 2:
            return False
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # ==========================================================
    # 主循环：Frontier scheduling（MUSS strict）
    # ==========================================================
    def run(self):
        """
        Paper-faithful MUSS frontier:
          1) Run MUSS only on the 2Q-only DAG.
          2) 1Q gates do not participate in gate selection.
          3) After all 2Q gates are scheduled, replay the full DAG and insert 1Q gates
             at their earliest legal times so they still affect total time/fidelity.
        """
        in_degree = {n: self.ir.in_degree(n) for n in self.ir.nodes}

        # 严格固定 FCFS 语义：
        #   - 初始 frontier 按 2Q DAG 的静态拓扑顺序进入 ready 队列
        #   - 后续新进入 frontier 的 gate 按“第一次变为 ready”的顺序追加到队尾
        #   - 在每个时刻，先从当前 ready 队列中挑 local executable gate；
        #     若有多个，则取进入 ready 队列最早者（FCFS）
        #   - 若没有 local gate，则在 remote gate 中同样按 FCFS 取最早者
        ready_seq = 0
        ready_queue = []
        ready_stamp = {}
        for n in self.static_topo_list:
            if in_degree[n] == 0:
                ready_stamp[n] = ready_seq
                ready_queue.append(n)
                ready_seq += 1

        processed_count = 0
        total_gates = len(self.ir.nodes)

        while processed_count < total_gates:
            if not ready_queue:
                break

            local_candidates = []
            remote_candidates = []

            for g in ready_queue:
                if self.is_executable_local(g):
                    local_candidates.append(g)
                else:
                    remote_candidates.append(g)

            if local_candidates:
                best_gate = min(local_candidates, key=lambda x: (ready_stamp[x], self.static_topo_order.get(x, float("inf"))))
            else:
                best_gate = min(remote_candidates, key=lambda x: (ready_stamp[x], self.static_topo_order.get(x, float("inf"))))

            gate_idx = self.static_topo_order.get(best_gate, 0)
            self.schedule_gate(best_gate, gate_idx=gate_idx)

            ready_queue.remove(best_gate)
            processed_count += 1

            succs = list(self.ir.successors(best_gate))
            succs.sort(key=lambda x: self.static_topo_order.get(x, float("inf")))
            for successor in succs:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_stamp[successor] = ready_seq
                    ready_queue.append(successor)
                    ready_seq += 1

        self._schedule_delayed_one_qubit_gates()
