# -*- coding: utf-8 -*-
"""
muss_schedule7.py
============================================================
MUSS Scheduler (V7) —— 第二阶段 large-scale 调度器

本文件在 muss_schedule6.py 的严格 small-scale 版本基础上扩展：
1) 保留 V6 的 small-scale 行为，不改坏现有 table2 / 小规模复现路径；
2) 支持同一 QCCD/module 内的本地两比特门调度；
3) 支持跨 QCCD/module 的 optical/fiber 两比特门调度；
4) 实现论文 3.3 的跨 QCCD SWAP insertion，仅在跨模块门后触发。

设计原则：
- small path 直接复用 V6，不重写、不改变其语义；
- large path 只新增“能跑起来且接口干净”的最小必要能力：
  * 同 module：按 multi-level 目标 trap 选择 + 局部 LRU 驱逐 + 片上 shuttle
  * 跨 module：先各自路由到 optical trap，再执行 fiber gate
- 不修改 analyzer / schedule.py 的现有接口；fiber gate 通过 Gate 事件 metadata 显式标记。

当前版本实现范围：
- 支持论文 3.2 的 multi-level scheduling / LRU conflict handling 在 large 模式下运行；
- 支持论文 3.3 的 SWAP insertion：仅对跨 QCCD 门触发，使用固定 look-ahead k=8 与阈值 T=4；
- 仍不展开更复杂的多 optical-zone 负载均衡启发式（单/双 optical zone 仍可运行，但只做贪心选点）。

因此，V7 的目标是：在不破坏现有 small 路径的前提下，把 large-scale 主线与论文 3.3 的 SWAP insertion 补齐。
============================================================
"""

from __future__ import annotations

import collections
import networkx as nx
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.schedule.muss.v6 import MUSSSchedule as MUSSScheduleV6
from src.machine.core import Trap
from src.route import FreeTrapRoute
from src.schedule.events import Schedule
from src.schedule.muss.schedule_v7 import ScheduleV7
from src.machine.state import MachineState


class GateScheduleResult(Enum):
    """V7 大规模路径下的单门调度结果。"""

    EXECUTED = "EXECUTED"
    BLOCKED = "BLOCKED"
    FATAL = "FATAL"


@dataclass
class GateAttemptResult:
    """封装一次 gate 调度尝试的结果。"""

    result: GateScheduleResult
    finish_time: int = 0
    reason: str = ""


@dataclass
class BlockedGateInfo:
    """记录当前 frontier 中某个被阻塞 gate 的诊断信息。"""

    gate_id: object
    reason: str
    round_idx: int


class DeltaJournal:
    """
    V7 large-mode 的局部增量回滚日志。

    设计目标：
    - 只记录本次候选尝试真正改到的局部状态；
    - 候选失败时按逆序撤销这些局部修改；
    - 不再复制整份 schedule / sys_state / log。
    """

    def __init__(self):
        self.undo: List = []

    def mark(self) -> int:
        return len(self.undo)

    def record(self, undo_fn) -> None:
        self.undo.append(undo_fn)

    def rollback_to(self, mark: int = 0) -> None:
        mark = int(mark)
        while len(self.undo) > mark:
            undo_fn = self.undo.pop()
            undo_fn()

    def commit(self) -> None:
        self.undo.clear()


class MUSSSchedule(MUSSScheduleV6):
    """
    V7 继承 V6：
    - small 模式完全走 V6 逻辑；
    - large 模式在 schedule_gate 中分流到：
        1) 同 module 本地 gate
        2) 跨 module optical/fiber gate
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
        super().__init__(
            ir_or_parse,
            gate_info_or_machine,
            M_or_init_map,
            init_map_or_serial_trap_ops,
            serial_trap_ops,
            serial_comm,
            global_serial_lock,
        )

        # V7 明确标识：用于 run.py / 调试日志识别。
        self.scheduler_name = "MUSS_V7"
        self.supports_large_mode = True
        self.supports_cross_qccd_fiber = True

        # ------------------------------
        # 论文 3.3：固定参数
        # ------------------------------
        # 论文明确采用 look-ahead k = 8，阈值 T = 4。
        self.swap_lookahead_k = int(getattr(getattr(self.machine, "mparams", object()), "swap_lookahead_k", 8))
        self.swap_threshold_T = int(getattr(getattr(self.machine, "mparams", object()), "swap_threshold_T", 4))
        self.enable_cross_qccd_swap_insertion = bool(
            getattr(getattr(self.machine, "mparams", object()), "enable_cross_qccd_swap_insertion", True)
        )

        # 逻辑 SWAP 完成后，调度器需要知道“逻辑 qubit 当前在哪个 trap”。
        # 由于 schedule/analyzer 原生事件并不显式表达“标签交换”，这里用 override 保持
        # 调度期的一致性；Analyzer 只关心每个 trap 的链长与事件开销，因此不受影响。
        # 结构：ion -> (override_ts, trap_id)
        self.logical_position_overrides: Dict[int, Tuple[int, int]] = {}
        self._inserted_swap_counter = 0

        # V7 大规模前沿调度需要记录“被阻塞后延期”的 gate。
        # 这些结构均只在 V7 large-mode 的 run() 中使用，不影响 V6/small 路径。
        self.deferred_queue = collections.deque()
        self.blocked_reason_count = collections.Counter()
        self.last_blocked_reasons: Dict[object, BlockedGateInfo] = {}
        self.v7_event_log: List[Dict] = []
        self.round_idx = 0

        # 这些计数器主要给 run.py 摘要行与 paper_report_v7.py 使用。
        self.remote_gate_count = 0
        self.swap_insert_gate_count = 0
        self.swap_insert_logical_count = 0

        # large-mode 采用独立的 ScheduleV7，实现 O(1) 的事件时间缓存，
        # 避免把论文 3.2 的调度代价退化为对整个历史 schedule 的重复扫描。
        if self.is_large_mode:
            self.schedule = ScheduleV7(self.machine)

        # 运行期缓存：逻辑 qubit 当前所在 trap 与最近一次操作完成时间。
        # 这两个缓存由 V7 在 commit 阶段增量维护，避免 large-mode 下反复扫描
        # 整个 schedule.events 去反推当前位置。
        self.ion_current_trap: Dict[int, int] = {}
        self.ion_ready_time: Dict[int, int] = {}
        for trap_id, ions in self.init_map.items():
            for ion in ions:
                self.ion_current_trap[int(ion)] = int(trap_id)
                self.ion_ready_time[int(ion)] = 0

        # 仅供论文 3.3 的 look-ahead 使用的增量 frontier 结构。
        self._runtime_in_degree = None
        self._runtime_ready_queue = None
        self._runtime_remaining_twoq: Set[object] = set()

    # ==========================================================
    # V7 large-mode：状态快照 / 日志 / 阻塞诊断
    # ==========================================================
    def _snapshot_runtime_state(self):
        """
        为 V7 的“试探性调度”保存完整快照。

        说明：
        - 论文流程要求先尝试候选 target / target-pair，再决定是否真正接受该方案；
        - 现有 fire_shuttle / add_gate 会直接修改 schedule 与 sys_state，
          因此这里使用深拷贝快照来保证失败尝试不会污染全局状态；
        - 该逻辑仅在 V7 large-mode 内部使用，不会影响 small/V6。
        """
        return {
            "schedule": deepcopy(self.schedule),
            "sys_state": self.sys_state.clone(),
            "gate_finish_times": dict(self.gate_finish_times),
            "ion_last_used": dict(self.ion_last_used),
            "logical_position_overrides": dict(self.logical_position_overrides),
            "count_rebalance": int(self.count_rebalance),
            "split_swap_counter": int(self.split_swap_counter),
            "shuttle_counter": int(self.shuttle_counter),
            "shuttle_log": deepcopy(self.shuttle_log),
            "chain_reorder_log": deepcopy(self.chain_reorder_log),
            "current_shuttle_id": self._current_shuttle_id,
            "current_shuttle_route": deepcopy(self._current_shuttle_route),
            "current_shuttle_ion": self._current_shuttle_ion,
            "current_shuttle_src": self._current_shuttle_src,
            "current_shuttle_dst": self._current_shuttle_dst,
            "inserted_swap_counter": int(self._inserted_swap_counter),
            "remote_gate_count": int(self.remote_gate_count),
            "swap_insert_gate_count": int(self.swap_insert_gate_count),
            "swap_insert_logical_count": int(self.swap_insert_logical_count),
            "protected_ions": set(self.protected_ions),
        }

    def _restore_runtime_state(self, snapshot):
        """恢复由 _snapshot_runtime_state 保存的试探性状态。"""
        self.schedule = snapshot["schedule"]
        self.sys_state = snapshot["sys_state"]
        self.router = FreeTrapRoute(self.machine, self.sys_state)
        self.gate_finish_times = dict(snapshot["gate_finish_times"])
        self.ion_last_used = dict(snapshot["ion_last_used"])
        self.logical_position_overrides = dict(snapshot["logical_position_overrides"])
        self.count_rebalance = int(snapshot["count_rebalance"])
        self.split_swap_counter = int(snapshot["split_swap_counter"])
        self.shuttle_counter = int(snapshot["shuttle_counter"])
        self.shuttle_log = deepcopy(snapshot["shuttle_log"])
        self.chain_reorder_log = deepcopy(snapshot["chain_reorder_log"])
        self._current_shuttle_id = snapshot["current_shuttle_id"]
        self._current_shuttle_route = deepcopy(snapshot["current_shuttle_route"])
        self._current_shuttle_ion = snapshot["current_shuttle_ion"]
        self._current_shuttle_src = snapshot["current_shuttle_src"]
        self._current_shuttle_dst = snapshot["current_shuttle_dst"]
        self._inserted_swap_counter = int(snapshot["inserted_swap_counter"])
        self.remote_gate_count = int(snapshot["remote_gate_count"])
        self.swap_insert_gate_count = int(snapshot["swap_insert_gate_count"])
        self.swap_insert_logical_count = int(snapshot["swap_insert_logical_count"])
        self.protected_ions = set(snapshot["protected_ions"])

    def _emit_event(self, event_name: str, **kwargs):
        """V7 内部统一事件记录接口，便于 large-scale 调试与批量日志分析。"""
        payload = {"event": event_name, "round": int(self.round_idx)}
        payload.update(kwargs)
        self.v7_event_log.append(payload)

    def _journal_mark(self, journal: Optional[DeltaJournal]) -> Tuple[int, int]:
        """返回 (journal_mark, schedule_mark)，供候选尝试失败时做局部回滚。"""
        if journal is None:
            raise RuntimeError("journal is required in V7 large-mode delta rollback path")
        schedule_mark = self.schedule.mark() if hasattr(self.schedule, "mark") else 0
        return journal.mark(), int(schedule_mark)

    def _rollback_to_mark(self, journal: Optional[DeltaJournal], mark: Tuple[int, int]) -> None:
        if journal is None:
            return
        jmark, smark = int(mark[0]), int(mark[1])
        journal.rollback_to(jmark)
        if hasattr(self.schedule, "rollback"):
            self.schedule.rollback(smark)

    def _journal_save_attr(self, journal: DeltaJournal, obj, attr_name: str):
        old_value = getattr(obj, attr_name)
        journal.record(lambda obj=obj, attr_name=attr_name, old_value=old_value: setattr(obj, attr_name, old_value))

    def _journal_save_dict_value(self, journal: DeltaJournal, d: Dict, key, default_missing=False):
        existed = key in d
        old_value = d.get(key)
        def undo():
            if existed:
                d[key] = old_value
            else:
                d.pop(key, None)
        journal.record(undo)

    def _journal_save_list_len(self, journal: DeltaJournal, lst: list):
        old_len = len(lst)
        journal.record(lambda lst=lst, old_len=old_len: lst.__delitem__(slice(old_len, None)))

    def _journal_save_trap_contents(self, journal: DeltaJournal, trap_ids: Sequence[int]):
        saved = {}
        for tid in trap_ids:
            tid = int(tid)
            if tid in saved:
                continue
            saved[tid] = list(self.sys_state.trap_ions[tid])
        def undo(saved=saved):
            for tid, ions in saved.items():
                self.sys_state.trap_ions[tid] = list(ions)
        journal.record(undo)

    def _fire_shuttle_with_journal(self, src_trap, dest_trap, ion, gate_fire_time, route=None, journal: Optional[DeltaJournal] = None):
        """
        V7 large-mode 的 journal-aware shuttle 封装。

        实现要点：
        - 仍然复用 V6 的论文式 Split/Move/Merge 执行语义；
        - 但候选失败时只回滚 route 涉及的 trap、事件尾部、相关缓存与计数器；
        - 不再对整份状态做 deep copy。
        """
        if journal is None:
            return self.fire_shuttle(src_trap, dest_trap, ion, gate_fire_time, route=route or [])

        if route is None or len(route) == 0:
            route = self._find_route_or_none(src_trap, dest_trap)
            if route is None:
                raise RuntimeError(f"No legal route found for shuttle: T{src_trap} -> T{dest_trap}")

        trap_ids = []
        for node in route:
            if isinstance(node, Trap):
                trap_ids.append(int(node.id))
        self._journal_save_trap_contents(journal, trap_ids)
        self._journal_save_attr(journal, self, 'split_swap_counter')
        self._journal_save_attr(journal, self, 'shuttle_counter')
        self._journal_save_attr(journal, self, '_current_shuttle_id')
        self._journal_save_attr(journal, self, '_current_shuttle_route')
        self._journal_save_attr(journal, self, '_current_shuttle_ion')
        self._journal_save_attr(journal, self, '_current_shuttle_src')
        self._journal_save_attr(journal, self, '_current_shuttle_dst')
        self._journal_save_list_len(journal, self.shuttle_log)
        self._journal_save_list_len(journal, self.chain_reorder_log)
        self._journal_save_dict_value(journal, self.ion_current_trap, int(ion))
        self._journal_save_dict_value(journal, self.ion_ready_time, int(ion))

        finish_time = MUSSScheduleV6.fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=route)
        dst_id = dest_trap.id if isinstance(dest_trap, Trap) else int(dest_trap)
        self.ion_current_trap[int(ion)] = int(dst_id)
        self.ion_ready_time[int(ion)] = int(finish_time)
        return finish_time

    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        finish_time = MUSSScheduleV6.fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=route)
        dst_id = dest_trap.id if isinstance(dest_trap, Trap) else int(dest_trap)
        self.ion_current_trap[int(ion)] = int(dst_id)
        self.ion_ready_time[int(ion)] = int(finish_time)
        return finish_time

    def _set_logical_override_journaled(self, ion: int, trap_id: int, ts: int, journal: Optional[DeltaJournal]):
        if journal is not None:
            self._journal_save_dict_value(journal, self.logical_position_overrides, int(ion))
        self.logical_position_overrides[int(ion)] = (int(ts), int(trap_id))

    def _swap_ions_in_sys_state_journaled(self, ion_a: int, ion_b: int, journal: Optional[DeltaJournal]):
        trap_a = self.sys_state.find_trap_id_by_ion(ion_a)
        trap_b = self.sys_state.find_trap_id_by_ion(ion_b)
        if trap_a == -1 or trap_b == -1:
            raise RuntimeError(f"Logical swap failed: cannot find ions ({ion_a}, {ion_b}) in current system state.")
        if journal is not None:
            self._journal_save_trap_contents(journal, [trap_a, trap_b])
        idx_a = self.sys_state.trap_ions[trap_a].index(ion_a)
        idx_b = self.sys_state.trap_ions[trap_b].index(ion_b)
        self.sys_state.trap_ions[trap_a][idx_a] = ion_b
        self.sys_state.trap_ions[trap_b][idx_b] = ion_a

    def _remove_from_deferred(self, gate):
        """兼容旧接口；严格论文轮次下不再维护跨轮 deferred 队列。"""
        _ = gate

    def _record_blocked_gate(self, gate, reason: str):
        """
        记录当前轮次内某个 frontier gate 的阻塞原因。

        与上一版不同：
        - 不再把 gate 推入跨轮 deferred 队列；
        - 仅保留“本轮为什么没法执行”的诊断信息。
        """
        reason = str(reason or "BLOCKED")
        self.last_blocked_reasons[gate] = BlockedGateInfo(gate_id=gate, reason=reason, round_idx=int(self.round_idx))
        self.blocked_reason_count[reason] += 1
        self._emit_event("BLOCKED_GATE", gate_id=gate, reason=reason)

    def _build_executable_set_fcfs(self, ready_queue: List) -> List:
        """
        论文式 frontier 选择：
        1) 只看当前 ready_queue；
        2) 若存在已经满足硬件执行条件的 gate，则优先这些 gate；
        3) 否则保持 FCFS。

        这里不再引入跨轮 deferred 优先级，以减少工程增强对论文主流程的干扰。
        """
        ordered = list(ready_queue)
        local_ready = [g for g in ordered if self.is_executable_local(g)]
        if local_ready:
            return local_ready
        return ordered

    def _detect_true_deadlock_or_raise(self, ready_queue: List):
        """
        V7 large-mode 的真死锁判定。

        这里不采用“单门失败立即报错”的旧行为，而是仅当：
        - 当前 frontier 中所有候选 gate 都已尝试；
        - 没有任何 gate 能在当前状态下被真正执行；
        才给出 TRUE_DEADLOCK。

        说明：论文 3.3 的 SWAP insertion 是“跨 QCCD gate 执行完成后”的后置优化，
        而不是一个普适性的死锁恢复器。因此这里不把“无进展时全局强行插 SWAP”
        作为主流程的一部分，以避免偏离论文技术路径。
        """
        frontier = list(ready_queue)
        reason_map = {g: info.reason for g, info in self.last_blocked_reasons.items() if g in frontier}
        occupancy = {tid: list(ions) for tid, ions in self.sys_state.trap_ions.items() if ions}
        self._emit_event(
            "TRUE_DEADLOCK",
            frontier_gate_ids=frontier,
            blocked_reasons=reason_map,
            trap_occupancy=occupancy,
        )
        raise RuntimeError(
            "TRUE_DEADLOCK under MUSS V7 large-mode scheduling. "
            f"frontier={frontier}; blocked_reasons={reason_map}; trap_occupancy={occupancy}"
        )

    # ==========================================================
    # 基础辅助：QCCD / zone / fiber 查询
    # ==========================================================
    def _qccd_id_of_trap(self, trap_id: int) -> int:
        if not hasattr(self.machine, "get_trap"):
            return 0
        return int(getattr(self.machine.get_trap(trap_id), "qccd_id", 0))

    def _trap_supports_local_2q(self, trap_id: int) -> bool:
        if not hasattr(self.machine, "get_trap"):
            return self._trap_can_execute_twoq(trap_id)
        trap = self.machine.get_trap(trap_id)
        if hasattr(trap, "can_execute_local_2q"):
            return bool(trap.can_execute_local_2q)
        return self._trap_can_execute_twoq(trap_id)

    def _trap_supports_remote_2q(self, trap_id: int) -> bool:
        if not hasattr(self.machine, "get_trap"):
            return False
        trap = self.machine.get_trap(trap_id)
        return bool(getattr(trap, "can_execute_remote_2q", False))

    def _qccd_local_exec_traps(self, qccd_id: int) -> List[int]:
        """返回某个 QCCD 内允许执行本地 2Q gate 的 trap。"""
        tids = []
        for tr in self.machine.traps:
            if self._qccd_id_of_trap(tr.id) != qccd_id:
                continue
            if self._trap_supports_local_2q(tr.id) and tr.capacity >= 2:
                tids.append(tr.id)
        return tids

    def _qccd_optical_traps(self, qccd_id: int) -> List[int]:
        if hasattr(self.machine, "get_qccd_optical_traps"):
            return [t.id for t in self.machine.get_qccd_optical_traps(qccd_id)]
        return [
            tr.id for tr in self.machine.traps
            if self._qccd_id_of_trap(tr.id) == qccd_id and self._trap_zone_type(tr.id) == "optical"
        ]

    def _fiber_gate_duration(self) -> int:
        return int(round(float(getattr(self.machine.mparams, "qccd_fiber_latency_us", 200.0))))

    def _fiber_gate_fidelity(self) -> float:
        return float(getattr(self.machine.mparams, "qccd_fiber_fidelity", 0.99))

    def _has_fiber_link_between_traps(self, trap_a: int, trap_b: int) -> bool:
        """
        严格判断两个 optical trap 之间是否存在显式登记的 fiber link。

        论文严格复现要求：
        - remote gate 的合法性必须完全由 machine topology 决定；
        - 不允许“只要都是 optical trap 就默认可做 fiber gate”的隐式 fallback。
        """
        if trap_a == trap_b:
            return False
        if not self._trap_supports_remote_2q(trap_a) or not self._trap_supports_remote_2q(trap_b):
            return False

        qccd_a = self._qccd_id_of_trap(trap_a)
        qccd_b = self._qccd_id_of_trap(trap_b)
        if qccd_a == qccd_b:
            return False

        if hasattr(self.machine, "get_fiber_links_between"):
            for link in self.machine.get_fiber_links_between(qccd_a, qccd_b):
                if {int(link.src_trap_id), int(link.dst_trap_id)} == {int(trap_a), int(trap_b)}:
                    return True
            return False

        if hasattr(self.machine, "get_fiber_link"):
            link = self.machine.get_fiber_link(qccd_a, qccd_b)
            if link is None:
                return False
            return {int(link.src_trap_id), int(link.dst_trap_id)} == {int(trap_a), int(trap_b)}

        raise RuntimeError(
            "Machine topology is incomplete: strict V7 requires get_fiber_link(s) to validate remote gates."
        )

    # ==========================================================
    # 位置一致性：扩展 ion_ready_info 以支持 fiber gate 与逻辑 SWAP
    # ==========================================================
    def ion_ready_info(self, ion_id):
        """
        返回 (该 ion 最近一次操作完成时间, 当前所在 trap_id)。

        V7 large-mode 使用增量缓存维护 ion 的当前位置与最近完成时间，
        从而与论文 3.2 的 O(g(n+z+c)) 复杂度保持一致；
        不再通过扫描整个 schedule.events 反推出当前位置。
        """
        if self.is_small_mode:
            return super().ion_ready_info(ion_id)

        ion_id = int(ion_id)
        trap_id = self.ion_current_trap.get(ion_id, self.sys_state.find_trap_id_by_ion(ion_id))
        ready_ts = int(self.ion_ready_time.get(ion_id, 0))

        if trap_id == -1:
            raise AssertionError(f"Did not find ion {ion_id} in runtime cache/sys_state")

        actual_trap = self.sys_state.find_trap_id_by_ion(ion_id)
        if int(trap_id) != int(actual_trap):
            print("ion location mismatch", ion_id, trap_id, actual_trap)
            self.sys_state.print_state()
            raise AssertionError("ion location mismatch between runtime cache and sys_state")

        return ready_ts, int(trap_id)

    # ==========================================================
    # 论文 3.3：look-ahead / 权重表 / SWAP insertion
    # ==========================================================
    def _remaining_twoq_generations(self, exclude_gate=None) -> List[List]:
        """
        增量式前 k 层生成视图。

        与旧版不同：
        - 不再为每次 SWAP 判定重新构造 remaining subgraph 并重新做 topological_generations；
        - 直接基于 run() 当前维护的 frontier indegree / ready queue 做局部层展开。

        这与论文 3.3 对 look-ahead 的 O(kn) 判定复杂度保持一致。
        """
        if self._runtime_in_degree is None or self._runtime_ready_queue is None:
            remaining = [g for g in self.ir.nodes if g != exclude_gate and g not in self.gate_finish_times]
            if not remaining:
                return []
            subg = self.ir.subgraph(remaining).copy()
            gens = []
            for layer in nx.topological_generations(subg):
                ordered = sorted(list(layer), key=lambda x: self.static_topo_order.get(x, float("inf")))
                gens.append(ordered)
            return gens

        remaining = set(self._runtime_remaining_twoq)
        if exclude_gate in remaining:
            remaining.remove(exclude_gate)

        ready_set = {g for g in self._runtime_ready_queue if g in remaining}
        indegree_override = {}
        queued = set(ready_set)

        if exclude_gate is not None and exclude_gate in self.ir:
            for succ in self.ir.successors(exclude_gate):
                if succ not in remaining:
                    continue
                new_deg = int(self._runtime_in_degree.get(succ, self.ir.in_degree(succ))) - 1
                indegree_override[succ] = new_deg
                if new_deg == 0 and succ not in queued:
                    ready_set.add(succ)
                    queued.add(succ)

        frontier = sorted(ready_set, key=lambda x: self.static_topo_order.get(x, float("inf")))
        layers: List[List] = []
        local_remaining = set(remaining)

        while frontier:
            layer = list(frontier)
            layers.append(layer)
            next_ready = set()
            for node in layer:
                if node in local_remaining:
                    local_remaining.remove(node)
                for succ in self.ir.successors(node):
                    if succ not in local_remaining:
                        continue
                    base_deg = indegree_override.get(succ, int(self._runtime_in_degree.get(succ, self.ir.in_degree(succ))))
                    new_deg = base_deg - 1
                    indegree_override[succ] = new_deg
                    if new_deg == 0 and succ not in queued:
                        next_ready.add(succ)
                        queued.add(succ)
            frontier = sorted(next_ready, key=lambda x: self.static_topo_order.get(x, float("inf")))

        return layers

    def _lookahead_gates_after(self, current_gate, k: Optional[int] = None) -> List:
        if k is None:
            k = self.swap_lookahead_k
        gens = self._remaining_twoq_generations(exclude_gate=current_gate)
        out = []
        for layer in gens[: max(int(k), 0)]:
            out.extend(layer)
        return out

    def _current_qccd_of_ion(self, ion: int) -> int:
        _, trap_id = self.ion_ready_info(ion)
        return self._qccd_id_of_trap(trap_id)

    def _ions_in_qccd(self, qccd_id: int) -> List[int]:
        ions = []
        for trap_id, qlist in self.sys_state.trap_ions.items():
            if self._qccd_id_of_trap(trap_id) != qccd_id:
                continue
            ions.extend(list(qlist))
        return sorted(ions)

    def _build_swap_weight_table(self, current_gate, k: Optional[int] = None) -> Dict[int, Dict[int, int]]:
        """
        构建论文中的权重表 W(q_i, c_j)：
            W(q_i, c_j) = 在前 k 层 DAG 中，q_i 与当前位于 QCCD c_j 上的 qubit 交互的 gate 数。

        注意：
        - 这里采用“当前调度时刻”的 module 布局；
        - 因为论文只给出了权重定义，没有规定额外折扣项，因此本实现不加入工程化权重修正。
        """
        gates = self._lookahead_gates_after(current_gate, k=k)
        W: Dict[int, Dict[int, int]] = {}

        for g in gates:
            _, qubits, _ = self._gate_payload(g)
            if len(qubits) != 2:
                continue
            qa, qb = qubits[0], qubits[1]
            ca = self._current_qccd_of_ion(qa)
            cb = self._current_qccd_of_ion(qb)

            W.setdefault(qa, {})
            W.setdefault(qb, {})
            W[qa][cb] = W[qa].get(cb, 0) + 1
            W[qb][ca] = W[qb].get(ca, 0) + 1

        return W

    def _weight_lookup(self, W: Dict[int, Dict[int, int]], ion: int, qccd_id: int) -> int:
        return int(W.get(ion, {}).get(qccd_id, 0))

    def _total_future_weight(self, W: Dict[int, Dict[int, int]], ion: int) -> int:
        return int(sum(W.get(ion, {}).values()))

    def _candidate_target_qccds_for_swap(self, W: Dict[int, Dict[int, int]], ion: int, src_qccd: int) -> List[int]:
        """
        论文条件：
        - 仅当 W(q_a, c_a) == 0 时才考虑；
        - 候选 c_j 需满足 W(q_a, c_j) > T。
        论文没有进一步规定多候选 tie-break，这里仅做稳定排序：
          1) 权重更大优先
          2) 与源 QCCD 的 module 距离更近优先
          3) qccd_id 更小优先
        """
        if self._weight_lookup(W, ion, src_qccd) != 0:
            return []

        cand = []
        for qccd_id in sorted({self._qccd_id_of_trap(t.id) for t in self.machine.traps}):
            if qccd_id == src_qccd:
                continue
            w = self._weight_lookup(W, ion, qccd_id)
            if w > self.swap_threshold_T:
                cand.append((-w, self._module_distance(src_qccd, qccd_id), qccd_id))
        cand.sort()
        return [x[-1] for x in cand]

    def _trap_distance(self, trap_a: int, trap_b: int) -> int:
        """
        统一的 trap 距离查询入口。

        修复背景：
        - 你上一版运行报错的直接原因，是这里有路径仍然调用了
          self.machine.distance(...)
        - 但当前 Machine 实现提供的是 trap_distance(...) / dist_cache，
          并没有 distance(...) 接口。

        设计上不去改其它调度器，也不强行要求 Machine 新增别名，
        而是在 V7 内部统一收口，保证 large-scale 技术尽量集中在 V7。
        """
        if trap_a == trap_b:
            return 0

        if hasattr(self.machine, "trap_distance"):
            try:
                return int(self.machine.trap_distance(trap_a, trap_b))
            except Exception:
                pass

        try:
            return int(self.machine.dist_cache.get((trap_a, trap_b), 10 ** 6))
        except Exception:
            return 10 ** 6

    def _module_distance(self, qccd_a: int, qccd_b: int) -> int:
        if qccd_a == qccd_b:
            return 0
        best = 10 ** 9
        for ta in [tr.id for tr in self.machine.traps if self._qccd_id_of_trap(tr.id) == qccd_a]:
            for tb in [tr.id for tr in self.machine.traps if self._qccd_id_of_trap(tr.id) == qccd_b]:
                d = self._trap_distance(ta, tb)
                if d < best:
                    best = d
        return best

    def _candidate_partner_ions_for_swap(self, W: Dict[int, Dict[int, int]], target_qccd: int) -> List[int]:
        """
        论文条件：在目标 c_j 中寻找某个 q_c，使得 W(q_c, c_j) = 0。
        论文未给出多个 q_c 的 tie-break，本实现采用稳定、保守的排序：
          1) 总未来权重更小优先（更“冷”）
          2) 最近使用时间更早优先（LRU 风格）
          3) 逻辑编号更小优先
        """
        cand = []
        for ion in self._ions_in_qccd(target_qccd):
            if self._weight_lookup(W, ion, target_qccd) != 0:
                continue
            total_w = self._total_future_weight(W, ion)
            lru = int(self.ion_last_used.get(ion, 0))
            cand.append((total_w, lru, ion))
        cand.sort()
        return [x[-1] for x in cand]

    def _set_logical_override(self, ion: int, trap_id: int, ts: int):
        self.logical_position_overrides[int(ion)] = (int(ts), int(trap_id))

    def _swap_ions_in_sys_state(self, ion_a: int, ion_b: int):
        """
        在 sys_state 中交换两个逻辑 qubit 的所在 trap/链位置。
        注意：这里交换的是“逻辑标签”，并不新增物理 shuttle。
        """
        trap_a = self.sys_state.find_trap_id_by_ion(ion_a)
        trap_b = self.sys_state.find_trap_id_by_ion(ion_b)
        if trap_a == -1 or trap_b == -1:
            raise RuntimeError(f"Logical swap failed: cannot find ions ({ion_a}, {ion_b}) in current system state.")

        idx_a = self.sys_state.trap_ions[trap_a].index(ion_a)
        idx_b = self.sys_state.trap_ions[trap_b].index(ion_b)

        self.sys_state.trap_ions[trap_a][idx_a] = ion_b
        self.sys_state.trap_ions[trap_b][idx_b] = ion_a

    def _candidate_swap_target_pairs(self, ion_a: int, ion_c: int, trap_a: int, trap_c: int) -> List[Tuple[int, int]]:
        """
        为论文 3.3 的跨 QCCD SWAP insertion 枚举 optical target pair。

        修复原因：
        - 上一版 _execute_cross_qccd_swap() 还是“左右两边各自贪心选一个 optical trap”，
          然后再事后检查这两个 trap 是否有 fiber link。
        - 这种做法在单 optical-zone 时通常没问题，但在双 optical-zone 或更多 optical-zone
          情况下，会出现“其实存在合法 pair，但独立贪心没选中”的漏解。

        因此这里与正式跨 QCCD gate 一样，直接枚举已经登记过 fiber link 的 optical pair，
        更符合论文“先选 gate，再选目标 zone”的技术路径，也更稳。
        """
        return self._candidate_remote_target_pairs(ion_a, ion_c, trap_a, trap_c)

    def _execute_cross_qccd_swap(self, ion_a: int, ion_c: int, clk: int) -> Tuple[bool, int]:
        """
        在不同 QCCD 上对 (ion_a, ion_c) 执行一次论文 3.3 的逻辑 SWAP。

        本版严格使用 delta rollback：
        - 对每个 optical pair 只记录本次尝试真实改动的 delta；
        - pair 失败时只撤销本 pair 的 shuttles / fiber gate / 逻辑标签交换；
        - 不再复制 planning state。
        """
        time_a, trap_a = self.ion_ready_info(ion_a)
        time_c, trap_c = self.ion_ready_info(ion_c)
        base_clk = max(int(clk), int(time_a), int(time_c))

        qccd_a = self._qccd_id_of_trap(trap_a)
        qccd_c = self._qccd_id_of_trap(trap_c)
        if qccd_a == qccd_c:
            return False, base_clk

        target_pairs = self._candidate_swap_target_pairs(ion_a, ion_c, trap_a, trap_c)
        if not target_pairs:
            return False, base_clk

        journal = DeltaJournal()
        last_clk = base_clk
        for target_a, target_c in target_pairs:
            mark = self._journal_mark(journal)
            cur_clk = base_clk
            try:
                ok_a, cur_clk = self._ensure_space_on_trap_large(
                    target_a,
                    cur_clk,
                    required_incoming=(0 if trap_a == target_a else 1),
                    forbidden_ions={ion_a},
                    journal=journal,
                )
                if not ok_a:
                    raise RuntimeError("SWAP_LEFT_TARGET_FULL")

                ok_c, cur_clk = self._ensure_space_on_trap_large(
                    target_c,
                    cur_clk,
                    required_incoming=(0 if trap_c == target_c else 1),
                    forbidden_ions={ion_c},
                    journal=journal,
                )
                if not ok_c:
                    raise RuntimeError("SWAP_RIGHT_TARGET_FULL")

                if trap_a != target_a:
                    route_a = self._find_route_or_none(trap_a, target_a)
                    if route_a is None:
                        raise RuntimeError("SWAP_LEFT_ROUTE_MISSING")
                    cur_clk = self._fire_shuttle_with_journal(trap_a, target_a, ion_a, cur_clk, route=route_a, journal=journal)

                if trap_c != target_c:
                    route_c = self._find_route_or_none(trap_c, target_c)
                    if route_c is None:
                        raise RuntimeError("SWAP_RIGHT_ROUTE_MISSING")
                    cur_clk = self._fire_shuttle_with_journal(trap_c, target_c, ion_c, cur_clk, route=route_c, journal=journal)

                if ion_a not in self.sys_state.trap_ions[target_a] or ion_c not in self.sys_state.trap_ions[target_c]:
                    raise RuntimeError("SWAP_ENDPOINT_NOT_REACHED")

                self._journal_save_attr(journal, self, '_inserted_swap_counter')
                self._journal_save_attr(journal, self, 'swap_insert_logical_count')
                self._journal_save_dict_value(journal, self.ion_last_used, int(ion_a))
                self._journal_save_dict_value(journal, self.ion_last_used, int(ion_c))

                swap_tag = ("inserted_swap", self._inserted_swap_counter, ion_a, ion_c)
                self._inserted_swap_counter += 1
                swap_insert_id = self._inserted_swap_counter - 1

                for phase in range(3):
                    cur_clk = self._add_gate_op_with_metadata(
                        cur_clk,
                        target_a,
                        [ion_a, ion_c],
                        gate=(swap_tag, phase),
                        gate_type="swap_fiber",
                        is_fiber=True,
                        remote_trap=target_c,
                        qccd_pair=(qccd_a, qccd_c),
                        extra_metadata={
                            "is_swap_insert": True,
                            "swap_insert_id": swap_insert_id,
                            "swap_insert_phase": phase,
                        },
                        journal=journal,
                    )

                self.swap_insert_logical_count += 1
                self._swap_ions_in_sys_state_journaled(ion_a, ion_c, journal)
                self._set_logical_override_journaled(ion_a, target_c, cur_clk, journal)
                self._set_logical_override_journaled(ion_c, target_a, cur_clk, journal)
                self.ion_current_trap[int(ion_a)] = int(target_c)
                self.ion_current_trap[int(ion_c)] = int(target_a)
                self.ion_ready_time[int(ion_a)] = int(cur_clk)
                self.ion_ready_time[int(ion_c)] = int(cur_clk)
                self.ion_last_used[ion_a] = cur_clk
                self.ion_last_used[ion_c] = cur_clk
                journal.commit()
                return True, cur_clk
            except Exception:
                self._rollback_to_mark(journal, mark)
                last_clk = cur_clk
                continue

        return False, last_clk

    def _maybe_insert_swap_for_ion(self, current_gate, ion: int, clk: int) -> int:
        """
        论文 3.3 的单边检查：
        After executing a two-qubit gate involving q_a, q_b on different QCCDs,
        if W(q_a, c_a) == 0 and there exists another QCCD c_j with W(q_a, c_j) > T
        and a qubit q_c in c_j such that W(q_c, c_j) == 0, then insert a SWAP(q_a, q_c).

        这里严格只在“跨 QCCD 门之后”由调用方触发。
        """
        if not self.enable_cross_qccd_swap_insertion:
            return clk

        _, trap_a = self.ion_ready_info(ion)
        src_qccd = self._qccd_id_of_trap(trap_a)
        W = self._build_swap_weight_table(current_gate, k=self.swap_lookahead_k)

        candidate_qccds = self._candidate_target_qccds_for_swap(W, ion, src_qccd)
        if not candidate_qccds:
            return clk

        for target_qccd in candidate_qccds:
            partner_ions = self._candidate_partner_ions_for_swap(W, target_qccd)
            for partner in partner_ions:
                if partner == ion:
                    continue
                ok, new_clk = self._execute_cross_qccd_swap(ion, partner, clk)
                if ok:
                    return new_clk
        return clk

    def _maybe_insert_swap_after_cross_qccd_gate(self, current_gate, ion1: int, ion2: int, clk: int) -> int:
        """
        论文 3.3：对刚完成的跨 QCCD gate，两侧 q_a / q_b 都做一次相同检查。
        若第一侧成功插入了 SWAP，再基于更新后的布局重新检查第二侧。
        """
        clk = self._maybe_insert_swap_for_ion(current_gate, ion1, clk)
        clk = self._maybe_insert_swap_for_ion(current_gate, ion2, clk)
        return clk

    # ==========================================================
    # Large-mode 目标选择 / 冲突处理
    # ==========================================================
    def _candidate_meeting_traps_large(self, ion1_trap: int, ion2_trap: int) -> List[int]:
        """
        large 模式下，同 module 本地 2Q gate 的目标 trap 选择。

        规则：
        1) 只在双方所在的同一个 qccd 内选择；
        2) 只选可执行本地 2Q 的 trap（operation / optical）；
        3) 排序仍遵循 V6 的论文式原则：available + level + distance。
        """
        qccd_id = self._qccd_id_of_trap(ion1_trap)
        if qccd_id != self._qccd_id_of_trap(ion2_trap):
            return []

        src_level_ref = max(self._trap_level(ion1_trap), self._trap_level(ion2_trap))
        candidates = []
        for tid in self._qccd_local_exec_traps(qccd_id):
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
            candidates.append((available_penalty, level_gap, dist_sum, tid))

        candidates.sort()
        return [x[-1] for x in candidates]

    def _choose_partition_target_large(self, ion1_trap: int, ion2_trap: int, ion1: int, ion2: int):
        """large 同 module 模式的目标选择，结构保持和 V6 一致。"""
        ordered_choices = []
        src_level_ref = max(self._trap_level(ion1_trap), self._trap_level(ion2_trap))

        for target in self._candidate_meeting_traps_large(ion1_trap, ion2_trap):
            plan_info = self._build_move_plan_for_target(ion1_trap, ion2_trap, ion1, ion2, target)
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

    def _candidate_traps_for_eviction_large(self, src_trap: int) -> List[int]:
        """
        large 模式下，优先在同一 qccd 内做局部驱逐，避免无意义跨模块迁移。

        论文 3.2 的 multi-level conflict handling 是“从高层 zone 向更低层/相近层回落”，
        因此这里的排序遵循：
        1) 同 qccd 优先；
        2) 优先迁往不高于当前 trap level 的 zone；
        3) 再按 level gap / 图距离排序。
        """
        src_level = self._trap_level(src_trap)
        src_qccd = self._qccd_id_of_trap(src_trap)
        cand = []

        for trap in self.machine.traps:
            tid = trap.id
            if tid == src_trap:
                continue
            if not self._trap_has_free_slot(tid, incoming=1):
                continue

            dst_qccd = self._qccd_id_of_trap(tid)
            same_qccd_penalty = 0 if dst_qccd == src_qccd else 1000
            dst_level = self._trap_level(tid)
            downward_penalty = 0 if dst_level <= src_level else 1
            level_gap = abs(src_level - dst_level)
            graph_dist = self.machine.dist_cache.get((src_trap, tid), 10 ** 6)
            cand.append((same_qccd_penalty, downward_penalty, level_gap, graph_dist, tid))

        cand.sort()
        return [x[-1] for x in cand]

    def _ensure_space_on_trap_large(
        self,
        trap_id: int,
        fire_time: int,
        required_incoming: int = 1,
        forbidden_ions: Optional[Set[int]] = None,
        journal: Optional[DeltaJournal] = None,
    ) -> Tuple[bool, int]:
        """
        large 模式局部冲突处理（论文 3.2 的 LRU conflict handling）。

        关键修复：
        - 不再复制 planning state；
        - 直接在当前真实状态上尝试局部驱逐；
        - 每次 victim 尝试仅对新增的 delta 做 rollback。
        """
        if forbidden_ions is None:
            forbidden_ions = set()

        cur_time = int(fire_time)
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
            cur_time = max(cur_time, int(victim_ready))

            for dst_trap in self._candidate_traps_for_eviction_large(victim_trap):
                if self._qccd_id_of_trap(dst_trap) != self._qccd_id_of_trap(victim_trap):
                    continue
                route = self._find_route_or_none(victim_trap, dst_trap)
                if route is None:
                    continue

                if journal is None:
                    cur_time = self.fire_shuttle(victim_trap, dst_trap, victim, cur_time, route=route)
                    moved = True
                    break

                mark = self._journal_mark(journal)
                try:
                    cur_time = self._fire_shuttle_with_journal(victim_trap, dst_trap, victim, cur_time, route=route, journal=journal)
                    moved = True
                    break
                except Exception:
                    self._rollback_to_mark(journal, mark)
                    continue

            if not moved:
                return False, cur_time

        return True, cur_time

    def _prepare_meeting_trap_large(
        self,
        ion1_trap: int,
        ion2_trap: int,
        ion1: int,
        ion2: int,
        fire_time: int,
    ) -> Tuple[bool, Optional[Dict], int]:
        """large 同 module 本地 gate 的 target trap 选择与冲突处理。"""
        cur_time = fire_time
        ordered_choices = self._choose_partition_target_large(ion1_trap, ion2_trap, ion1, ion2)
        if not ordered_choices:
            return False, None, cur_time

        choice = self._select_reachable_target(ordered_choices)
        if choice is None:
            return False, None, cur_time

        ok, cur_time = self._ensure_space_on_trap_large(
            choice["target_trap"],
            cur_time,
            required_incoming=choice["incoming_needed"],
            forbidden_ions={ion1, ion2},
        )
        if not ok:
            return False, None, cur_time

        if not self._routes_exist_for_target(choice["plan"]):
            return False, None, cur_time

        return True, choice, cur_time

    def _select_optical_target_for_ion(self, ion: int, current_trap: int) -> Optional[int]:
        """
        为跨模块 fiber gate 选择 optical 目标 trap。

        当前阶段策略：
        - 优先选当前 qccd 的 optical trap；
        - 若有多个 optical trap，则按 available + distance 排序。
        """
        qccd_id = self._qccd_id_of_trap(current_trap)
        opticals = self._qccd_optical_traps(qccd_id)
        if not opticals:
            return None

        candidates = []
        for tid in opticals:
            incoming_needed = 0 if current_trap == tid else 1
            available_penalty = 0 if self._trap_has_free_slot(tid, incoming=incoming_needed) else 1
            dist = self.machine.dist_cache.get((current_trap, tid), 10 ** 6)
            candidates.append((available_penalty, dist, tid))
        candidates.sort()
        return candidates[0][-1]

    def _candidate_remote_target_pairs(self, ion1: int, ion2: int, trap1: int, trap2: int):
        """
        为跨 QCCD gate 枚举合法的 optical trap pair。

        与旧实现不同，这里不再先分别给两颗离子独立选 optical trap、
        再事后检查 pair 是否可连；而是直接对 gate 选一对已经登记过
        fiber link 的目标 optical traps。这与论文 3.2 中“先选 gate，再选
        target zone”的技术路径一致。
        """
        q1 = self._qccd_id_of_trap(trap1)
        q2 = self._qccd_id_of_trap(trap2)
        opt1 = self._qccd_optical_traps(q1)
        opt2 = self._qccd_optical_traps(q2)

        candidates = []
        for t1 in opt1:
            for t2 in opt2:
                if not self._has_fiber_link_between_traps(t1, t2):
                    continue
                incoming1 = 0 if trap1 == t1 else 1
                incoming2 = 0 if trap2 == t2 else 1
                conflict_penalty = 0
                if len(self.sys_state.trap_ions[t1]) + incoming1 > self.machine.get_trap(t1).capacity:
                    conflict_penalty += 1
                if len(self.sys_state.trap_ions[t2]) + incoming2 > self.machine.get_trap(t2).capacity:
                    conflict_penalty += 1
                dist_score = self._trap_distance(trap1, t1) + self._trap_distance(trap2, t2)
                candidates.append((conflict_penalty, dist_score, t1, t2))

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return [(t1, t2) for _, _, t1, t2 in candidates]

    # ==========================================================
    # Fiber gate / local gate 执行原语
    # ==========================================================
    def _add_gate_op_with_metadata(
        self,
        clk: int,
        trap_id: int,
        ions: Sequence[int],
        gate,
        gate_type: str,
        *,
        is_fiber: bool = False,
        remote_trap: Optional[int] = None,
        qccd_pair: Optional[Tuple[int, int]] = None,
        remote_latency_us: Optional[int] = None,
        extra_metadata: Optional[Dict] = None,
        journal: Optional[DeltaJournal] = None,
    ) -> int:
        """
        写入 Gate 事件。

        说明：
        - 为兼容现有 analyzer / schedule 事件格式，仍然使用 add_gate；
        - fiber gate 通过 metadata 标识：is_fiber / remote_trap / qccd_pair。
        """
        fire_time = clk
        if self.SerialTrapOps == 1:
            last_event_time_on_trap = self.schedule.last_event_time_on_trap(trap_id)
            fire_time = max(fire_time, last_event_time_on_trap)
            if is_fiber and remote_trap is not None:
                last_remote = self.schedule.last_event_time_on_trap(remote_trap)
                fire_time = max(fire_time, last_remote)

        if self.GlobalSerialLock == 1:
            fire_time = max(fire_time, self.schedule.get_last_event_ts())

        if journal is not None:
            for ion in ions:
                self._journal_save_dict_value(journal, self.ion_current_trap, int(ion))
                self._journal_save_dict_value(journal, self.ion_ready_time, int(ion))
            if is_fiber:
                self._journal_save_attr(journal, self, "remote_gate_count")
            if extra_metadata and bool(extra_metadata.get("is_swap_insert", False)):
                self._journal_save_attr(journal, self, "swap_insert_gate_count")

        if is_fiber:
            duration = self._fiber_gate_duration() if remote_latency_us is None else int(remote_latency_us)
            zone_type = "optical"
            self.schedule.add_gate(
                fire_time,
                fire_time + duration,
                list(ions),
                trap_id,
                gate_type=gate_type,
                gate_id=gate,
                zone_type=zone_type,
                is_fiber=True,
            )
            last_event = self.schedule.events[-1]
            last_event[4]["remote_trap"] = remote_trap
            last_event[4]["qccd_pair"] = qccd_pair
            last_event[4]["fiber_fidelity"] = self._fiber_gate_fidelity()
            self.remote_gate_count += 1
        else:
            duration = self.machine.gate_time(self.sys_state, trap_id, ions[0], ions[1])
            zone_type = getattr(self.machine.get_trap(trap_id), "zone_type", None) if hasattr(self.machine, "get_trap") else None
            self.schedule.add_gate(
                fire_time,
                fire_time + duration,
                list(ions),
                trap_id,
                gate_type=gate_type,
                gate_id=gate,
                zone_type=zone_type,
            )

        if extra_metadata:
            self.schedule.events[-1][4].update(dict(extra_metadata))
            if bool(extra_metadata.get("is_swap_insert", False)):
                self.swap_insert_gate_count += 1

        finish_time = fire_time + duration
        if is_fiber:
            if len(ions) >= 1:
                self.ion_current_trap[int(ions[0])] = int(trap_id)
                self.ion_ready_time[int(ions[0])] = int(finish_time)
            if len(ions) >= 2 and remote_trap is not None:
                self.ion_current_trap[int(ions[1])] = int(remote_trap)
                self.ion_ready_time[int(ions[1])] = int(finish_time)
        else:
            for ion in ions:
                self.ion_current_trap[int(ion)] = int(trap_id)
                self.ion_ready_time[int(ion)] = int(finish_time)
        return finish_time

    def _rank_local_meeting_traps(self, ion1_trap: int, ion2_trap: int, ion1: int, ion2: int) -> List[int]:
        """
        按论文 3.2 的 multi-level 语义，对本地 2Q gate 的候选会合 trap 做稳定排序。

        排序键：
        1) 当前容量是否足以接收本门需要搬入的离子；
        2) 与源 zone level 的接近程度；
        3) 当前两条搬运路径的距离和；
        4) trap_id（保证可复现）。
        """
        qccd_id = self._qccd_id_of_trap(ion1_trap)
        if qccd_id != self._qccd_id_of_trap(ion2_trap):
            return []

        src_level_ref = max(self._trap_level(ion1_trap), self._trap_level(ion2_trap))
        ranked = []
        for tid in self._qccd_local_exec_traps(qccd_id):
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
            ranked.append((available_penalty, level_gap, dist_sum, tid))

        ranked.sort()
        return [item[-1] for item in ranked]

    def _try_schedule_intra_target(
        self,
        gate,
        ion1: int,
        ion2: int,
        gate_type: str,
        fire_time: int,
        dest_trap: int,
    ) -> GateAttemptResult:
        """
        在已经给定本地 meeting trap 的前提下，尝试完整执行一次 intra-QCCD 2Q gate。

        注意：
        - 本函数允许直接修改 schedule / sys_state；
        - 调用方必须通过快照决定“接受还是回滚”这次尝试。
        """
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        clk = max(int(fire_time), int(ion1_time), int(ion2_time))

        plan_info = self._build_move_plan_for_target(ion1_trap, ion2_trap, ion1, ion2, dest_trap)
        move_plan = plan_info["plan"]
        incoming_needed = plan_info["incoming_needed"]

        ok, clk = self._ensure_space_on_trap_large(
            dest_trap,
            clk,
            required_incoming=incoming_needed,
            forbidden_ions={ion1, ion2},
        )
        if not ok:
            return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "LOCAL_TARGET_FULL")

        if not self._routes_exist_for_target(move_plan):
            return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "NO_ROUTE_TO_LOCAL_TARGET")

        for moving_ion, source_trap, _ in move_plan:
            if source_trap == dest_trap:
                continue
            route = self._find_route_or_none(source_trap, dest_trap)
            if route is None:
                return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "ROUTE_VANISHED_AFTER_LOCAL_PREP")
            clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, clk, route=route)

        dest_ions = self.sys_state.trap_ions[dest_trap]
        if ion1 not in dest_ions or ion2 not in dest_ions:
            return GateAttemptResult(GateScheduleResult.FATAL, clk, "LOCAL_GATE_IONS_NOT_COLOCATED")

        finish_time = self._add_gate_op_with_metadata(clk, dest_trap, [ion1, ion2], gate, gate_type)
        return GateAttemptResult(GateScheduleResult.EXECUTED, finish_time, "")

    def _schedule_intra_qccd_gate(
        self,
        gate,
        ion1: int,
        ion2: int,
        gate_type: str,
        fire_time: int,
    ) -> GateAttemptResult:
        """
        安排同一 QCCD 内的本地 2Q gate。

        本版严格使用 delta rollback：
        - 对每个候选 target trap，只在真实状态上做局部尝试；
        - 失败时仅回滚该 target 尝试产生的 delta；
        - 不再复制 planning state。
        """
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        clk = max(int(fire_time), int(ion1_time), int(ion2_time))

        if ion1_trap == ion2_trap:
            finish_time = self._add_gate_op_with_metadata(clk, ion1_trap, [ion1, ion2], gate, gate_type)
            return GateAttemptResult(GateScheduleResult.EXECUTED, finish_time, "")

        ranked_targets = self._rank_local_meeting_traps(ion1_trap, ion2_trap, ion1, ion2)
        if not ranked_targets:
            return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "NO_LOCAL_MEETING_TRAP_NOW")

        journal = DeltaJournal()
        last_reason = "NO_LOCAL_MEETING_TRAP_NOW"
        for dest_trap in ranked_targets:
            mark = self._journal_mark(journal)
            try:
                plan_info = self._build_move_plan_for_target(ion1_trap, ion2_trap, ion1, ion2, dest_trap)
                move_plan = plan_info["plan"]
                incoming_needed = plan_info["incoming_needed"]

                ok, clk_local = self._ensure_space_on_trap_large(
                    dest_trap,
                    clk,
                    required_incoming=incoming_needed,
                    forbidden_ions={ion1, ion2},
                    journal=journal,
                )
                if not ok:
                    last_reason = "LOCAL_TARGET_FULL"
                    raise RuntimeError(last_reason)

                for moving_ion, source_trap, _ in move_plan:
                    if source_trap == dest_trap:
                        continue
                    route = self._find_route_or_none(source_trap, dest_trap)
                    if route is None:
                        last_reason = "NO_ROUTE_TO_LOCAL_TARGET"
                        raise RuntimeError(last_reason)
                    clk_local = self._fire_shuttle_with_journal(source_trap, dest_trap, moving_ion, clk_local, route=route, journal=journal)

                dest_ions = self.sys_state.trap_ions[dest_trap]
                if ion1 not in dest_ions or ion2 not in dest_ions:
                    last_reason = "LOCAL_GATE_IONS_NOT_COLOCATED"
                    raise RuntimeError(last_reason)

                finish_time = self._add_gate_op_with_metadata(
                    clk_local,
                    dest_trap,
                    [ion1, ion2],
                    gate,
                    gate_type,
                    journal=journal,
                )
                journal.commit()
                return GateAttemptResult(GateScheduleResult.EXECUTED, finish_time, "")
            except Exception as exc:
                if str(exc):
                    last_reason = str(exc)
                self._rollback_to_mark(journal, mark)
                continue

        return GateAttemptResult(GateScheduleResult.BLOCKED, clk, last_reason)

    def _try_schedule_cross_pair(
        self,
        gate,
        ion1: int,
        ion2: int,
        gate_type: str,
        fire_time: int,
        target1: int,
        target2: int,
    ) -> GateAttemptResult:
        """对一组已经选定的 optical trap pair 尝试一次完整的跨 QCCD gate 调度。"""
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        clk = max(int(fire_time), int(ion1_time), int(ion2_time))

        q1 = self._qccd_id_of_trap(ion1_trap)
        q2 = self._qccd_id_of_trap(ion2_trap)
        if q1 == q2:
            return self._schedule_intra_qccd_gate(gate, ion1, ion2, gate_type, clk)

        ok1, clk = self._ensure_space_on_trap_large(
            target1,
            clk,
            required_incoming=(0 if ion1_trap == target1 else 1),
            forbidden_ions={ion1},
        )
        ok2, clk = self._ensure_space_on_trap_large(
            target2,
            clk,
            required_incoming=(0 if ion2_trap == target2 else 1),
            forbidden_ions={ion2},
        )
        if not ok1 or not ok2:
            return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "OPTICAL_TARGET_FULL")

        if ion1_trap != target1:
            route1 = self._find_route_or_none(ion1_trap, target1)
            if route1 is None:
                return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "NO_ROUTE_TO_OPTICAL_TARGET_LEFT")
            clk = self.fire_shuttle(ion1_trap, target1, ion1, clk, route=route1)

        if ion2_trap != target2:
            route2 = self._find_route_or_none(ion2_trap, target2)
            if route2 is None:
                return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "NO_ROUTE_TO_OPTICAL_TARGET_RIGHT")
            clk = self.fire_shuttle(ion2_trap, target2, ion2, clk, route=route2)

        if ion1 not in self.sys_state.trap_ions[target1] or ion2 not in self.sys_state.trap_ions[target2]:
            return GateAttemptResult(GateScheduleResult.FATAL, clk, "FIBER_GATE_IONS_NOT_AT_OPTICAL_ENDPOINTS")

        clk = self._add_gate_op_with_metadata(
            clk,
            target1,
            [ion1, ion2],
            gate,
            gate_type,
            is_fiber=True,
            remote_trap=target2,
            qccd_pair=(q1, q2),
        )

        # 论文 3.3：只有在“跨 QCCD 2Q gate 已成功执行之后”才检查 SWAP insertion。
        clk = self._maybe_insert_swap_after_cross_qccd_gate(gate, ion1, ion2, clk)
        return GateAttemptResult(GateScheduleResult.EXECUTED, clk, "")

    def _schedule_cross_qccd_gate(
        self,
        gate,
        ion1: int,
        ion2: int,
        gate_type: str,
        fire_time: int,
    ) -> GateAttemptResult:
        """
        安排跨 module 的 optical/fiber 2Q gate。

        本版严格使用 delta rollback：
        - 对每个 optical pair 直接在真实状态上尝试；
        - 失败时仅撤销该 pair 产生的局部 delta；
        - 成功后再保留 fiber gate 及其后置 SWAP insertion 的结果。
        """
        ion1_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_time, ion2_trap = self.ion_ready_info(ion2)
        clk = max(int(fire_time), int(ion1_time), int(ion2_time))

        q1 = self._qccd_id_of_trap(ion1_trap)
        q2 = self._qccd_id_of_trap(ion2_trap)
        if q1 == q2:
            return self._schedule_intra_qccd_gate(gate, ion1, ion2, gate_type, clk)

        target_pairs = self._candidate_remote_target_pairs(ion1, ion2, ion1_trap, ion2_trap)
        if not target_pairs:
            return GateAttemptResult(GateScheduleResult.BLOCKED, clk, "NO_REGISTERED_FIBER_LINK")

        journal = DeltaJournal()
        last_reason = "NO_REGISTERED_FIBER_LINK"
        for target1, target2 in target_pairs:
            mark = self._journal_mark(journal)
            try:
                ok1, clk_local = self._ensure_space_on_trap_large(
                    target1,
                    clk,
                    required_incoming=(0 if ion1_trap == target1 else 1),
                    forbidden_ions={ion1},
                    journal=journal,
                )
                if not ok1:
                    last_reason = "OPTICAL_TARGET_FULL_LEFT"
                    raise RuntimeError(last_reason)

                ok2, clk_local = self._ensure_space_on_trap_large(
                    target2,
                    clk_local,
                    required_incoming=(0 if ion2_trap == target2 else 1),
                    forbidden_ions={ion2},
                    journal=journal,
                )
                if not ok2:
                    last_reason = "OPTICAL_TARGET_FULL_RIGHT"
                    raise RuntimeError(last_reason)

                if ion1_trap != target1:
                    route1 = self._find_route_or_none(ion1_trap, target1)
                    if route1 is None:
                        last_reason = "NO_ROUTE_TO_OPTICAL_TARGET_LEFT"
                        raise RuntimeError(last_reason)
                    clk_local = self._fire_shuttle_with_journal(ion1_trap, target1, ion1, clk_local, route=route1, journal=journal)

                if ion2_trap != target2:
                    route2 = self._find_route_or_none(ion2_trap, target2)
                    if route2 is None:
                        last_reason = "NO_ROUTE_TO_OPTICAL_TARGET_RIGHT"
                        raise RuntimeError(last_reason)
                    clk_local = self._fire_shuttle_with_journal(ion2_trap, target2, ion2, clk_local, route=route2, journal=journal)

                if ion1 not in self.sys_state.trap_ions[target1] or ion2 not in self.sys_state.trap_ions[target2]:
                    last_reason = "FIBER_GATE_IONS_NOT_AT_OPTICAL_ENDPOINTS"
                    raise RuntimeError(last_reason)

                clk_local = self._add_gate_op_with_metadata(
                    clk_local,
                    target1,
                    [ion1, ion2],
                    gate,
                    gate_type,
                    is_fiber=True,
                    remote_trap=target2,
                    qccd_pair=(q1, q2),
                    journal=journal,
                )
                clk_local = self._maybe_insert_swap_after_cross_qccd_gate(gate, ion1, ion2, clk_local)
                journal.commit()
                return GateAttemptResult(GateScheduleResult.EXECUTED, clk_local, "")
            except Exception as exc:
                if str(exc):
                    last_reason = str(exc)
                self._rollback_to_mark(journal, mark)
                continue

        return GateAttemptResult(GateScheduleResult.BLOCKED, clk, last_reason)

    def _schedule_gate_internal(self, gate, specified_time=0, gate_idx=None) -> GateAttemptResult:
        """
        V7 内部 gate 调度入口。

        返回 GateAttemptResult，而不是像 V6 那样默认“调度一定成功”。
        这样当某个 frontier gate 当前不可调度时，主循环可以选择 defer，
        而不是直接 crash。
        """
        gate_data, qubits, gate_type = self._gate_payload(gate)
        if gate_data is None:
            ready_ts = self.gate_ready_time(gate)
            self.gate_finish_times[gate] = ready_ts
            return GateAttemptResult(GateScheduleResult.EXECUTED, ready_ts, "")

        if len(qubits) != 2:
            # 1Q gate 不参与 V7 的 frontier 调度；保持与 V6 一致。
            return GateAttemptResult(GateScheduleResult.EXECUTED, self.gate_ready_time(gate), "SKIP_NON_2Q")

        if self.is_small_mode:
            super().schedule_gate(gate, specified_time=specified_time, gate_idx=gate_idx)
            return GateAttemptResult(GateScheduleResult.EXECUTED, self.gate_finish_times.get(gate, 0), "")

        ion1, ion2 = qubits[0], qubits[1]
        self.protected_ions = {ion1, ion2}

        try:
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            ion2_time, ion2_trap = self.ion_ready_info(ion2)
            fire_time = max(ready, ion1_time, ion2_time, specified_time)

            q1 = self._qccd_id_of_trap(ion1_trap)
            q2 = self._qccd_id_of_trap(ion2_trap)
            if q1 == q2:
                attempt = self._schedule_intra_qccd_gate(gate, ion1, ion2, gate_type, fire_time)
            else:
                attempt = self._schedule_cross_qccd_gate(gate, ion1, ion2, gate_type, fire_time)

            if attempt.result == GateScheduleResult.EXECUTED:
                self.gate_finish_times[gate] = int(attempt.finish_time)
                self.ion_last_used[ion1] = int(attempt.finish_time)
                self.ion_last_used[ion2] = int(attempt.finish_time)
            return attempt
        except Exception as exc:
            return GateAttemptResult(GateScheduleResult.FATAL, 0, str(exc))
        finally:
            self.protected_ions = set()

    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        """
        对外兼容接口。

        - small 模式保持 V6 行为；
        - large 模式返回 GateAttemptResult，供 V7.run() 判定 EXECUTED/BLOCKED/FATAL。
        """
        if self.is_small_mode:
            return super().schedule_gate(gate, specified_time=specified_time, gate_idx=gate_idx)
        return self._schedule_gate_internal(gate, specified_time=specified_time, gate_idx=gate_idx)

    # ==========================================================
    # 执行性判断：large 允许“同 trap / 同 qccd 本地 / 跨 qccd fiber”三类 ready gate
    # ==========================================================
    def is_executable_local(self, gate):
        """
        与 V6 不同：
        - small: 仍只有“同 trap 立即可执行”才算 local；
        - large: 若两离子已同处 optical trap 且存在 fiber link，也视为可立即执行；
                 若已同 qccd 且同 trap，也视为 local。

        注意：这里的 "local" 只是 frontier 优先级概念，
        不代表完全无需后续动作；真正的调度仍在 schedule_gate 中完成。
        """
        _, qubits, _ = self._gate_payload(gate)
        if len(qubits) != 2:
            return False

        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])

        if self.is_small_mode:
            return t1 == t2

        if t1 == t2:
            return True

        if self._trap_zone_type(t1) == "optical" and self._trap_zone_type(t2) == "optical":
            return self._has_fiber_link_between_traps(t1, t2)

        return False

    # ==========================================================
    # V7 large-mode 主循环：支持 BLOCKED/defer，而不是单门失败即崩溃
    # ==========================================================
    def run(self):
        """
        V7 主循环。

        相比旧实现，当前版本去掉了 gate-attempt 级别的 schedule/sys_state 深拷贝回滚；
        frontier 上每个 gate 只做一次纯规划，成功后再提交，从而与论文 3.2/3.3
        的复杂度设定保持一致。
        """
        if self.is_small_mode:
            return super().run()

        in_degree = {n: self.ir.in_degree(n) for n in self.ir.nodes}
        ready_queue = []
        for n in self.static_topo_list:
            if in_degree[n] == 0:
                ready_queue.append(n)

        total_gates = len(self.ir.nodes)
        processed_count = 0

        self.deferred_queue = collections.deque()
        self.blocked_reason_count = collections.Counter()
        self.last_blocked_reasons = {}
        self.v7_event_log = []
        self.round_idx = 0
        self.remote_gate_count = 0
        self.swap_insert_gate_count = 0
        self.swap_insert_logical_count = 0

        self._runtime_in_degree = in_degree
        self._runtime_ready_queue = ready_queue
        self._runtime_remaining_twoq = set(self.ir.nodes)

        while processed_count < total_gates:
            if not ready_queue:
                self._detect_true_deadlock_or_raise(ready_queue)

            ordered_candidates = self._build_executable_set_fcfs(ready_queue)
            progressed = False

            for gate in ordered_candidates:
                gate_idx = self.static_topo_order.get(gate, 0)
                attempt = self._schedule_gate_internal(gate, gate_idx=gate_idx)

                if attempt.result == GateScheduleResult.EXECUTED:
                    progressed = True
                    if gate in ready_queue:
                        ready_queue.remove(gate)
                    self._runtime_remaining_twoq.discard(gate)
                    self.last_blocked_reasons.pop(gate, None)
                    processed_count += 1

                    succs = list(self.ir.successors(gate))
                    succs.sort(key=lambda x: self.static_topo_order.get(x, float("inf")))
                    for successor in succs:
                        in_degree[successor] -= 1
                        if in_degree[successor] == 0:
                            ready_queue.append(successor)

                    self._emit_event("GATE_EXECUTED", gate_id=gate, finish_time=int(attempt.finish_time))
                    break

                if attempt.result == GateScheduleResult.BLOCKED:
                    self._record_blocked_gate(gate, attempt.reason)
                    continue

                raise RuntimeError(f"V7 fatal scheduling error on gate {gate}: {attempt.reason}")

            if not progressed:
                self._emit_event(
                    "ROUND_NO_PROGRESS",
                    frontier_gate_ids=list(ready_queue),
                    blocked_reasons={g: info.reason for g, info in self.last_blocked_reasons.items() if g in ready_queue},
                    deferred_size=0,
                )
                self._detect_true_deadlock_or_raise(ready_queue)

            self.round_idx += 1

        self._schedule_delayed_one_qubit_gates()
