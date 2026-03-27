"""
ScheduleV7
===========

为 MUSS V7 large-mode 提供与原 schedule.py 事件语义兼容的调度表对象，
并补充两类能力：
1) O(1) 级别的最近事件缓存；
2) 面向 V7 候选试探的增量 mark / rollback。

设计原则：
- 不修改原 schedule.py，避免影响其它调度器；
- 事件 tuple 结构、事件类型编号、打印 / 查询接口尽量保持兼容；
- rollback 只回退“最近一次候选尝试新增的尾部事件”，不做整表 deep copy。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Dict, List, Tuple

from utils import *


class ScheduleV7:
    Gate = 1
    Split = 2
    Merge = 3
    Move = 4

    def __init__(self, machine):
        self.event_id = 0
        # events 始终按 finish time 排序，保持与旧 schedule 的遍历语义一致。
        self.events: List[tuple] = []
        self._finish_times: List[int] = []
        self.machine = machine

        # 增量缓存
        self.last_event_ts = 0
        self.last_comm_event_ts = 0
        self.last_event_time_by_trap: Dict[int, int] = {}
        self.ion_event_end_times: Dict[int, List[int]] = {}
        self.ion_events: Dict[int, List[tuple]] = {}

        # 为 rollback 维护“按写入顺序”的事件日志和缓存恢复点。
        self._append_log: List[tuple] = []
        self._append_restore_log: List[dict] = []

    # ------------------------------------------------------------------
    # 内部：事件注册 / 回滚
    # ------------------------------------------------------------------
    def mark(self) -> int:
        """返回当前追加日志长度，供 V7 候选尝试设置回滚点。"""
        return len(self._append_log)

    def rollback(self, mark: int) -> None:
        """
        回滚到给定 mark。

        只撤销 mark 之后新增的事件与其缓存修改，不触碰更早的历史。
        这样 large-mode 候选失败时的代价只与“本次尝试产生了多少事件”有关。
        """
        mark = int(mark)
        while len(self._append_log) > mark:
            ev = self._append_log.pop()
            restore = self._append_restore_log.pop()
            self._remove_event_instance(ev)
            self._restore_cache_snapshot(restore)
            self.event_id -= 1

    def _remove_event_instance(self, ev: tuple) -> None:
        """从按 finish-time 排序的 events 中删除指定事件实例。"""
        finish = int(ev[3])
        left = bisect_left(self._finish_times, finish)
        right = bisect_right(self._finish_times, finish)
        found = -1
        for idx in range(left, right):
            if self.events[idx] is ev or self.events[idx] == ev:
                found = idx
                break
        if found < 0:
            raise RuntimeError("ScheduleV7 rollback failed: appended event not found")
        self.events.pop(found)
        self._finish_times.pop(found)

    def _restore_cache_snapshot(self, restore: dict) -> None:
        self.last_event_ts = int(restore["last_event_ts"])
        self.last_comm_event_ts = int(restore["last_comm_event_ts"])

        trap_id = restore.get("trap_id")
        if trap_id is not None:
            if restore["trap_existed"]:
                self.last_event_time_by_trap[int(trap_id)] = int(restore["trap_old"])
            else:
                self.last_event_time_by_trap.pop(int(trap_id), None)

        for ion, info in restore.get("ions", {}).items():
            ion = int(ion)
            old_len = int(info["old_len"])
            if old_len == 0:
                self.ion_event_end_times.pop(ion, None)
                self.ion_events.pop(ion, None)
                continue
            self.ion_event_end_times[ion] = self.ion_event_end_times.get(ion, [])[:old_len]
            self.ion_events[ion] = self.ion_events.get(ion, [])[:old_len]

    def _register_event(self, ev: tuple):
        _, etype, _start_time, end_time, info = ev
        end_time = int(end_time)

        # 记录缓存恢复点（只保存本事件会改到的局部字段）
        restore = {
            "last_event_ts": int(self.last_event_ts),
            "last_comm_event_ts": int(self.last_comm_event_ts),
            "trap_id": None,
            "trap_existed": False,
            "trap_old": 0,
            "ions": {},
        }

        if etype in [ScheduleV7.Gate, ScheduleV7.Split, ScheduleV7.Merge]:
            trap_id = info.get("trap")
            if trap_id is not None:
                trap_id = int(trap_id)
                restore["trap_id"] = trap_id
                restore["trap_existed"] = trap_id in self.last_event_time_by_trap
                restore["trap_old"] = int(self.last_event_time_by_trap.get(trap_id, 0))

        for ion in info.get("ions", []):
            ion = int(ion)
            restore["ions"][ion] = {
                "old_len": len(self.ion_event_end_times.get(ion, [])),
            }

        pos = bisect_right(self._finish_times, end_time)
        self._finish_times.insert(pos, end_time)
        self.events.insert(pos, ev)
        self._append_log.append(ev)
        self._append_restore_log.append(restore)
        self.event_id += 1

        self.last_event_ts = max(self.last_event_ts, end_time)
        if etype in [ScheduleV7.Split, ScheduleV7.Move, ScheduleV7.Merge]:
            self.last_comm_event_ts = max(self.last_comm_event_ts, end_time)

        if etype in [ScheduleV7.Gate, ScheduleV7.Split, ScheduleV7.Merge]:
            trap_id = info.get("trap")
            if trap_id is not None:
                trap_id = int(trap_id)
                self.last_event_time_by_trap[trap_id] = max(self.last_event_time_by_trap.get(trap_id, 0), end_time)

        for ion in info.get("ions", []):
            ion = int(ion)
            self.ion_event_end_times.setdefault(ion, []).append(end_time)
            self.ion_events.setdefault(ion, []).append(ev)

    # ------------------------------------------------------------------
    # 事件写入接口（与原 schedule.py 保持兼容）
    # ------------------------------------------------------------------
    def add_gate(self, start_time, end_time, ions, trap_id, gate_type=None, is_fiber=False, zone_type=None, gate_id=None):
        gate_dict = {
            "ions": ions,
            "trap": trap_id,
        }
        if gate_type is not None:
            gate_dict["gate_type"] = gate_type
        if is_fiber:
            gate_dict["is_fiber"] = True
        if zone_type is not None:
            gate_dict["zone_type"] = zone_type
        if gate_id is not None:
            gate_dict["gate_id"] = gate_id
        self._register_event((self.event_id, ScheduleV7.Gate, start_time, end_time, gate_dict))

    def add_split_or_merge(self, start_time, end_time, ions, trap_id, seg_id, op_type, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops):
        split_dict = {
            "ions": ions,
            "trap": trap_id,
            "seg": seg_id,
            "swap_cnt": split_swap_count,
            "swap_hops": split_swap_hops,
            "ion_hops": ion_swap_hops,
            "i1": i1,
            "i2": i2,
        }
        self._register_event((self.event_id, op_type, start_time, end_time, split_dict))

    def add_move(self, start_time, end_time, ions, seg_id1, seg_id2):
        move_dict = {
            "ions": ions,
            "source_seg": seg_id1,
            "dest_seg": seg_id2,
        }
        self._register_event((self.event_id, ScheduleV7.Move, start_time, end_time, move_dict))

    # ------------------------------------------------------------------
    # 与原 schedule.py 兼容的查询 / 打印接口
    # ------------------------------------------------------------------
    def print_stats(self):
        cnt_splits = 0
        cnt_merge = 0
        cnt_moves = 0
        cnt_gates = 0
        for item in list(self.events):
            if item[1] == ScheduleV7.Gate:
                cnt_gates += 1
            elif item[1] == ScheduleV7.Split:
                cnt_splits += 1
            elif item[1] == ScheduleV7.Move:
                cnt_moves += 1
            elif item[1] == ScheduleV7.Merge:
                cnt_merge += 1
        print("Split:", cnt_splits, "Merge:", cnt_merge, "Moves:", cnt_moves, "Gates:", cnt_gates)
        return [cnt_splits, cnt_merge, cnt_moves, cnt_gates]

    def print_events(self):
        for item in list(self.events):
            if item[1] == ScheduleV7.Gate:
                print("GAT", item[4]["ions"], trap_name(item[4]["trap"]), (item[2], item[3]))
            elif item[1] == ScheduleV7.Split:
                print("SPL", item[4]["ions"], trap_name(item[4]["trap"]) + "->" + seg_name(item[4]["seg"]), (item[2], item[3]))
            elif item[1] == ScheduleV7.Move:
                print("MOV", item[4]["ions"], seg_name(item[4]["source_seg"]) + "->" + seg_name(item[4]["dest_seg"]), (item[2], item[3]))
            elif item[1] == ScheduleV7.Merge:
                print("MER", item[4]["ions"], seg_name(item[4]["seg"]) + "->" + trap_name(item[4]["trap"]), (item[2], item[3]))

    def get_last_event_ts(self):
        return int(self.last_event_ts)

    def events_ge_ts(self, ts):
        idx = bisect_left(self._finish_times, int(ts))
        return self.events[idx:]

    def events_lt_ts(self, ts):
        idx = bisect_left(self._finish_times, int(ts))
        return self.events[:idx]

    def events_in_interval(self, ts1, ts2):
        ret_list = []
        for item in self.events_ge_ts(ts1):
            st = item[2]
            fin = item[3]
            if fin <= ts1:
                continue
            if st >= ts2:
                continue
            ret_list.append(item)
        return ret_list

    def last_ion_event_before_ts(self, ts, ion_id):
        ion_id = int(ion_id)
        ends = self.ion_event_end_times.get(ion_id, [])
        if not ends:
            return []
        idx = bisect_left(ends, int(ts)) - 1
        if idx >= 0:
            return self.ion_events[ion_id][idx]
        return []

    def filter_gate_ops(self, items):
        return list(filter(lambda x: x[1] == ScheduleV7.Gate, items))

    def filter_seg_ops(self, items):
        return list(filter(lambda x: x[1] > ScheduleV7.Gate, items))

    def filter_by_ion(self, items, ion_id):
        if items is self.events:
            return list(self.ion_events.get(int(ion_id), []))
        return list(filter(lambda x: ion_id in x[4]["ions"], items))

    def filter_by_segment(self, items, segment_id):
        return list(
            filter(
                lambda x: ((x[1] == ScheduleV7.Split or x[1] == ScheduleV7.Merge) and (x[4]["seg"] == segment_id))
                or (x[1] == ScheduleV7.Move and (x[4]["source_seg"] == segment_id or x[4]["dest_seg"] == segment_id)),
                items,
            )
        )

    def filter_by_trap(self, items, trap_id):
        return list(filter(lambda x: ((x[1] == ScheduleV7.Gate or x[1] == ScheduleV7.Split or x[1] == ScheduleV7.Merge) and (trap_id == x[4]["trap"])), items))

    def filter_by_junction(self, items, junction):
        seg_list = []
        for v in self.machine.graph[junction]:
            seg_list.append(self.machine.graph[junction][v]["seg"].id)
        return list(filter(lambda x: ((x[1] == ScheduleV7.Move) and (x[4]["source_seg"] in seg_list and x[4]["dest_seg"] in seg_list)), items))

    def last_event_time_on_trap(self, trap_id):
        return int(self.last_event_time_by_trap.get(int(trap_id), 0))

    def last_comm_event_time(self):
        return int(self.last_comm_event_ts)

    def identify_start_time(self, move_path, start_time, est_time):
        events_after_st = self.filter_seg_ops(self.events_ge_ts(start_time))
        max_time = start_time
        for i in range(len(move_path) - 1):
            u = move_path[i]
            v = move_path[i + 1]
            segment = self.machine.graph[u][v]["seg"]
            rel_events = self.filter_by_segment(events_after_st, segment.id)
            if len(rel_events):
                last_event = rel_events[-1]
                max_time = max(max_time, last_event[3])
        return max_time

    def junction_traffic_crossing(self, src_seg, dest_seg, junct, start_time, end_time):
        events_after_st = self.filter_seg_ops(self.events_ge_ts(start_time))
        worst_case_start_time = 0
        duration = end_time - start_time
        for item in events_after_st:
            worst_case_start_time = max(worst_case_start_time, item[3])
        for i in range(start_time, worst_case_start_time + 1):
            overlapping_events = self.filter_by_junction(self.events_in_interval(i, i + duration), junct)
            if len(overlapping_events) == 0:
                return i, i + duration
        return start_time, end_time

    def pretty_print(self, num_traps, num_segs):
        last_time = self.get_last_event_ts()
        out = []
        for _i in range(last_time):
            out.append(["          "] * (num_traps + num_segs))
        for item in list(self.events):
            if item[1] == ScheduleV7.Gate:
                trap = item[4]["trap"]
                txt = "G(" + str(item[4]["ions"][0]) + "," + str(item[4]["ions"][1]) + "," + str(trap) + ")"
                for i in range(item[2], item[3]):
                    out[i][trap] = "{:<10}".format(txt)
            elif item[1] == ScheduleV7.Split:
                trap = item[4]["trap"]
                seg = item[4]["seg"]
                txt = "S(" + str(item[4]["ions"][0]) + "," + str(trap) + "," + str(seg) + ")"
                for i in range(item[2], item[3]):
                    out[i][item[4]["trap"]] = "{:<10}".format(txt)
                    out[i][num_traps + item[4]["seg"]] = "{:<10}".format(txt)
            elif item[1] == ScheduleV7.Move:
                txt = "Tr(" + str(item[4]["ions"][0]) + "," + str(item[4]["source_seg"]) + "," + str(item[4]["dest_seg"]) + ")"
                for i in range(item[2], item[3]):
                    out[i][num_traps + item[4]["source_seg"]] = "{:<10}".format(txt)
                    out[i][num_traps + item[4]["dest_seg"]] = "{:<10}".format(txt)
            elif item[1] == ScheduleV7.Merge:
                trap = item[4]["trap"]
                seg = item[4]["seg"]
                txt = "M(" + str(item[4]["ions"][0]) + "," + str(trap) + "," + str(seg) + ")"
                for i in range(item[2], item[3]):
                    out[i][item[4]["trap"]] = "{:<10}".format(txt)
                    out[i][num_traps + item[4]["seg"]] = "{:<10}".format(txt)
        for item in out:
            print(item)
