"""
Machine state snapshot and manipulation helpers.

本文件保留原项目对 MachineState 的最小接口，同时补充少量“第一阶段 large 架构”
真正会用到的只读辅助函数。这里不引入调度策略逻辑，避免影响小规模路径。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional

from utils import *


class MachineState:
    """
    Machine state at a given point is:
      - which qubits are sitting in which traps    --> self.trap_ions
      - which qubits are sitting in which segments --> self.seg_ions

    兼容性原则：
      - 继续保留原来的构造参数与 process_split/process_merge/process_move。
      - 新增的方法均为纯辅助接口，不改变既有行为。
    """

    def __init__(self, ts, trap_ions, seg_ions):
        self.ts = ts  # timestamp
        self.trap_ions = trap_ions
        self.seg_ions = seg_ions

        # 第一阶段仅添加无副作用元数据容器，后续 large scheduler 可复用。
        self.last_used_time = {}
        # 与 V7 大规模路径统一命名；保持与旧字段指向同一份字典，确保完全兼容。
        self.last_used_ts = self.last_used_time

    # ------------------------------------------------------------------
    # 原有状态更新接口（保持行为不变）
    # ------------------------------------------------------------------
    def process_split(self, trap, seg, ions):
        """将 ions 从 trap 移到 seg。"""
        if seg not in self.seg_ions:
            self.seg_ions[seg] = []
        for ion in ions:
            self.trap_ions[trap].remove(ion)
            self.seg_ions[seg].append(ion)

    def process_merge(self, trap, seg, ions):
        """将 ions 从 seg 合并回 trap。"""
        if trap not in self.trap_ions:
            self.trap_ions[trap] = []
        for ion in ions:
            self.trap_ions[trap].append(ion)
            self.seg_ions[seg].remove(ion)

    def process_move(self, seg1, seg2, ions):
        """将 ions 从 seg1 移到 seg2。"""
        if seg2 not in self.seg_ions:
            self.seg_ions[seg2] = []
        for ion in ions:
            self.seg_ions[seg1].remove(ion)
            self.seg_ions[seg2].append(ion)

    # ------------------------------------------------------------------
    # 原有查询接口（保持行为不变）
    # ------------------------------------------------------------------
    def find_trap_id_by_ion(self, ion_id):
        for trap_id in self.trap_ions.keys():
            if ion_id in self.trap_ions[trap_id]:
                return trap_id
        return -1

    def check_ion_in_a_trap(self, ion_id):
        if self.find_trap_id_by_ion(ion_id) != -1:
            return 1
        return 0

    def print_state(self):
        print("Machine State")
        for t in self.trap_ions.keys():
            print(trap_name(t), len(self.trap_ions[t]), self.trap_ions[t])
        # for s in self.seg_ions.keys():
        #     print(seg_name(s), len(self.seg_ions[s]), self.seg_ions[s])

    # ------------------------------------------------------------------
    # 第一阶段新增的只读/低风险辅助接口
    # ------------------------------------------------------------------
    def clone(self):
        """返回当前状态的深拷贝，便于 large path 试探性规划。"""
        copied = MachineState(self.ts, deepcopy(self.trap_ions), deepcopy(self.seg_ions))
        # 注意：__init__ 中要求 last_used_ts 与 last_used_time 指向同一份字典。
        # 如果这里只重写 copied.last_used_time 而不同时回绑 last_used_ts，
        # 那么 clone 后两个名字会指向不同对象，V7 的快照/回滚路径就会出现
        # “last_used_time 已复制、last_used_ts 却还是旧空字典”的一致性问题。
        copied.last_used_time = deepcopy(self.last_used_time)
        copied.last_used_ts = copied.last_used_time
        return copied

    def ions_in_trap(self, trap_id) -> List[int]:
        return list(self.trap_ions.get(trap_id, []))

    def ions_in_segment(self, seg_id) -> List[int]:
        return list(self.seg_ions.get(seg_id, []))

    def trap_load(self, trap_id) -> int:
        return len(self.trap_ions.get(trap_id, []))

    def segment_load(self, seg_id) -> int:
        return len(self.seg_ions.get(seg_id, []))

    def find_segment_id_by_ion(self, ion_id):
        for seg_id in self.seg_ions.keys():
            if ion_id in self.seg_ions[seg_id]:
                return seg_id
        return -1

    def ion_location(self, ion_id):
        """
        返回离子所在位置：
          ("trap", trap_id) / ("segment", seg_id) / (None, -1)
        """
        tid = self.find_trap_id_by_ion(ion_id)
        if tid != -1:
            return ("trap", tid)

        sid = self.find_segment_id_by_ion(ion_id)
        if sid != -1:
            return ("segment", sid)

        return (None, -1)

    def touch_ions(self, ions, ts=None):
        """记录 ions 的最近使用时间；small path 不依赖，大路径可直接复用。"""
        stamp = self.ts if ts is None else ts
        for ion in ions:
            self.last_used_time[ion] = stamp

    def touch_ion(self, ion, ts=None):
        """V7 兼容包装：记录单个离子的最近使用时间。"""
        self.touch_ions([ion], ts=ts)

    def most_idle_ion(self, ions: Iterable[int]):
        """从给定集合中按最近最少使用（LRU）挑一个；若没有记录，则按 ion id 稳定打破平局。"""
        ions = list(ions)
        if not ions:
            return None

        def key_fn(ion_id):
            return (self.last_used_time.get(ion_id, -1), ion_id)

        return min(ions, key=key_fn)
