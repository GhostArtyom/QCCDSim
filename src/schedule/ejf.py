"""
EJF Schedule - Earliest Job First scheduler

迁移自项目根目录 ejf_schedule.py
"""

import networkx as nx
import numpy as np

from src.machine.state import MachineState
from src.route.basic import BasicRoute
from src.schedule.events import Schedule
from src.machine.core import Trap, Segment, Junction


class EJFSchedule:
    """Earliest Job First scheduler.

    Schedules gates in topologically sorted order.
    Policy is similar to the famous job scheduling policy.
    """

    def __init__(self, ir, gate_info, M, init_map, serial_trap_ops, serial_comm, global_serial_lock):
        self.ir = ir
        self.gate_info = gate_info
        self.machine = M
        self.init_map = init_map

        self.SerialTrapOps = serial_trap_ops
        self.SerialCommunication = serial_comm
        self.GlobalSerialLock = global_serial_lock

        self.schedule = Schedule(M)
        self.router = BasicRoute(M)
        self.gate_finish_times = {}

        self.count_rebalance = 0
        self.split_swap_counter = 0

        # Create system state
        trap_ions = {}
        seg_ions = {}
        for i in M.traps:
            if init_map[i.id]:
                trap_ions[i.id] = init_map[i.id][:]
            else:
                trap_ions[i.id] = []
        for i in M.segments:
            seg_ions[i.id] = []
        self.sys_state = MachineState(0, trap_ions, seg_ions)

    def gate_ready_time(self, gate):
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
        return ready_time

    def ion_ready_info(self, ion_id):
        s = self.schedule
        this_ion_ops = s.filter_by_ion(s.events, ion_id)
        this_ion_last_op_time = 0
        this_ion_trap = None
        if len(this_ion_ops):
            assert (this_ion_ops[-1][1] == Schedule.Gate) or (this_ion_ops[-1][1] == Schedule.Merge)
            this_ion_last_op_time = this_ion_ops[-1][3]
            this_ion_trap = this_ion_ops[-1][4]["trap"]
        else:
            did_not_find = True
            for trap_id in self.init_map.keys():
                if ion_id in self.init_map[trap_id]:
                    this_ion_trap = trap_id
                    did_not_find = False
                    break
            if did_not_find:
                print("Did not find:", ion_id)
            assert did_not_find == False

        if this_ion_trap != self.sys_state.find_trap_id_by_ion(ion_id):
            print(ion_id, this_ion_trap, self.sys_state.find_trap_id_by_ion(ion_id))
            self.sys_state.print_state()
            assert 0
        return this_ion_last_op_time, this_ion_trap

    def add_split_op(self, clk, src_trap, dest_seg, ion):
        m = self.machine
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

        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops = m.split_time(
            self.sys_state, src_trap.id, dest_seg.id, ion
        )
        self.split_swap_counter += split_swap_count
        split_end = split_start + split_duration
        self.schedule.add_split_or_merge(
            split_start, split_end, [ion], src_trap.id, dest_seg.id,
            Schedule.Split, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops
        )
        return split_end

    def add_merge_op(self, clk, dest_trap, src_seg, ion):
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

        merge_duration = m.merge_time(dest_trap.id)
        merge_end = merge_start + merge_duration
        self.schedule.add_split_or_merge(
            merge_start, merge_end, [ion], dest_trap.id, src_seg.id,
            Schedule.Merge, 0, 0, 0, 0, 0
        )
        return merge_end

    def schedule_gate(self, gate, gate_ready_clock):
        from src.schedule.events import Schedule as Sched

        q1, q2 = self.gate_info[gate]
        ion1 = q1
        ion2 = q2

        # Get ion locations
        ion1_ready_time, ion1_trap = self.ion_ready_info(ion1)
        ion2_ready_time, ion2_trap = self.ion_ready_info(ion2)

        start_time = max(gate_ready_clock, ion1_ready_time, ion2_ready_time)

        # If ions are in the same trap, just execute the gate
        if ion1_trap == ion2_trap:
            end_time = start_time + self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
            self.schedule.add_gate(start_time, end_time, [ion1, ion2], ion1_trap, gate_id=gate)
            self.gate_finish_times[gate] = end_time
            return end_time

        # If ions are in different traps, shuttle one ion to the other
        route = self.router.find_route(ion1_trap, ion2_trap)
        assert len(route) >= 2

        # Move ion1 along the route
        for i in range(len(route) - 1):
            src = route[i]
            dst = route[i + 1]
            if isinstance(src, Trap) and isinstance(dst, Trap):
                continue

            src_id = src.id if hasattr(src, 'id') else src
            dst_id = dst.id if hasattr(dst, 'id') else dst

            if isinstance(src, Trap):
                seg = self.machine.graph[src][dst]['seg']
                end_time = self.add_split_op(start_time, src, seg, ion1)
                start_time = self.add_merge_op(end_time, dst, seg, ion1)
            elif isinstance(dst, Trap):
                seg = self.machine.graph[src][dst]['seg']
                end_time = self.add_split_op(start_time, dst, seg, ion1)
                start_time = self.add_merge_op(end_time, src, seg, ion1)

        # Execute gate
        end_time = start_time + self.machine.gate_time(self.sys_state, ion2_trap, ion1, ion2)
        self.schedule.add_gate(start_time, end_time, [ion1, ion2], ion2_trap, gate_id=gate)
        self.gate_finish_times[gate] = end_time
        return end_time

    def run(self):
        """执行调度"""
        sorted_gates = list(nx.topological_sort(self.ir))
        for gate in sorted_gates:
            if gate not in self.gate_info:
                continue
            gate_ready_clock = self.gate_ready_time(gate)
            self.schedule_gate(gate, gate_ready_clock)
        return self.schedule
