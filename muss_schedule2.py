import networkx as nx
import numpy as np
import collections
from machine_state import MachineState
from utils import *
from route import *
from schedule import *
from machine import Trap, Segment, Junction
from rebalance import *


# 严格实现论文要求版本。
class MUSSSchedule:
    # Inputs are
    # 1. gate dependency graph - IR
    # 2. gate_info = what are the qubits used by a two-qubit gate?
    # 3. M = machine object
    # 4. init_map = initial qubit mapping
    def __init__(self, ir, gate_info, M, init_map, serial_trap_ops, serial_comm, global_serial_lock):
        self.ir = ir
        self.gate_info = gate_info
        self.machine = M
        self.init_map = init_map

        # Setup scheduler
        self.machine.add_comm_capacity(2)
        # Add space for 2 extra ions in each trap
        self.SerialTrapOps = serial_trap_ops
        self.SerialCommunication = serial_comm
        self.GlobalSerialLock = global_serial_lock

        self.schedule = Schedule(M)
        self.router = BasicRoute(M)
        self.gate_finish_times = {}

        # Scheduling statistics
        self.count_rebalance = 0
        self.split_swap_counter = 0

        # Create sys_state
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

        # Precompute distances
        if not hasattr(self.machine, "dist_cache") or not self.machine.dist_cache:
            self.machine.precompute_distances()

        # === MUSS Strict Requirement: LRU Tracking ===
        # Track the time when an ion was last used.
        # Initialized to -1.
        all_ions = set()
        for t_ions in trap_ions.values():
            all_ions.update(t_ions)
        self.ion_last_used = {ion: -1 for ion in all_ions}

        # Protect currently active ions from being evicted during rebalancing
        self.protected_ions = set()

        # Precompute static order for Lookahead/FCFS tie-breaking
        try:
            self.static_topo_list = list(nx.topological_sort(self.ir))
        except:
            self.static_topo_list = list(self.ir.nodes)
        self.static_topo_order = {g: i for i, g in enumerate(self.static_topo_list)}

        # Access self.gates for lookahead functions (compatibility)
        self.gates = self.static_topo_list

    # Find the earliest time at which a gate can be scheduled
    def gate_ready_time(self, gate):
        ready_time = 0
        for in_edge in self.ir.in_edges(gate):
            in_gate = in_edge[0]
            if in_gate in self.gate_finish_times:
                ready_time = max(ready_time, self.gate_finish_times[in_gate])
            else:
                continue
        return ready_time

    # Find the time at which a particular qubit/ion is ready for another operation
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

    # Basic Operations (Split, Merge, Move, Gate)
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
        split_duration, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops = m.split_time(self.sys_state, src_trap.id, dest_seg.id, ion)
        self.split_swap_counter += split_swap_count
        split_end = split_start + split_duration
        self.schedule.add_split_or_merge(split_start, split_end, [ion], src_trap.id, dest_seg.id, Schedule.Split, split_swap_count, split_swap_hops, i1, i2, ion_swap_hops)
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
        merge_end = merge_start + m.merge_time(dest_trap.id)
        self.schedule.add_split_or_merge(merge_start, merge_end, [ion], dest_trap.id, src_seg.id, Schedule.Merge, 0, 0, 0, 0, 0)
        return merge_end

    def add_move_op(self, clk, src_seg, dest_seg, junct, ion):
        m = self.machine
        move_start = clk
        if self.GlobalSerialLock == 1:
            last_event_time_in_system = self.schedule.get_last_event_ts()
            move_start = max(move_start, last_event_time_in_system)

        if self.SerialCommunication == 1:
            last_comm_time = self.schedule.last_comm_event_time()
            move_start = max(move_start, last_comm_time)

        move_end = move_start + m.move_time(src_seg.id, dest_seg.id) + m.junction_cross_time(junct)
        move_start, move_end = self.schedule.junction_traffic_crossing(src_seg, dest_seg, junct, move_start, move_end)
        self.schedule.add_move(move_start, move_end, [ion], src_seg.id, dest_seg.id)
        return move_end

    def add_gate_op(self, clk, trap_id, gate, ion1, ion2):
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

    # === Lookahead Logic for Shuttling (Compatible with User Preference) ===
    def get_future_score(self, ion, current_gate_idx, target_trap_id):
        score = 0
        gamma = 0.7
        weight = 1.0
        lookahead_depth = 8
        found_count = 0

        # Use static topological list for lookahead
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
        m = self.machine
        ALPHA = 0.8

        if current_gate_idx is None:
            return ion1_trap, ion2_trap

        # Option A: Meet at Trap 2 (Ion 1 moves)
        cost_current_move1 = m.dist_cache.get((ion1_trap, ion2_trap), 100)
        future_score_1 = self.get_future_score(ion1, current_gate_idx, ion2_trap)
        future_score_2 = self.get_future_score(ion2, current_gate_idx, ion2_trap)
        total_score_meet_at_t2 = cost_current_move1 + ALPHA * (future_score_1 + future_score_2)

        # Option B: Meet at Trap 1 (Ion 2 moves)
        cost_current_move2 = m.dist_cache.get((ion2_trap, ion1_trap), 100)
        future_score_1_at_t1 = self.get_future_score(ion1, current_gate_idx, ion1_trap)
        future_score_2_at_t1 = self.get_future_score(ion2, current_gate_idx, ion1_trap)
        total_score_meet_at_t1 = cost_current_move2 + ALPHA * (future_score_1_at_t1 + future_score_2_at_t1)

        # Capacity Constraints
        ss = self.sys_state
        cap1 = m.traps[ion1_trap].capacity - len(ss.trap_ions[ion1_trap])
        cap2 = m.traps[ion2_trap].capacity - len(ss.trap_ions[ion2_trap])

        if cap1 <= 0 and cap2 > 0:
            return ion1_trap, ion2_trap
        if cap2 <= 0 and cap1 > 0:
            return ion2_trap, ion1_trap

        if total_score_meet_at_t2 < total_score_meet_at_t1:
            return ion1_trap, ion2_trap
        else:
            return ion2_trap, ion1_trap

    def fire_shuttle(self, src_trap, dest_trap, ion, gate_fire_time, route=[]):
        s = self.schedule
        m = self.machine
        if len(route):
            rpath = route
        else:
            rpath = self.router.find_route(src_trap, dest_trap)

        t_est = 0
        for i in range(len(rpath) - 1):
            src = rpath[i]
            dest = rpath[i + 1]
            if type(src) == Trap and type(dest) == Junction:
                t_est += m.mparams.split_merge_time
            elif type(src) == Junction and type(dest) == Junction:
                t_est += m.move_time(src.id, dest.id)
            elif type(src) == Junction and type(dest) == Trap:
                t_est += m.merge_time(dest.id)

        clk = self.schedule.identify_start_time(rpath, gate_fire_time, t_est)
        clk = self._add_shuttle_ops(rpath, ion, clk)
        return clk

    def _add_shuttle_ops(self, spath, ion, clk):
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
                self.sys_state.trap_ions[spath[trap_pos[i + 1]].id].append(ion)
            else:
                self.sys_state.trap_ions[spath[trap_pos[i + 1]].id].insert(0, ion)
        return clk

    def _add_partial_shuttle_ops(self, spath, ion, clk):
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

    # === MUSS Strict Requirement: Conflict Handling via LRU ===
    def rebalance_traps(self, focus_traps, fire_time):
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
        self.count_rebalance += 1
        rebal = RebalanceTraps(self.machine, self.sys_state)
        # RebalanceTraps calculates how many ions need to move from Trap A to B to satisfy capacity
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
                # Find a path in the flow graph
                for tnode in T:
                    if T.out_degree(tnode) == 0 and tnode != node:
                        try:
                            shuttle_route = nx.shortest_path(T, node, tnode)
                            break
                        except:
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

                # === LRU EVICTION IMPLEMENTATION ===
                # Instead of taking the first ion, we take the Least Recently Used ion
                # that is NOT in the protected set (currently active).
                candidates = self.sys_state.trap_ions[node.id]
                valid_candidates = [ion for ion in candidates if ion not in self.protected_ions]

                if not valid_candidates:
                    # If all are protected (rare), fallback to first available
                    moving_ion = candidates[0]
                else:
                    # Sort by last_used time (ascending). -1 or small timestamp means unused for longest duration.
                    # According to MUSS paper: "prioritizing eviction of qubits that have remained unused for the longest duration"
                    moving_ion = min(valid_candidates, key=lambda ion: self.ion_last_used.get(ion, -1))

                ion_time, _ = self.ion_ready_info(moving_ion)
                fire_time = max(fire_time, ion_time)

                fin_time_new = self.fire_shuttle(node.id, tnode.id, moving_ion, fire_time, route=shuttle_route)
                fin_time = max(fin_time, fin_time_new)
        return fin_time

    # === Main Gate Scheduling Logic ===
    def schedule_gate(self, gate, specified_time=0, gate_idx=None):
        if gate not in self.gate_info:
            return

        gate_data = self.gate_info[gate]
        qubits = gate_data if isinstance(gate_data, list) else gate_data["qubits"]

        # Set Protected Ions to prevent eviction of these specific qubits during this operation
        self.protected_ions = set(qubits)

        finish_time = 0

        if len(qubits) == 1:
            ion1 = qubits[0]
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            fire_time = max(ready, ion1_time, specified_time)

            duration = 5
            if hasattr(self.machine, "single_qubit_gate_time"):
                duration = self.machine.single_qubit_gate_time(gate_data.get("type", "u3"))

            self.schedule.add_gate(fire_time, fire_time + duration, [ion1], ion1_trap)
            self.gate_finish_times[gate] = fire_time + duration
            finish_time = fire_time + duration

            # Update LRU
            self.ion_last_used[ion1] = finish_time

        elif len(qubits) == 2:
            ion1 = qubits[0]
            ion2 = qubits[1]
            ready = self.gate_ready_time(gate)
            ion1_time, ion1_trap = self.ion_ready_info(ion1)
            ion2_time, ion2_trap = self.ion_ready_info(ion2)
            fire_time = max(ready, ion1_time, ion2_time, specified_time)

            if ion1_trap == ion2_trap:
                gate_duration = self.machine.gate_time(self.sys_state, ion1_trap, ion1, ion2)
                self.schedule.add_gate(fire_time, fire_time + gate_duration, [ion1, ion2], ion1_trap)
                self.gate_finish_times[gate] = fire_time + gate_duration
                finish_time = fire_time + gate_duration
            else:
                rebal_flag, new_fin_time = self.rebalance_traps(focus_traps=[ion1_trap, ion2_trap], fire_time=fire_time)

                if not rebal_flag:
                    source_trap, dest_trap = self.shuttling_direction(ion1_trap, ion2_trap, ion1, ion2, gate_idx)
                    moving_ion = ion1 if source_trap == ion1_trap else ion2

                    clk = self.fire_shuttle(source_trap, dest_trap, moving_ion, fire_time)

                    gate_duration = self.machine.gate_time(self.sys_state, dest_trap, ion1, ion2)
                    self.schedule.add_gate(clk, clk + gate_duration, [ion1, ion2], dest_trap)
                    self.gate_finish_times[gate] = clk + gate_duration
                    finish_time = clk + gate_duration
                else:
                    # Clear protection before recursion (recursion will re-set it)
                    self.protected_ions = set()
                    self.schedule_gate(gate, specified_time=new_fin_time, gate_idx=gate_idx)
                    return  # Recursion handles the rest

            # Update LRU
            self.ion_last_used[ion1] = finish_time
            self.ion_last_used[ion2] = finish_time

        # Clear Protection
        self.protected_ions = set()

    def is_executable_local(self, gate):
        """Helper to check if gate can be executed without movement"""
        if gate not in self.gate_info:
            return True
        qubits = self.gate_info[gate]
        if isinstance(qubits, dict):
            qubits = qubits["qubits"]

        if len(qubits) < 2:
            return True
        _, t1 = self.ion_ready_info(qubits[0])
        _, t2 = self.ion_ready_info(qubits[1])
        return t1 == t2

    # === MUSS Strict Requirement: Dynamic Frontier Scheduling ===
    def run(self):
        # Initial Frontier
        in_degree = {n: self.ir.in_degree(n) for n in self.ir.nodes}
        ready_gates = [n for n in self.ir.nodes if in_degree[n] == 0]

        processed_count = 0
        total_gates = len(self.ir.nodes)

        while processed_count < total_gates:
            if not ready_gates:
                break  # Should not happen unless cycle

            # Filter gates that are LOCAL (no move required)
            # MUSS: "prioritize executing those that can be executed right away"
            local_candidates = []
            remote_candidates = []

            for g in ready_gates:
                if self.is_executable_local(g):
                    local_candidates.append(g)
                else:
                    remote_candidates.append(g)

            best_gate = None

            # Tie-breaking rule: "first-come, first-served" (Static Topological Index)
            if local_candidates:
                best_gate = min(local_candidates, key=lambda x: self.static_topo_order.get(x, float("inf")))
            else:
                # If no local gates, pick the oldest available remote gate
                best_gate = min(remote_candidates, key=lambda x: self.static_topo_order.get(x, float("inf")))

            # Schedule the selected gate
            gate_idx = self.static_topo_order.get(best_gate, 0)
            self.schedule_gate(best_gate, gate_idx=gate_idx)

            # Remove from frontier and update successors
            ready_gates.remove(best_gate)
            processed_count += 1

            for successor in self.ir.successors(best_gate):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    ready_gates.append(successor)
