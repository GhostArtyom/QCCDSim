"""
Schedule events - event data structure for ion shuttling schedules

迁移自项目根目录 schedule.py
"""

from operator import itemgetter

from src.utils.sorted_collection import SortedCollection


class Schedule:
    """Schedule events sorted by finish time.

    Event tuple structure:
        (id, event_type, start_time, end_time, event_info_dict)

    Event types:
        Gate = 1
        Split = 2
        Merge = 3
        Move = 4
    """

    Gate = 1
    Split = 2
    Merge = 3
    Move = 4

    def __init__(self, machine):
        self.event_id = 0
        self.events = SortedCollection(key=itemgetter(3))  # sorted by finish time
        self.machine = machine

    def add_gate(self, start_time, end_time, ions, trap_id, gate_type=None, is_fiber=False, zone_type=None, gate_id=None):
        gate_dict = {"ions": ions, "trap": trap_id}
        if gate_type is not None:
            gate_dict["gate_type"] = gate_type
        if is_fiber:
            gate_dict["is_fiber"] = True
        if zone_type is not None:
            gate_dict["zone_type"] = zone_type
        if gate_id is not None:
            gate_dict["gate_id"] = gate_id
        self.events.insert((self.event_id, Schedule.Gate, start_time, end_time, gate_dict))
        self.event_id += 1

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
        self.events.insert((self.event_id, op_type, start_time, end_time, split_dict))
        self.event_id += 1

    def add_move(self, start_time, end_time, ions, seg_id1, seg_id2):
        move_dict = {
            "ions": ions,
            "source_seg": seg_id1,
            "dest_seg": seg_id2,
        }
        self.events.insert((self.event_id, Schedule.Move, start_time, end_time, move_dict))
        self.event_id += 1

    def print_stats(self):
        cnt_splits = 0
        cnt_merge = 0
        cnt_moves = 0
        cnt_gates = 0
        for item in list(self.events):
            if item[1] == Schedule.Gate:
                cnt_gates += 1
            elif item[1] == Schedule.Split:
                cnt_splits += 1
            elif item[1] == Schedule.Move:
                cnt_moves += 1
            elif item[1] == Schedule.Merge:
                cnt_merge += 1
        print("Split:", cnt_splits, "Merge:", cnt_merge, "Moves:", cnt_moves, "Gates:", cnt_gates)
        return [cnt_splits, cnt_merge, cnt_moves, cnt_gates]

    def print_events(self):
        from src.utils.helpers import trap_name, seg_name
        for item in list(self.events):
            if item[1] == Schedule.Gate:
                print("GAT", item[4]["ions"], trap_name(item[4]["trap"]), (item[2], item[3]))
            elif item[1] == Schedule.Split:
                print("SPL", item[4]["ions"], trap_name(item[4]["trap"]) + "->" + seg_name(item[4]["seg"]), (item[2], item[3]))
            elif item[1] == Schedule.Move:
                print("MOV", item[4]["ions"], seg_name(item[4]["source_seg"]) + "->" + seg_name(item[4]["dest_seg"]), (item[2], item[3]))
            elif item[1] == Schedule.Merge:
                print("MER", item[4]["ions"], seg_name(item[4]["seg"]) + "->" + trap_name(item[4]["trap"]), (item[2], item[3]))

    def get_last_event_ts(self):
        max_time = 0
        for item in list(self.events):
            if item[3] > max_time:
                max_time = item[3]
        return max_time

    def events_ge_ts(self, ts):
        try:
            return self.events.find_all_ge(ts)
        except ValueError:
            return []

    def events_lt_ts(self, ts):
        try:
            return self.events.find_all_lt(ts)
        except ValueError:
            return []

    def events_in_interval(self, ts1, ts2):
        items = self.events_ge_ts(ts1)
        ret_list = []
        for item in items:
            st = item[2]
            fin = item[3]
            if fin <= ts1:
                continue
            if st >= ts2:
                continue
            ret_list.append(item)
        return ret_list

    def last_ion_event_before_ts(self, ts, ion_id):
        try:
            items = self.filter_by_ion(self.events_lt_ts(ts), ion_id)
            if len(items):
                return items[-1]
        except ValueError:
            pass
        return []

    def filter_gate_ops(self, items):
        return list(filter(lambda x: x[1] == Schedule.Gate, items))

    def filter_seg_ops(self, items):
        return list(filter(lambda x: x[1] > Schedule.Gate, items))

    def filter_by_ion(self, items, ion_id):
        return list(filter(lambda x: ion_id in x[4]["ions"], items))

    def filter_by_segment(self, items, segment_id):
        return list(
            filter(
                lambda x: ((x[1] == Schedule.Split or x[1] == Schedule.Merge) and (x[4]["seg"] == segment_id))
                or (x[1] == Schedule.Move and (x[4]["source_seg"] == segment_id or x[4]["dest_seg"] == segment_id)),
                items,
            )
        )

    def filter_by_trap(self, items, trap_id):
        return list(filter(lambda x: ((x[1] == Schedule.Gate or x[1] == Schedule.Split or x[1] == Schedule.Merge) and (trap_id == x[4]["trap"])), items))

    def filter_by_junction(self, items, junction):
        seg_list = []
        for v in self.machine.graph[junction]:
            seg_list.append(self.machine.graph[junction][v]["seg"].id)
        return list(filter(lambda x: ((x[1] == Schedule.Move) and (x[4]["source_seg"] in seg_list and x[4]["dest_seg"] in seg_list)), items))

    def last_event_time_on_trap(self, trap_id):
        tmp = self.filter_by_trap(self.events, trap_id)
        if tmp:
            return tmp[-1][3]
        else:
            return 0

    def last_comm_event_time(self):
        tmp = self.filter_seg_ops(self.events)
        if tmp:
            return tmp[-1][3]
        else:
            return 0

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
        assert 0
        return start_time, end_time
