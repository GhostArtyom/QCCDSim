"""
Free trap route - capacity-aware routing

迁移自项目根目录 route.py
"""

import networkx as nx
from src.machine.core import Trap, Segment, Junction


class FreeTrapRoute:
    def __init__(self, machine, sys_state):
        self.machine = machine
        self.ss = sys_state

    def find_route(self, source_trap, dest_trap):
        m = self.machine
        ss = self.ss
        edge_states = {}
        trap_free_space = {}
        for k in self.ss.trap_ions:
            trap_free_space[k] = m.traps[k].capacity - len(ss.trap_ions[k])
        for u, v in m.graph.edges:
            if type(u) == Trap and type(v) == Junction:
                e0 = u
                e1 = v
            elif type(u) == Junction and type(v) == Trap:
                e0 = v
                e1 = u
            elif type(u) == Junction and type(v) == Junction:
                edge_states[(u, v)] = 0
                edge_states[(v, u)] = 0
                continue

            if trap_free_space[e0.id] == 0 and e0.id != source_trap:
                edge_states[(e0, e1)] = 10**9
                edge_states[(e1, e0)] = 10**9
            else:
                edge_states[(e0, e1)] = 0
                edge_states[(e1, e0)] = 0

        nx.set_edge_attributes(m.graph, edge_states, "block_status")
        ret = nx.shortest_path(m.graph, source=m.traps[source_trap], target=m.traps[dest_trap], weight="block_status")
        cost = 0
        for i in range(len(ret) - 1):
            u = ret[i]
            v = ret[i + 1]
            if (u, v) in edge_states:
                cost += edge_states[(u, v)]
            elif (v, u) in edge_states:
                cost += edge_states[(v, u)]
        if cost > 1:
            return 1, ret
        else:
            return 0, ret
