"""
Rebalance traps - network simplex based trap rebalancing

迁移自项目根目录 rebalance.py，作为 MUSS 调度器内部模块
"""

import networkx as nx
import numpy as np

from src.machine.core import Trap, Segment, Junction


class RebalanceTraps:
    """Trap rebalancing using network simplex."""

    def __init__(self, machine, system_state):
        self.machine = machine
        self.ss = system_state

    def clear_all_blocks(self):
        m = self.machine
        graph = nx.DiGraph(m.graph)
        demand = {}
        weight = {}
        capacity = {}
        ss = self.ss
        trap_free_space = {}
        for k in self.ss.trap_ions:
            trap_free_space[k] = m.traps[k].capacity - len(ss.trap_ions[k])
        req_free_space = 0
        for k in self.ss.trap_ions:
            if trap_free_space[k] == 0:
                req_free_space += 1
                demand[m.traps[k]] = -1
        for k in self.ss.trap_ions:
            if req_free_space != 0 and trap_free_space[k] > 1:
                offer = min(trap_free_space[k] - 1, req_free_space)
                req_free_space -= offer
                demand[m.traps[k]] = offer
        nx.set_node_attributes(graph, demand, "demand")
        for u, v in graph.edges:
            weight[(u, v)] = 1
            capacity[(u, v)] = 100
        nx.set_edge_attributes(graph, weight, "weight")
        nx.set_edge_attributes(graph, capacity, "capacity")
        flowCost, flowDict = nx.network_simplex(graph)
        return flowDict
