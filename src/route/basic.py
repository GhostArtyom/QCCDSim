"""
Basic route - simple shortest path routing

迁移自项目根目录 route.py
"""

import networkx as nx
from src.machine.core import Trap, Segment, Junction


class BasicRoute:
    def __init__(self, machine):
        self.machine = machine

    def find_route(self, source_trap, dest_trap):
        graph = self.machine.graph
        tsrc = self.machine.traps[source_trap]
        tdest = self.machine.traps[dest_trap]
        path = nx.shortest_path(graph, source=tsrc, target=tdest)
        return path
