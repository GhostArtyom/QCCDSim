"""
Parser for a simplified OpenQASM subset used by the QCCD / MUSS experiments.

Design goals for the paper-faithful flow:
1) Preserve the full gate dependency DAG (1Q + 2Q) for timing / fidelity replay.
2) Build an explicit 2Q-only DAG for MUSS scheduling and SABRE2 mapping.
3) Keep every gate's metadata in all_gate_map:
      gate_id -> {"type": <str>, "qubits": [..]}
4) Keep the original cx interaction graph for legacy mappers / reorderers.
"""

import sys
import networkx as nx

gset1 = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "measure"]
gset2 = ["rx", "ry", "rz"]
gset3 = ["cx"]


class InputParse:
    def __init__(self):
        self.cx_graph = nx.Graph()
        self.cx_graph.graph["edge_weight_attr"] = "weight"
        self.cx_graph.graph["node_weight_attr"] = "node_weight"

        self.edge_weights = {}
        self.prev_gate = {}
        self.global_gate_id = 0

        self.cx_gate_map = {}
        self.oneq_gate_map = {}
        self.all_gate_map = {}

        self.gate_graph = nx.DiGraph()
        self.twoq_gate_graph = nx.DiGraph()

        self.gset = []
        self.gset.extend(gset1)
        self.gset.extend(gset2)
        self.gset.extend(gset3)

        self.qbit_count = 0
        self.two_qubit_gate_list = []
        self.one_qubit_gate_list = []

    def find_dep_gate(self, qbit):
        if qbit in self.prev_gate:
            return [self.prev_gate[qbit]]
        return []

    def update_dep_gate(self, qbit, gate_id):
        self.prev_gate[qbit] = gate_id

    def check_valid_gate(self, line):
        for g in self.gset:
            if line.startswith(g):
                return 1
        return 0

    def add_edge_pair(self, q1, q2):
        c = min(q1, q2)
        t = max(q1, q2)
        if c not in self.edge_weights:
            self.edge_weights[c] = {}
        if t not in self.edge_weights[c]:
            self.edge_weights[c][t] = 0
        self.edge_weights[c][t] += 1
        self.cx_graph.add_edge(c, t)
        self.cx_graph.adj[c][t]["weight"] = self.edge_weights[c][t]
        self.cx_graph.nodes[c]["node_weight"] = 1
        self.cx_graph.nodes[t]["node_weight"] = 1

    def process_gate(self, line):
        for g in gset1:
            if line.startswith(g):
                qbit = int(line.split("[")[1].split("]")[0])
                if not self.check_valid_qbit(qbit):
                    sys.exit("qbit " + str(qbit) + " not in range")

                gate_id = self.global_gate_id
                dep_gates = self.find_dep_gate(qbit)

                self.gate_graph.add_node(gate_id)
                for dgate in dep_gates:
                    self.gate_graph.add_edge(dgate, gate_id)

                self.oneq_gate_map[gate_id] = [qbit]
                self.all_gate_map[gate_id] = {"type": g, "qubits": [qbit]}
                self.one_qubit_gate_list.append(gate_id)

                self.update_dep_gate(qbit, gate_id)
                self.global_gate_id += 1
                return

        for g in gset2:
            if line.startswith(g):
                qbit = int(line.split("[")[1].split("]")[0])
                if not self.check_valid_qbit(qbit):
                    sys.exit("qbit " + str(qbit) + " not in range")

                gate_id = self.global_gate_id
                dep_gates = self.find_dep_gate(qbit)

                self.gate_graph.add_node(gate_id)
                for dgate in dep_gates:
                    self.gate_graph.add_edge(dgate, gate_id)

                self.oneq_gate_map[gate_id] = [qbit]
                self.all_gate_map[gate_id] = {"type": g, "qubits": [qbit]}
                self.one_qubit_gate_list.append(gate_id)

                self.update_dep_gate(qbit, gate_id)
                self.global_gate_id += 1
                return

        for g in gset3:
            if line.startswith(g):
                base = "".join(line.split()).split(",")
                qbit1 = int(base[0].split("[")[1].split("]")[0])
                qbit2 = int(base[1].split("[")[1].split("]")[0])

                if not self.check_valid_qbit(qbit1):
                    sys.exit("qbit " + str(qbit1) + " not in range")
                if not self.check_valid_qbit(qbit2):
                    sys.exit("qbit " + str(qbit2) + " not in range")

                gate_id = self.global_gate_id
                self.add_edge_pair(qbit1, qbit2)

                dep_gates = []
                dep_gates.extend(self.find_dep_gate(qbit1))
                dep_gates.extend(self.find_dep_gate(qbit2))
                dep_gates = list(dict.fromkeys(dep_gates))

                self.gate_graph.add_node(gate_id)
                self.twoq_gate_graph.add_node(gate_id)
                for dgate in dep_gates:
                    self.gate_graph.add_edge(dgate, gate_id)
                    if dgate in self.cx_gate_map:
                        self.twoq_gate_graph.add_edge(dgate, gate_id)

                self.cx_gate_map[gate_id] = [qbit1, qbit2]
                self.all_gate_map[gate_id] = {"type": g, "qubits": [qbit1, qbit2]}
                self.two_qubit_gate_list.append(gate_id)

                self.update_dep_gate(qbit1, gate_id)
                self.update_dep_gate(qbit2, gate_id)
                self.global_gate_id += 1
                return

    def parse_ir(self, fname):
        with open(fname, "r") as f:
            for raw_line in f.readlines():
                line = " ".join(raw_line.split())
                if not line:
                    continue
                if line.startswith("OPENQASM"):
                    continue
                elif line.startswith("include"):
                    continue
                elif line.startswith("qreg"):
                    self.qbit_count = int(line.split("[")[1].split("]")[0])
                elif line.startswith("creg"):
                    continue
                else:
                    if self.check_valid_gate(line):
                        self.process_gate(line)

        if not nx.is_directed_acyclic_graph(self.gate_graph):
            raise ValueError("Full gate dependency graph is not a DAG.")
        if not nx.is_directed_acyclic_graph(self.twoq_gate_graph):
            raise ValueError("2Q-only gate dependency graph is not a DAG.")

    def print_gates(self):
        for edge in self.gate_graph.edges:
            print(edge)

    def get_ir(self):
        return self.cx_gate_map, self.twoq_gate_graph

    def visualize_graph(self, fname):
        nx.write_gexf(self.cx_graph, fname)

    def check_valid_qbit(self, qbit):
        return qbit >= 0 and qbit < self.qbit_count
