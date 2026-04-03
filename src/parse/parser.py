"""
OpenQASM Parser for QCCD/MUSS experiments.

迁移自项目根目录 parse.py

功能：
1) 构建完整的量子门依赖 DAG（1Q + 2Q），供调度 / fidelity replay 使用
2) 构建仅包含 2Q 门的依赖 DAG，供 MUSS 调度和 SABRE2 映射使用
3) 在 all_gate_map 中保存每个真实量子门的元信息
4) 保留原始 cx interaction graph，兼容旧 mapper / reorderer
5) 支持 barrier 语义
6) 对未知门直接报错，避免静默漏门
"""

import re
import networkx as nx


# 支持的单比特门
GSET1 = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "measure"]

# 支持的参数化单比特门
GSET2 = ["rx", "ry", "rz"]

# 支持的双比特门
GSET3 = ["cx"]

# 特殊语句
SPECIAL_GATES = ["barrier"]


class InputParse:
    """QASM 解析器"""

    def __init__(self):
        # 旧版 mapper / reorderer 所依赖的 CX interaction graph
        self.cx_graph = nx.Graph()
        self.cx_graph.graph["edge_weight_attr"] = "weight"
        self.cx_graph.graph["node_weight_attr"] = "node_weight"

        self.edge_weights = {}

        # 每个 qubit 上"最近一个真实量子门"的 gate_id
        self.prev_gate = {}

        # barrier 不会生成真实 gate_id 节点，因此需要额外记录：
        # 对每个 qubit，下一次真实量子门需要额外依赖哪些 gate_id
        self.pending_barrier_deps = {}

        self.global_gate_id = 0

        # 旧接口保留
        self.cx_gate_map = {}
        self.oneq_gate_map = {}
        self.all_gate_map = {}

        # 完整依赖图（仅真实量子门）
        self.gate_graph = nx.DiGraph()

        # 仅 2Q 门依赖图（仅真实 2Q 门）
        self.twoq_gate_graph = nx.DiGraph()

        # 支持门集
        self.gset = []
        self.gset.extend(GSET1)
        self.gset.extend(GSET2)
        self.gset.extend(GSET3)
        self.gset.extend(SPECIAL_GATES)

        self.qbit_count = 0
        self.two_qubit_gate_list = []
        self.one_qubit_gate_list = []

        # 记录 qreg 名称
        self.qreg_names = set()

        # gate summary：记录每种门/语句出现次数
        self.gate_summary = {}

    # ============================================================
    # 基础工具函数
    # ============================================================
    def inc_gate_summary(self, gate_name):
        if gate_name not in self.gate_summary:
            self.gate_summary[gate_name] = 0
        self.gate_summary[gate_name] += 1

    def strip_comments(self, line):
        """去掉 // 注释，并去掉首尾空白。"""
        if "//" in line:
            line = line.split("//", 1)[0]
        return line.strip()

    def normalize_line(self, line):
        """统一空白格式，但保留原有语法结构。"""
        line = self.strip_comments(line)
        if not line:
            return ""
        return " ".join(line.split())

    def extract_gate_name(self, line):
        """
        从一行语句中精确提取 gate 名。
        例如：
          "rz(pi/2) q[0];" -> "rz"
          "cx q[0],q[1];"  -> "cx"
          "barrier q;"     -> "barrier"
          "measure q[0] -> c[0];" -> "measure"
        """
        if not line:
            return None

        first_token = line.split()[0]
        if "(" in first_token:
            return first_token.split("(")[0]
        return first_token

    def extract_indexed_refs(self, line):
        """
        提取所有形如 name[idx] 的引用。
        返回 [(name, idx), ...]
        """
        matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", line)
        return [(name, int(idx)) for name, idx in matches]

    def find_dep_gate(self, qbit):
        """返回该 qubit 当前的真实量子门前驱。"""
        deps = []
        if qbit in self.prev_gate:
            deps.append(self.prev_gate[qbit])
        return deps

    def find_pending_barrier_deps(self, qbit):
        """返回该 qubit 因 barrier 引入的待消费依赖。"""
        if qbit not in self.pending_barrier_deps:
            return []
        return list(self.pending_barrier_deps[qbit])

    def consume_barrier_deps(self, qubits):
        """消费 barrier 依赖，避免重复附加。"""
        for q in qubits:
            if q in self.pending_barrier_deps:
                self.pending_barrier_deps[q] = set()

    def update_dep_gate(self, qbit, gate_id):
        """更新该 qubit 上最近一个真实量子门。"""
        self.prev_gate[qbit] = gate_id

    def check_valid_qbit(self, qbit):
        return 0 <= qbit < self.qbit_count

    def ensure_valid_qbit(self, qbit, line_no):
        if not self.check_valid_qbit(qbit):
            raise ValueError(f"Line {line_no}: qbit {qbit} out of range [0, {self.qbit_count - 1}]")

    def check_valid_gate(self, gate_name):
        return gate_name in self.gset

    def add_edge_pair(self, q1, q2):
        """在 cx interaction graph 中记录一条 2Q 交互边，并累计边权。"""
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

    # ============================================================
    # barrier 处理
    # ============================================================
    def parse_barrier_qubits(self, line, line_no):
        """解析 barrier 作用的 qubit 集合。"""
        indexed_refs = self.extract_indexed_refs(line)

        # barrier q[0],q[1],...
        if indexed_refs:
            qubits = []
            for _, idx in indexed_refs:
                self.ensure_valid_qbit(idx, line_no)
                qubits.append(idx)
            return sorted(set(qubits))

        # barrier q; -> 整个寄存器
        rest = line[len("barrier"):].strip().rstrip(";").strip()
        if not rest:
            raise ValueError(f"Line {line_no}: malformed barrier statement '{line}'")

        return list(range(self.qbit_count))

    def process_barrier(self, line, line_no):
        """处理 barrier：将同步依赖延迟附加到 barrier 之后第一个真实量子门上。"""
        barrier_qubits = self.parse_barrier_qubits(line, line_no)

        dep_gates = []
        for q in barrier_qubits:
            dep_gates.extend(self.find_dep_gate(q))
            dep_gates.extend(self.find_pending_barrier_deps(q))

        dep_gates = list(dict.fromkeys(dep_gates))

        for q in barrier_qubits:
            if q not in self.pending_barrier_deps:
                self.pending_barrier_deps[q] = set()
            self.pending_barrier_deps[q].update(dep_gates)

    # ============================================================
    # 真实量子门处理
    # ============================================================
    def _process_single_qubit_gate(self, gate_name, line, line_no):
        """处理单比特门（有无参数均适用）"""
        indexed_refs = self.extract_indexed_refs(line)
        if not indexed_refs:
            raise ValueError(f"Line {line_no}: cannot parse qubit from '{line}'")

        qbit = indexed_refs[0][1]
        self.ensure_valid_qbit(qbit, line_no)

        gate_id = self.global_gate_id

        dep_gates = []
        dep_gates.extend(self.find_dep_gate(qbit))
        dep_gates.extend(self.find_pending_barrier_deps(qbit))
        dep_gates = list(dict.fromkeys(dep_gates))

        self.gate_graph.add_node(gate_id)
        for dgate in dep_gates:
            self.gate_graph.add_edge(dgate, gate_id)

        self.oneq_gate_map[gate_id] = [qbit]
        self.all_gate_map[gate_id] = {"type": gate_name, "qubits": [qbit]}
        self.one_qubit_gate_list.append(gate_id)

        self.update_dep_gate(qbit, gate_id)
        self.consume_barrier_deps([qbit])

        self.global_gate_id += 1

    def _process_two_qubit_gate(self, gate_name, line, line_no):
        """处理双比特门"""
        indexed_refs = self.extract_indexed_refs(line)
        if len(indexed_refs) < 2:
            raise ValueError(f"Line {line_no}: cannot parse two qubits from '{line}'")

        qbit1 = indexed_refs[0][1]
        qbit2 = indexed_refs[1][1]

        self.ensure_valid_qbit(qbit1, line_no)
        self.ensure_valid_qbit(qbit2, line_no)

        gate_id = self.global_gate_id

        self.add_edge_pair(qbit1, qbit2)

        dep_gates = []
        dep_gates.extend(self.find_dep_gate(qbit1))
        dep_gates.extend(self.find_dep_gate(qbit2))
        dep_gates.extend(self.find_pending_barrier_deps(qbit1))
        dep_gates.extend(self.find_pending_barrier_deps(qbit2))
        dep_gates = list(dict.fromkeys(dep_gates))

        self.gate_graph.add_node(gate_id)
        self.twoq_gate_graph.add_node(gate_id)

        for dgate in dep_gates:
            self.gate_graph.add_edge(dgate, gate_id)
            if dgate in self.cx_gate_map:
                self.twoq_gate_graph.add_edge(dgate, gate_id)

        self.cx_gate_map[gate_id] = [qbit1, qbit2]
        self.all_gate_map[gate_id] = {"type": gate_name, "qubits": [qbit1, qbit2]}
        self.two_qubit_gate_list.append(gate_id)

        self.update_dep_gate(qbit1, gate_id)
        self.update_dep_gate(qbit2, gate_id)
        self.consume_barrier_deps([qbit1, qbit2])

        self.global_gate_id += 1

    def process_gate(self, gate_name, line, line_no):
        """处理一个真实量子门（1Q / 2Q）"""
        # 单比特门（无参数）
        if gate_name in GSET1:
            self._process_single_qubit_gate(gate_name, line, line_no)
            return

        # 单比特门（有参数）
        if gate_name in GSET2:
            self._process_single_qubit_gate(gate_name, line, line_no)
            return

        # 双比特门
        if gate_name in GSET3:
            self._process_two_qubit_gate(gate_name, line, line_no)
            return

        raise ValueError(f"Line {line_no}: unsupported internal gate dispatch for '{gate_name}'")

    # ============================================================
    # QASM 主解析入口
    # ============================================================
    def parse_ir(self, fname):
        """解析 QASM 文件"""
        with open(fname, "r") as f:
            for line_no, raw_line in enumerate(f.readlines(), start=1):
                line = self.normalize_line(raw_line)
                if not line:
                    continue

                # 跳过头部 / 声明
                if line.startswith("OPENQASM"):
                    continue
                elif line.startswith("include"):
                    continue
                elif line.startswith("qreg"):
                    refs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", line)
                    if len(refs) != 1:
                        raise ValueError(f"Line {line_no}: malformed qreg declaration '{line}'")

                    qreg_name, qcount = refs[0][0], int(refs[0][1])
                    self.qreg_names.add(qreg_name)
                    self.qbit_count = qcount
                    continue
                elif line.startswith("creg"):
                    continue

                gate_name = self.extract_gate_name(line)
                if gate_name is None:
                    continue

                # 记录 gate-set summary
                self.inc_gate_summary(gate_name)

                # 精确判断门名是否合法
                if not self.check_valid_gate(gate_name):
                    raise ValueError(
                        f"Line {line_no}: unsupported gate '{gate_name}' in line: {line}"
                    )

                # barrier 单独处理
                if gate_name == "barrier":
                    self.process_barrier(line, line_no)
                else:
                    self.process_gate(gate_name, line, line_no)

        # 解析结束后做 DAG 检查
        if not nx.is_directed_acyclic_graph(self.gate_graph):
            raise ValueError("Full gate dependency graph is not a DAG.")
        if not nx.is_directed_acyclic_graph(self.twoq_gate_graph):
            raise ValueError("2Q-only gate dependency graph is not a DAG.")

        # 打印 gate-set summary
        self.print_gate_summary()

    # ============================================================
    # 对外接口
    # ============================================================
    def print_gates(self):
        for edge in self.gate_graph.edges:
            print(edge)

    def get_ir(self):
        return self.cx_gate_map, self.twoq_gate_graph

    def visualize_graph(self, fname):
        nx.write_gexf(self.cx_graph, fname)

    def print_gate_summary(self):
        """打印本次 QASM 中实际出现过的门集及次数。"""
        print("========== Gate Set Summary ==========")
        if not self.gate_summary:
            print("No gate statements found.")
            print("======================================")
            return

        for gate_name in sorted(self.gate_summary.keys()):
            print(f"{gate_name}: {self.gate_summary[gate_name]}")
        print("======================================")
