"""
Generates an initial qubit mapping
Three mappers:
    QubitMapPO - Program Order based
    QubitMapMetis - Metis clustering
    QubitMapAgg - Agglomerative cluster

Metis mapping didnt work very well, use either PO or Agg mappers

Output of this step is a partitioning of the qubits into regions given by a dictionary
prog_qubit -> region id

TODO mrm suggested comparing to lpfs. Need to check if its feasible to implement the comparison
"""

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AggClus
import copy
from route import BasicRoute
from machine import Machine
from parse import InputParse
import networkx as nx

class QubitMapGreedy:
    def __init__(self, parse_obj, machine_obj):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.build_program_graph()
        self.pending_program_edges = []
        self.mapping = []
        self.remaining_capacity = []
        trap_capacity = self.machine_obj.traps[0].capacity
        for i in range(len(machine_obj.traps)):
            self.mapping.append([])
            self.remaining_capacity.append(trap_capacity)
        self.router = BasicRoute(machine_obj)

    def gate_tuple(self, g):
        return (min(g), max(g))

    def build_program_graph(self):
        self.prog_graph = nx.Graph()
        edge_weights = {}
        # add support for single qubit (not cx gates)
        # check gate id or get node label in networkx
        # if single qubit, skip over this whole part
        # try printing out parse_obj - the gate ids are there
        # when you get an id, check if it exists in the cx_gate_map
        # if it does, then it is a 2 qubit gate
        # if not, then continue
        for g in self.parse_obj.gate_graph:
            # 只处理存在于 cx_gate_map 中的门（CX门）
            if g not in self.parse_obj.cx_gate_map:
                continue  # 跳过非CX门
            g_qubits = self.parse_obj.cx_gate_map[g]
            tup = self.gate_tuple(g_qubits)
            if tup in edge_weights:
                edge_weights[tup] += 1
            else:
                edge_weights[tup] = 1
        # print(edge_weights)
        for key in edge_weights:
            self.prog_graph.add_edge(*key, weight=edge_weights[key])
        # print(prog_graph.edges)

    def _is_mapped(self, qubit):
        for item in self.mapping:
            if qubit in item:
                return True
        return False

    def _trap(self, qubit):
        for i, item in enumerate(self.mapping):
            if qubit in item:
                return i
        assert 0

    def _select_next_edge(self):
        """Select the next edge.
        If there is an edge with one endpoint mapped, return it.
        Else return in the first edge
        """
        for edge in self.pending_program_edges:
            q1_mapped = self._is_mapped(edge[0])
            q2_mapped = self._is_mapped(edge[1])
            assert not (q1_mapped and q2_mapped)
            if q1_mapped or q2_mapped:
                return edge
        return self.pending_program_edges[0]

    def _map_qubit(self, qubit):
        # Iterate through traps and pick the best one
        all_dist = []
        for target_trap in range(len(self.machine_obj.traps)):
            if self.remaining_capacity[target_trap] == 0:
                all_dist.append(float("inf"))
            else:
                sum_distances = 0
                for n in self.prog_graph.neighbors(qubit):
                    if self._is_mapped(n):
                        src_trap = self._trap(n)
                        path = self.router.find_route(src_trap, target_trap)
                        sum_distances += len(path)
                all_dist.append(sum_distances)
        if all_dist:
            return all_dist.index(min(all_dist))
        else:
            for i, val in enumerate(self.remaining_capacity):
                if val > 0:
                    return i
        assert 0

    def compute_mapping(self):
        for end1, end2, _ in sorted(self.prog_graph.edges(data=True), key=lambda x: x[2]["weight"], reverse=True):
            self.pending_program_edges.append((end1, end2))
        while self.pending_program_edges:
            edge = self._select_next_edge()
            q1_mapped = self._is_mapped(edge[0])
            q2_mapped = self._is_mapped(edge[1])
            if not q1_mapped:
                q1_trap = self._map_qubit(edge[0])
                self.mapping[q1_trap].append(edge[0])
                self.remaining_capacity[q1_trap] -= 1
            if not q2_mapped:
                q2_trap = self._map_qubit(edge[1])
                self.mapping[q2_trap].append(edge[1])
                self.remaining_capacity[q2_trap] -= 1
            tmplist = [x for x in self.pending_program_edges if not (self._is_mapped(x[0]) and self._is_mapped(x[1]))]
            self.pending_program_edges = tmplist
        output_partition = {}
        for i in range(len(self.mapping)):
            output_partition[i] = self.mapping[i]
        return output_partition


class QubitMapLPFS:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity

    def compute_mapping(self):
        gate_graph = copy.deepcopy(self.parse_obj.gate_graph)
        k = len(self.machine_obj.traps)
        cap = self.machine_obj.traps[0].capacity
        already_mapped = []
        mapping = []
        for i in range(k):
            path = nx.algorithms.dag.dag_longest_path(gate_graph)
            qubit_set = []
            used_gates = []
            for g in path:
                if len(qubit_set) >= cap - 1:
                    break
                # 只处理CX门，跳过非CX门
                if g not in self.parse_obj.cx_gate_map:
                    continue
                g_qubits = self.parse_obj.cx_gate_map[g]
                if g == 992:
                    print(g_qubits)
                for qubit in g_qubits:
                    if qubit in already_mapped:
                        continue
                    qubit_set.append(qubit)
                    already_mapped.append(qubit)
                    if not g in used_gates:
                        used_gates.append(g)
            mapping.append(qubit_set)
            for g in used_gates:
                gate_graph.remove_node(g)
        num_qubits = len(list(self.parse_obj.cx_graph.nodes))
        output_partition = {}
        for i, qubit_set in enumerate(mapping):
            for q in qubit_set:
                output_partition[q] = i
        num_qubits = len(list(self.parse_obj.cx_graph.nodes))
        for i in range(num_qubits):
            if not i in output_partition:
                output_partition[i] = k - 1
        return output_partition


class QubitMapRandom:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity

    def compute_mapping(self):
        num_qubits = len(list(self.parse_obj.cx_graph.nodes))
        partition = []
        trap_sizes = []
        for t in self.machine_obj.traps:
            trap_sizes.append(t.capacity - self.excess_capacity)
        for i in range(len(trap_sizes)):
            partition.extend([i] * trap_sizes[i])
        partition = partition[:num_qubits]
        np.random.shuffle(partition)
        output_partition = {}
        for i in range(len(partition)):
            output_partition[i] = partition[i]
        return output_partition


class QubitMapPO:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity

    def compute_mapping(self):
        # 1. 获取真实的量子比特 ID 列表（关键修正）
        # cx_graph.nodes 包含了所有参与 CX 门的量子比特 ID（例如 0, 4, 32...）
        qubit_nodes = list(self.parse_obj.cx_graph.nodes)
        num_qubits = len(qubit_nodes)

        # 2. 准备物理阱的空槽位列表
        partition = []
        trap_sizes = []
        for t in self.machine_obj.traps:
            # 计算每个阱的有效容量
            current_cap = t.capacity - self.excess_capacity
            if current_cap < 0:
                current_cap = 0
            trap_sizes.append(current_cap)

        # 生成初始分配池 [0, 0, ..., 1, 1, ..., 5, 5]
        for i in range(len(trap_sizes)):
            partition.extend([i] * trap_sizes[i])

        # 3. 安全性检查：容量是否足够？
        if len(partition) < num_qubits:
            print(f"Warning: Not enough capacity! Need {num_qubits}, Have {len(partition)}. Force filling (Results may be invalid physically).")
            # 强制循环填充以避免代码崩溃，但这在物理上意味着超载
            while len(partition) < num_qubits:
                partition.extend(partition[: num_qubits - len(partition)])

        # 4. 截取所需的长度
        partition = partition[:num_qubits]

        # 5. 生成映射表 (使用真实的 qubit_nodes ID)
        output_partition = {}
        for i, qubit_id in enumerate(qubit_nodes):
            output_partition[qubit_id] = partition[i]  # 修正：Key 使用真实的 qubit_id (如 32)

        # 6. 处理孤立量子比特 (未参与 CX 门的 qubits)
        # 如果 QASM 中有 q[1] 但它只做了 H 门没做 CX，它不会在 cx_graph 中。
        # 这里将其默认分配到最后一个阱，防止调度器找不到它。
        # (注：InputParse 可能需要 cx_graph 包含所有 nodes，以此为准)
        all_qubits_count = len(self.parse_obj.cx_graph.nodes)  # 这里假设 parse_obj 只解析了相关比特
        # 如果需要更鲁棒，可以遍历 parse_obj.gate_graph 的所有节点

        return output_partition


class QubitMapMetis:
    def __init__(self, parse_obj, machine_obj):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj

    def partition_graph(self, parts, cx_graph):
        tpwgts = []
        ubvec = [1.1]
        for i in range(parts):
            tpwgts.append((1.0 / parts))
        out = mt.part_graph(cx_graph, nparts=parts, tpwgts=tpwgts, ubvec=ubvec)
        return out

    def compute_mapping(self):
        num_parts = len(self.machine_obj.traps)
        parts = self.partition_graph(num_parts, self.parse_obj.cx_graph)
        # TODO: can we partition with lesser parts?
        # TODO: initial mapping may exceed bounds
        # TODO: init mapping not aware of cluster distances
        # TODO: adjust mapping partition: full set of clusters with tail cluster
        tot_wt = 0
        for c in self.parse_obj.edge_weights.keys():
            for t in self.parse_obj.edge_weights[c].keys():
                tot_wt += self.parse_obj.edge_weights[c][t]
        output_partition = {}
        for i in range(len(parts[1])):
            output_partition[i] = parts[1][i]
        return output_partition


class QubitMapAgg:
    def __init__(self, parse_obj, machine_obj):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.num_traps = len(self.machine_obj.traps)
        self.num_nodes = len(self.parse_obj.cx_graph.nodes)
        print(self.num_nodes)
        self.trap_capacity = self.machine_obj.traps[0].capacity
        self.occupied_traps = 0
        self.qubit_mapping = {}
        self.trap_empty_space = {}
        for i in range(self.num_traps):
            self.trap_empty_space[i] = self.trap_capacity

    #
    def select_from_clusters(self, u, nclusters):
        curr_clusters = []
        for i in range(nclusters):
            curr_clusters.append([])

        for i in range(len(u)):
            curr_clusters[u[i]].append(i)
        bad_cluster = False
        for clus in curr_clusters:
            if len(clus) > self.trap_capacity:
                bad_cluster = True
        if bad_cluster:
            return 0

        curr_clusters.sort(key=len, reverse=True)
        top_k = min(3, self.num_traps)
        top_k = min(top_k, nclusters)
        for i in range(top_k):
            clus = curr_clusters[i]
            # print("Map", clus, "trap", i)
            for pq in clus:
                self.qubit_mapping[pq] = i
            self.trap_empty_space[i] -= len(clus)

        # print("unmapped")
        # print("caps:", self.trap_empty_space)
        for clus in curr_clusters[top_k:]:
            # if clus fits fully in some trap, assign it there
            is_assigned = False
            for i in range(self.num_traps):
                if self.trap_empty_space[i] >= len(clus):
                    # print("Map", clus, "trap", i)
                    for pq in clus:
                        self.qubit_mapping[pq] = i
                    self.trap_empty_space[i] -= len(clus)
                    is_assigned = True
                    break
            if not is_assigned:
                for i in range(self.num_traps):
                    available_capacity = self.trap_empty_space[i]
                    # print("Map", clus, "trap", i)
                    for pq in clus[:available_capacity]:
                        self.qubit_mapping[pq] = i
                    self.trap_empty_space[i] -= available_capacity
                    new_clus = clus[available_capacity:]

        return 1

    def compute_mapping(self):
        # compute affinity matrix of distances
        # distance function 1 - f/T
        affinity_matrix = np.ones([self.num_nodes, self.num_nodes])
        T = 0
        for u, v, d in self.parse_obj.cx_graph.edges(data=True):
            T = max(T, d["weight"])
        for u, v, d in self.parse_obj.cx_graph.edges(data=True):
            f = d["weight"]
            factor = float(f) / T
            affinity_matrix[u][v] = 1.0 - (factor)
            affinity_matrix[v][u] = 1.0 - (factor)
        for i in range(1, self.num_nodes):
            agg = AggClus(n_clusters=i, affinity="precomputed", linkage="average")
            u = agg.fit_predict(affinity_matrix)
            # print("Clustering level", i)
            done = self.select_from_clusters(u, i)
            if done == 1:
                break
        return self.qubit_mapping


"""
Reorders qubits within a region according to fidelity
Simple heuristic for now: places qubits with lot of gates
around the the center of the chain
"""


class QubitOrdering:
    def __init__(self, parse_obj, machine_obj, qubit_mapping):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.mapping = qubit_mapping
        self.trap_capacity = self.machine_obj.traps[0].capacity
        self.num_traps = len(self.machine_obj.traps)

    def reorder_naive(self):
        output_layout = {}
        for i in range(self.num_traps):
            this_layout = []
            for q in self.mapping.keys():
                if self.mapping[q] == i:  # q belongs to trap i
                    this_layout.append(q)
            output_layout[i] = this_layout
        return output_layout

    def reorder_fidelity(self):
        output_layout = {}
        for i in range(self.num_traps):
            this_layout = []
            candidates = []
            for q in self.mapping.keys():
                if self.mapping[q] == i:  # q belongs to trap i
                    candidates.append(q)

            candidates_with_wt = []
            for c in candidates:
                wt = 0
                for u, v, d in self.parse_obj.cx_graph.edges(data=True):
                    if u == c or v == c:
                        wt += d["weight"]
                candidates_with_wt.append((c, wt))
            # Find weight of each qubit as the no. of gates using the qubits
            # Sort qubits according to descending order of weight
            candidates_with_wt.sort(key=lambda tup: tup[1], reverse=True)
            coin = 0
            # Places qubits in an odd-even fashion around the center of the chain
            for item in candidates_with_wt:
                if coin == 0:
                    this_layout.append(item[0])
                    coin = 1
                elif coin == 1:
                    this_layout.insert(0, item[0])
                    coin = 0
            output_layout[i] = this_layout
        return output_layout

class QubitMapTrivial:
    """
    Paper-faithful trivial mapper.

    Rule from the MUSS-TI paper:
      - prioritize zones ordered by their levels from highest to lowest
      - place qubits sequentially in that order

    Interpretation used here:
      - if zone_level exists, sort traps by zone_level ascending
        (smaller level value means higher priority)
      - otherwise, if zone_type exists, use:
            optical > operation > storage
      - otherwise, fall back to trap id order
      - then place logical qubits sequentially into traps
    """

    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity

    def _effective_capacity(self, trap_id):
        return self.machine_obj.traps[trap_id].capacity - self.excess_capacity

    def _trap_priority_key(self, trap_obj):
        zone_level = getattr(trap_obj, "zone_level", None)
        if zone_level is not None:
            return (zone_level, trap_obj.id)

        zone_type = getattr(trap_obj, "zone_type", None)
        if zone_type is not None:
            order = {"optical": 0, "operation": 1, "storage": 2}
            return (order.get(zone_type, 9), trap_obj.id)

        return (0, trap_obj.id)

    def _logical_qubits(self):
        if hasattr(self.parse_obj, "nqubits"):
            return list(range(self.parse_obj.nqubits))

        qubit_set = set()
        for gate_id, data in self.parse_obj.all_gate_map.items():
            if isinstance(data, dict):
                qlist = data.get("qubits", [])
            else:
                qlist = list(data)
            for q in qlist:
                qubit_set.add(q)
        return sorted(qubit_set)

    def compute_mapping(self):
        trap_order = sorted(self.machine_obj.traps, key=self._trap_priority_key)
        logical_qubits = self._logical_qubits()

        mapping = {}
        q_idx = 0

        for trap in trap_order:
            cap = self._effective_capacity(trap.id)
            for _ in range(cap):
                if q_idx >= len(logical_qubits):
                    break
                mapping[logical_qubits[q_idx]] = trap.id
                q_idx += 1
            if q_idx >= len(logical_qubits):
                break

        if q_idx != len(logical_qubits):
            raise RuntimeError(
                f"QubitMapTrivial failed: only mapped {q_idx}/{len(logical_qubits)} qubits."
            )

        return mapping



import networkx as nx
import copy


# 多次sabre，但未严格限制容量。
class QubitMapSABRE1:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        # 确保距离矩阵已计算
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def get_trap_load(self, mapping):
        """辅助函数：计算当前每个 Trap 里的量子比特数量"""
        load = [0] * len(self.machine_obj.traps)
        for q, t in mapping.items():
            load[t] += 1
        return load

    def run_pass(self, gate_list, start_mapping):
        """
        执行一次 SABRE 遍历，并返回结束时的布局和移动代价
        """
        current_layout = start_mapping.copy()
        traps = self.machine_obj.traps
        move_count = 0  # 记录移动次数作为 Cost

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue

            qubits = self.parse_obj.cx_gate_map[g]
            q1, q2 = qubits[0], qubits[1]

            t1 = current_layout[q1]
            t2 = current_layout[q2]

            # 如果不在同一个 Trap，需要发生移动
            if t1 != t2:
                loads = self.get_trap_load(current_layout)
                cap1 = traps[t1].capacity - self.excess_capacity
                cap2 = traps[t2].capacity - self.excess_capacity

                can_move_to_t1 = loads[t1] < cap1
                can_move_to_t2 = loads[t2] < cap2

                moved = False
                # 贪心策略：优先移动到有空位的 Trap，如果都有，移动负载小的
                if can_move_to_t1 and not can_move_to_t2:
                    current_layout[q2] = t1
                    moved = True
                elif can_move_to_t2 and not can_move_to_t1:
                    current_layout[q1] = t2
                    moved = True
                elif can_move_to_t1 and can_move_to_t2:
                    if loads[t1] <= loads[t2]:
                        current_layout[q2] = t1
                    else:
                        current_layout[q1] = t2
                    moved = True
                else:
                    # 都没有位置 (拥堵)，强制移动到 t1 (后续调度器处理溢出)
                    current_layout[q2] = t1
                    moved = True

                if moved:
                    move_count += 1

        return current_layout, move_count

    def compute_mapping(self):
        # 1. 准备门序列
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except:
            forward_gates = list(self.parse_obj.gate_graph.nodes)

        backward_gates = forward_gates[::-1]

        # 2. 初始种子：强制使用 Trivial Mapping
        # 这是为了利用 QFT 等电路的线性结构优势
        base_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed_mapping = base_mapper.compute_mapping()

        # 3. 迭代优化 (Ping-Pong Loop)
        # current_start_mapping: 当前这一轮 Forward 开始前的布局
        current_start_mapping = seed_mapping.copy()

        best_mapping = seed_mapping.copy()
        min_total_moves = float("inf")

        # 迭代次数 (建议 3-10 次，通常 5 次左右收敛)
        num_iterations = 10
        print(f"SABRE: Starting iterative optimization ({num_iterations} rounds) from Trivial...")

        for i in range(num_iterations):
            # --- Forward Pass ---
            # 作用：评估当前初始布局的好坏，并生成一个适合电路结尾的布局
            layout_after_fwd, fwd_cost = self.run_pass(forward_gates, current_start_mapping)

            # 记录最佳结果
            # 我们评价一个初始布局好不好，是看它跑 Forward Pass 时产生的移动多不多
            if fwd_cost < min_total_moves:
                min_total_moves = fwd_cost
                best_mapping = current_start_mapping.copy()
                # print(f"  Round {i}: Found better mapping, cost = {fwd_cost}")

            # --- Backward Pass ---
            # 作用：利用电路的可逆性，从电路结尾倒推，寻找一个更适合电路开头的布局
            # 关键：输入的布局是 Forward 的终点
            layout_after_bwd, bwd_cost = self.run_pass(backward_gates, layout_after_fwd)

            # --- Update for Next Round ---
            # 关键：Backward 的终点，就是下一轮 Forward 的起点
            current_start_mapping = layout_after_bwd

        print(f"SABRE: Optimization Done. Best Forward Cost: {min_total_moves}")

        # 返回产生最小代价的那个“初始布局”
        return best_mapping

class QubitMapSABRE2:
    """
    最终推荐版 SABRE2（面向论文小规模 Table 2 复现）。

    设计原则：
      1) 严格运行在 2Q-only dependency DAG 上；
      2) seed layout 使用论文中的 Trivial mapping；
      3) 使用两次搜索：
           - 正向 G
           - 反向 G^R
      4) 每个 pass 中遵循 SABRE 原则：
           - 若 front layer 中已有可本地执行门，则优先执行这些门；
           - 否则仅考虑论文允许的 relocation 候选：
                 q1 -> trap(q2)
                 q2 -> trap(q1)
           - 使用论文主代价函数：
                 H = avg(front distances) + W * avg(extended distances)
      5) 严格 trap 容量限制，trap 满时仅使用 LRU victim replacement；
      6) 保留 SABRE pass 内部形成的 trap 内 qubit 顺序，不再丢弃；
      7) 输出结构化 bundle：
           - layout: q -> trap
           - trap_to_qubits: trap -> ordered qubit list

    为什么这是“最终推荐版”：
      - 相比最早版本：修复了“只返回 q->trap，导致 trap 内链顺序丢失”的问题；
      - 相比过强增强版：去掉了热点 qubit 惩罚、working-set gain 等额外偏好，
        避免在 BV / ADDER / QFT 这类 benchmark 上形成过强的局部团块化；
      - 保留的 tie-break 只有：
           1) 论文主代价 H
           2) front 中可立即本地执行门数（越多越好）
           3) front 总距离和（越小越好）
           4) moving_q / target_t / gate_id 稳定 tie-break
        这几项都属于“同分时更稳地落地论文 SABRE”，而不是论文外的大新启发式。
    """

    def __init__(
        self,
        parse_obj,
        machine_obj,
        excess_capacity=0,
        extended_set_size=20,
        extended_weight=0.5,
    ):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        self.extended_set_size = int(extended_set_size)
        self.extended_weight = float(extended_weight)

        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    # ============================================================
    # 基础工具函数
    # ============================================================
    def _effective_capacity(self, trap_id):
        """
        返回 trap 的有效容量。
        excess_capacity > 0 时，可用于给后续调度预留少量空位。
        """
        return self.machine_obj.traps[trap_id].capacity - self.excess_capacity

    def _all_trap_ids(self):
        return list(range(len(self.machine_obj.traps)))

    def _trap_distance(self, t1, t2):
        if t1 == t2:
            return 0
        return self.machine_obj.dist_cache.get((t1, t2), float("inf"))

    def _gate_qubits(self, gate_id):
        return tuple(self.parse_obj.cx_gate_map[gate_id])

    def _load_layout(self, mapping):
        """
        将 q -> trap 装载成两份运行时状态：
          - loads: trap 当前负载
          - trap_to_qubits: trap 当前有序 qubit 列表

        注意：
        这里保留 dict 的遍历顺序，因此若初始 mapping 本身有稳定顺序，
        该顺序会自然成为 SABRE pass 的起始链顺序。
        """
        loads = {t: 0 for t in self._all_trap_ids()}
        trap_to_qubits = {t: [] for t in self._all_trap_ids()}
        for q, t in mapping.items():
            loads[t] += 1
            trap_to_qubits[t].append(q)
        return loads, trap_to_qubits

    def _normalize_trap_layout(self, trap_to_qubits):
        """
        规范化 trap -> ordered qubit list：
          - 补齐空 trap
          - 保证每个条目都是普通 Python list
        """
        out = {}
        for t in self._all_trap_ids():
            out[t] = list(trap_to_qubits.get(t, []))
        return out

    # ============================================================
    # DAG 构造
    # ============================================================
    def _build_two_qubit_dag(self):
        """
        优先使用 parse_obj.twoq_gate_graph；
        若不存在，则从 gate_graph 中截取 2Q 子图。
        """
        dag = getattr(self.parse_obj, "twoq_gate_graph", None)
        if dag is None or dag.number_of_nodes() == 0:
            dag = self.parse_obj.gate_graph.subgraph(
                list(self.parse_obj.cx_gate_map.keys())
            ).copy()
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("SABRE2 requires a 2Q-only DAG.")
        return dag

    def _build_reversed_two_qubit_dag(self, dag_2q):
        """
        构造真反向 DAG G^R。
        """
        rev = dag_2q.reverse(copy=True)
        if not nx.is_directed_acyclic_graph(rev):
            raise ValueError("Reversed 2Q DAG is not a DAG.")
        return rev

    def _trivial_seed_mapping(self):
        """
        论文要求：SABRE 的初始种子布局来自 Trivial mapping。
        """
        trivial_mapper = QubitMapTrivial(
            self.parse_obj, self.machine_obj, self.excess_capacity
        )
        return trivial_mapper.compute_mapping()

    # ============================================================
    # SABRE 主评分与辅助统计
    # ============================================================
    def _gate_distance_under_layout(self, gate_id, layout):
        q1, q2 = self._gate_qubits(gate_id)
        t1 = layout[q1]
        t2 = layout[q2]
        return self._trap_distance(t1, t2)

    def _front_layer(self, dag, indegree_map, done_set):
        return [g for g in dag.nodes if g not in done_set and indegree_map[g] == 0]

    def _extended_set(self, dag, front, done_set):
        """
        论文式 extended set：
        从 front layer 往后做宽搜，取有限大小的后继集合。
        """
        ext = []
        seen = set(front)
        queue = list(front)

        while queue and len(ext) < self.extended_set_size:
            cur = queue.pop(0)
            for succ in dag.successors(cur):
                if succ in seen or succ in done_set:
                    continue
                seen.add(succ)
                ext.append(succ)
                queue.append(succ)
                if len(ext) >= self.extended_set_size:
                    break
        return ext

    def _heuristic_cost(self, layout, front, ext):
        """
        论文 SABRE 主代价函数：
            H = avg(front distances) + W * avg(extended distances)
        """
        if not front:
            return 0.0

        front_cost = sum(
            self._gate_distance_under_layout(g, layout) for g in front
        ) / float(len(front))

        if not ext:
            return front_cost

        ext_cost = sum(
            self._gate_distance_under_layout(g, layout) for g in ext
        ) / float(len(ext))

        return front_cost + self.extended_weight * ext_cost

    def _count_executable_front_gates(self, front, layout):
        """
        统计在给定 layout 下，front layer 中有多少门已经能本地执行。
        这是 tie-break 用的辅助量，不替代论文主代价函数 H。
        """
        cnt = 0
        for g in front:
            q1, q2 = self._gate_qubits(g)
            if layout[q1] == layout[q2]:
                cnt += 1
        return cnt

    def _total_front_distance(self, front, layout):
        """
        front layer 总 trap 距离和。
        仅用于 H 相同时的 tie-break。
        """
        return sum(self._gate_distance_under_layout(g, layout) for g in front)

    # ============================================================
    # 容量与 LRU replacement
    # ============================================================
    def _select_lru_victim(self, target_trap, trap_to_qubits, last_used, forbidden=None):
        """
        从目标 trap 中选择 LRU victim。
        forbidden 中的 qubit 不允许被驱逐（通常是当前 gate 的 partner）。
        """
        forbidden = forbidden or set()
        candidates = [q for q in trap_to_qubits[target_trap] if q not in forbidden]
        if not candidates:
            return None
        return min(candidates, key=lambda q: last_used.get(q, -1))

    def _apply_relocation(
        self,
        layout,
        loads,
        trap_to_qubits,
        last_used,
        moving_qubit,
        target_trap,
        partner_qubit=None,
    ):
        """
        执行一个 relocation 候选动作：
          - 将 moving_qubit 迁到 target_trap；
          - 若 target_trap 已满，则使用 LRU victim replacement；
          - 同时维护：
                layout: q -> trap
                loads: trap 当前负载
                trap_to_qubits: trap 内部有序 qubit 列表

        注意：
        这里维护的 trap_to_qubits 顺序不是“额外 heuristic”，
        只是把 SABRE pass 自然产生的顺序保留下来。
        """
        new_layout = dict(layout)
        new_loads = dict(loads)
        new_trap_to_qubits = {t: list(qs) for t, qs in trap_to_qubits.items()}

        src_trap = new_layout[moving_qubit]
        if src_trap == target_trap:
            return new_layout, new_loads, new_trap_to_qubits

        # 目标 trap 还有空位：直接迁移
        if new_loads[target_trap] < self._effective_capacity(target_trap):
            new_trap_to_qubits[src_trap].remove(moving_qubit)
            new_trap_to_qubits[target_trap].append(moving_qubit)
            new_loads[src_trap] -= 1
            new_loads[target_trap] += 1
            new_layout[moving_qubit] = target_trap
            return new_layout, new_loads, new_trap_to_qubits

        # 目标 trap 已满：按论文小规模解释，使用 LRU replacement
        victim = self._select_lru_victim(
            target_trap,
            new_trap_to_qubits,
            last_used,
            forbidden={partner_qubit} if partner_qubit is not None else set(),
        )
        if victim is None:
            return None, None, None

        # 先把 victim 送回 src_trap
        new_trap_to_qubits[target_trap].remove(victim)
        new_trap_to_qubits[src_trap].append(victim)
        new_layout[victim] = src_trap

        # 再把 moving_qubit 放入 target_trap
        new_trap_to_qubits[src_trap].remove(moving_qubit)
        new_trap_to_qubits[target_trap].append(moving_qubit)
        new_layout[moving_qubit] = target_trap

        # 负载不变，因此 new_loads 不需要修改
        return new_layout, new_loads, new_trap_to_qubits

    # ============================================================
    # 候选动作生成与选择
    # ============================================================
    def _generate_candidate_actions(self, front, layout, loads, trap_to_qubits, last_used):
        """
        仅针对 front layer 中当前仍不本地的门，生成两类论文内候选：
          - q1 -> trap(q2)
          - q2 -> trap(q1)

        不引入 front 之外的任意迁移，不做论文外的大范围动作集扩展。
        """
        candidates = []

        for g in front:
            q1, q2 = self._gate_qubits(g)
            t1 = layout[q1]
            t2 = layout[q2]

            # 已经本地可执行的门，不需要生成 relocation
            if t1 == t2:
                continue

            # 候选1：把 q1 迁到 q2 所在 trap
            cand_layout, cand_loads, cand_trap_to_qubits = self._apply_relocation(
                layout,
                loads,
                trap_to_qubits,
                last_used,
                moving_qubit=q1,
                target_trap=t2,
                partner_qubit=q2,
            )
            if cand_layout is not None:
                candidates.append(
                    (g, q1, t2, cand_layout, cand_loads, cand_trap_to_qubits)
                )

            # 候选2：把 q2 迁到 q1 所在 trap
            cand_layout, cand_loads, cand_trap_to_qubits = self._apply_relocation(
                layout,
                loads,
                trap_to_qubits,
                last_used,
                moving_qubit=q2,
                target_trap=t1,
                partner_qubit=q1,
            )
            if cand_layout is not None:
                candidates.append(
                    (g, q2, t1, cand_layout, cand_loads, cand_trap_to_qubits)
                )

        return candidates

    def _choose_best_candidate(self, front, ext, candidates):
        """
        选择最佳 relocation 候选。

        主排序仍严格遵循论文 SABRE 的 H 代价函数；
        仅在 H 完全相同或非常接近时，使用更稳定的 tie-break：
          1) H 越小越好
          2) 动作后 front 中可立即执行门数越多越好
          3) front 总距离和越小越好
          4) moving_q / target_t / gate_id 越小越好（稳定性）
        """
        best_payload = None

        for gate_id, moving_q, target_t, cand_layout, cand_loads, cand_trap_to_qubits in candidates:
            score = self._heuristic_cost(cand_layout, front, ext)
            local_exec = self._count_executable_front_gates(front, cand_layout)
            front_dist_sum = self._total_front_distance(front, cand_layout)

            key = (
                score,
                -local_exec,
                front_dist_sum,
                moving_q,
                target_t,
                gate_id,
            )

            if best_payload is None or key < best_payload[-1]:
                best_payload = (
                    gate_id,
                    moving_q,
                    target_t,
                    cand_layout,
                    cand_loads,
                    cand_trap_to_qubits,
                    key,
                )

        return best_payload

    # ============================================================
    # 单次 SABRE pass
    # ============================================================
    def _simulate_pass(self, dag, initial_mapping):
        """
        执行一次 SABRE pass。

        返回：
          - layout: q -> trap
          - trap_to_qubits: trap -> ordered qubit list

        与旧版相比，这里不再只返回 layout，而是同时返回 trap_to_qubits，
        以保留 SABRE 搜索过程中形成的 trap 顺序。
        """
        layout = dict(initial_mapping)
        loads, trap_to_qubits = self._load_layout(layout)
        indegree_map = {g: dag.in_degree(g) for g in dag.nodes}
        done_set = set()

        last_used = {q: -1 for q in layout.keys()}
        timestamp = 0

        while len(done_set) < dag.number_of_nodes():
            front = self._front_layer(dag, indegree_map, done_set)

            executable = []
            for g in front:
                q1, q2 = self._gate_qubits(g)
                if layout[q1] == layout[q2]:
                    executable.append(g)

            if executable:
                for g in executable:
                    q1, q2 = self._gate_qubits(g)
                    timestamp += 1
                    last_used[q1] = timestamp
                    last_used[q2] = timestamp
                    done_set.add(g)
                    for succ in dag.successors(g):
                        indegree_map[succ] -= 1
                continue

            ext = self._extended_set(dag, front, done_set)
            candidates = self._generate_candidate_actions(
                front, layout, loads, trap_to_qubits, last_used
            )

            if not candidates:
                raise RuntimeError(
                    "SABRE2: no executable gate and no valid relocation candidate."
                )

            best_payload = self._choose_best_candidate(front, ext, candidates)

            _, moved_q, _, layout, loads, trap_to_qubits, _ = best_payload
            timestamp += 1
            last_used[moved_q] = timestamp

        return dict(layout), self._normalize_trap_layout(trap_to_qubits)

    # ============================================================
    # 对外接口
    # ============================================================
    def compute_mapping(self):
        """
        返回结构化 bundle：

        {
            "layout": {q: trap_id, ...},
            "trap_to_qubits": {trap_id: [q1, q2, ...], ...},
        }

        这样 run.py 可以：
          - 若只想看传统 mapping，可读 bundle["layout"]
          - 若想保留 SABRE pass 内部形成的 trap 链顺序，可直接使用 bundle["trap_to_qubits"]
        """
        dag_2q = self._build_two_qubit_dag()
        dag_2q_rev = self._build_reversed_two_qubit_dag(dag_2q)

        seed_mapping = self._trivial_seed_mapping()

        # 正向 pass：G
        mapping_after_forward, _ = self._simulate_pass(dag_2q, seed_mapping)

        # 反向 pass：G^R
        mapping_after_backward, trap_layout_after_backward = self._simulate_pass(
            dag_2q_rev, mapping_after_forward
        )

        return {
            "layout": dict(mapping_after_backward),
            "trap_to_qubits": self._normalize_trap_layout(trap_layout_after_backward),
        }


# 严格离子阱容量限制版，加进行多次sabre
class QubitMapSABRE3:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        # 确保距离矩阵已计算
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def find_nearest_free_trap(self, target_trap, loads, traps, current_trap):
        """
        寻找一个有空位的 Trap，使其到 target_trap 的距离最小。
        用于当 target_trap 已满时，寻找一个“次优”的落脚点。
        """
        best_t = -1
        min_dist = float("inf")

        # 遍历所有 Trap
        for t_idx, t_obj in enumerate(traps):
            if t_idx == current_trap:
                continue  # 不移动也是一种选择，但在外部比较

            # 严格容量检查
            cap = t_obj.capacity - self.excess_capacity
            if loads[t_idx] < cap:
                # 检查距离
                dist = self.machine_obj.dist_cache.get((t_idx, target_trap), float("inf"))
                if dist < min_dist:
                    min_dist = dist
                    best_t = t_idx
                elif dist == min_dist:
                    # Tie-breaker: 选择负载更小的
                    if loads[t_idx] < loads[best_t]:
                        best_t = t_idx

        return best_t, min_dist

    def run_pass(self, gate_list, start_mapping):
        """
        执行一次 SABRE 遍历，严格遵守容量限制。
        """
        current_layout = start_mapping.copy()
        traps = self.machine_obj.traps
        move_count = 0

        # 初始化负载表 (O(N) setup, O(1) update)
        loads = [0] * len(traps)
        for t in current_layout.values():
            loads[t] += 1

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue

            qubits = self.parse_obj.cx_gate_map[g]
            q1, q2 = qubits[0], qubits[1]

            t1 = current_layout[q1]
            t2 = current_layout[q2]

            # 如果已经在同一个 Trap，无需移动
            if t1 == t2:
                continue

            cap1 = traps[t1].capacity - self.excess_capacity
            cap2 = traps[t2].capacity - self.excess_capacity

            can_move_to_t1 = loads[t1] < cap1
            can_move_to_t2 = loads[t2] < cap2

            target_t = None
            moving_q = None

            # === 决策逻辑 ===
            if can_move_to_t1 and can_move_to_t2:
                # 两个都有位置：移动到负载较轻的，或者距离中心更近的
                if loads[t1] <= loads[t2]:
                    target_t = t1
                    moving_q = q2
                else:
                    target_t = t2
                    moving_q = q1
            elif can_move_to_t1:
                # 只有 t1 有位置
                target_t = t1
                moving_q = q2
            elif can_move_to_t2:
                # 只有 t2 有位置
                target_t = t2
                moving_q = q1
            else:
                # === 关键修正：两个都满了 ===
                # 原版逻辑会强制移动导致超载。
                # 新逻辑：尝试找到离对方最近的“第三个”有空位的 Trap。

                # 方案 A: q2 移动到 t1 附近的某个空闲 Trap
                alt_t1, dist_q2_new = self.find_nearest_free_trap(t1, loads, traps, t2)

                # 方案 B: q1 移动到 t2 附近的某个空闲 Trap
                alt_t2, dist_q1_new = self.find_nearest_free_trap(t2, loads, traps, t1)

                # 当前距离
                current_dist = self.machine_obj.dist_cache.get((t1, t2), 100)

                # 比较：只有当能缩短距离（或至少不变得太远但解决拥堵）时才移动
                # 这里简单策略：谁能变得更近就选谁
                best_alt_dist = min(dist_q2_new, dist_q1_new)

                if best_alt_dist < current_dist:
                    if dist_q2_new <= dist_q1_new:
                        target_t = alt_t1
                        moving_q = q2
                    else:
                        target_t = alt_t2
                        moving_q = q1
                else:
                    # 如果找不到更好的位置，保持原地不动 (合法性优先于聚集性)
                    target_t = None

            # === 执行移动 ===
            if target_t is not None and target_t != -1:
                old_t = current_layout[moving_q]
                if old_t != target_t:
                    loads[old_t] -= 1
                    loads[target_t] += 1
                    current_layout[moving_q] = target_t
                    move_count += 1

        return current_layout, move_count

    def compute_mapping(self):
        # 1. 准备门序列
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except:
            forward_gates = list(self.parse_obj.gate_graph.nodes)

        backward_gates = forward_gates[::-1]

        # 2. 初始种子：必须使用合法的 Trivial Mapping
        # QubitMapTrivial 会均匀分配，保证不超过 base_cap (如果总容量够的话)
        base_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed_mapping = base_mapper.compute_mapping()

        # 3. 迭代优化
        current_start_mapping = seed_mapping.copy()
        best_mapping = seed_mapping.copy()
        min_total_moves = float("inf")

        # 增加迭代次数，因为严格模式下移动更困难，需要更多轮次收敛
        num_iterations = 20
        print(f"SABRE (Strict): Starting iterative optimization ({num_iterations} rounds)...")

        for i in range(num_iterations):
            # Forward
            layout_after_fwd, fwd_cost = self.run_pass(forward_gates, current_start_mapping)

            # Record Best
            if fwd_cost < min_total_moves:
                min_total_moves = fwd_cost
                best_mapping = current_start_mapping.copy()

            # Backward
            layout_after_bwd, bwd_cost = self.run_pass(backward_gates, layout_after_fwd)

            # Update seed
            current_start_mapping = layout_after_bwd

        print(f"SABRE (Strict): Optimization Done. Best Forward Cost: {min_total_moves}")
        return best_mapping


# 贪婪聚类版，2*3,8，qft32结果是110


class QubitMapSABRE4:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        # 预计算距离矩阵
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def get_trap_load(self, mapping):
        """辅助函数：计算当前每个 Trap 里的量子比特数量"""
        load = [0] * len(self.machine_obj.traps)
        for q, t in mapping.items():
            load[t] += 1
        return load

    def calculate_urgency_score(self, qubit, gate_list, lookahead_limit=100):
        """
        计算离子的'紧迫性'。
        逻辑：离子越早在未来的门中出现，分数越高。
        这对应了 Scheduler 中的 Future Score 逻辑。
        """
        score = 0.0
        count = 0
        for g in gate_list:
            if count >= lookahead_limit:
                break
            if g not in self.parse_obj.cx_gate_map:
                continue

            qubits = self.parse_obj.cx_gate_map[g]
            if qubit in qubits:
                # 越近的门权重呈指数级增加
                score += 100.0 / (count + 1)
                # 只需要找到几次出现即可判断重要性
                if score > 200:
                    break
            count += 1
        return score

    def find_nearest_underloaded_trap(self, source_trap, current_loads, traps):
        """
        寻找最近的、未满的 Trap
        """
        best_t = -1
        min_dist = float("inf")

        for t_idx, t_obj in enumerate(traps):
            if t_idx == source_trap:
                continue

            cap = t_obj.capacity - self.excess_capacity
            if current_loads[t_idx] < cap:
                dist = self.machine_obj.dist_cache.get((source_trap, t_idx), float("inf"))
                if dist < min_dist:
                    min_dist = dist
                    best_t = t_idx
                elif dist == min_dist:
                    # Tie-breaker: load
                    if current_loads[t_idx] < current_loads[best_t]:
                        best_t = t_idx
        return best_t

    def legalize_mapping(self, mapping, gate_list):
        """
        [关键步骤] 合法化：将超载 Trap 中的'冷数据'驱逐出去。
        """
        traps = self.machine_obj.traps
        legal_mapping = mapping.copy()

        # 1. 构建 Trap 内容视图
        trap_contents = {i: [] for i in range(len(traps))}
        for q, t in legal_mapping.items():
            trap_contents[t].append(q)

        # 2. 遍历每个 Trap 检查是否超载
        for t_idx in range(len(traps)):
            cap = traps[t_idx].capacity - self.excess_capacity
            current_ions = trap_contents[t_idx]

            if len(current_ions) > cap:
                excess_count = len(current_ions) - cap
                # print(f"  Mapping Fix: Trap {t_idx} overloaded by {excess_count}. Evicting...")

                # 3. 计算该 Trap 内所有离子的紧迫性分数
                ion_scores = []
                for ion in current_ions:
                    s = self.calculate_urgency_score(ion, gate_list)
                    ion_scores.append((ion, s))

                # 4. 排序：分数高的保留，分数低的驱逐 (升序排列，前 excess_count 个是分数最低的)
                ion_scores.sort(key=lambda x: x[1])

                eviction_candidates = [x[0] for x in ion_scores[:excess_count]]
                kept_ions = [x[0] for x in ion_scores[excess_count:]]

                # 更新当前 Trap 内容
                trap_contents[t_idx] = kept_ions

                # 5. 将被驱逐者移动到最近的空位
                # 为了防止驱逐造成的连锁反应，我们先计算当前的全局负载
                current_loads = [len(trap_contents[i]) for i in range(len(traps))]

                for evict_ion in eviction_candidates:
                    target_t = self.find_nearest_underloaded_trap(t_idx, current_loads, traps)

                    if target_t != -1:
                        legal_mapping[evict_ion] = target_t
                        trap_contents[target_t].append(evict_ion)
                        current_loads[target_t] += 1
                    else:
                        print("CRITICAL WARNING: No space left in machine to rebalance mapping!")

        return legal_mapping

    def run_pass_relaxed(self, gate_list, start_mapping):
        """
        宽松版 SABRE：允许暂时违反容量限制，以此发现全局最优的聚类结构。
        这重现了之前产生 83 Shuttle 结果时的行为。
        """
        current_layout = start_mapping.copy()
        traps = self.machine_obj.traps

        # 为了让离子充分聚类，我们使用一个虚拟的、较大的容量限制
        # 或者完全忽略容量，只看距离

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue
            qubits = self.parse_obj.cx_gate_map[g]
            q1, q2 = qubits[0], qubits[1]

            t1 = current_layout[q1]
            t2 = current_layout[q2]

            if t1 == t2:
                continue

            # 贪婪聚类：总是试图移动到对方的位置
            # 为了防止乒乓效应，我们可以引入一些 tie-breaking

            # 策略：如果移动能显著减少距离，就移。
            # 这里我们简化逻辑，模仿之前的'强制移动'

            # 计算将 q2 移到 t1 的得分 (更近了吗？)
            # 在拓扑结构中，我们简单地将两者设为同一位置

            # 这里我们使用一个简单的启发式：
            # 总是把 '较远' 的那个移向 '较中心' 的那个，或者随机
            # 但为了复现之前的效果，我们允许堆叠。

            # 我们检查目前的堆叠程度，稍微做一点限制防止所有 32 个离子都去 Trap 0
            # 比如限制为 2 倍容量
            soft_limit = (traps[0].capacity - self.excess_capacity) * 2

            loads = self.get_trap_load(current_layout)

            can_go_t1 = loads[t1] < soft_limit
            can_go_t2 = loads[t2] < soft_limit

            if can_go_t1:
                current_layout[q2] = t1
            elif can_go_t2:
                current_layout[q1] = t2

        return current_layout

    def compute_mapping(self):
        # 1. 准备门序列
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except:
            forward_gates = list(self.parse_obj.gate_graph.nodes)

        backward_gates = forward_gates[::-1]

        # 2. 种子：Trivial Mapping (均匀分布)
        base_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed_mapping = base_mapper.compute_mapping()

        current_start_mapping = seed_mapping.copy()

        # 3. 迭代优化 (Relaxed Mode)
        # 允许超载，寻找 Golden Cluster
        print("SABRE (Golden-Cluster Mode): Starting optimization...")
        for i in range(10):  # 10 rounds is sufficient
            # Forward
            layout_after_fwd = self.run_pass_relaxed(forward_gates, current_start_mapping)
            # Backward
            layout_after_bwd = self.run_pass_relaxed(backward_gates, layout_after_fwd)
            # Update
            current_start_mapping = layout_after_bwd

        # 4. 此时 current_start_mapping 包含了一个高度聚类但可能非法的映射
        # 例如 Trap 0 有 15 个离子
        raw_mapping = current_start_mapping

        # 5. 合法化 (Legalization)
        # 使用基于程序顺序 (Future Score) 的智能驱逐
        print("SABRE: Legalizing mapping using interaction urgency...")
        final_mapping = self.legalize_mapping(raw_mapping, forward_gates)

        return final_mapping


# 暴力聚类 + 贪婪静态疏散，qft32 2*3 8的结果是91
class QubitMapSABRE5:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def get_trap_load(self, mapping):
        load = [0] * len(self.machine_obj.traps)
        for q, t in mapping.items():
            load[t] += 1
        return load

    def run_pass(self, gate_list, start_mapping):
        """
        [完全保留 QubitMapSABRE3 的逻辑]
        允许强制移动导致超载，这是产生'黄金聚类'的关键。
        """
        current_layout = start_mapping.copy()
        traps = self.machine_obj.traps
        move_count = 0

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue

            qubits = self.parse_obj.cx_gate_map[g]
            q1, q2 = qubits[0], qubits[1]
            t1, t2 = current_layout[q1], current_layout[q2]

            if t1 != t2:
                loads = self.get_trap_load(current_layout)
                cap1 = traps[t1].capacity - self.excess_capacity
                cap2 = traps[t2].capacity - self.excess_capacity

                can_move_to_t1 = loads[t1] < cap1
                can_move_to_t2 = loads[t2] < cap2

                moved = False
                if can_move_to_t1 and not can_move_to_t2:
                    current_layout[q2] = t1
                    moved = True
                elif can_move_to_t2 and not can_move_to_t1:
                    current_layout[q1] = t2
                    moved = True
                elif can_move_to_t1 and can_move_to_t2:
                    if loads[t1] <= loads[t2]:
                        current_layout[q2] = t1
                    else:
                        current_layout[q1] = t2
                    moved = True
                else:
                    # [关键点] 即使满了，也强制堆叠到 t1
                    # 这会产生超载，但创造了极好的局部性
                    current_layout[q2] = t1
                    moved = True

                if moved:
                    move_count += 1
        return current_layout, move_count

    def legalize_mapping(self, mapping, gate_list):
        """
        [新增] 合法化步骤
        模拟 Scheduler 的 Future Score 逻辑，把超载部分'聪明地'移走。
        """
        traps = self.machine_obj.traps
        final_map = mapping.copy()

        # 1. 预计算每个离子下一次被使用的时间 (Next Use Time)
        # 用作驱逐的依据：越晚被使用的，越该被踢走
        next_use = {q: float("inf") for q in final_map.keys()}

        for idx, g in enumerate(gate_list):
            if g not in self.parse_obj.cx_gate_map:
                continue
            qubits = self.parse_obj.cx_gate_map[g]
            for q in qubits:
                if next_use.get(q) == float("inf"):
                    next_use[q] = idx

        # 2. 检查并修复超载
        # 我们反复迭代直到所有 Trap 合法
        max_fix_rounds = 100
        for _ in range(max_fix_rounds):
            loads = self.get_trap_load(final_map)
            overloaded_traps = []
            for t_idx, load in enumerate(loads):
                cap = traps[t_idx].capacity - self.excess_capacity
                if load > cap:
                    overloaded_traps.append((t_idx, load - cap))

            if not overloaded_traps:
                break  # 全部合法，退出

            # 对每个超载的 Trap 进行驱逐
            for t_id, excess_count in overloaded_traps:
                # 找出该 Trap 里的所有离子
                ions_in_trap = [q for q, t in final_map.items() if t == t_id]

                # 按 Next Use Time 降序排列 (晚使用的排前面 -> 优先驱逐)
                # 如果都没人用 (inf)，则按 ID 排序保持确定性
                ions_in_trap.sort(key=lambda x: (next_use.get(x, float("inf")), x), reverse=True)

                # 选出要驱逐的受害者
                victims = ions_in_trap[:excess_count]

                for victim in victims:
                    # 寻找最近的、未满的 Trap
                    best_dest = -1
                    min_dist = float("inf")

                    # 重新计算当前负载 (因为可能刚刚移入了离子)
                    current_loads = self.get_trap_load(final_map)

                    for cand_t in range(len(traps)):
                        if cand_t == t_id:
                            continue
                        cap = traps[cand_t].capacity - self.excess_capacity

                        if current_loads[cand_t] < cap:
                            dist = self.machine_obj.dist_cache.get((t_id, cand_t), float("inf"))
                            if dist < min_dist:
                                min_dist = dist
                                best_dest = cand_t
                            elif dist == min_dist:
                                # Tie-break: load
                                if current_loads[cand_t] < current_loads[best_dest]:
                                    best_dest = cand_t

                    if best_dest != -1:
                        final_map[victim] = best_dest
                    else:
                        print(f"Warning: Could not rebalance ion {victim} from trap {t_id}. Machine full?")

        return final_map

    def compute_mapping(self):
        # 1. 准备门序列
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except:
            forward_gates = list(self.parse_obj.gate_graph.nodes)

        backward_gates = forward_gates[::-1]

        # 2. 种子
        base_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed_mapping = base_mapper.compute_mapping()

        current_start_mapping = seed_mapping.copy()
        best_mapping = seed_mapping.copy()
        min_moves = float("inf")

        print("SABRE (Overload+Legalize): Starting optimization...")

        # 3. SABRE 迭代 (使用允许超载的 run_pass)
        for i in range(10):
            # Forward
            layout_after_fwd, fwd_cost = self.run_pass(forward_gates, current_start_mapping)

            if fwd_cost < min_moves:
                min_moves = fwd_cost
                best_mapping = current_start_mapping.copy()

            # Backward
            layout_after_bwd, _ = self.run_pass(backward_gates, layout_after_fwd)
            current_start_mapping = layout_after_bwd

        # 此时 best_mapping 是那个包含"黄金聚类"但超载的映射
        print(f"SABRE: Optimization Done. Best Cost (Overloaded): {min_moves}")

        # 4. 合法化
        # 使用 forward_gates 计算未来紧迫性，模拟 Scheduler 的决策
        print("SABRE: Legalizing final mapping...")
        final_legal_mapping = self.legalize_mapping(best_mapping, forward_gates)

        return final_legal_mapping


# 暴力聚类（Mapper产生的超载）” + “全局最优疏散，静态版本，在传给调度器之前就已经进行了合法化操作。当前结果是91
import networkx as nx
import math


class QubitMapSABRE6:
    """
    SABRE-style greedy "violent clustering" (may overload traps),
    then global legalization via Min-Cost Max-Flow (MCMF) with:
      - distance cost (shuttle)
      - convex chain-length marginal cost (approx Δ(N^2)=2N+1 per added ion)
      - urgency scaling using next_use (ions used sooner are more expensive to move far / into long chains)

    NOTE: This class only produces a legal qubit->trap mapping.
    """

    def __init__(
        self,
        parse_obj,
        machine_obj,
        excess_capacity=0,
        # -------------------- Tunable knobs (重点可调参数) --------------------
        # Distance weight. Bigger => legalization prefers shorter moves (fewer shuttles),
        # may tolerate longer chains if needed.
        w_dist=1.0,
        # Chain-length marginal penalty weight. Bigger => avoids long chains more aggressively,
        # but may move ions farther (increasing shuttle distance).
        w_chain=0.8,
        # Urgency strength for next_use scaling.
        # Bigger => ions that will be used soon become MUCH harder to move far / into long chains.
        urgency_strength=3.0,
        # If next_use is inf (never used again), optionally discount its distance cost,
        # letting "dead" ions be moved more freely to satisfy legalization.
        # 1.0 means no discount; smaller (e.g. 0.5) makes dead ions cheaper to move.
        dead_ion_dist_discount=0.7,
        # Safety cap when reading dist_cache.
        default_dist=100,
        # Optional: soft cap for "violent stacking" in run_pass.
        # If set (e.g. 7), we will not force-stack beyond this load (after excess_capacity),
        # instead will pick the better direction or do nothing.
        # None keeps your original "always force stack" behavior.
        n_max_soft=None,
    ):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity

        # Tunables
        self.w_dist = float(w_dist)
        self.w_chain = float(w_chain)
        self.urgency_strength = float(urgency_strength)
        self.dead_ion_dist_discount = float(dead_ion_dist_discount)
        self.default_dist = int(default_dist)
        self.n_max_soft = n_max_soft  # None or int

        # Ensure distance matrix computed
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def get_trap_load(self, mapping):
        load = [0] * len(self.machine_obj.traps)
        for q, t in mapping.items():
            load[t] += 1
        return load

    def _compute_next_use(self, mapping_keys, gate_list):
        """
        next_use[q] = earliest index in gate_list where q appears in a CX gate.
        If never appears => inf
        """
        next_use = {q: float("inf") for q in mapping_keys}
        for idx, g in enumerate(gate_list):
            if g not in self.parse_obj.cx_gate_map:
                continue
            qubits = self.parse_obj.cx_gate_map[g]
            for q in qubits:
                if q in next_use and next_use[q] == float("inf"):
                    next_use[q] = idx
        return next_use

    def _urgency_factor(self, next_use_val):
        """
        Turn next_use into a multiplicative factor for distance cost.
        Smaller next_use (used soon) => larger factor.
        """
        if next_use_val == float("inf"):
            return self.dead_ion_dist_discount
        # 1 + U/(t+1)
        return 1.0 + (self.urgency_strength / (next_use_val + 1.0))

    def run_pass(self, gate_list, start_mapping):
        """
        暴力聚类逻辑（保留原始功能）。
        可选：如果 n_max_soft != None，则在“暴力堆叠”分支加一个软上限，
        防止链长被推得过大。
        """
        current_layout = start_mapping.copy()
        traps = self.machine_obj.traps
        move_count = 0

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue

            qubits = self.parse_obj.cx_gate_map[g]
            q1, q2 = qubits[0], qubits[1]
            t1, t2 = current_layout[q1], current_layout[q2]

            if t1 != t2:
                loads = self.get_trap_load(current_layout)
                cap1 = traps[t1].capacity - self.excess_capacity
                cap2 = traps[t2].capacity - self.excess_capacity

                can_move_to_t1 = loads[t1] < cap1
                can_move_to_t2 = loads[t2] < cap2

                moved = False
                if can_move_to_t1 and not can_move_to_t2:
                    current_layout[q2] = t1
                    moved = True
                elif can_move_to_t2 and not can_move_to_t1:
                    current_layout[q1] = t2
                    moved = True
                elif can_move_to_t1 and can_move_to_t2:
                    if loads[t1] <= loads[t2]:
                        current_layout[q2] = t1
                    else:
                        current_layout[q1] = t2
                    moved = True
                else:
                    # 暴力堆叠：即使满了也往 t1 塞（原始行为）
                    # 可选软上限：超过 n_max_soft 就不再硬塞（避免超长链）
                    if self.n_max_soft is None:
                        current_layout[q2] = t1
                        moved = True
                    else:
                        # soft cap uses "effective cap" = n_max_soft - excess_capacity
                        soft_cap = max(0, int(self.n_max_soft) - int(self.excess_capacity))
                        if loads[t1] < soft_cap:
                            current_layout[q2] = t1
                            moved = True
                        elif loads[t2] < soft_cap:
                            current_layout[q1] = t2
                            moved = True
                        else:
                            # both beyond soft cap: do nothing (or you can pick nearer, but keep simple & safe)
                            moved = False

                if moved:
                    move_count += 1

        return current_layout, move_count

    def legalize_mapping_max_flow(self, mapping, gate_list):
        """
        基于最小费用最大流 (Min-Cost Max-Flow) 的全局最优合法化（升级版）：

        - Supply：超载trap里挑 victim（next_use 最晚的）
        - Demand：未满trap提供空位
        - MCMF cost（Ion -> Slot）：
              cost = w_dist * dist(current_trap, dest_trap) * urgency_factor(ion)
                   + w_chain * deltaN2(dest_trap, slot_index)
          其中 deltaN2 ≈ 2N+1，N为目的trap当前链长，slot_index让成本“凸”起来（越塞越贵）
        """
        traps = self.machine_obj.traps
        final_map = mapping.copy()

        # 1) next_use (for victim selection + urgency factor)
        next_use = self._compute_next_use(final_map.keys(), gate_list)

        # 2) supply/demand
        loads = self.get_trap_load(final_map)
        supply_nodes = []  # (ion_id, current_trap_id)
        demand_nodes = []  # (trap_id, free_capacity)

        # demand
        for t_idx, load in enumerate(loads):
            cap = traps[t_idx].capacity - self.excess_capacity
            if load < cap:
                demand_nodes.append((t_idx, cap - load))

        # supply: for each overloaded trap, pick latest-used ions as victims
        for t_idx, load in enumerate(loads):
            cap = traps[t_idx].capacity - self.excess_capacity
            if load > cap:
                excess_count = load - cap
                ions_in_trap = [q for q, t in final_map.items() if t == t_idx]
                ions_in_trap.sort(key=lambda x: (next_use.get(x, float("inf")), x), reverse=True)
                victims = ions_in_trap[:excess_count]
                for v in victims:
                    supply_nodes.append((v, t_idx))

        if not supply_nodes:
            return final_map  # already legal

        # Quick feasibility sanity: total free slots must cover total supply
        total_supply = len(supply_nodes)
        total_demand_slots = sum(free for _, free in demand_nodes)
        if total_demand_slots < total_supply:
            print("CRITICAL: Not enough free slots to legalize. Machine might be physically full.")
            return final_map

        # 3) build flow network
        G = nx.DiGraph()
        SOURCE = "S"
        SINK = "T"
        G.add_node(SOURCE)
        G.add_node(SINK)

        # Pre-create slot nodes for convex chain cost
        # For each dest_trap with free_slots, create slot_1..slot_free_slots
        # Each slot has its own marginal chain cost via deltaN2 = 2*(L + (k-1)) + 1
        slot_nodes = []  # list of (dest_trap, k, slot_name)
        for dest_trap, free_slots in demand_nodes:
            L = loads[dest_trap]  # current chain length in that trap
            for k in range(1, free_slots + 1):
                slot = f"trap_{dest_trap}_slot_{k}"
                slot_nodes.append((dest_trap, k, slot))
                # slot -> sink (each slot capacity 1)
                G.add_edge(slot, SINK, capacity=1, weight=0)

        # Supply edges and ion->slot edges
        for ion, current_trap in supply_nodes:
            ion_node = f"ion_{ion}"
            G.add_edge(SOURCE, ion_node, capacity=1, weight=0)

            urg = self._urgency_factor(next_use.get(ion, float("inf")))

            for dest_trap, k, slot in slot_nodes:
                dist = self.machine_obj.dist_cache.get((current_trap, dest_trap), self.default_dist)

                # convex marginal chain penalty: Δ(N^2) where N increases as we fill slots
                # N_before = L + (k-1)
                L = loads[dest_trap]
                n_before = L + (k - 1)
                deltaN2 = 2 * n_before + 1  # ≈ (N+1)^2 - N^2

                cost = (self.w_dist * float(dist) * float(urg)) + (self.w_chain * float(deltaN2))
                G.add_edge(ion_node, slot, capacity=1, weight=int(cost))

        # 4) solve min-cost max-flow
        try:
            flow_dict = nx.max_flow_min_cost(G, SOURCE, SINK)
        except nx.NetworkXUnfeasible:
            print("CRITICAL: Max flow unfeasible. Machine might be physically full or disconnected.")
            return final_map

        # 5) apply result
        # Each ion_node should send flow 1 to exactly one slot node
        for ion, _ in supply_nodes:
            ion_node = f"ion_{ion}"
            if ion_node not in flow_dict:
                continue

            chosen_trap = None
            for dest_node, flow_val in flow_dict[ion_node].items():
                if flow_val > 0 and dest_node.startswith("trap_") and "_slot_" in dest_node:
                    # dest_node format: trap_{dest_trap}_slot_{k}
                    # parse dest_trap
                    try:
                        parts = dest_node.split("_")
                        # ["trap", "{t}", "slot", "{k}"]
                        dest_trap_id = int(parts[1])
                        chosen_trap = dest_trap_id
                        break
                    except Exception:
                        continue

            if chosen_trap is not None:
                final_map[ion] = chosen_trap

        return final_map

    def compute_mapping(self):
        # 1) prepare gate order
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except Exception:
            forward_gates = list(self.parse_obj.gate_graph.nodes)

        backward_gates = forward_gates[::-1]

        # 2) seed mapping
        base_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed_mapping = base_mapper.compute_mapping()

        current_start_mapping = seed_mapping.copy()
        best_mapping = seed_mapping.copy()
        min_moves = float("inf")

        print("SABRE (Overload+MaxFlow+Chain+Urgency): Starting optimization...")

        # 3) SABRE iterations (overloaded)
        for _ in range(10):
            # Forward
            layout_after_fwd, fwd_cost = self.run_pass(forward_gates, current_start_mapping)

            # Keep the best overloaded mapping (as you originally did)
            if fwd_cost < min_moves:
                min_moves = fwd_cost
                best_mapping = current_start_mapping.copy()

            # Backward
            layout_after_bwd, _ = self.run_pass(backward_gates, layout_after_fwd)
            current_start_mapping = layout_after_bwd

        print(f"SABRE: Optimization Done. Best Cost (Overloaded): {min_moves}")

        # 4) global legalization
        print("SABRE: Legalizing using Min-Cost Max-Flow (distance + convex chain + urgency)...")
        final_legal_mapping = self.legalize_mapping_max_flow(best_mapping, forward_gates)

        return final_legal_mapping


# 暴力聚类（Mapper产生的超载）” + “全局最优疏散，动态版本，传给调度器超载版本的映射。对应的muss调度要进行修改，采用第四版本的调度。
# mappers.py

class QubitMapSABRE7:
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        # 预计算距离
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def get_trap_load(self, mapping):
        load = [0] * len(self.machine_obj.traps)
        for q, t in mapping.items():
            load[t] += 1
        return load

    def run_pass(self, gate_list, start_mapping):
        """
        允许超载的 SABRE 核心逻辑。
        为了让离子形成最佳聚类，我们允许 Trap 暂时超载。
        """
        current_layout = start_mapping.copy()
        traps = self.machine_obj.traps
        move_count = 0

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue

            qubits = self.parse_obj.cx_gate_map[g]
            q1, q2 = qubits[0], qubits[1]
            t1, t2 = current_layout[q1], current_layout[q2]

            if t1 != t2:
                loads = self.get_trap_load(current_layout)
                cap1 = traps[t1].capacity - self.excess_capacity
                cap2 = traps[t2].capacity - self.excess_capacity

                can_move_to_t1 = loads[t1] < cap1
                can_move_to_t2 = loads[t2] < cap2

                moved = False
                # 贪心策略：优先去空位，如果都满，强制去 t1 (制造超载)
                if can_move_to_t1 and not can_move_to_t2:
                    current_layout[q2] = t1
                    moved = True
                elif can_move_to_t2 and not can_move_to_t1:
                    current_layout[q1] = t2
                    moved = True
                elif can_move_to_t1 and can_move_to_t2:
                    # Tie-breaking
                    if loads[t1] <= loads[t2]:
                        current_layout[q2] = t1
                    else:
                        current_layout[q1] = t2
                    moved = True
                else:
                    # === 关键点：允许超载堆叠 ===
                    # 这里不检查容量，强制移动，把压力交给调度器的 Loading Phase
                    current_layout[q2] = t1
                    moved = True

                if moved:
                    move_count += 1
        return current_layout, move_count

    def compute_mapping(self):
        # 1. 门序列
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except:
            forward_gates = list(self.parse_obj.gate_graph.nodes)
        backward_gates = forward_gates[::-1]

        # 2. 种子 (均匀分布)
        base_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed_mapping = base_mapper.compute_mapping()

        current_start_mapping = seed_mapping.copy()
        best_mapping = seed_mapping.copy()
        min_moves = float("inf")

        print("SABRE (Golden Cluster Mode): Starting optimization...")

        # 3. 迭代 (不进行任何 Legalization)
        for i in range(10):
            layout_after_fwd, fwd_cost = self.run_pass(forward_gates, current_start_mapping)
            if fwd_cost < min_moves:
                min_moves = fwd_cost
                best_mapping = current_start_mapping.copy()

            layout_after_bwd, _ = self.run_pass(backward_gates, layout_after_fwd)
            current_start_mapping = layout_after_bwd

        print(f"SABRE: Generated Golden Cluster (Cost: {min_moves}). Passing to Scheduler for loading.")

        # 直接返回超载的 Map
        return best_mapping
    
    
class QubitMapSABRELarge:
    """
    面向 large / EML-QCCD 的 SABRE 初始映射器。

    目标：
      1) 支持每个 QCCD 的最大载荷约束（默认 32 qubits）；
      2) 支持 module-aware 初始映射：
           - 先把逻辑 qubit 映射到 qccd/module
           - 再在 module 内做 zone-aware trap 落位
      3) 支持 zone-aware seed：
           - optical > operation > storage
           - 或按 zone_level 从高到低（数值越小优先级越高）
      4) 保持与 QubitMapSABRE2 相同的返回风格：
           {
               "layout": {q: trap_id, ...},
               "trap_to_qubits": {trap_id: [q1, q2, ...], ...},
           }

    说明：
      - 这个类只负责“大规模初始映射”，不负责 look-ahead SWAP insertion；
      - 它的 SABRE 搜索运行在“QCCD module 粒度”上，而不是 trap 粒度；
      - module 内部的最终 trap 落位是 zone-aware 的贪心分配。
    """

    def __init__(
        self,
        parse_obj,
        machine_obj,
        excess_capacity=0,
        max_qubits_per_qccd=None,
        extended_set_size=20,
        extended_weight=0.5,
    ):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = int(excess_capacity)
        if max_qubits_per_qccd is None:
            max_qubits_per_qccd = getattr(getattr(self.machine_obj, "mparams", None), "max_qubits_per_qccd", 32)
        self.max_qubits_per_qccd = int(max_qubits_per_qccd)
        self.extended_set_size = int(extended_set_size)
        self.extended_weight = float(extended_weight)

        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

        # 预处理机器上的 QCCD / trap / 容量信息
        self._build_module_metadata()

    # ============================================================
    # 一、基础元数据与工具函数
    # ============================================================
    def _logical_qubits(self):
        """
        获取需要映射的逻辑 qubit 列表。
        优先使用 parse_obj.nqubits；若没有，则从 all_gate_map/cx_graph 推断。
        """
        if hasattr(self.parse_obj, "nqubits"):
            return list(range(self.parse_obj.nqubits))

        qubit_set = set()

        if hasattr(self.parse_obj, "all_gate_map"):
            for _, data in self.parse_obj.all_gate_map.items():
                if isinstance(data, dict):
                    qlist = data.get("qubits", [])
                else:
                    qlist = list(data)
                for q in qlist:
                    qubit_set.add(q)

        if hasattr(self.parse_obj, "cx_graph"):
            for q in self.parse_obj.cx_graph.nodes:
                qubit_set.add(q)

        return sorted(qubit_set)

    def _build_module_metadata(self):
        """
        从 machine 中提取 QCCD / trap 元数据。

        期望 Stage 1 的 machine / test_machines 已经为 trap 补充：
          - qccd_id
          - zone_type
          - zone_level

        若缺失 qccd_id，则退化为“所有 trap 属于一个 module”，
        这样不会误伤小规模路径。
        """
        self.module_to_traps = {}
        self.trap_to_module = {}
        self.trap_effective_capacity = {}
        self.module_capacity = {}
        self.module_order = []

        traps = getattr(self.machine_obj, "traps", [])
        if not traps:
            raise RuntimeError("QubitMapSABRELarge: machine_obj.traps is empty.")

        for trap in traps:
            trap_id = trap.id
            module_id = getattr(trap, "qccd_id", 0)

            self.trap_to_module[trap_id] = module_id
            self.module_to_traps.setdefault(module_id, []).append(trap_id)

            eff_cap = int(getattr(trap, "capacity", 0)) - self.excess_capacity
            if eff_cap < 0:
                eff_cap = 0
            self.trap_effective_capacity[trap_id] = eff_cap

        self.module_order = sorted(self.module_to_traps.keys())

        # 每个模块的有效容量：
        #   - 多 module / EML：min(模块 trap 总容量, max_qubits_per_qccd)
        #   - 单 module fallback：使用真实总容量，不做 32 截断
        single_module_fallback = (len(self.module_order) == 1)
        for module_id in self.module_order:
            total_cap = sum(self.trap_effective_capacity[t] for t in self.module_to_traps[module_id])
            if single_module_fallback:
                self.module_capacity[module_id] = total_cap
            else:
                self.module_capacity[module_id] = min(total_cap, self.max_qubits_per_qccd)

        total_capacity = sum(self.module_capacity[m] for m in self.module_order)
        nqubits = len(self._logical_qubits())
        if total_capacity < nqubits:
            raise RuntimeError(
                f"QubitMapSABRELarge failed: machine total module capacity {total_capacity} "
                f"is smaller than required logical qubits {nqubits}."
            )

    def _trap_priority_key(self, trap_obj):
        """
        module 内部 zone-aware 的 trap 排序键：
          1) zone_level 越小优先级越高
          2) 若没有 zone_level，则按 zone_type:
               optical > operation > storage
          3) 最后按 trap id 稳定排序
        """
        zone_level = getattr(trap_obj, "zone_level", None)
        if zone_level is not None:
            return (int(zone_level), trap_obj.id)

        zone_type = getattr(trap_obj, "zone_type", None)
        if zone_type is not None:
            order = {"optical": 0, "operation": 1, "storage": 2}
            return (order.get(zone_type, 9), trap_obj.id)

        return (0, trap_obj.id)

    def _sorted_traps_in_module(self, module_id):
        """
        返回某个 module 内按 zone 优先级排序后的 trap 列表。
        """
        trap_objs = [self.machine_obj.traps[t] for t in self.module_to_traps[module_id]]
        trap_objs.sort(key=self._trap_priority_key)
        return [t.id for t in trap_objs]

    def _trap_distance(self, t1, t2):
        if t1 == t2:
            return 0
        return self.machine_obj.dist_cache.get((t1, t2), float("inf"))

    def _module_distance(self, m1, m2):
        """
        module 间距离。

        优先级：
          1) 若 machine 有 qccd_graph，则走 qccd_graph 最短路
          2) 否则用两个 module 中 trap 间最小 trap distance 近似
        """
        if m1 == m2:
            return 0

        qccd_graph = getattr(self.machine_obj, "qccd_graph", None)
        if qccd_graph is not None:
            try:
                import networkx as nx
                return nx.shortest_path_length(qccd_graph, source=m1, target=m2)
            except Exception:
                pass

        best = float("inf")
        for t1 in self.module_to_traps[m1]:
            for t2 in self.module_to_traps[m2]:
                d = self._trap_distance(t1, t2)
                if d < best:
                    best = d
        return best

    def _gate_qubits(self, gate_id):
        return tuple(self.parse_obj.cx_gate_map[gate_id])

    def _build_two_qubit_dag(self):
        """
        与 SABRE2 保持一致：优先使用 2Q-only DAG。
        """
        dag = getattr(self.parse_obj, "twoq_gate_graph", None)
        if dag is None or dag.number_of_nodes() == 0:
            dag = self.parse_obj.gate_graph.subgraph(
                list(self.parse_obj.cx_gate_map.keys())
            ).copy()

        import networkx as nx
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("QubitMapSABRELarge requires a 2Q-only DAG.")
        return dag

    def _build_reversed_two_qubit_dag(self, dag_2q):
        import networkx as nx
        rev = dag_2q.reverse(copy=True)
        if not nx.is_directed_acyclic_graph(rev):
            raise ValueError("Reversed 2Q DAG is not a DAG.")
        return rev

    def _build_program_graph_weights(self):
        """
        构建逻辑 qubit 交互图的边权。
        用于 module-aware seed。
        """
        edge_weights = {}
        for g in self.parse_obj.gate_graph:
            if g not in self.parse_obj.cx_gate_map:
                continue
            q1, q2 = self.parse_obj.cx_gate_map[g]
            a, b = min(q1, q2), max(q1, q2)
            edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1
        return edge_weights

    def _qubit_weight(self):
        """
        每个逻辑 qubit 的总交互权重。
        """
        weights = {q: 0 for q in self._logical_qubits()}
        for (q1, q2), w in self._build_program_graph_weights().items():
            weights[q1] += w
            weights[q2] += w
        return weights

    # ============================================================
    # 二、module-aware trivial seed
    # ============================================================
    def _ordered_module_priority(self):
        """
        给 module 排序，便于 trivial seed 使用。
        当前规则：
          - module 内最高优先级 trap 越“高级”，module 越靠前
          - 然后按 module id 稳定排序
        """
        ranked = []
        for m in self.module_order:
            traps = [self.machine_obj.traps[t] for t in self.module_to_traps[m]]
            best_key = min(self._trap_priority_key(t) for t in traps)
            ranked.append((best_key, m))
        ranked.sort()
        return [m for _, m in ranked]

    def _trivial_module_seed(self):
        """
        大规模 trivial seed：
          - 先按 qubit 编号顺序分配到 module
          - module 顺序按 zone-aware 优先级
          - 严格遵守每 module <= 32 限制
        """
        logical_qubits = self._logical_qubits()
        modules = self._ordered_module_priority()

        module_layout = {}
        module_loads = {m: 0 for m in modules}

        q_idx = 0
        for m in modules:
            cap = self.module_capacity[m]
            for _ in range(cap):
                if q_idx >= len(logical_qubits):
                    break
                module_layout[logical_qubits[q_idx]] = m
                module_loads[m] += 1
                q_idx += 1
            if q_idx >= len(logical_qubits):
                break

        if q_idx != len(logical_qubits):
            raise RuntimeError(
                f"QubitMapSABRELarge seed failed: only mapped {q_idx}/{len(logical_qubits)} qubits into modules."
            )

        return module_layout, module_loads

    # ============================================================
    # 三、SABRE at module level
    # ============================================================
    def _front_layer(self, dag, indegree_map, done_set):
        return [g for g in dag.nodes if g not in done_set and indegree_map[g] == 0]

    def _extended_set(self, dag, front, done_set):
        ext = []
        seen = set(front)
        queue = list(front)

        while queue and len(ext) < self.extended_set_size:
            cur = queue.pop(0)
            for succ in dag.successors(cur):
                if succ in seen or succ in done_set:
                    continue
                seen.add(succ)
                ext.append(succ)
                queue.append(succ)
                if len(ext) >= self.extended_set_size:
                    break
        return ext

    def _gate_module_distance_under_layout(self, gate_id, module_layout):
        q1, q2 = self._gate_qubits(gate_id)
        m1 = module_layout[q1]
        m2 = module_layout[q2]
        return self._module_distance(m1, m2)

    def _heuristic_cost(self, module_layout, front, ext):
        """
        SABRE 的 module-level 代价函数：
            H = avg(front module distances) + W * avg(extended module distances)
        """
        if not front:
            return 0.0

        front_cost = sum(
            self._gate_module_distance_under_layout(g, module_layout) for g in front
        ) / float(len(front))

        if not ext:
            return front_cost

        ext_cost = sum(
            self._gate_module_distance_under_layout(g, module_layout) for g in ext
        ) / float(len(ext))

        return front_cost + self.extended_weight * ext_cost

    def _count_executable_front_gates(self, front, module_layout):
        """
        统计当前 front 中有多少 gate 已处于同 module。
        注意：这里是“同 module 可本地化”的判断，不要求同 trap。
        """
        cnt = 0
        for g in front:
            q1, q2 = self._gate_qubits(g)
            if module_layout[q1] == module_layout[q2]:
                cnt += 1
        return cnt

    def _total_front_distance(self, front, module_layout):
        return sum(self._gate_module_distance_under_layout(g, module_layout) for g in front)

    def _load_module_layout(self, module_layout):
        loads = {m: 0 for m in self.module_order}
        module_to_qubits = {m: [] for m in self.module_order}
        for q, m in module_layout.items():
            loads[m] += 1
            module_to_qubits[m].append(q)
        return loads, module_to_qubits

    def _select_lru_victim(self, target_module, module_to_qubits, last_used, forbidden=None):
        forbidden = forbidden or set()
        candidates = [q for q in module_to_qubits[target_module] if q not in forbidden]
        if not candidates:
            return None
        return min(candidates, key=lambda q: last_used.get(q, -1))

    def _apply_module_relocation(
        self,
        module_layout,
        module_loads,
        module_to_qubits,
        last_used,
        moving_qubit,
        target_module,
        partner_qubit=None,
    ):
        """
        module 级 relocation：
          - 若 target module 还有空位，则直接迁入
          - 若 target module 已满，则使用 LRU replacement
        """
        new_layout = dict(module_layout)
        new_loads = dict(module_loads)
        new_module_to_qubits = {m: list(qs) for m, qs in module_to_qubits.items()}

        src_module = new_layout[moving_qubit]
        if src_module == target_module:
            return new_layout, new_loads, new_module_to_qubits

        # 目标 module 还有空位：直接移动
        if new_loads[target_module] < self.module_capacity[target_module]:
            new_module_to_qubits[src_module].remove(moving_qubit)
            new_module_to_qubits[target_module].append(moving_qubit)
            new_loads[src_module] -= 1
            new_loads[target_module] += 1
            new_layout[moving_qubit] = target_module
            return new_layout, new_loads, new_module_to_qubits

        # 目标 module 满了：使用 LRU victim replacement
        victim = self._select_lru_victim(
            target_module,
            new_module_to_qubits,
            last_used,
            forbidden={partner_qubit} if partner_qubit is not None else set(),
        )
        if victim is None:
            return None, None, None

        # victim: target_module -> src_module
        new_module_to_qubits[target_module].remove(victim)
        new_module_to_qubits[src_module].append(victim)
        new_layout[victim] = src_module

        # moving_qubit: src_module -> target_module
        new_module_to_qubits[src_module].remove(moving_qubit)
        new_module_to_qubits[target_module].append(moving_qubit)
        new_layout[moving_qubit] = target_module

        # 负载不变
        return new_layout, new_loads, new_module_to_qubits

    def _generate_candidate_actions(self, front, module_layout, module_loads, module_to_qubits, last_used):
        """
        仅生成论文风格的双向候选：
          - q1 -> module(q2)
          - q2 -> module(q1)
        """
        candidates = []

        for g in front:
            q1, q2 = self._gate_qubits(g)
            m1 = module_layout[q1]
            m2 = module_layout[q2]

            if m1 == m2:
                continue

            cand_layout, cand_loads, cand_module_to_qubits = self._apply_module_relocation(
                module_layout,
                module_loads,
                module_to_qubits,
                last_used,
                moving_qubit=q1,
                target_module=m2,
                partner_qubit=q2,
            )
            if cand_layout is not None:
                candidates.append((g, q1, m2, cand_layout, cand_loads, cand_module_to_qubits))

            cand_layout, cand_loads, cand_module_to_qubits = self._apply_module_relocation(
                module_layout,
                module_loads,
                module_to_qubits,
                last_used,
                moving_qubit=q2,
                target_module=m1,
                partner_qubit=q1,
            )
            if cand_layout is not None:
                candidates.append((g, q2, m1, cand_layout, cand_loads, cand_module_to_qubits))

        return candidates

    def _choose_best_candidate(self, front, ext, candidates):
        """
        module 级 SABRE 候选选择。
        """
        best_payload = None

        for gate_id, moving_q, target_m, cand_layout, cand_loads, cand_module_to_qubits in candidates:
            score = self._heuristic_cost(cand_layout, front, ext)
            local_exec = self._count_executable_front_gates(front, cand_layout)
            front_dist_sum = self._total_front_distance(front, cand_layout)

            key = (
                score,
                -local_exec,
                front_dist_sum,
                moving_q,
                target_m,
                gate_id,
            )

            if best_payload is None or key < best_payload[-1]:
                best_payload = (
                    gate_id,
                    moving_q,
                    target_m,
                    cand_layout,
                    cand_loads,
                    cand_module_to_qubits,
                    key,
                )

        return best_payload

    def _simulate_module_pass(self, dag, initial_module_layout):
        """
        在 module 粒度执行一次 SABRE pass。
        """
        module_layout = dict(initial_module_layout)
        module_loads, module_to_qubits = self._load_module_layout(module_layout)

        indegree_map = {g: dag.in_degree(g) for g in dag.nodes}
        done_set = set()

        last_used = {q: -1 for q in module_layout.keys()}
        timestamp = 0

        while len(done_set) < dag.number_of_nodes():
            front = self._front_layer(dag, indegree_map, done_set)

            executable = []
            for g in front:
                q1, q2 = self._gate_qubits(g)
                if module_layout[q1] == module_layout[q2]:
                    executable.append(g)

            if executable:
                for g in executable:
                    q1, q2 = self._gate_qubits(g)
                    timestamp += 1
                    last_used[q1] = timestamp
                    last_used[q2] = timestamp
                    done_set.add(g)
                    for succ in dag.successors(g):
                        indegree_map[succ] -= 1
                continue

            ext = self._extended_set(dag, front, done_set)
            candidates = self._generate_candidate_actions(
                front, module_layout, module_loads, module_to_qubits, last_used
            )

            if not candidates:
                raise RuntimeError(
                    "QubitMapSABRELarge: no executable gate and no valid module relocation candidate."
                )

            best_payload = self._choose_best_candidate(front, ext, candidates)
            _, moved_q, _, module_layout, module_loads, module_to_qubits, _ = best_payload

            timestamp += 1
            last_used[moved_q] = timestamp

        return dict(module_layout), {m: list(qs) for m, qs in module_to_qubits.items()}

    # ============================================================
    # 四、module -> trap 的 zone-aware 落位
    # ============================================================
    def _build_zone_seed_order_in_module(self, module_id, qubits_in_module):
        """
        给 module 内 qubit 生成一个优先落位顺序。

        当前策略：
          - 按 qubit 的总交互权重降序
          - 权重大的先尝试占据 optical / operation
          - 同权时按 qubit id 稳定排序
        """
        q_weight = self._qubit_weight()
        ordered = list(qubits_in_module)
        ordered.sort(key=lambda q: (-q_weight.get(q, 0), q))
        return ordered

    def _assign_module_qubits_to_traps(self, module_id, qubits_in_module):
        """
        在固定 module 内进行 zone-aware trap 分配。
        规则：
          - trap 按 zone_level / zone_type 优先级排序
          - qubit 按交互权重排序
          - 顺序填充，不超过各 trap 有效容量
        """
        trap_order = self._sorted_traps_in_module(module_id)
        ordered_qubits = self._build_zone_seed_order_in_module(module_id, qubits_in_module)

        trap_to_qubits = {t: [] for t in self.module_to_traps[module_id]}
        layout = {}

        idx = 0
        for trap_id in trap_order:
            cap = self.trap_effective_capacity[trap_id]
            for _ in range(cap):
                if idx >= len(ordered_qubits):
                    break
                q = ordered_qubits[idx]
                trap_to_qubits[trap_id].append(q)
                layout[q] = trap_id
                idx += 1
            if idx >= len(ordered_qubits):
                break

        if idx != len(ordered_qubits):
            raise RuntimeError(
                f"QubitMapSABRELarge failed inside module {module_id}: "
                f"only placed {idx}/{len(ordered_qubits)} qubits."
            )

        return layout, trap_to_qubits

    def _materialize_trap_layout(self, module_layout, module_to_qubits):
        """
        将 module 级布局具体化为 trap 级布局。
        """
        final_layout = {}
        final_trap_to_qubits = {t.id: [] for t in self.machine_obj.traps}

        for module_id in self.module_order:
            qubits = module_to_qubits.get(module_id, [])
            local_layout, local_trap_to_qubits = self._assign_module_qubits_to_traps(
                module_id, qubits
            )

            for q, t in local_layout.items():
                final_layout[q] = t
            for t, qlist in local_trap_to_qubits.items():
                final_trap_to_qubits[t] = list(qlist)

        return final_layout, final_trap_to_qubits

    # ============================================================
    # 五、对外接口
    # ============================================================
    def compute_mapping(self):
        """
        主入口：
          1) 构造 2Q DAG
          2) 用 module-aware trivial seed 初始化
          3) 正向 SABRE（module 粒度）
          4) 反向 SABRE（module 粒度）
          5) 在每个 module 内做 zone-aware trap 落位
          6) 返回结构化 bundle
        """
        dag_2q = self._build_two_qubit_dag()
        dag_2q_rev = self._build_reversed_two_qubit_dag(dag_2q)

        # 第一步：module-aware seed
        seed_module_layout, _ = self._trivial_module_seed()

        # 第二步：正向 pass
        module_layout_after_forward, _ = self._simulate_module_pass(dag_2q, seed_module_layout)

        # 第三步：反向 pass
        module_layout_after_backward, module_to_qubits_after_backward = self._simulate_module_pass(
            dag_2q_rev, module_layout_after_forward
        )

        # 第四步：module 内 zone-aware trap 落位
        final_layout, final_trap_to_qubits = self._materialize_trap_layout(
            module_layout_after_backward,
            module_to_qubits_after_backward,
        )

        return {
            "layout": dict(final_layout),
            "trap_to_qubits": {t: list(qs) for t, qs in final_trap_to_qubits.items()},
        }
