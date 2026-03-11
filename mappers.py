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
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity

    def compute_mapping(self):
        # 获取所有量子比特 ID
        # 注意：这里最好用 parse_obj.cx_graph.nodes 确保涵盖所有相关比特
        all_qubits = sorted(list(self.parse_obj.cx_graph.nodes))
        num_qubits = len(all_qubits)

        output_partition = {}
        traps = self.machine_obj.traps
        num_traps = len(traps)

        # 计算每个 Trap 的有效容量
        # 为了负载均衡，通常平均分配
        base_cap = self.machine_obj.traps[0].capacity - self.excess_capacity

        # Trivial Mapping: 简单的取模分配 或 顺序填满
        # MUSS-TI 通常倾向于顺序填满 (0,0,0, 1,1,1...) 以保持局部性

        current_trap = 0
        current_load = 0

        for q in all_qubits:
            output_partition[q] = current_trap
            current_load += 1
            if current_load >= base_cap:
                current_trap = (current_trap + 1) % num_traps
                current_load = 0

        return output_partition


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


# 论文标准版。
# mappers.py
class QubitMapSABRE2:
    """
    忠实实现 MUSS-TI 论文第 3.4 节描述的 SABRE 初始映射。
    流程：
      1. 从 trivial mapping 开始
      2. 在原始电路上模拟执行一遍（forward pass），得到映射 π'
      3. 以 π' 为初始映射，在反向电路上模拟执行一遍（backward pass），得到映射 π''
      4. 将 π'' 作为最终初始映射返回
    模拟执行严格遵守容量限制（不允许超载），使用 LRU 替换策略处理容量冲突。
    输出字典：{qubit: trap_id}
    """
    def __init__(self, parse_obj, machine_obj, excess_capacity=0):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    def _simulate_pass(self, gate_list, initial_mapping):
        """
        模拟执行一遍电路，返回最终映射。
        - 遵守容量限制，不允许超载
        - 使用 LRU 策略驱逐 qubit 以腾出空间
        - 驱逐目标 trap 选择距离源 trap 最近的空闲 trap
        """
        # 复制初始映射
        mapping = initial_mapping.copy()
        num_traps = len(self.machine_obj.traps)
        cap = self.machine_obj.traps[0].capacity - self.excess_capacity

        # 当前每个 trap 的负载（索引与 trap ID 一致，因机器构造时 ID 连续）
        loads = [0] * num_traps
        for t in mapping.values():
            loads[t] += 1

        # 记录每个 qubit 的最后使用时间（门索引）
        last_used = {q: -1 for q in mapping.keys()}
        time = 0

        # 辅助函数：移动 qubit src_q 到 dst_trap
        def move_qubit(src_q, dst_trap):
            nonlocal loads
            src_trap = mapping[src_q]
            loads[src_trap] -= 1
            loads[dst_trap] += 1
            mapping[src_q] = dst_trap

        # 辅助函数：从指定 trap 中驱逐一个 LRU qubit 到最近的空闲 trap
        def evict_lru_from_trap(src_trap):
            # 找出 src_trap 中所有 qubit
            qubits_in_trap = [q for q, t in mapping.items() if t == src_trap]
            # 按最后使用时间升序排序（最久未用的在前）
            qubits_in_trap.sort(key=lambda q: last_used[q])
            victim = qubits_in_trap[0]  # 最久未用

            # 找一个有空位的目标 trap
            free_traps = [i for i in range(num_traps) if loads[i] < cap]
            if free_traps:
                # 选择距离 src_trap 最近的空闲 trap
                # 注意：dist_cache 键为 (trap_id, trap_id)，此处索引即 ID
                dest_trap = min(free_traps, key=lambda t: self.machine_obj.dist_cache.get((src_trap, t), 1000))
            else:
                # 所有 trap 都满（理论上不会发生，因为总容量足够）
                # 回退：放到负载最小的 trap（允许轻微超载，但此处保持合规）
                dest_trap = min(range(num_traps), key=lambda i: loads[i])
            move_qubit(victim, dest_trap)
            return victim

        for g in gate_list:
            if g not in self.parse_obj.cx_gate_map:
                continue
            q1, q2 = self.parse_obj.cx_gate_map[g]

            # 更新时间
            time += 1
            last_used[q1] = time
            last_used[q2] = time

            t1, t2 = mapping[q1], mapping[q2]
            if t1 == t2:
                continue  # 已在同一 trap，可直接执行

            # --- 尝试将 q2 移动到 t1 ---
            if loads[t1] < cap:
                move_qubit(q2, t1)
                continue

            # --- 尝试将 q1 移动到 t2 ---
            if loads[t2] < cap:
                move_qubit(q1, t2)
                continue

            # --- 两个 trap 都满，需要驱逐 ---
            # 策略：从 t1 中驱逐一个 LRU qubit 到任意有空位的 trap，然后将 q2 移入 t1
            free_traps = [i for i in range(num_traps) if loads[i] < cap]
            if free_traps:
                # 从 t1 驱逐（evict 内部会使用最近空闲 trap）
                evict_lru_from_trap(t1)
                # 现在 t1 有了空位，将 q2 移入
                move_qubit(q2, t1)
            else:
                # 理论上不应进入，但若发生，尝试从 t2 驱逐
                evict_lru_from_trap(t2)
                move_qubit(q1, t2)

        return mapping

    def compute_mapping(self):
        # 1. 准备门序列
        try:
            forward_gates = list(nx.topological_sort(self.parse_obj.gate_graph))
        except:
            forward_gates = list(self.parse_obj.gate_graph.nodes)

        backward_gates = forward_gates[::-1]

        # 2. 从 trivial mapping 开始
        trivial_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        seed = trivial_mapper.compute_mapping()

        # 3. 正向 pass
        mapping_fwd = self._simulate_pass(forward_gates, seed)

        # 4. 反向 pass，以 mapping_fwd 为初始映射
        mapping_bwd = self._simulate_pass(backward_gates, mapping_fwd)

        # 5. 返回反向最终映射作为初始映射（符合论文描述）
        return mapping_bwd

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

#严格实现sabre版本。
class QubitMapSABRE8:
    """
    MUSS-TI paper-style SABRE initial mapper, implemented with true SABRE principles.

    Key properties:
      1. Build a 2Q-only DAG for mapping
      2. Start from trivial mapping
      3. Forward pass on original 2Q DAG
      4. Backward pass on reversed 2Q DAG
      5. Return the final layout after backward pass

    Inside each pass:
      - Execute all currently local/executable 2Q gates first
      - If none executable, choose one relocation action using SABRE-style heuristic:
            H = avg(front_layer_distance) + W * avg(extended_set_distance)
      - Respect trap capacity strictly
      - Resolve full-target conflict by LRU replacement
    """

    def __init__(self, parse_obj, machine_obj, excess_capacity=0,
                 extended_set_size=20, extended_weight=0.5):
        self.parse_obj = parse_obj
        self.machine_obj = machine_obj
        self.excess_capacity = excess_capacity
        self.extended_set_size = extended_set_size
        self.extended_weight = extended_weight

        if not hasattr(self.machine_obj, "dist_cache") or not self.machine_obj.dist_cache:
            self.machine_obj.precompute_distances()

    # -----------------------------
    # Basic helpers
    # -----------------------------
    def _effective_capacity(self, trap_id):
        return self.machine_obj.traps[trap_id].capacity - self.excess_capacity

    def _all_trap_ids(self):
        return list(range(len(self.machine_obj.traps)))

    def _trap_distance(self, t1, t2):
        if t1 == t2:
            return 0
        return self.machine_obj.dist_cache.get((t1, t2), float("inf"))

    def _gate_qubits(self, gate_id):
        return tuple(self.parse_obj.cx_gate_map[gate_id])

    def _is_two_qubit_gate(self, gate_id):
        return gate_id in self.parse_obj.cx_gate_map

    def _load_layout(self, mapping):
        loads = {t: 0 for t in self._all_trap_ids()}
        trap_to_qubits = {t: [] for t in self._all_trap_ids()}
        for q, t in mapping.items():
            loads[t] += 1
            trap_to_qubits[t].append(q)
        return loads, trap_to_qubits

    def _build_two_qubit_dag(self):
        """
        Build a 2Q-only DAG by taking the induced subgraph of gate_graph on cx_gate_map keys.
        """
        dag = self.parse_obj.gate_graph.subgraph(list(self.parse_obj.cx_gate_map.keys())).copy()
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("2Q induced gate graph is not a DAG.")
        return dag

    def _build_reversed_two_qubit_dag(self, dag_2q):
        """
        True reversed DAG, not just reversed topo list.
        """
        rev = dag_2q.reverse(copy=True)
        if not nx.is_directed_acyclic_graph(rev):
            raise ValueError("Reversed 2Q DAG is not a DAG.")
        return rev

    def _trivial_seed_mapping(self):
        """
        Paper-style seed entry point.
        Reuse current trivial mapper interface.
        """
        trivial_mapper = QubitMapTrivial(self.parse_obj, self.machine_obj, self.excess_capacity)
        return trivial_mapper.compute_mapping()

    # -----------------------------
    # SABRE scoring
    # -----------------------------
    def _gate_distance_under_layout(self, gate_id, layout):
        q1, q2 = self._gate_qubits(gate_id)
        t1 = layout[q1]
        t2 = layout[q2]
        return self._trap_distance(t1, t2)

    def _front_layer(self, dag, indegree_map, done_set):
        """
        Nodes with indegree 0 and not yet executed.
        """
        out = []
        for g in dag.nodes:
            if g not in done_set and indegree_map[g] == 0:
                out.append(g)
        return out

    def _extended_set(self, dag, front, done_set, indegree_map):
        """
        Collect a bounded look-ahead set behind the front layer, using BFS over successors.
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
        SABRE-style heuristic:
            H = average distance(front) + W * average distance(ext)
        """
        if not front:
            return 0.0

        front_cost = 0.0
        for g in front:
            front_cost += self._gate_distance_under_layout(g, layout)
        front_cost /= float(len(front))

        if not ext:
            return front_cost

        ext_cost = 0.0
        for g in ext:
            ext_cost += self._gate_distance_under_layout(g, layout)
        ext_cost /= float(len(ext))

        return front_cost + self.extended_weight * ext_cost

    # -----------------------------
    # LRU-based relocation
    # -----------------------------
    def _select_lru_victim(self, target_trap, trap_to_qubits, last_used, forbidden=None):
        """
        Pick the least recently used qubit in target_trap, excluding any forbidden qubits.
        """
        if forbidden is None:
            forbidden = set()
        candidates = [q for q in trap_to_qubits[target_trap] if q not in forbidden]
        if not candidates:
            return None
        return min(candidates, key=lambda q: last_used.get(q, -1))

    def _apply_relocation(self, layout, loads, trap_to_qubits, last_used,
                          moving_qubit, target_trap, partner_qubit=None):
        """
        Relocate moving_qubit to target_trap while respecting strict capacity.

        If target_trap has room:
            move directly.

        If target_trap is full:
            evict one LRU victim from target_trap to the source_trap of moving_qubit.
            This is capacity-safe because moving_qubit leaves source_trap first.

        Returns:
            new_layout, new_loads, new_trap_to_qubits
        """
        new_layout = dict(layout)
        new_loads = dict(loads)
        new_trap_to_qubits = {t: list(qs) for t, qs in trap_to_qubits.items()}

        src_trap = new_layout[moving_qubit]
        if src_trap == target_trap:
            return new_layout, new_loads, new_trap_to_qubits

        # direct move if space exists
        if new_loads[target_trap] < self._effective_capacity(target_trap):
            new_trap_to_qubits[src_trap].remove(moving_qubit)
            new_trap_to_qubits[target_trap].append(moving_qubit)
            new_loads[src_trap] -= 1
            new_loads[target_trap] += 1
            new_layout[moving_qubit] = target_trap
            return new_layout, new_loads, new_trap_to_qubits

        # full target -> LRU replacement
        forbidden = {partner_qubit} if partner_qubit is not None else set()
        victim = self._select_lru_victim(target_trap, new_trap_to_qubits, last_used, forbidden=forbidden)
        if victim is None:
            # if partner is the only qubit or all are forbidden, relocation is impossible
            return None, None, None

        # Step 1: victim target_trap -> src_trap
        new_trap_to_qubits[target_trap].remove(victim)
        new_trap_to_qubits[src_trap].append(victim)
        new_layout[victim] = src_trap

        # Step 2: moving_qubit src_trap -> target_trap
        new_trap_to_qubits[src_trap].remove(moving_qubit)
        new_trap_to_qubits[target_trap].append(moving_qubit)
        new_layout[moving_qubit] = target_trap

        # loads unchanged overall under replacement
        return new_layout, new_loads, new_trap_to_qubits

    def _generate_candidate_actions(self, front, layout, loads, trap_to_qubits, last_used):
        """
        Generate SABRE candidate actions from the front layer.

        For every non-local front gate (q1,q2), generate:
            - move q1 to trap(q2)
            - move q2 to trap(q1)

        Each candidate is materialized as a resulting temporary layout.
        """
        candidates = []

        for g in front:
            q1, q2 = self._gate_qubits(g)
            t1 = layout[q1]
            t2 = layout[q2]
            if t1 == t2:
                continue

            # candidate: move q1 -> t2
            cand_layout, cand_loads, cand_trap_to_qubits = self._apply_relocation(
                layout, loads, trap_to_qubits, last_used,
                moving_qubit=q1, target_trap=t2, partner_qubit=q2
            )
            if cand_layout is not None:
                candidates.append((g, q1, t2, cand_layout, cand_loads, cand_trap_to_qubits))

            # candidate: move q2 -> t1
            cand_layout, cand_loads, cand_trap_to_qubits = self._apply_relocation(
                layout, loads, trap_to_qubits, last_used,
                moving_qubit=q2, target_trap=t1, partner_qubit=q1
            )
            if cand_layout is not None:
                candidates.append((g, q2, t1, cand_layout, cand_loads, cand_trap_to_qubits))

        return candidates

    # -----------------------------
    # Pass execution
    # -----------------------------
    def _simulate_pass(self, dag, initial_mapping):
        """
        True SABRE-style pass:
          - maintain front layer from DAG
          - execute all local gates immediately
          - otherwise choose one best relocation by SABRE heuristic
          - continue until DAG exhausted

        Returns the final layout after this pass.
        """
        layout = dict(initial_mapping)
        loads, trap_to_qubits = self._load_layout(layout)

        indegree_map = {g: dag.in_degree(g) for g in dag.nodes}
        done_set = set()

        # initialize LRU timestamps for all qubits in layout
        last_used = {q: -1 for q in layout.keys()}
        timestamp = 0

        while len(done_set) < dag.number_of_nodes():
            front = self._front_layer(dag, indegree_map, done_set)

            # 1) execute all currently local gates
            executable = []
            for g in front:
                q1, q2 = self._gate_qubits(g)
                if layout[q1] == layout[q2]:
                    executable.append(g)

            if executable:
                # execute all local gates in current front
                for g in executable:
                    q1, q2 = self._gate_qubits(g)
                    timestamp += 1
                    last_used[q1] = timestamp
                    last_used[q2] = timestamp

                    done_set.add(g)
                    for succ in dag.successors(g):
                        indegree_map[succ] -= 1
                continue

            # 2) no executable gate -> choose best SABRE relocation
            ext = self._extended_set(dag, front, done_set, indegree_map)
            candidates = self._generate_candidate_actions(front, layout, loads, trap_to_qubits, last_used)

            if not candidates:
                raise RuntimeError("SABRE2: no executable gate and no valid relocation candidate.")

            best_score = float("inf")
            best_payload = None

            for (_gate_id, moving_q, target_t, cand_layout, cand_loads, cand_trap_to_qubits) in candidates:
                score = self._heuristic_cost(cand_layout, front, ext)

                # deterministic tie-break:
                # 1. smaller score
                # 2. smaller moving qubit id
                # 3. smaller target trap id
                key = (score, moving_q, target_t)

                if best_payload is None:
                    best_score = score
                    best_payload = (_gate_id, moving_q, target_t, cand_layout, cand_loads, cand_trap_to_qubits, key)
                else:
                    if key < best_payload[-1]:
                        best_score = score
                        best_payload = (_gate_id, moving_q, target_t, cand_layout, cand_loads, cand_trap_to_qubits, key)

            _, moved_q, _, layout, loads, trap_to_qubits, _ = best_payload
            timestamp += 1
            last_used[moved_q] = timestamp

        return layout

    # -----------------------------
    # Public API
    # -----------------------------
    def compute_mapping(self):
        """
        Paper-style:
          1. trivial seed π
          2. forward pass on G -> π'
          3. backward pass on reversed G -> π''
          4. return π''
        """
        dag_2q = self._build_two_qubit_dag()
        dag_2q_rev = self._build_reversed_two_qubit_dag(dag_2q)

        seed_mapping = self._trivial_seed_mapping()
        mapping_after_forward = self._simulate_pass(dag_2q, seed_mapping)
        mapping_after_backward = self._simulate_pass(dag_2q_rev, mapping_after_forward)

        return mapping_after_backward
