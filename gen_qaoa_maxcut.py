import collections

edge_list = [(47, 54), (47, 40), (47, 38), (54, 33), (54, 50), (39, 59), (39, 8), (39, 13),
             (59, 24), (59, 41), (7, 25), (7, 18), (7, 45), (25, 16), (25, 3), (37, 48),
             (37, 44), (37, 5), (48, 36), (48, 51), (9, 21), (9, 58), (9, 53), (21, 30),
             (21, 60), (44, 56), (44, 14), (56, 38), (56, 11), (16, 61), (16, 36), (2, 6),
             (2, 61), (2, 58), (6, 23), (6, 22), (13, 35), (13, 14), (35, 49), (35, 63),
             (29, 30), (29, 10), (29, 57), (30, 10), (23, 53), (23, 28), (53, 12), (38, 1),
             (28, 49), (28, 43), (11, 41), (11, 32), (41, 5), (36, 55), (55, 52), (55, 46),
             (1, 62), (1, 27), (10, 12), (5, 42), (24, 17), (24, 45), (4, 31), (4, 18),
             (4, 34), (31, 20), (31, 32), (33, 40), (33, 17), (40, 52), (17, 19), (49, 3),
             (18, 57), (57, 52), (20, 22), (20, 34), (0, 8), (0, 50), (0, 26), (8, 27),
             (46, 63), (46, 19), (63, 45), (22, 43), (43, 32), (3, 15), (15, 42), (15, 60),
             (26, 61), (26, 19), (51, 58), (51, 14), (50, 42), (60, 12), (62, 27), (62, 34)]

# 选 32 个节点
deg = collections.Counter()
for u, v in edge_list:
    deg[u] += 1
    deg[v] += 1

selected_nodes = [node for node, _ in deg.most_common(32)]
selected_set = set(selected_nodes)

sub_edges = []
for u, v in edge_list:
    if u in selected_set and v in selected_set:
        sub_edges.append((u, v))

selected_nodes_sorted = sorted(selected_nodes)
remap = {old: new for new, old in enumerate(selected_nodes_sorted)}
sub_edges_remapped = [(remap[u], remap[v]) for u, v in sub_edges]

# QAOA parameters
p = 1
gamma = 0.6
beta = 0.4

with open("QAOA32_maxcut_true.qasm", "w", encoding="utf-8") as f:
    f.write('OPENQASM 2.0;\n')
    f.write('include "qelib1.inc";\n')
    f.write('qreg q[32];\n')

    # initial H layer
    for q in range(32):
        f.write(f"h q[{q}];\n")

    # p layers
    for _ in range(p):
        # cost layer on graph edges
        for u, v in sub_edges_remapped:
            f.write(f"cx q[{u}],q[{v}];\n")
            f.write(f"rz({2*gamma}) q[{v}];\n")
            f.write(f"cx q[{u}],q[{v}];\n")

        # mixer layer
        for q in range(32):
            f.write(f"rx({2*beta}) q[{q}];\n")

print("32-node subgraph edge count:", len(sub_edges_remapped))
