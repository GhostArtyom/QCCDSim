import subprocess as sp

PATH = "./programs"

# 选择电路
PROG = ["BV32", "GHZ32"]

# 选择架构
MACHINE = {"G2x2": "12", "G2x3": "8"}
# 可以设置多个架构，格式如下
# MACHINE = ["G2x3", "L6", "H6"]

# 可以设置多个不同容量的离子阱列表，进行对比测试，格式如下
# for i in range(14, 35, 2):
#     IONS.append(str(i))
# print(IONS)

# 选择映射方式，mapper (映射策略 - 决定离子去哪个阱)，可供的选择有：Greedy, PO, LPFS, Agg, Random, Trivial, SABRE
mapper = "SABRE"
# "Greedy":  贪心算法。优先聚合高频交互对。通常效果很好且稳定。
# "PO":      图分割优化 (Placement Optimization)。适合大电路最小化跨阱通信。（文献55推荐的映射方式）
# "LPFS":    最长路径优先。优化关键路径延迟。
# "Agg":     层次聚类。将紧密子图聚类到同一个阱。
# "Random":  随机映射。用于验证算法有效性的下界。
# "Trivial": MUSS-TI，不包括SABRE。
# "SABRE":   SABRE映射。

# 选择重排方式
reorder = "Naive"  # 根据run中设置，当mapper的选择为Greedy时，reorder无效。
# "Naive":    不优化顺序。直接按分配顺序排列。（MUSS论文似乎没有更多的要求）
# "Fidelity": 优化阱内顺序。让交互多的离子在链上相邻，减少局部SWAP。（文献55推荐的映射方式）

for prog in PROG:  # 遍历所有电路
    for machine, ions in MACHINE.items():  # 遍历所有架构和对应的离子数
        output_file = open(f"./output/{prog}_{machine}_{ions}_{mapper}.log", "w")
        sp.call(["python", "run.py", f"{PATH}/{prog}.qasm", machine, ions, mapper, reorder, "1", "0", "0", "FM", "GateSwap"], stdout=output_file)
