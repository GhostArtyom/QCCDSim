import subprocess as sp
import os

PATH = "./programs"

# 选择电路
PROG = ["BV32", "GHZ32", "ADDER32", "QAOA32", "QFT32","SQRT30"]
#PROG = ["QFT32"]  # 只跑一个电路，便于调试和验证

# 选择架构与容量
MACHINE = {"G2x2": "12", "G2x3": "8"}

# 映射方式
mapper = "SABRE"
# "Greedy":  贪心算法。优先聚合高频交互对。通常效果很好且稳定。
# "PO":      图分割优化 (Placement Optimization)。适合大电路最小化跨阱通信。（文献55推荐的映射方式）
# "LPFS":    最长路径优先。优化关键路径延迟。
# "Agg":     层次聚类。将紧密子图聚类到同一个阱。
# "Random":  随机映射。用于验证算法有效性的下界。
# "Trivial": MUSS-TI，不包括SABRE。
# "SABRE":   SABRE映射。

# 重排序方式
reorder = "Naive"
# "Naive":    不优化顺序。直接按分配顺序排列。（MUSS论文似乎没有更多的要求）
# "Fidelity": 优化阱内顺序。让交互多的离子在链上相邻，减少局部SWAP。（文献55推荐的映射方式）

# 调度器组合（可自由增删）
SCHEDULERS = [
    ("MUSS", "V2"),   # 论文版本
    #("MUSS", "V3"),   # 创新版本
    #("MUSS", "V4"),   # 创新版本调度基础上的降低加热版本。
    #("EJF",  ""),     # 文献[55]EJF，也可以扩展
]

# [PATCH] 显式抽出 gate / swap 类型，便于统一切换与日志命名
gate_type = "FM"
swap_type = "PaperSwapDirect"

# [PATCH] 保证输出目录存在
os.makedirs("./output", exist_ok=True)

for prog in PROG:
    for machine, ions in MACHINE.items():
        for family, version in SCHEDULERS:
            # 日志文件名带上版本信息，便于区分
            log_name = f"./output/{prog}_{machine}_{ions}_{mapper}_{family}_{version}_{gate_type}_{swap_type}.log"
            
            args = [
                "python", "run.py",
                f"{PATH}/{prog}.qasm",
                machine,
                ions,
                mapper,
                reorder,
                "1", "1", "1",          # serial_trap, serial_comm, serial_all
                gate_type, swap_type
            ]
            
            # 只在需要 MUSS 家族时才传版本
            if family == "MUSS":
                args.extend([family, version])
            else:
                args.append(family)     # EJF 等只需要 family

            print(f"Running: {' '.join(args)} -> {log_name}")
            
            with open(log_name, "w", encoding="utf-8") as f:
                sp.call(args, stdout=f, stderr=sp.STDOUT)
