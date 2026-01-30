import sys
from parse import InputParse
from mappers import *
from machine import Machine, MachineParams, Trap, Segment
from ejf_schedule import Schedule, EJFSchedule
from analyzer import *
from test_machines import *
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from muss_schedule2 import MUSSSchedule  # 导入新调度器

np.random.seed(12345)

# Command line args
# Machine attributes
openqasm_file_name = sys.argv[1]
machine_type = sys.argv[2]
num_ions_per_region = int(sys.argv[3])
mapper_choice = sys.argv[4]
reorder_choice = sys.argv[5]

serial_trap_ops = int(sys.argv[6])  # 1 = 开启离子阱内操作串行，0 = 并行(如果有多个激光束)
serial_comm = int(sys.argv[7])  # 1 = 通信串行化 (全芯片同一时间只能有一组离子在移动)，0 = 通信并行化 (允许不同区域同时移动离子)
serial_all = int(sys.argv[8])  # 门和移动操作全串行化，1 = 全串行化 (同一时间只能有一个操作在进行)，0 = 非全串行化 (允许不同类型的操作并行进行)
gate_type = sys.argv[9]  # 选择门类型，如PM、FM
swap_type = sys.argv[10]  # 定义如何交换离子顺序，如GateSwap、IonSwap

# sp.call(["python", "run.py", p, m, i, mapper, reorder, "1", "0", "0", "PM", "GateSwap"], stdout=output_file)
##########################################################
"""
mpar_model1 = MachineParams()
mpar_model1.alpha = 0.003680029  # 加热率
mpar_model1.beta = 39.996319971  # 量子比特寿命
mpar_model1.split_merge_time = 80  # 离子分裂和合并时间
mpar_model1.shuttle_time = 5  # 离子移动时间
mpar_model1.junction2_cross_time = 5  # 二维交叉口穿越时间
mpar_model1.junction3_cross_time = 100  # 三叉交叉口穿越时间
mpar_model1.junction4_cross_time = 120  # 四叉交叉口穿越时间
mpar_model1.gate_type = gate_type  # 选择门类型
mpar_model1.swap_type = swap_type  # 选择交换类型
mpar_model1.ion_swap_time = 42  # 离子交换时间
machine_model = "MPar1"
"""
"""
mpar_model2 = MachineParams()
mpar_model2.alpha = 0.003680029
mpar_model2.beta = 39.996319971
mpar_model2.split_merge_time = 80
mpar_model2.shuttle_time = 5
mpar_model2.junction2_cross_time = 5
mpar_model2.junction3_cross_time = 100
mpar_model2.junction4_cross_time = 120
mpar_model2.alpha
machine_model = "MPar2"
"""

mpar_model1 = MachineParams()
# === MUSS-TI Table 1 Parameters ===
mpar_model1.split_merge_time = 80  # Split/Merge 都是 80
mpar_model1.shuttle_time = 5  # Move: 假设 seg长10um / 2um/us = 5us
mpar_model1.ion_swap_time = 40  # Swap: 40us
mpar_model1.junction2_cross_time = 5  # 假设过结点也是 5us
mpar_model1.junction3_cross_time = 5
mpar_model1.junction4_cross_time = 5

# 保真度相关常数 (用于 analyzer)
mpar_model1.T1 = 600e6  # 600 seconds
mpar_model1.k_heating = 0.001
mpar_model1.epsilon = 1.0 / 25600.0  # Gate error coeff

# 其他设置
mpar_model1.gate_type = gate_type  # 将会被我们在 machine.py 里覆盖
mpar_model1.swap_type = swap_type
machine_model = "MUSS_Params"


print("Simulation")
print("Program:", openqasm_file_name)
print("Machine:", machine_type)
print("Model:", machine_model)
print("Ions:", num_ions_per_region)
print("Mapper:", mapper_choice)
print("Reorder:", reorder_choice)
print("SerialTrap:", serial_trap_ops)
print("SerialComm:", serial_comm)
print("SerialAll:", serial_all)
print("Gatetype:", gate_type)
print("Swaptype:", swap_type)

# Create a test machine
if machine_type == "G2x3":
    m = test_trap_2x3(num_ions_per_region, mpar_model1)
elif machine_type == "G2x2":  # <--- 新增这行
    m = test_trap_2x2(num_ions_per_region, mpar_model1)
elif machine_type == "L6":
    m = make_linear_machine(6, num_ions_per_region, mpar_model1)
elif machine_type == "H6":
    m = make_single_hexagon_machine(num_ions_per_region, mpar_model1)
else:
    assert 0

# Parse the input program DAG
ip = InputParse()
ip.parse_ir(openqasm_file_name)
ip.visualize_graph("visualize_graph_2.gexf")  # dumps parser graph into file

qc = QuantumCircuit.from_qasm_file(openqasm_file_name)
dag = circuit_to_dag(qc)
# dag_drawer(dag, filename=f"{openqasm_file_name[:-5]}.svg")

print("parse object map:")
print(ip.cx_gate_map)
print("parse object graph:")
print(ip.gate_graph)

# Map the program onto the machine regions
# For every program qubit, this gives a region id
if mapper_choice == "LPFS":  # LPFS映射的具体方式 见mappers.py中的QubitMapLPFS类，下面类似
    qm = QubitMapLPFS(ip, m)
elif mapper_choice == "Agg":
    qm = QubitMapAgg(ip, m)
elif mapper_choice == "Random":
    qm = QubitMapRandom(ip, m)
elif mapper_choice == "PO":
    qm = QubitMapPO(ip, m)
elif mapper_choice == "Greedy":
    qm = QubitMapGreedy(ip, m)
# Trival
elif mapper_choice == "Trivial":
    qm = QubitMapTrivial(ip, m)  # 使用新 Mapper
# === 新增 SABRE 分支 ===
elif mapper_choice == "SABRE":
    qm = QubitMapSABRE3(ip, m)
else:
    assert 0
mapping = qm.compute_mapping()

# Reorder qubits within a region to increse the use of high fidelity operations
if mapper_choice == "Greedy":
    init_qubit_layout = mapping
else:
    qo = QubitOrdering(ip, m, mapping)
    if reorder_choice == "Naive":
        init_qubit_layout = qo.reorder_naive()
    elif reorder_choice == "Fidelity":
        init_qubit_layout = qo.reorder_fidelity()
    else:
        assert 0

print(init_qubit_layout)

# Schedule gates in the prorgam in topological sorted order
# EJF = earliest job first, here it refers to earliest gate first
# This step performs the shuttling
if machine_type == "MUSS_TI_Mode" or mapper_choice in ["Trivial", "SABRE"]:
    print(f"Using MUSS-TI Scheduler with {mapper_choice} Mapping")
    # 创建调度器
    scheduler = MUSSSchedule(ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout, serial_trap_ops, serial_comm, serial_all)
    # 运行调度
    scheduler.run()

    # 【关键修改】：这里必须传 scheduler 实例，不要传 scheduler.schedule
    analyzer = Analyzer(scheduler, m, init_qubit_layout)

    # 运行检查
    analyzer.move_check()

    print("SplitSWAP:", scheduler.split_swap_counter)

else:
    # 兼容旧的 EJF 调度器
    ejfs = EJFSchedule(ip.gate_graph, ip.all_gate_map, m, init_qubit_layout, serial_trap_ops, serial_comm, serial_all)
    ejfs.run()

    # 如果 EJFSchedule 也有 sys_state，可以类似传递，或者 Analyzer 需要做兼容
    # 假设你主要跑 MUSS_TI 模式，上面那个 if 块对了就行
    analyzer = Analyzer(ejfs, m, init_qubit_layout)
    analyzer.move_check()

# print("SplitSWAP:", scheduler.split_swap_counter)
# analyzer.print_events()
print("----------------")
