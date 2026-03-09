import sys
from parse import InputParse
from mappers import *
from machine import Machine, MachineParams, Trap, Segment
from ejf_schedule import Schedule, EJFSchedule
from analyzer import Analyzer, AnalyzerKnobs
from test_machines import *
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer

# 导入三个版本的 MUSS 调度器
from muss_schedule2 import MUSSSchedule as MUSSScheduleV2   # 论文版本调度
from muss_schedule3 import MUSSSchedule as MUSSScheduleV3   # 创新版本调度
from muss_schedule4 import MUSSSchedule as MUSSScheduleV4   # 创新版本 + 降低加热版本

np.random.seed(12345)


# ============================================================
# Helper: 命令行参数读取
# ============================================================
def get_arg(idx, default=None):
    return sys.argv[idx] if len(sys.argv) > idx else default


def has_shuttle_id_annotations(scheduler) -> bool:
    """
    检查 scheduler.schedule.events 里是否存在带 shuttle_id 的 Split/Move/Merge 事件。
    若存在，则说明 analyzer 可以安全使用 aggregate 模式。
    若不存在，则回退 per_event，兼容尚未修补 shuttle_id 的旧调度器（如 V3/V4）。
    """
    if not hasattr(scheduler, "schedule"):
        return False
    if not hasattr(scheduler.schedule, "events"):
        return False

    for ev in scheduler.schedule.events:
        try:
            etype = ev[1]
            info = ev[4]
            if etype in [Schedule.Split, Schedule.Move, Schedule.Merge]:
                if isinstance(info, dict) and ("shuttle_id" in info):
                    return True
        except Exception:
            continue
    return False


# ============================================================
# Command line args
# ============================================================
if len(sys.argv) < 11:
    print("用法:")
    print("python run.py <qasm> <machine_type> <ions_per_region> <mapper> <reorder> "
          "<serial_trap_ops> <serial_comm> <serial_all> <gate_type> <swap_type> "
          "[sched_family] [sched_version]")
    print("")
    print("示例:")
    print("python run.py ghz32.qasm G2x2 12 SABRE Fidelity 1 1 0 FM PaperSwapDirect MUSS V2")
    sys.exit(1)

openqasm_file_name = sys.argv[1]
machine_type = sys.argv[2]
num_ions_per_region = int(sys.argv[3])
mapper_choice = sys.argv[4]
reorder_choice = sys.argv[5]

serial_trap_ops = int(sys.argv[6])   # 阱内操作是否串行
serial_comm = int(sys.argv[7])       # 通信是否串行
serial_all = int(sys.argv[8])        # 所有操作是否全串行

gate_type = sys.argv[9]              # PM / FM / Duan / Trout
swap_type = sys.argv[10]             # PaperSwapDirect / GateSwap / IonSwap

# 新增：调度器家族和版本（可缺省）
sched_family = get_arg(11, "MUSS").upper()
sched_version = get_arg(12, "V2").upper()   # 不锁死 V2，保留你原来的默认习惯


# ============================================================
# Machine 参数（MUSS-TI Table 1 + 可调扩展参数）
# ============================================================
mpar = MachineParams()

# ---- Table 1 核心时间参数 ----
mpar.split_merge_time = 80
mpar.shuttle_time = 5
mpar.ion_swap_time = 40
mpar.junction2_cross_time = 5
mpar.junction3_cross_time = 5
mpar.junction4_cross_time = 5
mpar.move_speed_um_per_us = 2.0       # Table 1: Move speed 2 μm/us

# ---- 这些是实现/拟合相关参数 ----
mpar.segment_length_um = 53.0         # 默认段长；后续可按机器类型再调
mpar.inter_ion_spacing_um = 1.0       # gate_time 距离项使用
mpar.alpha_bg = 0.0                   # 论文对准时通常先关掉背景 Bi

# ---- Analyzer 会读到的物理参数（兼容保留）----
mpar.T1 = 600e6                       # us
mpar.k_heating = 0.001
mpar.epsilon = 1.0 / 25600.0

# ---- 命令行控制 ----
mpar.gate_type = gate_type
mpar.swap_type = swap_type

machine_model = "MUSS_Params"


# ============================================================
# 打印基本信息
# ============================================================
print("Simulation")
print("Program:          ", openqasm_file_name)
print("Machine:          ", machine_type)
print("Model:            ", machine_model)
print("Ions/Region:      ", num_ions_per_region)
print("Mapper:           ", mapper_choice)
print("Reorder:          ", reorder_choice)
print("SerialTrap:       ", serial_trap_ops)
print("SerialComm:       ", serial_comm)
print("SerialAll:        ", serial_all)
print("GateType:         ", gate_type)
print("SwapType:         ", swap_type)
print("Scheduler Family: ", sched_family)
print("Scheduler Version:", sched_version)


# ============================================================
# 创建测试机器
# ============================================================
if machine_type == "G2x3":
    m = test_trap_2x3(num_ions_per_region, mpar)
elif machine_type == "G2x2":
    m = test_trap_2x2(num_ions_per_region, mpar)
elif machine_type == "L6":
    m = make_linear_machine(6, num_ions_per_region, mpar)
elif machine_type == "H6":
    m = make_single_hexagon_machine(num_ions_per_region, mpar)
else:
    print(f"不支持的机器类型: {machine_type}")
    sys.exit(1)


# ============================================================
# 解析电路
# ============================================================
ip = InputParse()
ip.parse_ir(openqasm_file_name)
ip.visualize_graph("visualize_graph_2.gexf")

qc = QuantumCircuit.from_qasm_file(openqasm_file_name)
dag = circuit_to_dag(qc)
# dag_drawer(dag, filename=f"{openqasm_file_name[:-5]}.svg")   # 可选输出

print("parse object map:")
print(ip.cx_gate_map)
print("parse object graph:")
print(ip.gate_graph)


# ============================================================
# 初始映射
# ============================================================
if mapper_choice == "LPFS":
    qm = QubitMapLPFS(ip, m)
elif mapper_choice == "Agg":
    qm = QubitMapAgg(ip, m)
elif mapper_choice == "Random":
    qm = QubitMapRandom(ip, m)
elif mapper_choice == "PO":
    qm = QubitMapPO(ip, m)
elif mapper_choice == "Greedy":
    qm = QubitMapGreedy(ip, m)
elif mapper_choice == "Trivial":
    qm = QubitMapTrivial(ip, m)
elif mapper_choice == "SABRE":
    # 根据调度器版本选择对应 SABRE 变体
    if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
        if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
            print("→ Using SABRE2 mapper (matches muss_schedule2 paper version)")
            qm = QubitMapSABRE3(ip, m)
        elif sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
            print("→ Using SABRE6 mapper (matches muss_schedule3 improved version)")
            qm = QubitMapSABRE6(ip, m)
        elif sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
            print("→ Using SABRE6 mapper (matches muss_schedule4 improved version)")
            qm = QubitMapSABRE6(ip, m)
        else:
            print(f"Warning: Unknown scheduler version '{sched_version}', fallback to SABRE3")
            qm = QubitMapSABRE3(ip, m)
    else:
        print("→ Using default SABRE3 mapper (non-MUSS scheduler)")
        qm = QubitMapSABRE3(ip, m)
else:
    print(f"Error: Unsupported mapper choice '{mapper_choice}'")
    sys.exit(1)

mapping = qm.compute_mapping()


# ============================================================
# 阱内重排序
# ============================================================
if mapper_choice == "Greedy":
    init_qubit_layout = mapping
else:
    qo = QubitOrdering(ip, m, mapping)
    if reorder_choice == "Naive":
        init_qubit_layout = qo.reorder_naive()
    elif reorder_choice == "Fidelity":
        init_qubit_layout = qo.reorder_fidelity()
    else:
        print(f"不支持的 reorder: {reorder_choice}")
        sys.exit(1)

print("init_qubit_layout:")
print(init_qubit_layout)


# ============================================================
# 调度阶段
# ============================================================
print(f"Using {sched_family} Scheduler ({sched_version}) with {mapper_choice} Mapping")

if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
    if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
        print("→ muss_schedule2.py old_vision")
        scheduler = MUSSScheduleV2(
            ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
        print("→ muss_schedule3.py new_vision")
        scheduler = MUSSScheduleV3(
            ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
        print("→ muss_schedule4.py new_vision")
        scheduler = MUSSScheduleV4(
            ip.gate_graph, ip.cx_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    else:
        print(f"error：不支持的调度器版本 {sched_version}，支持：V2 / V3 / V4")
        sys.exit(1)
else:
    print("→ 回退使用 EJF 调度器")
    scheduler = EJFSchedule(
        ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
        serial_trap_ops, serial_comm, serial_all
    )

scheduler.run()


# ============================================================
# Analyzer 配置
# 核心兼容逻辑：
#   - 若 schedule 事件里带 shuttle_id，则用 aggregate（新 analyzer 论文对准模式）
#   - 否则自动回退 per_event（兼容还没补 shuttle_id 的 V3/V4/旧版本）
# ============================================================
use_aggregate = has_shuttle_id_annotations(scheduler)

if use_aggregate:
    print("Analyzer mode: aggregate shuttle fidelity (detected shuttle_id annotations)")
else:
    print("Analyzer mode: per_event shuttle fidelity (no shuttle_id detected; compatibility fallback)")

knobs = AnalyzerKnobs(
    bg_model="exp",
    alpha_bg=0.001,
    inject_norm="none",
    swap_norm="none",
    move_heat_use_distance=True,
    move_heat_const=0.1,
    move_bg_fraction=0.175,
    gate_env_time_mode="duration",
    gate_use_env=False, 
    gate_use_bg=True,
    shuttle_fidelity_mode="aggregate",
    merge_equalize=True,
    debug_events=False,
    debug_summary=True,
)


analyzer = Analyzer(scheduler, m, init_qubit_layout, knobs)
result = analyzer.analyze_and_return()


# ============================================================
# 输出
# ============================================================
print("\n========== ANALYSIS RESULT ==========")
print(result)

print("\n========== SHUTTLE TRACE ==========")
if hasattr(scheduler, "dump_shuttle_trace"):
    print(scheduler.dump_shuttle_trace())
else:
    print("(scheduler does not support dump_shuttle_trace)")

if hasattr(scheduler, "split_swap_counter"):
    print("SplitSWAP:", scheduler.split_swap_counter)

if hasattr(scheduler, "shuttle_counter"):
    print("TotalShuttle:", scheduler.shuttle_counter)

print("----------------")
