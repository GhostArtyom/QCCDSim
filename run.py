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
from muss_schedule2 import MUSSSchedule as MUSSScheduleV2

try:
    from muss_schedule3 import MUSSSchedule as MUSSScheduleV3
except Exception:
    MUSSScheduleV3 = None

try:
    from muss_schedule4 import MUSSSchedule as MUSSScheduleV4
except Exception:
    MUSSScheduleV4 = None

try:
    from muss_schedule5 import MUSSSchedule as MUSSScheduleV5
except Exception:
    MUSSScheduleV5 = None

try:
    from muss_schedule6 import MUSSSchedule as MUSSScheduleV6
except Exception:
    MUSSScheduleV6 = None

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


def should_run_qubit_ordering(mapper_choice: str) -> bool:
    """
    决定是否执行额外的 qubit ordering。
    设计原则：
      - 老 QCCD 项目附带的 mapper：保留历史行为，继续做 ordering
      - 新增的论文复现 mapper（Trivial / SABRE）：不再做 ordering
    """
    return mapper_choice not in ["Trivial", "SABRE"]


def is_trap_layout(mapping, machine_obj) -> bool:
    """
    判断 mapping 是否已经是 trap_id -> [ion_ids...] 这种布局格式。
    """
    if not isinstance(mapping, dict):
        return False

    trap_ids = set(t.id for t in machine_obj.traps)

    for k, v in mapping.items():
        if k not in trap_ids:
            return False
        if not isinstance(v, list):
            return False

    return True


def canonicalize_mapping_to_layout(mapping, machine_obj):
    """
    将 mapper 输出统一转换成 scheduler 可接受的 trap_id -> [ion_ids...] 格式。

    支持两类输入：
      1) trap_id -> [ion_ids...]      （例如 Greedy）
      2) qubit_id -> trap_id          （例如 Trivial / SABRE2 / LPFS / PO / Random / Agg）

    转换原则：
      - 如果已经是 trap->list，则只补齐空 trap，并复制列表
      - 如果是 qubit->trap，则按 mapping 的遍历顺序稳定地装入 trap
        （Python 3.7+ dict 保持插入顺序）
    """
    trap_ids = [t.id for t in machine_obj.traps]

    # 情况1：已经是 scheduler 需要的格式
    if is_trap_layout(mapping, machine_obj):
        output_layout = {}
        for tid in trap_ids:
            output_layout[tid] = list(mapping.get(tid, []))
        return output_layout

    # 情况2：认为它是 qubit -> trap
    output_layout = {tid: [] for tid in trap_ids}

    if not isinstance(mapping, dict):
        raise TypeError("Mapping must be a dict.")

    trap_id_set = set(trap_ids)
    for qubit_id, trap_id in mapping.items():
        if trap_id not in trap_id_set:
            raise RuntimeError(
                f"Invalid raw mapping: qubit {qubit_id} assigned to unknown trap {trap_id}."
            )
        output_layout[trap_id].append(qubit_id)

    return output_layout


def is_mapping_bundle(mapping, machine_obj) -> bool:
    """
    判断 mapper 输出是否为新的结构化 bundle：
      {
          "layout": q->trap,
          "trap_to_qubits": trap->ordered_qubit_list
      }
    """
    if not isinstance(mapping, dict):
        return False
    if "layout" not in mapping or "trap_to_qubits" not in mapping:
        return False
    if not isinstance(mapping["layout"], dict):
        return False
    if not is_trap_layout(mapping["trap_to_qubits"], machine_obj):
        return False
    return True


def extract_layout_bundle(mapping, machine_obj):
    """
    将 mapper 输出统一拆解成两部分：
      - layout_mapping: q->trap 或旧式原始输出
      - mapper_trap_layout: 若 mapper 已提供有序 trap 布局，则直接返回；否则为 None
    """
    if is_mapping_bundle(mapping, machine_obj):
        return mapping["layout"], canonicalize_mapping_to_layout(mapping["trap_to_qubits"], machine_obj)
    return mapping, None


def describe_layout_policy(mapper_choice: str, reorder_choice: str, raw_mapping=None, machine_obj=None) -> str:
    """
    用于日志打印，描述本次 initial layout 的生成策略。
    """
    if machine_obj is not None and is_mapping_bundle(raw_mapping, machine_obj):
        return "mapping_bundle(layout + trap_to_qubits_from_mapper)"
    if should_run_qubit_ordering(mapper_choice):
        return f"mapping + qubit_ordering({reorder_choice})"
    return "mapping_only + canonicalize_to_trap_layout"


# ============================================================
# Command line args
# ============================================================
if len(sys.argv) < 11:
    print("Usage:")
    print("python run.py <qasm> <machine_type> <ions_per_region> <mapper> <reorder> "
          "<serial_trap_ops> <serial_comm> <serial_all> <gate_type> <swap_type> "
          "[sched_family] [sched_version] [analyzer_mode] [architecture_scale]")
    print("")
    print("Example:")
    print("python run.py ghz32.qasm G2x2 12 SABRE Fidelity 1 1 0 FM PaperSwapDirect MUSS V2 PAPER SMALL")
    sys.exit(1)

openqasm_file_name = sys.argv[1]
machine_type = sys.argv[2]
num_ions_per_region = int(sys.argv[3])
mapper_choice = sys.argv[4]
reorder_choice = sys.argv[5]

serial_trap_ops = int(sys.argv[6])
serial_comm = int(sys.argv[7])
serial_all = int(sys.argv[8])

gate_type = sys.argv[9]
swap_type = sys.argv[10]

sched_family = get_arg(11, "MUSS").upper()
sched_version = get_arg(12, "V2").upper()
analyzer_mode = get_arg(13, "PAPER").upper()
architecture_scale = get_arg(14, None)


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
mpar.move_speed_um_per_us = 2.0

# ---- 这些是实现/拟合相关参数 ----
mpar.segment_length_um = 28.0
mpar.inter_ion_spacing_um = 1.0

# 论文复现默认值：
# 保留 B_i 接口，因此不给 0；若你后续要完全关闭 B_i，可显式改回 0.0
mpar.alpha_bg = 0.0

mpar.architecture_scale = "small"
mpar.enable_partition = False

# ---- Analyzer 会读到的物理参数（兼容保留）----
mpar.T1 = 600e6
mpar.k_heating = 0.001
mpar.epsilon = 1.0 / 25600.0

# ---- 命令行控制 ----
mpar.gate_type = gate_type
mpar.swap_type = swap_type

machine_model = "MUSS_Params"


# ============================================================
# Architecture scale: explicit small / large switch
# ============================================================
if architecture_scale is None:
    if machine_type in ["G2x2", "G2x3"]:
        architecture_scale = "SMALL"
    else:
        architecture_scale = "LARGE"
architecture_scale = architecture_scale.upper()

if architecture_scale in ["SMALL", "S", "TABLE2"]:
    mpar.architecture_scale = "small"
    mpar.enable_partition = False
elif architecture_scale in ["LARGE", "L", "EML", "EML-QCCD"]:
    mpar.architecture_scale = "large"
    mpar.enable_partition = True
else:
    print(f"Warning: unknown architecture_scale '{architecture_scale}', fallback SMALL")
    mpar.architecture_scale = "small"
    mpar.enable_partition = False


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
print("Analyzer Mode:    ", analyzer_mode)
print("Arch Scale:       ", architecture_scale)


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
    print(f"Error: Unsupported machine type '{machine_type}'")
    sys.exit(1)


# ============================================================
# 解析电路
# ============================================================
ip = InputParse()
ip.parse_ir(openqasm_file_name)
ip.visualize_graph("visualize_graph_2.gexf")

qc = QuantumCircuit.from_qasm_file(openqasm_file_name)
dag = circuit_to_dag(qc)  #后续均没用到qc，这是对比遗留
# dag_drawer(dag, filename=f"{openqasm_file_name[:-5]}.svg")

print("Parse object map:")
print(ip.cx_gate_map)
print("Parse object graph:")
print(ip.gate_graph)


# ============================================================
# 初始映射：选择 mapper
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
    if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
        if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
            print("Using QubitMapSABRE2 mapper (matches muss_schedule2 paper version)")
            qm = QubitMapSABRE2(ip, m)
        elif sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
            print("Using QubitMapSABRE6 mapper (matches muss_schedule3 improved version)")
            qm = QubitMapSABRE6(ip, m)
        elif sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
            print("Using SABRE6 mapper (matches muss_schedule4 improved version)")
            qm = QubitMapSABRE6(ip, m)
        elif sched_version in ["V5", "5", "MUSS_SCHEDULE5", "INNOV3"]:
            print("Using SABRE2 mapper (matches muss_schedule5 improved version)")
            qm = QubitMapSABRE2(ip, m)
        elif sched_version in ["V6", "6", "MUSS_SCHEDULE6", "INNOV4"]:
            print("Using QubitMapSABRE2 mapper (matches muss_schedule6 improved version)")
            qm = QubitMapSABRE2(ip, m)
        else:
            print(f"Warning: Unknown scheduler version '{sched_version}', fallback to SABRE2")
            qm = QubitMapSABRE2(ip, m)
    else:
        print("Using default SABRE2 mapper (non-MUSS scheduler)")
        qm = QubitMapSABRE2(ip, m)
else:
    print(f"Error: Unsupported mapper choice '{mapper_choice}'")
    sys.exit(1)

raw_mapping = qm.compute_mapping()

print("Raw mapping:")
print(raw_mapping)

layout_mapping, mapper_trap_layout = extract_layout_bundle(raw_mapping, m)

# ============================================================
# 初始布局生成：
#   - 老 mapper：mapping -> QubitOrdering -> trap_id -> [ion_ids]
#   - Trivial：跳过 ordering，但仍要 canonicalize 成 trap_id -> [ion_ids]
#   - 新 SABRE2：若 mapper 已直接返回有序 trap 布局，则直接使用该布局
# ============================================================
print(f"Initial layout policy: {describe_layout_policy(mapper_choice, reorder_choice, raw_mapping, m)}")

run_ordering = should_run_qubit_ordering(mapper_choice)

if mapper_trap_layout is not None:
    # 对新的结构化 mapper 输出，直接使用 mapper 内部形成的有序 trap 布局
    # 这里不再做额外 qubit ordering，也不再做 canonicalize 重建，以避免抹掉 SABRE 的 pass 结果。
    if reorder_choice not in [None, "", "None", "NONE", "Disabled", "DISABLED"]:
        print(f"Note: reorder_choice='{reorder_choice}' is ignored because mapper already provides ordered trap layout")

    print(f"Use mapper-provided ordered trap layout for mapper '{mapper_choice}'")
    init_qubit_layout = mapper_trap_layout

elif not run_ordering:
    # 对论文复现路径中的旧式输出，不做额外 ordering
    # 但仍需把 q->trap 形式转成 scheduler 所需的 trap->list 形式
    if reorder_choice not in [None, "", "None", "NONE", "Disabled", "DISABLED"]:
        print(f"Note: reorder_choice='{reorder_choice}' is ignored for mapper '{mapper_choice}'")

    print(f"Skip qubit ordering for mapper '{mapper_choice}'")
    init_qubit_layout = canonicalize_mapping_to_layout(layout_mapping, m)
else:
    # 对老 mapper，保留历史行为
    print(f"Apply qubit ordering for mapper '{mapper_choice}' with mode '{reorder_choice}'")

    # 这里仍然要求传给 QubitOrdering 的是 q->trap 格式
    # 如果某个老 mapper 已经直接返回了 trap->list（例如 Greedy），则无需再 ordering
    if is_trap_layout(layout_mapping, m):
        print("Mapper output is already trap-layout; skip ordering and use it directly")
        init_qubit_layout = canonicalize_mapping_to_layout(layout_mapping, m)
    else:
        qo = QubitOrdering(ip, m, layout_mapping)
        if reorder_choice == "Naive":
            init_qubit_layout = qo.reorder_naive()
        elif reorder_choice == "Fidelity":
            init_qubit_layout = qo.reorder_fidelity()
        else:
            print(f"Error: Unsupported reorder choice '{reorder_choice}'")
            sys.exit(1)

print("Initial qubit layout:")
print(init_qubit_layout)


# ============================================================
# 调度阶段
# ============================================================
print(f"Using {sched_family} Scheduler ({sched_version}) with {mapper_choice} Mapping")

if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
    if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
        print("Using muss_schedule2.py paper-faithful fixed version")
        scheduler = MUSSScheduleV2(
            ip, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
        if MUSSScheduleV3 is None:
            print("Error: muss_schedule3 is not available")
            sys.exit(1)
        print("Using muss_schedule3.py new_vision")
        scheduler = MUSSScheduleV3(
            ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
        if MUSSScheduleV4 is None:
            print("Error: muss_schedule4 is not available")
            sys.exit(1)
        print("Using muss_schedule4.py new_vision")
        scheduler = MUSSScheduleV4(
            ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V5", "45", "MUSS_SCHEDULE5", "INNOV2"]:
        if MUSSScheduleV5 is None:
            print("Error: muss_schedule5 is not available")
            sys.exit(1)
        print("Using muss_schedule5.py new_vision")
        scheduler = MUSSScheduleV5(
            ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    elif sched_version in ["V6", "6", "MUSS_SCHEDULE6", "INNOV4"]:
        if MUSSScheduleV6 is None:
            print("Error: muss_schedule6 is not available")
            sys.exit(1)
        print("Using muss_schedule6.py new_vision")
        scheduler = MUSSScheduleV6(
            ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
            serial_trap_ops, serial_comm, serial_all
        )
    else:
        print(f"Error: Unsupported scheduler version '{sched_version}', supported: V2 / V3 / V4 / V5 / V6")
        sys.exit(1)
else:
    print("Fallback to EJF scheduler")
    scheduler = EJFSchedule(
        ip.gate_graph, ip.all_gate_map, m, init_qubit_layout,
        serial_trap_ops, serial_comm, serial_all
    )

scheduler.run()


# ============================================================
# Analyzer 配置
#   PAPER: 更贴论文 Table 2 口径
#   EXTENDED: 保留原扩展热背景模型
# ============================================================
use_aggregate = has_shuttle_id_annotations(scheduler)

if use_aggregate:
    print("Analyzer shuttle mode: aggregate (detected shuttle_id annotations)")
else:
    print("Analyzer shuttle mode: per_event (no shuttle_id detected; compatibility fallback)")

shuttle_mode = ("aggregate" if use_aggregate else "per_event")
if analyzer_mode in ["PAPER", "TABLE2", "P"]:
    knobs = AnalyzerKnobs.paper_mode(
        shuttle_fidelity_mode=shuttle_mode,
        debug_summary=True
    )
elif analyzer_mode in ["EXTENDED", "EXP", "E"]:
    knobs = AnalyzerKnobs.extended_mode(
        shuttle_fidelity_mode=shuttle_mode,
        debug_summary=True
    )
else:
    print(f"Warning: unknown analyzer mode '{analyzer_mode}', fallback to PAPER")
    knobs = AnalyzerKnobs.paper_mode(
        shuttle_fidelity_mode=shuttle_mode,
        debug_summary=True
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
    print("SchedulerFiredShuttle:", scheduler.shuttle_counter)

print("ReportedTotalShuttle:", result.get("total_shuttle"))
print("----------------")
program_name = openqasm_file_name.split("/")[-1].replace(".qasm", "")
mapper_label = "SABRE2" if mapper_choice == "SABRE" else mapper_choice

summary_line = (
    "SUMMARY|"
    f"program={program_name}|"
    f"machine={machine_type}|"
    f"version={sched_version}|"
    f"mapper={mapper_label}|"
    f"total_shuttle={result.get('total_shuttle')}|"
    f"execution_time_us={result.get('time')}|"
    f"fidelity={result.get('fidelity')}"
)
print(summary_line)
