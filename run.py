# -*- coding: utf-8 -*-
"""
run.py
============================================================
最终入口版：同时兼容 small / large 两条主线

本版目标：
1) 保留原有 small-scale 功能，不改坏 Table 2 复现路径；
2) 支持 large-scale 架构参数从环境变量读取；
3) 支持大规模入口的调度器 V7；
4) 支持大规模 mapper QubitMapSABRELarge；
5) 保持与现有 run_batch.py / run_batch_large.py 的命令行接口兼容；
6) 保持与 analyzer / scheduler / mapper 的既有接口兼容。

命令行接口（保持兼容）：
    python run.py <qasm> <machine_type> <ions_per_region> <mapper> <reorder>
                  <serial_trap_ops> <serial_comm> <serial_all>
                  <gate_type> <swap_type> [sched_family] [sched_version]
                  [analyzer_mode] [architecture_scale]

大规模 sweep 参数通过环境变量传入：
    MUSS_TRAP_CAPACITY
    MUSS_SWAP_LOOKAHEAD_K
    MUSS_SWAP_THRESHOLD
    MUSS_MAX_QUBITS_PER_QCCD
    MUSS_NUM_OPTICAL_ZONES
    MUSS_ENABLE_SWAP_INSERT

说明：
- small 路径默认不读取这些大规模参数，除非显式设置 architecture_scale=LARGE；
- large 路径优先使用 V7 + SABRELarge；
- 若相应类不存在，会给出明确错误，而不是静默退化到不一致实现。
============================================================
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from analyzer import Analyzer, AnalyzerKnobs

# V7 论文专用 analyzer：若文件不存在，则自动回退到旧 analyzer。
try:
    from analyzer_v7 import AnalyzerV7, AnalyzerV7Knobs
except Exception:
    AnalyzerV7 = None
    AnalyzerV7Knobs = None
from ejf_schedule import EJFSchedule, Schedule
from machine import MachineParams
from mappers import *
from parse import InputParse
from test_machines import *

# ------------------------------
# 旧版 / 小规模 MUSS 调度器
# ------------------------------
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

# ------------------------------
# 新增：large-scale MUSS V7
# ------------------------------
try:
    from muss_schedule7 import MUSSSchedule as MUSSScheduleV7
except Exception:
    MUSSScheduleV7 = None

np.random.seed(12345)


# ============================================================
# Helper: 命令行与环境变量读取
# ============================================================
def get_arg(idx: int, default: Optional[str] = None) -> Optional[str]:
    return sys.argv[idx] if len(sys.argv) > idx else default


def getenv_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return str(value)


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Environment variable {name} must be int, got {value!r}") from exc


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Environment variable {name} must be float, got {value!r}") from exc


def getenv_bool_int(name: str, default: int) -> bool:
    return bool(getenv_int(name, default))


# ============================================================
# Helper: scheduler/analyzer compatibility
# ============================================================
def has_shuttle_id_annotations(scheduler) -> bool:
    """
    检查 scheduler.schedule.events 里是否存在带 shuttle_id 的 Split/Move/Merge 事件。
    若存在，则说明 analyzer 可以安全使用 aggregate 模式。
    若不存在，则回退 per_event，兼容尚未修补 shuttle_id 的旧调度器。
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


def _build_v7_knobs(analyzer_mode: str, shuttle_mode: str):
    """
    构造 V7 专用 analyzer 的 knobs。

    设计原则：
    - 优先调用 AnalyzerV7Knobs 自己提供的模式工厂；
    - 若某模式在 analyzer_v7 中未实现，则尽量平滑回退到 paper_mode；
    - 保持 run.py 简洁，不把 V7 细节散落到主流程。
    """
    mode = analyzer_mode.upper()

    if AnalyzerV7Knobs is None:
        raise RuntimeError("AnalyzerV7Knobs is not available")

    # 先优先使用 analyzer_v7 自己定义的模式工厂。
    if mode in ["PAPER", "TABLE2", "P"]:
        if hasattr(AnalyzerV7Knobs, "paper_mode"):
            try:
                return AnalyzerV7Knobs.paper_mode(
                    shuttle_fidelity_mode=shuttle_mode,
                    debug_summary=True,
                )
            except TypeError:
                # 当前 analyzer_v7.py 若未暴露 shuttle_fidelity_mode 参数，则保持兼容。
                return AnalyzerV7Knobs.paper_mode(debug_summary=True)

    if mode in ["EXTENDED", "EXP", "E"] and hasattr(AnalyzerV7Knobs, "extended_mode"):
        try:
            return AnalyzerV7Knobs.extended_mode(
                shuttle_fidelity_mode=shuttle_mode,
                debug_summary=True,
            )
        except TypeError:
            return AnalyzerV7Knobs.extended_mode(debug_summary=True)

    # V7 当前若只提供 paper_mode，则统一回退到论文模式。
    if hasattr(AnalyzerV7Knobs, "paper_mode"):
        if mode not in ["PAPER", "TABLE2", "P", "EXTENDED", "EXP", "E"]:
            print(f"Warning: unknown analyzer mode '{analyzer_mode}', fallback to PAPER for analyzer_v7")
        elif mode in ["EXTENDED", "EXP", "E"]:
            print("Warning: analyzer_v7.py has no extended_mode(); fallback to PAPER")

        try:
            return AnalyzerV7Knobs.paper_mode(
                shuttle_fidelity_mode=shuttle_mode,
                debug_summary=True,
            )
        except TypeError:
            return AnalyzerV7Knobs.paper_mode(debug_summary=True)

    raise RuntimeError("analyzer_v7.py is available, but AnalyzerV7Knobs exposes no usable mode factory")


def build_analyzer(
    sched_family: str,
    sched_version: str,
    analyzer_mode: str,
    scheduler,
    machine_obj,
    init_qubit_layout,
    use_aggregate: bool,
):
    """
    统一创建 analyzer。

    目标：
    1) 对外接口保持不变；
    2) 仅当运行的是 MUSS V7 时，自动切到 analyzer_v7；
    3) 其它所有版本保持原 analyzer 行为，不受影响。
    """
    shuttle_mode = ("aggregate" if use_aggregate else "per_event")
    family = sched_family.upper()
    version = sched_version.upper()

    use_v7_analyzer = (
        family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]
        and version in ["V7", "7", "MUSS_SCHEDULE7", "LARGE"]
        and AnalyzerV7 is not None
        and AnalyzerV7Knobs is not None
    )

    if use_v7_analyzer:
        knobs = _build_v7_knobs(analyzer_mode, shuttle_mode)
        print("Using analyzer_v7.py (paper-faithful analyzer for MUSS V7)")
        return AnalyzerV7(scheduler, machine_obj, init_qubit_layout, knobs)

    # 旧路径保持原样。
    if analyzer_mode in ["PAPER", "TABLE2", "P"]:
        knobs = AnalyzerKnobs.paper_mode(
            shuttle_fidelity_mode=shuttle_mode,
            debug_summary=True,
        )
    elif analyzer_mode in ["EXTENDED", "EXP", "E"]:
        knobs = AnalyzerKnobs.extended_mode(
            shuttle_fidelity_mode=shuttle_mode,
            debug_summary=True,
        )
    else:
        print(f"Warning: unknown analyzer mode '{analyzer_mode}', fallback to PAPER")
        knobs = AnalyzerKnobs.paper_mode(
            shuttle_fidelity_mode=shuttle_mode,
            debug_summary=True,
        )

    print("Using legacy analyzer.py")
    return Analyzer(scheduler, machine_obj, init_qubit_layout, knobs)


# ============================================================
# Helper: initial layout handling
# ============================================================
def should_run_qubit_ordering(mapper_choice: str) -> bool:
    """
    决定是否执行额外 qubit ordering。

    设计原则：
      - 老项目自带 mapper：保留历史行为，继续做 ordering；
      - 新增论文复现 mapper（Trivial / SABRE / SABRELARGE）：
        若其已返回结构化 trap_to_qubits，则不再额外排序。
    """
    return mapper_choice.upper() not in ["TRIVIAL", "SABRE", "SABRELARGE"]



def is_trap_layout(mapping, machine_obj) -> bool:
    """判断 mapping 是否已是 trap_id -> [ion_ids...] 格式。"""
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
      1) trap_id -> [ion_ids...]
      2) qubit_id -> trap_id
    """
    trap_ids = [t.id for t in machine_obj.traps]

    if is_trap_layout(mapping, machine_obj):
        output_layout = {}
        for tid in trap_ids:
            output_layout[tid] = list(mapping.get(tid, []))
        return output_layout

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
    """用于日志打印，描述本次 initial layout 的生成策略。"""
    if machine_obj is not None and is_mapping_bundle(raw_mapping, machine_obj):
        return "mapping_bundle(layout + trap_to_qubits_from_mapper)"
    if should_run_qubit_ordering(mapper_choice):
        return f"mapping + qubit_ordering({reorder_choice})"
    return "mapping_only + canonicalize_to_trap_layout"


# ============================================================
# Helper: Large 参数装配
# ============================================================
def apply_large_env_overrides(mpar: MachineParams, architecture_scale: str) -> Dict[str, Any]:
    """
    将 run_batch_large.py 通过环境变量传下来的 large 参数写入 mpar。

    注意：
    - SMALL 路径保持原行为，不消费 large sweep 参数；
    - LARGE 路径中，这些参数会直接影响 machine 构造、V7 与 analyzer。
    """
    effective: Dict[str, Any] = {}

    if architecture_scale.upper() not in ["LARGE", "L", "EML", "EML-QCCD"]:
        # small path 保持稳定，不强行读取大规模 sweep 参数
        effective["trap_capacity"] = None
        effective["lookahead_k"] = None
        effective["swap_threshold"] = None
        effective["max_qubits_per_qccd"] = getattr(mpar, "max_qubits_per_qccd", 32)
        effective["num_optical_zones"] = getattr(mpar, "num_optical_zones", 1)
        effective["enable_swap_insert"] = False
        return effective

    # 对应 run_batch_large.py 里约定的变量名
    trap_capacity = getenv_int("MUSS_TRAP_CAPACITY", -1)
    lookahead_k = getenv_int("MUSS_SWAP_LOOKAHEAD_K", 8)
    swap_threshold = getenv_int("MUSS_SWAP_THRESHOLD", 4)
    max_qubits_per_qccd = getenv_int("MUSS_MAX_QUBITS_PER_QCCD", 32)
    num_optical_zones = getenv_int("MUSS_NUM_OPTICAL_ZONES", 1)
    enable_swap_insert = getenv_bool_int("MUSS_ENABLE_SWAP_INSERT", 1)

    # 写入 MachineParams，供 machine / scheduler / analyzer 统一读取。
    if trap_capacity > 0:
        effective["trap_capacity"] = trap_capacity
    else:
        effective["trap_capacity"] = None

    mpar.swap_lookahead_k = int(lookahead_k)
    mpar.swap_threshold_T = int(swap_threshold)
    mpar.max_qubits_per_qccd = int(max_qubits_per_qccd)
    mpar.num_optical_zones = int(num_optical_zones)
    mpar.enable_cross_qccd_swap_insertion = bool(enable_swap_insert)

    effective["lookahead_k"] = int(lookahead_k)
    effective["swap_threshold"] = int(swap_threshold)
    effective["max_qubits_per_qccd"] = int(max_qubits_per_qccd)
    effective["num_optical_zones"] = int(num_optical_zones)
    effective["enable_swap_insert"] = bool(enable_swap_insert)
    return effective


# ============================================================
# Helper: Mapper 选择
# ============================================================
def select_mapper(
    mapper_choice: str,
    architecture_scale: str,
    sched_family: str,
    sched_version: str,
    parse_obj,
    machine_obj,
):
    """
    统一 mapper 选择逻辑。

    规则：
    - SMALL + SABRE：保持历史行为；
    - LARGE + SABRE：优先 QubitMapSABRELarge；
    - 显式 SABRELARGE：只允许在有该类时使用。
    """
    mapper_choice_upper = mapper_choice.upper()
    arch_upper = architecture_scale.upper()

    if mapper_choice == "LPFS":
        return QubitMapLPFS(parse_obj, machine_obj)
    if mapper_choice == "Agg":
        return QubitMapAgg(parse_obj, machine_obj)
    if mapper_choice == "Random":
        return QubitMapRandom(parse_obj, machine_obj)
    if mapper_choice == "PO":
        return QubitMapPO(parse_obj, machine_obj)
    if mapper_choice == "Greedy":
        return QubitMapGreedy(parse_obj, machine_obj)
    if mapper_choice_upper == "TRIVIAL":
        return QubitMapTrivial(parse_obj, machine_obj)

    if mapper_choice_upper in ["SABRE", "SABRELARGE"]:
        # large path：优先新版 large mapper
        if arch_upper in ["LARGE", "L", "EML", "EML-QCCD"]:
            sabre_large_cls = globals().get("QubitMapSABRELarge", None)
            if mapper_choice_upper == "SABRELARGE":
                if sabre_large_cls is None:
                    raise RuntimeError(
                        "Mapper 'SABRELarge' was requested, but QubitMapSABRELarge is not available in mappers.py"
                    )
                print("Using QubitMapSABRELarge mapper (large-scale module-aware / zone-aware version)")
                return sabre_large_cls(
                    parse_obj,
                    machine_obj,
                    max_qubits_per_qccd=getattr(getattr(machine_obj, "mparams", None), "max_qubits_per_qccd", 32),
                )

            # mapper_choice == SABRE
            if sabre_large_cls is not None:
                print("Using QubitMapSABRELarge mapper (large-scale module-aware / zone-aware version)")
                return sabre_large_cls(
                    parse_obj,
                    machine_obj,
                    max_qubits_per_qccd=getattr(getattr(machine_obj, "mparams", None), "max_qubits_per_qccd", 32),
                )

            raise RuntimeError(
                "Large-scale run requested mapper 'SABRE', but QubitMapSABRELarge is not available. "
                "Please add the Stage-3 mapper implementation to mappers.py."
            )

        # small path：保留历史版本绑定
        if sched_family in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
            if sched_version in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
                print("Using QubitMapSABRE2 mapper (matches muss_schedule2 paper version)")
                return QubitMapSABRE2(parse_obj, machine_obj)
            if sched_version in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
                print("Using QubitMapSABRE6 mapper (matches muss_schedule3 improved version)")
                return QubitMapSABRE6(parse_obj, machine_obj)
            if sched_version in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
                print("Using SABRE6 mapper (matches muss_schedule4 improved version)")
                return QubitMapSABRE6(parse_obj, machine_obj)
            if sched_version in ["V5", "5", "MUSS_SCHEDULE5", "INNOV3"]:
                print("Using SABRE2 mapper (matches muss_schedule5 improved version)")
                return QubitMapSABRE2(parse_obj, machine_obj)
            if sched_version in ["V6", "6", "MUSS_SCHEDULE6", "INNOV4"]:
                print("Using QubitMapSABRE2 mapper (matches muss_schedule6 improved version)")
                return QubitMapSABRE2(parse_obj, machine_obj)
            if sched_version in ["V7", "7", "MUSS_SCHEDULE7", "LARGE"]:
                # 允许 small 上用 V7，但 small mapper 仍保持 SABRE2
                print("Using QubitMapSABRE2 mapper (small-scale compatibility path for V7)")
                return QubitMapSABRE2(parse_obj, machine_obj)

            print(f"Warning: Unknown scheduler version '{sched_version}', fallback to SABRE2")
            return QubitMapSABRE2(parse_obj, machine_obj)

        print("Using default SABRE2 mapper (non-MUSS scheduler)")
        return QubitMapSABRE2(parse_obj, machine_obj)

    raise RuntimeError(f"Unsupported mapper choice '{mapper_choice}'")


# ============================================================
# Helper: Scheduler 选择
# ============================================================
def build_scheduler(
    sched_family: str,
    sched_version: str,
    architecture_scale: str,
    ip,
    machine_obj,
    init_qubit_layout,
    serial_trap_ops: int,
    serial_comm: int,
    serial_all: int,
):
    """
    统一 scheduler 选择。

    规则：
    - SMALL: 保持既有 V2~V6 兼容；
    - LARGE + MUSS + V7: 走 muss_schedule7；
    - LARGE + MUSS + V6: 给出提示，但允许继续用旧逻辑（不推荐）；
    - 其它情况回退到 EJF。
    """
    sched_family_upper = sched_family.upper()
    sched_version_upper = sched_version.upper()
    arch_upper = architecture_scale.upper()

    if sched_family_upper in ["MUSS", "MUSS-TI", "MUSS_TI_MODE"]:
        if sched_version_upper in ["V2", "2", "MUSS_SCHEDULE2", "PAPER"]:
            print("Using muss_schedule2.py paper-faithful fixed version")
            return MUSSScheduleV2(
                ip, machine_obj, init_qubit_layout,
                serial_trap_ops, serial_comm, serial_all,
            )

        if sched_version_upper in ["V3", "3", "MUSS_SCHEDULE3", "INNOV"]:
            if MUSSScheduleV3 is None:
                raise RuntimeError("muss_schedule3 is not available")
            print("Using muss_schedule3.py new_vision")
            return MUSSScheduleV3(
                ip.gate_graph, ip.all_gate_map, machine_obj, init_qubit_layout,
                serial_trap_ops, serial_comm, serial_all,
            )

        if sched_version_upper in ["V4", "4", "MUSS_SCHEDULE4", "INNOV2"]:
            if MUSSScheduleV4 is None:
                raise RuntimeError("muss_schedule4 is not available")
            print("Using muss_schedule4.py new_vision")
            return MUSSScheduleV4(
                ip.gate_graph, ip.all_gate_map, machine_obj, init_qubit_layout,
                serial_trap_ops, serial_comm, serial_all,
            )

        if sched_version_upper in ["V5", "5", "MUSS_SCHEDULE5", "INNOV3"]:
            if MUSSScheduleV5 is None:
                raise RuntimeError("muss_schedule5 is not available")
            print("Using muss_schedule5.py new_vision")
            return MUSSScheduleV5(
                ip.gate_graph, ip.all_gate_map, machine_obj, init_qubit_layout,
                serial_trap_ops, serial_comm, serial_all,
            )

        if sched_version_upper in ["V6", "6", "MUSS_SCHEDULE6", "INNOV4"]:
            if MUSSScheduleV6 is None:
                raise RuntimeError("muss_schedule6 is not available")
            if arch_upper in ["LARGE", "L", "EML", "EML-QCCD"]:
                print("[INFO] Large-scale machine model is enabled, but V6 is mainly a small-scale strict scheduler.")
            print("Using muss_schedule6.py new_vision")
            return MUSSScheduleV6(
                ip.gate_graph, ip.all_gate_map, machine_obj, init_qubit_layout,
                serial_trap_ops, serial_comm, serial_all,
            )

        if sched_version_upper in ["V7", "7", "MUSS_SCHEDULE7", "LARGE"]:
            if MUSSScheduleV7 is None:
                raise RuntimeError(
                    "muss_schedule7 is not available. Please add the Stage-4 scheduler implementation."
                )
            print("Using muss_schedule7.py large-scale version")
            return MUSSScheduleV7(
                ip.gate_graph, ip.all_gate_map, machine_obj, init_qubit_layout,
                serial_trap_ops, serial_comm, serial_all,
            )

        raise RuntimeError(
            f"Unsupported scheduler version '{sched_version}', supported: V2 / V3 / V4 / V5 / V6 / V7"
        )

    print("Fallback to EJF scheduler")
    return EJFSchedule(
        ip.gate_graph, ip.all_gate_map, machine_obj, init_qubit_layout,
        serial_trap_ops, serial_comm, serial_all,
    )


# ============================================================
# Command line args
# ============================================================
if len(sys.argv) < 11:
    print("Usage:")
    print(
        "python run.py <qasm> <machine_type> <ions_per_region> <mapper> <reorder> "
        "<serial_trap_ops> <serial_comm> <serial_all> <gate_type> <swap_type> "
        "[sched_family] [sched_version] [analyzer_mode] [architecture_scale]"
    )
    print("")
    print("Examples:")
    print("python run.py ghz32.qasm G2x2 12 SABRE Fidelity 1 1 0 FM PaperSwapDirect MUSS V2 PAPER SMALL")
    print("python run.py qft128.qasm G3x4 16 SABRE Fidelity 1 1 1 FM PaperSwapDirect MUSS V7 PAPER LARGE")
    print("python run.py qft256.qasm EML 16 SABRE Fidelity 1 1 1 FM PaperSwapDirect MUSS V7 PAPER LARGE")
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
# Pre-parse QASM
#   large 架构（尤其 EML）需要根据 qubit 数决定模块数，
#   因此这里先解析 QASM，再建机器。
# ============================================================
ip = InputParse()
ip.parse_ir(openqasm_file_name)
ip.visualize_graph("visualize_graph_2.gexf")

qc = QuantumCircuit.from_qasm_file(openqasm_file_name)
dag = circuit_to_dag(qc)  # 保留现有调试/对比遗留接口
num_program_qubits = int(qc.num_qubits)
_ = dag  # 显式保留，避免静态检查误报未使用。


# ============================================================
# Machine 参数（MUSS-TI Table 1 + large sweep 扩展参数）
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
mpar.alpha_bg = 0.0

mpar.architecture_scale = "small"
mpar.enable_partition = False

# ---- Analyzer 会读到的物理参数 ----
mpar.T1 = 600e6
mpar.k_heating = 0.001
mpar.epsilon = 1.0 / 25600.0

# ---- Large-scale 默认参数 ----
mpar.max_qubits_per_qccd = 32
mpar.num_optical_zones = 1
mpar.qccd_fiber_latency_us = 200.0
mpar.qccd_fiber_fidelity = 0.99
mpar.swap_lookahead_k = 8
mpar.swap_threshold_T = 4
mpar.enable_cross_qccd_swap_insertion = True

# ---- 命令行控制 ----
mpar.gate_type = gate_type
mpar.swap_type = swap_type

machine_model = "MUSS_Params"


# ============================================================
# Architecture scale: explicit small / large switch
# ============================================================
if architecture_scale is None:
    if machine_type in ["G2x2", "G2x3", "L6", "H6"]:
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
    architecture_scale = "SMALL"

# large sweep 环境变量覆盖（仅在 large 模式生效）
effective_large_env = apply_large_env_overrides(mpar, architecture_scale)

# ions_per_region 对 small 路径保留历史含义；large 路径若给了 MUSS_TRAP_CAPACITY，优先使用它。
effective_capacity = int(num_ions_per_region)
if effective_large_env.get("trap_capacity") is not None:
    effective_capacity = int(effective_large_env["trap_capacity"])


# ============================================================
# 打印基本信息
# ============================================================
print("Simulation")
print("Program:          ", openqasm_file_name)
print("Machine:          ", machine_type)
print("Model:            ", machine_model)
print("Program Qubits:   ", num_program_qubits)
print("Ions/Region(arg): ", num_ions_per_region)
print("Effective Cap:    ", effective_capacity)
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
print("Large.Env.K:      ", effective_large_env.get("lookahead_k"))
print("Large.Env.T:      ", effective_large_env.get("swap_threshold"))
print("Large.Env.MaxQCCD:", effective_large_env.get("max_qubits_per_qccd"))
print("Large.Env.OptZone:", effective_large_env.get("num_optical_zones"))
print("Large.Env.SWAP:   ", effective_large_env.get("enable_swap_insert"))


# ============================================================
# 创建测试机器
#   small path 仍走原有 2x2 / 2x3 / L6 / H6；
#   large path 支持 G3x4 / G4x5 / EML / EML2Z。
# ============================================================
try:
    m = build_machine_by_type(
        machine_type,
        effective_capacity,
        mpar,
        num_qubits=num_program_qubits,
    )
except Exception as exc:
    print(f"Error: Failed to construct machine '{machine_type}': {exc}")
    sys.exit(1)

m.print_machine_stats()
print("Parse object map:")
print(ip.cx_gate_map)
print("Parse object graph:")
print(ip.gate_graph)


# ============================================================
# 初始映射：选择 mapper
# ============================================================
try:
    qm = select_mapper(
        mapper_choice=mapper_choice,
        architecture_scale=architecture_scale,
        sched_family=sched_family,
        sched_version=sched_version,
        parse_obj=ip,
        machine_obj=m,
    )
except Exception as exc:
    print(f"Error: mapper selection failed: {exc}")
    sys.exit(1)

raw_mapping = qm.compute_mapping()

print("Raw mapping:")
print(raw_mapping)

layout_mapping, mapper_trap_layout = extract_layout_bundle(raw_mapping, m)


# ============================================================
# 初始布局生成
# ============================================================
print(f"Initial layout policy: {describe_layout_policy(mapper_choice, reorder_choice, raw_mapping, m)}")

run_ordering = should_run_qubit_ordering(mapper_choice)

if mapper_trap_layout is not None:
    if reorder_choice not in [None, "", "None", "NONE", "Disabled", "DISABLED"]:
        print(
            f"Note: reorder_choice='{reorder_choice}' is ignored because mapper already provides ordered trap layout"
        )

    print(f"Use mapper-provided ordered trap layout for mapper '{mapper_choice}'")
    init_qubit_layout = mapper_trap_layout

elif not run_ordering:
    if reorder_choice not in [None, "", "None", "NONE", "Disabled", "DISABLED"]:
        print(f"Note: reorder_choice='{reorder_choice}' is ignored for mapper '{mapper_choice}'")

    print(f"Skip qubit ordering for mapper '{mapper_choice}'")
    init_qubit_layout = canonicalize_mapping_to_layout(layout_mapping, m)
else:
    print(f"Apply qubit ordering for mapper '{mapper_choice}' with mode '{reorder_choice}'")

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

scheduler_build_ts = time.perf_counter()
try:
    scheduler = build_scheduler(
        sched_family=sched_family,
        sched_version=sched_version,
        architecture_scale=architecture_scale,
        ip=ip,
        machine_obj=m,
        init_qubit_layout=init_qubit_layout,
        serial_trap_ops=serial_trap_ops,
        serial_comm=serial_comm,
        serial_all=serial_all,
    )
except Exception as exc:
    print(f"Error: scheduler construction failed: {exc}")
    sys.exit(1)

scheduler_build_elapsed = time.perf_counter() - scheduler_build_ts
print(f"Scheduler object construction time: {scheduler_build_elapsed:.6f}s")

scheduler_run_ts = time.perf_counter()
scheduler.run()
scheduler_run_elapsed = time.perf_counter() - scheduler_run_ts
print(f"Scheduler run time: {scheduler_run_elapsed:.6f}s")


# ============================================================
# Analyzer 配置
# ============================================================
use_aggregate = has_shuttle_id_annotations(scheduler)

if use_aggregate:
    print("Analyzer shuttle mode: aggregate (detected shuttle_id annotations)")
else:
    print("Analyzer shuttle mode: per_event (no shuttle_id detected; compatibility fallback)")

analyzer_ts = time.perf_counter()
analyzer = build_analyzer(
    sched_family=sched_family,
    sched_version=sched_version,
    analyzer_mode=analyzer_mode,
    scheduler=scheduler,
    machine_obj=m,
    init_qubit_layout=init_qubit_layout,
    use_aggregate=use_aggregate,
)
result = analyzer.analyze_and_return()
analyzer_elapsed = time.perf_counter() - analyzer_ts
print(f"Analyzer time: {analyzer_elapsed:.6f}s")


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
mapper_label = mapper_choice
if mapper_choice.upper() == "SABRE":
    if architecture_scale in ["LARGE", "L", "EML", "EML-QCCD"]:
        mapper_label = "SABRELarge"
    else:
        mapper_label = "SABRE2"

# 机器上若支持模块统计，则纳入摘要，便于 parse_output / CSV 汇总。
num_modules = None
if hasattr(m, "qccd_graph") and m.qccd_graph is not None:
    try:
        num_modules = int(m.qccd_graph.number_of_nodes())
    except Exception:
        num_modules = None
elif hasattr(m, "module_count"):
    try:
        num_modules = int(m.module_count)
    except Exception:
        num_modules = None

remote_gate_count = result.get("remote_gate_count")
if remote_gate_count is None and hasattr(scheduler, "remote_gate_count"):
    remote_gate_count = getattr(scheduler, "remote_gate_count")

compile_time_s = scheduler_build_elapsed + scheduler_run_elapsed + analyzer_elapsed

summary_fields = {
    "program": program_name,
    "machine": machine_type,
    "version": sched_version,
    "mapper": mapper_label,
    "scale": architecture_scale,
    "total_shuttle": result.get("total_shuttle"),
    "execution_time_us": result.get("time"),
    "fidelity": result.get("fidelity"),
    "trap_capacity": effective_capacity,
    "lookahead_k": effective_large_env.get("lookahead_k"),
    "swap_threshold": effective_large_env.get("swap_threshold"),
    "num_optical_zones": effective_large_env.get("num_optical_zones"),
    "max_qubits_per_qccd": effective_large_env.get("max_qubits_per_qccd"),
    "enable_swap_insert": int(bool(effective_large_env.get("enable_swap_insert"))),
    "num_modules": num_modules,
    "remote_gate_count": remote_gate_count,
    "compile_time_s": f"{compile_time_s:.6f}",
}

summary_line = "SUMMARY|" + "|".join(f"{k}={v}" for k, v in summary_fields.items())
print(summary_line)
