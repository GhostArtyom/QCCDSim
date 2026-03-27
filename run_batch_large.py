#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch_large.py
============================================================
大规模实验驱动脚本（第五阶段）

目标：
1) 在不影响现有 small-scale run_batch.py 的前提下，新增一个独立的大规模批量实验入口；
2) 覆盖论文大规模阶段最常用的四类实验：
   - architecture comparison
   - trap capacity sweep
   - look-ahead sweep
   - multi-zone comparison
3) 与当前项目的 run.py 接口保持兼容：
   python run.py <qasm> <machine_type> <ions_per_region> <mapper> <reorder>
                 <serial_trap_ops> <serial_comm> <serial_all>
                 <gate_type> <swap_type> [sched_family] [sched_version]
                 [analyzer_mode] [architecture_scale]
4) 为后续继续扩展 run.py / scheduler 参数留好接口：
   - 通过环境变量把 large-scale sweep 参数传给子进程；
   - 即便当前 run.py 尚未消费这些环境变量，实验清单与日志命名也会完整保留它们，
     后续只需在 run.py 读取对应环境变量即可无缝接上。

说明：
- 本脚本不替代原有 run_batch.py；原 run_batch.py 继续服务小规模 Table 2 复现。
- 本脚本统一将输出写入 output_large/，避免污染 small-scale 输出目录。
- 本脚本会生成 manifest.csv，供 parse_output.py 进行二次汇总。
"""

from __future__ import annotations

import csv
import os
import subprocess as sp
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ============================================================
# 目录配置
# ============================================================
ROOT = Path(".")
PROGRAM_DIR = ROOT / "programs"
OUTPUT_DIR = ROOT / "output_large"
LOG_DIR = OUTPUT_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 默认实验参数
# ============================================================
DEFAULT_MAPPER = "SABRE"
DEFAULT_REORDER = "Naive"  # 对 SABRE 路径会被 run.py 忽略，但保留接口一致性
DEFAULT_GATE_TYPE = "FM"
DEFAULT_SWAP_TYPE = "PaperSwapDirect"
DEFAULT_SCHED_FAMILY = "MUSS"
DEFAULT_SCHED_VERSION = "V7"
DEFAULT_ANALYZER_MODE = "PAPER"
DEFAULT_ARCH_SCALE = "LARGE"

DEFAULT_SERIAL_TRAP_OPS = "1"
DEFAULT_SERIAL_COMM = "1"
DEFAULT_SERIAL_ALL = "1"

DEFAULT_TRAP_CAPACITY = 16
DEFAULT_LOOKAHEAD_K = 8
DEFAULT_SWAP_THRESHOLD = 4
DEFAULT_MAX_QUBITS_PER_QCCD = 32
DEFAULT_NUM_OPTICAL_ZONES = 1
DEFAULT_ENABLE_SWAP_INSERT = 1


# ============================================================
# benchmark 集合
# 说明：
# - 这里只给出“可直接用项目目录名对齐”的默认集合；
# - 若某个 qasm 不存在，会自动跳过并给出 warning；
# - 你可以按自己的程序文件名直接替换。
# ============================================================
#DEFAULT_BENCHMARKS = {
#    "small": ["BV32", "GHZ32", "ADDER32", "QAOA32", "qft32_swap", "SQRT30"],
#    "medium": ["BV128", "GHZ128", "ADDER128", "QAOA128", "QFT128", "SQRT128"],
 #   "large": ["BV256", "GHZ256", "ADDER256", "QAOA256", "QFT256", "SQRT256"],
#}

DEFAULT_BENCHMARKS = {
    "small": ["BV32", "GHZ32", "ADDER32", "QAOA32", "qft32_swap", "SQRT30"],
    "medium": ["BV128", "GHZ128", "SQRT117"],
    "large": ["GHZ256", "SQRT299"],
}

# ============================================================
# 实验配置数据结构
# ============================================================
@dataclass
class JobConfig:
    suite: str
    benchmark: str
    qasm_path: str
    machine: str
    ions_per_region: int
    mapper: str = DEFAULT_MAPPER
    reorder: str = DEFAULT_REORDER
    serial_trap_ops: str = DEFAULT_SERIAL_TRAP_OPS
    serial_comm: str = DEFAULT_SERIAL_COMM
    serial_all: str = DEFAULT_SERIAL_ALL
    gate_type: str = DEFAULT_GATE_TYPE
    swap_type: str = DEFAULT_SWAP_TYPE
    sched_family: str = DEFAULT_SCHED_FAMILY
    sched_version: str = DEFAULT_SCHED_VERSION
    analyzer_mode: str = DEFAULT_ANALYZER_MODE
    architecture_scale: str = DEFAULT_ARCH_SCALE
    trap_capacity: int = DEFAULT_TRAP_CAPACITY
    lookahead_k: int = DEFAULT_LOOKAHEAD_K
    swap_threshold: int = DEFAULT_SWAP_THRESHOLD
    max_qubits_per_qccd: int = DEFAULT_MAX_QUBITS_PER_QCCD
    num_optical_zones: int = DEFAULT_NUM_OPTICAL_ZONES
    enable_swap_insert: int = DEFAULT_ENABLE_SWAP_INSERT
    notes: str = ""

    def to_env(self) -> Dict[str, str]:
        """
        为 run.py / scheduler 预留的大规模参数环境变量。
        当前项目若尚未读取这些环境变量，也不会影响批量脚本本身。
        """
        return {
            "MUSS_TRAP_CAPACITY": str(self.trap_capacity),
            "MUSS_SWAP_LOOKAHEAD_K": str(self.lookahead_k),
            "MUSS_SWAP_THRESHOLD": str(self.swap_threshold),
            "MUSS_MAX_QUBITS_PER_QCCD": str(self.max_qubits_per_qccd),
            "MUSS_NUM_OPTICAL_ZONES": str(self.num_optical_zones),
            "MUSS_ENABLE_SWAP_INSERT": str(self.enable_swap_insert),
            "MUSS_SUITE": self.suite,
        }

    def build_args(self) -> List[str]:
        """构造对子进程 run.py 的调用参数。"""
        return [
            sys.executable,
            "-u",
            "run.py",
            self.qasm_path,
            self.machine,
            str(self.ions_per_region),
            self.mapper,
            self.reorder,
            self.serial_trap_ops,
            self.serial_comm,
            self.serial_all,
            self.gate_type,
            self.swap_type,
            self.sched_family,
            self.sched_version,
            self.analyzer_mode,
            self.architecture_scale,
        ]

    def log_filename(self) -> str:
        safe_prog = Path(self.qasm_path).stem
        return (
            f"{self.suite}__{safe_prog}__{self.machine}__cap{self.trap_capacity}"
            f"__k{self.lookahead_k}__T{self.swap_threshold}__oz{self.num_optical_zones}"
            f"__{self.mapper}__{self.sched_family}{self.sched_version}.log"
        )


# ============================================================
# 工具函数
# ============================================================
def find_existing_qasm(stem: str) -> Optional[Path]:
    """在 programs/ 下寻找 <stem>.qasm。"""
    candidate = PROGRAM_DIR / f"{stem}.qasm"
    return candidate if candidate.exists() else None



def available_benchmarks(names: Iterable[str]) -> List[Path]:
    existing = []
    for name in names:
        qasm = find_existing_qasm(name)
        if qasm is not None:
            existing.append(qasm)
        else:
            print(f"[WARN] Skip missing benchmark: {name}.qasm")
    return existing



def parse_summary_line_from_log(log_path: Path) -> Optional[Dict[str, str]]:
    """
    提取 run.py 打印的 SUMMARY 行。

    当前 run.py 的 SUMMARY 至少包含：
      program / machine / version / mapper / total_shuttle / execution_time_us / fidelity
    """
    if not log_path.exists():
        return None

    summary_line = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("SUMMARY|"):
                summary_line = line

    if summary_line is None:
        return None

    out: Dict[str, str] = {}
    for item in summary_line.split("|")[1:]:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k] = v
    return out



def parse_compile_time_from_log(log_path: Path) -> Optional[float]:
    """
    从日志中提取编译时间。

    当前 run.py 至少打印了：
      Scheduler object construction time: <x>s

    若后续 run.py 扩展出总编译时间字段，这里可继续兼容。
    """
    if not log_path.exists():
        return None

    scheduler_build = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("Scheduler object construction time:"):
                try:
                    scheduler_build = float(line.split(":", 1)[1].strip().rstrip("s"))
                except Exception:
                    scheduler_build = None
    return scheduler_build



def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)



def write_markdown(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def maybe_generate_paper_report(manifest_csv: Path) -> None:
    """
    若 paper_report_v7.py 存在，则在批量实验结束后自动生成论文式汇总表。
    这样可避免把汇总逻辑继续堆进 run_batch_large.py。
    """
    report_script = ROOT / "paper_report_v7.py"
    if not report_script.exists():
        print(f"[INFO] Skip paper report: {report_script} not found")
        return

    cmd = [sys.executable, str(report_script), str(manifest_csv), str(OUTPUT_DIR)]
    print("[REPORT] " + " ".join(cmd))
    ret = sp.call(cmd)
    if ret != 0:
        print(f"[WARN] paper_report_v7.py exited with code {ret}")


# ============================================================
# 实验矩阵构造
# ============================================================
def build_architecture_comparison_jobs() -> List[JobConfig]:
    jobs: List[JobConfig] = []

    # medium: baseline 用 G3x4，EML 用 EML
    for qasm in available_benchmarks(DEFAULT_BENCHMARKS["medium"]):
        jobs.append(
            JobConfig(
                suite="architecture_comparison",
                benchmark=qasm.stem,
                qasm_path=str(qasm),
                machine="G3x4",
                ions_per_region=DEFAULT_TRAP_CAPACITY,
                trap_capacity=DEFAULT_TRAP_CAPACITY,
                notes="medium_baseline",
            )
        )
        jobs.append(
            JobConfig(
                suite="architecture_comparison",
                benchmark=qasm.stem,
                qasm_path=str(qasm),
                machine="EML",
                ions_per_region=DEFAULT_TRAP_CAPACITY,
                trap_capacity=DEFAULT_TRAP_CAPACITY,
                notes="medium_eml",
            )
        )

    # large: baseline 用 G4x5，EML 用 EML
    for qasm in available_benchmarks(DEFAULT_BENCHMARKS["large"]):
        jobs.append(
            JobConfig(
                suite="architecture_comparison",
                benchmark=qasm.stem,
                qasm_path=str(qasm),
                machine="G4x5",
                ions_per_region=DEFAULT_TRAP_CAPACITY,
                trap_capacity=DEFAULT_TRAP_CAPACITY,
                notes="large_baseline",
            )
        )
        jobs.append(
            JobConfig(
                suite="architecture_comparison",
                benchmark=qasm.stem,
                qasm_path=str(qasm),
                machine="EML",
                ions_per_region=DEFAULT_TRAP_CAPACITY,
                trap_capacity=DEFAULT_TRAP_CAPACITY,
                notes="large_eml",
            )
        )
    return jobs



def build_capacity_jobs() -> List[JobConfig]:
    jobs: List[JobConfig] = []
    for qasm in available_benchmarks(DEFAULT_BENCHMARKS["medium"]):
        for cap in [12, 14, 16, 18, 20]:
            jobs.append(
                JobConfig(
                    suite="capacity_sweep",
                    benchmark=qasm.stem,
                    qasm_path=str(qasm),
                    machine="EML",
                    ions_per_region=cap,
                    trap_capacity=cap,
                    notes="capacity_sweep_medium",
                )
            )
    return jobs



def build_lookahead_jobs() -> List[JobConfig]:
    jobs: List[JobConfig] = []
    for qasm in available_benchmarks(DEFAULT_BENCHMARKS["medium"]):
        for k in [4, 6, 8, 10, 12]:
            jobs.append(
                JobConfig(
                    suite="lookahead_sweep",
                    benchmark=qasm.stem,
                    qasm_path=str(qasm),
                    machine="EML",
                    ions_per_region=DEFAULT_TRAP_CAPACITY,
                    trap_capacity=DEFAULT_TRAP_CAPACITY,
                    lookahead_k=k,
                    notes="lookahead_sweep_medium",
                )
            )
    return jobs



def build_multizone_jobs() -> List[JobConfig]:
    jobs: List[JobConfig] = []
    for qasm in available_benchmarks(DEFAULT_BENCHMARKS["medium"] + DEFAULT_BENCHMARKS["large"]):
        # 1 optical zone
        jobs.append(
            JobConfig(
                suite="multi_zone",
                benchmark=qasm.stem,
                qasm_path=str(qasm),
                machine="EML",
                ions_per_region=DEFAULT_TRAP_CAPACITY,
                trap_capacity=DEFAULT_TRAP_CAPACITY,
                num_optical_zones=1,
                notes="one_optical_zone",
            )
        )
        # 2 optical zones
        jobs.append(
            JobConfig(
                suite="multi_zone",
                benchmark=qasm.stem,
                qasm_path=str(qasm),
                machine="EML2Z",
                ions_per_region=DEFAULT_TRAP_CAPACITY,
                trap_capacity=DEFAULT_TRAP_CAPACITY,
                num_optical_zones=2,
                notes="two_optical_zones",
            )
        )
    return jobs



def build_all_jobs(selected_suites: Iterable[str]) -> List[JobConfig]:
    selected = {s.strip().lower() for s in selected_suites}
    jobs: List[JobConfig] = []

    if "architecture" in selected or "architecture_comparison" in selected or "all" in selected:
        jobs.extend(build_architecture_comparison_jobs())
    if "capacity" in selected or "capacity_sweep" in selected or "all" in selected:
        jobs.extend(build_capacity_jobs())
    if "lookahead" in selected or "lookahead_sweep" in selected or "all" in selected:
        jobs.extend(build_lookahead_jobs())
    if "multizone" in selected or "multi_zone" in selected or "all" in selected:
        jobs.extend(build_multizone_jobs())

    return jobs


# ============================================================
# 单个任务执行
# ============================================================
def run_one_job(job: JobConfig) -> Dict[str, str]:
    log_path = LOG_DIR / job.log_filename()
    env = os.environ.copy()
    env.update(job.to_env())

    args = job.build_args()
    start_ts = time.perf_counter()

    print("=" * 100)
    print(f"[RUN] suite={job.suite} benchmark={job.benchmark} machine={job.machine} cap={job.trap_capacity} k={job.lookahead_k} oz={job.num_optical_zones}")
    print("[CMD] " + " ".join(args))
    print(f"[LOG] {log_path}")

    with open(log_path, "w", encoding="utf-8") as fh:
        ret = sp.call(args, stdout=fh, stderr=sp.STDOUT, env=env)

    wallclock_s = time.perf_counter() - start_ts

    summary = parse_summary_line_from_log(log_path) or {}
    scheduler_build_s = parse_compile_time_from_log(log_path)

    row = asdict(job)
    row.update(
        {
            "return_code": ret,
            "log_path": str(log_path),
            "wallclock_s": f"{wallclock_s:.6f}",
            "scheduler_build_s": "" if scheduler_build_s is None else f"{scheduler_build_s:.6f}",
            "program": summary.get("program", job.benchmark),
            "summary_machine": summary.get("machine", ""),
            "summary_version": summary.get("version", ""),
            "summary_mapper": summary.get("mapper", ""),
            "total_shuttle": summary.get("total_shuttle", ""),
            "execution_time_us": summary.get("execution_time_us", ""),
            "fidelity": summary.get("fidelity", ""),
        }
    )
    if summary:
        row.update(summary)

    if ret != 0:
        print(f"[WARN] Job failed with exit code {ret}: {log_path}")
    elif not summary:
        print(f"[WARN] No SUMMARY line found: {log_path}")
    else:
        print(f"[OK] total_shuttle={row['total_shuttle']} time_us={row['execution_time_us']} fidelity={row['fidelity']}")

    return row


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    selected_suites = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    jobs = build_all_jobs(selected_suites)

    print("=" * 100)
    print("Large-scale batch driver")
    print(f"Output dir         : {OUTPUT_DIR}")
    print(f"Selected suites    : {selected_suites}")
    print(f"Total jobs         : {len(jobs)}")
    print(f"Default scheduler  : {DEFAULT_SCHED_FAMILY} {DEFAULT_SCHED_VERSION}")
    print(f"Default analyzer   : {DEFAULT_ANALYZER_MODE}")
    print(f"Default mapper     : {DEFAULT_MAPPER}")
    print("=" * 100)

    manifest_rows: List[Dict] = []
    for job in jobs:
        manifest_rows.append(run_one_job(job))

    manifest_csv = OUTPUT_DIR / "manifest.csv"
    manifest_md = OUTPUT_DIR / "manifest.md"
    write_csv(manifest_rows, manifest_csv)
    write_markdown(manifest_rows, manifest_md)

    print("=" * 100)
    print(f"Manifest CSV : {manifest_csv}")
    print(f"Manifest MD  : {manifest_md}")
    maybe_generate_paper_report(manifest_csv)
    print("All large-scale jobs finished.")


if __name__ == "__main__":
    main()
