#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_report_v7.py
============================================================
根据 run_batch_large.py 生成的 manifest.csv 输出论文风格汇总表。

设计目标：
1) 不把汇总逻辑堆进 run_batch_large.py；
2) 兼容 analyzer_v7.py 输出的论文指标；
3) 同时适配 small / large（只要 manifest 中有相同字段即可）；
4) 生成扁平 CSV/Markdown，方便直接贴到论文实验章节或再做二次作图。
============================================================
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def choose(row: Dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in [None, ""]:
            return value
    return default


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    """输出一条适合论文表格的扁平结果。"""
    return {
        "suite": row.get("suite", ""),
        "benchmark": choose(row, "program", "benchmark"),
        "machine": choose(row, "summary_machine", "machine"),
        "version": choose(row, "summary_version", "sched_version"),
        "mapper": choose(row, "summary_mapper", "mapper"),
        "scale": row.get("scale", row.get("architecture_scale", "")),
        "trap_capacity": choose(row, "trap_capacity"),
        "lookahead_k": choose(row, "lookahead_k"),
        "swap_threshold": choose(row, "swap_threshold"),
        "num_optical_zones": choose(row, "num_optical_zones"),
        "max_qubits_per_qccd": choose(row, "max_qubits_per_qccd"),
        "enable_swap_insert": choose(row, "enable_swap_insert"),
        "num_modules": choose(row, "num_modules"),
        "total_shuttle": choose(row, "total_shuttle"),
        "logical_shuttle_count": choose(row, "logical_shuttle_count"),
        "physical_shuttle_leg_count": choose(row, "physical_shuttle_leg_count"),
        "execution_time_us": choose(row, "execution_time_us"),
        "fidelity": choose(row, "fidelity"),
        "remote_2q_count": choose(row, "remote_2q_count", "remote_gate_count"),
        "fiber_gate_count": choose(row, "fiber_gate_count"),
        "regular_fiber_gate_count": choose(row, "regular_fiber_gate_count"),
        "swap_insert_gate_count": choose(row, "swap_insert_gate_count"),
        "swap_insert_logical_count": choose(row, "swap_insert_logical_count"),
        "local_2q_count": choose(row, "local_2q_count"),
        "operation_local_2q_count": choose(row, "operation_local_2q_count"),
        "optical_local_2q_count": choose(row, "optical_local_2q_count"),
        "local_2q_time": choose(row, "local_2q_time"),
        "operation_local_2q_time": choose(row, "operation_local_2q_time"),
        "optical_local_2q_time": choose(row, "optical_local_2q_time"),
        "fiber_gate_time": choose(row, "fiber_gate_time"),
        "swap_insert_time": choose(row, "swap_insert_time"),
        "shuttle_time_total": choose(row, "shuttle_time_total"),
        "gate_mult": choose(row, "gate_mult"),
        "swap_insert_gate_mult": choose(row, "swap_insert_gate_mult"),
        "shuttle_mult": choose(row, "shuttle_mult"),
        "compile_time_s": choose(row, "compile_time_s", "scheduler_build_s"),
        "log_path": row.get("log_path", ""),
    }


def sort_key(row: Dict[str, str]):
    return (row.get("suite", ""), row.get("benchmark", ""), row.get("machine", ""), row.get("trap_capacity", ""), row.get("lookahead_k", ""), row.get("num_optical_zones", ""))


def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_large/manifest.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [normalize_row(r) for r in read_csv(manifest_path)]
    rows.sort(key=sort_key)

    all_csv = out_dir / "paper_results_v7.csv"
    all_md = out_dir / "paper_results_v7.md"
    write_csv(rows, all_csv)
    write_markdown(rows, all_md)

    suites = sorted({r.get("suite", "") for r in rows if r.get("suite", "")})
    for suite in suites:
        suite_rows = [r for r in rows if r.get("suite", "") == suite]
        write_csv(suite_rows, out_dir / f"paper_{suite}.csv")
        write_markdown(suite_rows, out_dir / f"paper_{suite}.md")

    print(f"Paper report CSV: {all_csv}")
    print(f"Paper report MD : {all_md}")
    for suite in suites:
        print(f"Paper suite report: {out_dir / ('paper_' + suite + '.csv')}")


if __name__ == "__main__":
    main()
