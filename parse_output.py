#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_output.py
============================================================
第五阶段结果汇总脚本。

设计目标：
1) 兼容新的 output_large/manifest.csv 输出；
2) 面向论文大规模阶段，生成以下四类聚合结果：
   - architecture comparison
   - capacity sweep
   - look-ahead sweep
   - multi-zone comparison
3) 不再沿用旧版 parse_output.py 中与历史日志格式强绑定的大量绘图逻辑；
   本脚本专注于“解析 + 清洗 + 导出图表所需 CSV”，避免耦合过时日志格式。

输入：
- 默认读取 output_large/manifest.csv

输出：
- output_large/parsed/all_results.csv
- output_large/parsed/architecture_comparison.csv
- output_large/parsed/capacity_sweep.csv
- output_large/parsed/lookahead_sweep.csv
- output_large/parsed/multi_zone.csv
- output_large/parsed/summary.md

说明：
- 若后续需要画图，建议在 notebook / pandas / matplotlib 中直接读取这些 CSV。
- 为了兼容当前 run.py 的 SUMMARY 输出，本脚本只依赖 manifest.csv 中的统一字段。
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


INPUT_MANIFEST = Path("output_large") / "manifest.csv"
PARSED_DIR = Path("output_large") / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 基础工具
# ============================================================
def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input csv not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))



def write_csv(rows: List[Dict], path: Path) -> None:
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



def write_markdown(rows: List[Dict], path: Path) -> None:
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



def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None



def to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None



def safe_log10(x: Optional[float]) -> Optional[float]:
    if x is None or x <= 0:
        return None
    return math.log10(x)



def group_by(rows: Iterable[Dict], keys: List[str]) -> Dict[tuple, List[Dict]]:
    out: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        out.setdefault(key, []).append(row)
    return out


# ============================================================
# 行标准化
# ============================================================
def normalize_rows(rows: List[Dict[str, str]]) -> List[Dict]:
    normalized: List[Dict] = []

    for row in rows:
        nr = dict(row)
        nr["return_code"] = to_int(row.get("return_code"))
        nr["ions_per_region"] = to_int(row.get("ions_per_region"))
        nr["trap_capacity"] = to_int(row.get("trap_capacity"))
        nr["lookahead_k"] = to_int(row.get("lookahead_k"))
        nr["swap_threshold"] = to_int(row.get("swap_threshold"))
        nr["max_qubits_per_qccd"] = to_int(row.get("max_qubits_per_qccd"))
        nr["num_optical_zones"] = to_int(row.get("num_optical_zones"))
        nr["enable_swap_insert"] = to_int(row.get("enable_swap_insert"))

        nr["wallclock_s"] = to_float(row.get("wallclock_s"))
        nr["scheduler_build_s"] = to_float(row.get("scheduler_build_s"))
        nr["execution_time_us"] = to_float(row.get("execution_time_us"))
        nr["fidelity"] = to_float(row.get("fidelity"))
        nr["total_shuttle"] = to_float(row.get("total_shuttle"))
        nr["log10_fidelity"] = safe_log10(nr["fidelity"])

        # scale 推断：便于后续 architecture comparison 切 medium / large
        bench = str(row.get("benchmark", "")).lower()
        if "256" in bench or "299" in bench:
            nr["scale_bucket"] = "large"
        elif "128" in bench or "117" in bench:
            nr["scale_bucket"] = "medium"
        else:
            nr["scale_bucket"] = "other"

        normalized.append(nr)

    return normalized


# ============================================================
# 套件级导出
# ============================================================
def export_all_results(rows: List[Dict]) -> None:
    write_csv(rows, PARSED_DIR / "all_results.csv")



def export_architecture_comparison(rows: List[Dict]) -> List[Dict]:
    subset = [r for r in rows if r.get("suite") == "architecture_comparison" and r.get("return_code") == 0]
    subset.sort(key=lambda r: (r.get("scale_bucket", ""), r.get("benchmark", ""), r.get("machine", "")))
    write_csv(subset, PARSED_DIR / "architecture_comparison.csv")
    return subset



def export_capacity_sweep(rows: List[Dict]) -> List[Dict]:
    subset = [r for r in rows if r.get("suite") == "capacity_sweep" and r.get("return_code") == 0]
    subset.sort(key=lambda r: (r.get("benchmark", ""), r.get("trap_capacity") or -1))
    write_csv(subset, PARSED_DIR / "capacity_sweep.csv")
    return subset



def export_lookahead_sweep(rows: List[Dict]) -> List[Dict]:
    subset = [r for r in rows if r.get("suite") == "lookahead_sweep" and r.get("return_code") == 0]
    subset.sort(key=lambda r: (r.get("benchmark", ""), r.get("lookahead_k") or -1))
    write_csv(subset, PARSED_DIR / "lookahead_sweep.csv")
    return subset



def export_multi_zone(rows: List[Dict]) -> List[Dict]:
    subset = [r for r in rows if r.get("suite") == "multi_zone" and r.get("return_code") == 0]
    subset.sort(key=lambda r: (r.get("benchmark", ""), r.get("num_optical_zones") or -1))
    write_csv(subset, PARSED_DIR / "multi_zone.csv")
    return subset


# ============================================================
# 聚合摘要
# ============================================================
def summarize_suite(rows: List[Dict], suite_name: str, pivot_key: str) -> List[Dict]:
    subset = [r for r in rows if r.get("suite") == suite_name and r.get("return_code") == 0]
    grouped = group_by(subset, [pivot_key])

    out: List[Dict] = []
    for (pivot_value,), items in sorted(grouped.items(), key=lambda x: str(x[0][0])):
        fidelity_vals = [r["fidelity"] for r in items if r.get("fidelity") is not None]
        exec_vals = [r["execution_time_us"] for r in items if r.get("execution_time_us") is not None]
        shuttle_vals = [r["total_shuttle"] for r in items if r.get("total_shuttle") is not None]

        out.append(
            {
                "suite": suite_name,
                pivot_key: pivot_value,
                "num_jobs": len(items),
                "avg_fidelity": "" if not fidelity_vals else f"{statistics.mean(fidelity_vals):.8e}",
                "avg_log10_fidelity": "" if not fidelity_vals else f"{statistics.mean(safe_log10(v) for v in fidelity_vals if v and v > 0):.6f}",
                "avg_execution_time_us": "" if not exec_vals else f"{statistics.mean(exec_vals):.3f}",
                "avg_total_shuttle": "" if not shuttle_vals else f"{statistics.mean(shuttle_vals):.3f}",
            }
        )
    return out



def export_summary_markdown(rows: List[Dict]) -> None:
    summary_rows: List[Dict] = []
    summary_rows.extend(summarize_suite(rows, "architecture_comparison", "machine"))
    summary_rows.extend(summarize_suite(rows, "capacity_sweep", "trap_capacity"))
    summary_rows.extend(summarize_suite(rows, "lookahead_sweep", "lookahead_k"))
    summary_rows.extend(summarize_suite(rows, "multi_zone", "num_optical_zones"))
    write_markdown(summary_rows, PARSED_DIR / "summary.md")


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_MANIFEST
    rows = normalize_rows(read_csv(manifest_path))

    export_all_results(rows)
    export_architecture_comparison(rows)
    export_capacity_sweep(rows)
    export_lookahead_sweep(rows)
    export_multi_zone(rows)
    export_summary_markdown(rows)

    print(f"[OK] Parsed manifest: {manifest_path}")
    print(f"[OK] Output directory: {PARSED_DIR}")
    print("[OK] Generated:")
    print(f"  - {PARSED_DIR / 'all_results.csv'}")
    print(f"  - {PARSED_DIR / 'architecture_comparison.csv'}")
    print(f"  - {PARSED_DIR / 'capacity_sweep.csv'}")
    print(f"  - {PARSED_DIR / 'lookahead_sweep.csv'}")
    print(f"  - {PARSED_DIR / 'multi_zone.csv'}")
    print(f"  - {PARSED_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
