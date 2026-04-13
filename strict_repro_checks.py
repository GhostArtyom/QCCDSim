# -*- coding: utf-8 -*-
"""
strict_repro_checks.py
============================================================
严格复现验收辅助脚本。

用途：
1) 读取 run.py 产生的日志/SUMMARY 行；
2) 检查关键字段是否齐全；
3) 检查是否出现兼容回退关键词；
4) 作为论文复现验收时的自动化守门脚本。

使用示例：
    python strict_repro_checks.py output_large/logs/example.log
============================================================
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_FALLBACK_PATTERNS = [
    r"fallback to PAPER",
    r"Fallback to EJF",
    r"compatibility fallback",
    r"Using legacy analyzer\.py",
]

REQUIRED_SUMMARY_KEYS = {
    "trap_capacity",
    "lookahead_k",
    "swap_threshold",
    "num_optical_zones",
    "max_qubits_per_qccd",
    "enable_swap_insert",
    "execution_time_us",
    "fidelity",
    "total_shuttle",
}


def parse_summary_line(text: str):
    for line in text.splitlines():
        if not line.startswith("SUMMARY|"):
            continue
        fields = {}
        for token in line.split("|")[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        return fields
    raise RuntimeError("未在日志中找到 SUMMARY 行")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("用法: python strict_repro_checks.py <log_file>")
        return 2

    path = Path(argv[1])
    text = path.read_text(encoding="utf-8", errors="replace")

    for pat in FORBIDDEN_FALLBACK_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            raise RuntimeError(f"日志中发现禁止的兼容回退痕迹：{pat}")

    summary = parse_summary_line(text)
    missing = sorted(REQUIRED_SUMMARY_KEYS - set(summary))
    if missing:
        raise RuntimeError(f"SUMMARY 缺少关键字段：{missing}")

    print("严格复现日志检查通过。")
    print("关键字段：")
    for key in sorted(REQUIRED_SUMMARY_KEYS):
        print(f"  {key} = {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
