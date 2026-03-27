# -*- coding: utf-8 -*-
"""
test.py
============================================================
SQRT299 + EML 定点插桩验证脚本（仅针对 MUSS V7 large 路径）

用途：
1) 只对以下热点函数做定点计时与调用统计：
   - _ensure_space_on_trap_large
   - _schedule_cross_qccd_gate
   - _execute_cross_qccd_swap
   - _maybe_insert_swap_after_cross_qccd_gate
2) 若当前代码中仍然存在 _copy_planning_state，则额外统计它的耗时与调用次数，
   用于验证瓶颈是否仍停留在“候选复制”。
3) 直接复用现有 run.py 的真实执行路径，不改论文参数语义。

推荐运行方式：
    python test.py --qasm programs/SQRT299.qasm

如果你的 qasm 不在 programs/ 下，也可以显式指定：
    python test.py --qasm /你的路径/SQRT299.qasm

默认参数严格对应你当前要测的 case：
- machine = EML
- trap_capacity = 16
- lookahead_k = 8
- swap_threshold = 4
- num_optical_zones = 1
- mapper = SABRE
- sched_family = MUSS
- sched_version = 7
- architecture_scale = LARGE

输出：
1) 正常打印 run.py 原始输出
2) 进程结束时额外打印：
   - INSTRUMENT_SUMMARY
   - 每个目标函数的调用次数 / 总耗时 / 平均耗时 / 最大单次耗时 / 占总运行时间比例
   - 若存在 _copy_planning_state，也会额外打印它
3) 同时将 JSON 报告写到：
   instrumentation_report.json

说明：
- 这是“定点插桩”，不是 profiler；目的就是精确回答你当前这 4 个函数是不是主瓶颈。
- 不改调度语义，不改论文技术路径，只是在外层做 monkey patch 计时。
============================================================
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# 统计器
# ------------------------------------------------------------

class CallStats:
    """单个函数的计时统计。"""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.total_s = 0.0
        self.max_s = 0.0
        self.min_s = None
        self.errors = 0
        self.samples: List[float] = []

    def add(self, dt: float, is_error: bool = False) -> None:
        self.calls += 1
        self.total_s += dt
        if dt > self.max_s:
            self.max_s = dt
        if self.min_s is None or dt < self.min_s:
            self.min_s = dt
        if is_error:
            self.errors += 1
        # 为了避免极端大样本占内存，只保留前 1000 个样本用于简单观察
        if len(self.samples) < 1000:
            self.samples.append(dt)

    @property
    def avg_s(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_s / self.calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "total_s": self.total_s,
            "avg_s": self.avg_s,
            "max_s": self.max_s,
            "min_s": self.min_s if self.min_s is not None else 0.0,
            "errors": self.errors,
        }


class InstrumentationRegistry:
    """全局统计注册表。"""

    def __init__(self):
        self.stats: Dict[str, CallStats] = {}
        self.run_started_at = time.perf_counter()
        self.run_finished_at: Optional[float] = None

    def get(self, name: str) -> CallStats:
        if name not in self.stats:
            self.stats[name] = CallStats(name)
        return self.stats[name]

    def finish(self) -> None:
        if self.run_finished_at is None:
            self.run_finished_at = time.perf_counter()

    @property
    def wall_time_s(self) -> float:
        end = self.run_finished_at if self.run_finished_at is not None else time.perf_counter()
        return end - self.run_started_at

    def report_dict(self) -> Dict[str, Any]:
        self.finish()
        funcs = []
        for _, st in sorted(self.stats.items(), key=lambda kv: kv[1].total_s, reverse=True):
            item = st.to_dict()
            item["share_of_total_wall_time"] = (
                item["total_s"] / self.wall_time_s if self.wall_time_s > 0 else 0.0
            )
            funcs.append(item)
        return {
            "total_wall_time_s": self.wall_time_s,
            "functions": funcs,
        }


REG = InstrumentationRegistry()


# ------------------------------------------------------------
# 包装器
# ------------------------------------------------------------

def make_timed_wrapper(cls, func_name: str):
    """
    给类方法打点。
    这里不改函数返回值，不吞异常，只做计时。
    """
    if not hasattr(cls, func_name):
        return False

    original = getattr(cls, func_name)
    if getattr(original, "__instrumented_by_test_py__", False):
        return True

    def wrapped(self, *args, **kwargs):
        start = time.perf_counter()
        is_error = False
        try:
            return original(self, *args, **kwargs)
        except Exception:
            is_error = True
            raise
        finally:
            dt = time.perf_counter() - start
            REG.get(func_name).add(dt, is_error=is_error)

    wrapped.__name__ = getattr(original, "__name__", func_name)
    wrapped.__doc__ = getattr(original, "__doc__", "")
    wrapped.__instrumented_by_test_py__ = True
    setattr(cls, func_name, wrapped)
    return True


# ------------------------------------------------------------
# 报告输出
# ------------------------------------------------------------

def dump_report(json_path: Path) -> None:
    REG.finish()
    report = REG.report_dict()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    total_wall = report["total_wall_time_s"]

    print("\n" + "=" * 100)
    print("INSTRUMENT_SUMMARY|SQRT299+EML 定点插桩结果")
    print(f"TOTAL_WALL_TIME_S|{total_wall:.6f}")
    print("-" * 100)
    print(
        "FUNCTION"
        .ljust(42)
        + "CALLS".rjust(10)
        + "TOTAL_S".rjust(14)
        + "AVG_MS".rjust(14)
        + "MAX_MS".rjust(14)
        + "ERR".rjust(8)
        + "SHARE".rjust(12)
    )
    print("-" * 100)

    for item in report["functions"]:
        print(
            f"{item['name']:<42}"
            f"{item['calls']:>10d}"
            f"{item['total_s']:>14.6f}"
            f"{item['avg_s'] * 1000:>14.3f}"
            f"{item['max_s'] * 1000:>14.3f}"
            f"{item['errors']:>8d}"
            f"{item['share_of_total_wall_time'] * 100:>11.2f}%"
        )

    print("-" * 100)
    print(f"JSON_REPORT|{json_path.resolve()}")
    print("=" * 100)

    # 额外给一个直接可读的判断提示
    heavy_names = {x["name"]: x for x in report["functions"]}
    cps = heavy_names.get("_copy_planning_state")
    if cps is not None:
        print(
            f"[判读提示] 检测到 _copy_planning_state：calls={cps['calls']}, "
            f"total_s={cps['total_s']:.6f}, share={cps['share_of_total_wall_time'] * 100:.2f}%"
        )
        if cps["share_of_total_wall_time"] > 0.10:
            print("[判读提示] _copy_planning_state 仍占比较高，说明“候选复制”仍可能是瓶颈。")
        else:
            print("[判读提示] _copy_planning_state 占比已不高，瓶颈更可能转向论文允许范围内的 routing/look-ahead。")
    else:
        print("[判读提示] 当前代码中未检测到 _copy_planning_state，说明候选复制路径已基本被移除。")

    key_targets = [
        "_ensure_space_on_trap_large",
        "_schedule_cross_qccd_gate",
        "_execute_cross_qccd_swap",
        "_maybe_insert_swap_after_cross_qccd_gate",
    ]
    print("[判读提示] 四个目标函数总占比：")
    total_share = 0.0
    for name in key_targets:
        it = heavy_names.get(name)
        if it is None:
            print(f"  - {name}: 未命中（可能该 workload 中未走到，或函数名已变化）")
            continue
        share = it["share_of_total_wall_time"] * 100
        total_share += share
        print(f"  - {name}: {share:.2f}%")
    print(f"  => 四者合计占总运行时间约 {total_share:.2f}%")
    print()


# ------------------------------------------------------------
# 主逻辑
# ------------------------------------------------------------

def find_default_qasm() -> Optional[Path]:
    """
    尝试在常见位置寻找 SQRT299.qasm
    """
    candidates = [
        Path("programs/SQRT299.qasm"),
        Path("programs/sqrt299.qasm"),
        Path("SQRT299.qasm"),
        Path("sqrt299.qasm"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQRT299 + EML 定点插桩验证脚本"
    )
    parser.add_argument(
        "--qasm",
        type=str,
        default=None,
        help="SQRT299.qasm 路径。默认自动在常见位置查找。",
    )
    parser.add_argument("--machine", type=str, default="EML")
    parser.add_argument("--ions-per-region", type=str, default="16")
    parser.add_argument("--mapper", type=str, default="SABRE")
    parser.add_argument("--reorder", type=str, default="na")
    parser.add_argument("--serial-trap-ops", type=str, default="0")
    parser.add_argument("--serial-comm", type=str, default="0")
    parser.add_argument("--serial-all", type=str, default="0")
    parser.add_argument("--gate-type", type=str, default="FM")
    parser.add_argument("--swap-type", type=str, default="GateSwap")
    parser.add_argument("--sched-family", type=str, default="MUSS")
    parser.add_argument("--sched-version", type=str, default="7")
    parser.add_argument("--analyzer-mode", type=str, default="paper")
    parser.add_argument("--architecture-scale", type=str, default="LARGE")

    parser.add_argument("--trap-capacity", type=str, default="16")
    parser.add_argument("--lookahead-k", type=str, default="8")
    parser.add_argument("--swap-threshold", type=str, default="4")
    parser.add_argument("--max-qubits-per-qccd", type=str, default="32")
    parser.add_argument("--num-optical-zones", type=str, default="1")
    parser.add_argument("--enable-swap-insert", type=str, default="1")

    parser.add_argument(
        "--json-out",
        type=str,
        default="instrumentation_report.json",
        help="插桩结果 JSON 输出文件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    qasm_path: Optional[Path]
    if args.qasm is None:
        qasm_path = find_default_qasm()
        if qasm_path is None:
            raise FileNotFoundError(
                "没有找到默认的 SQRT299.qasm，请显式指定：python test.py --qasm /path/to/SQRT299.qasm"
            )
    else:
        qasm_path = Path(args.qasm).resolve()
        if not qasm_path.exists():
            raise FileNotFoundError(f"QASM 文件不存在: {qasm_path}")

    # 设置与论文复现实验一致的 large 参数环境变量
    os.environ["MUSS_TRAP_CAPACITY"] = str(args.trap_capacity)
    os.environ["MUSS_SWAP_LOOKAHEAD_K"] = str(args.lookahead_k)
    os.environ["MUSS_SWAP_THRESHOLD"] = str(args.swap_threshold)
    os.environ["MUSS_MAX_QUBITS_PER_QCCD"] = str(args.max_qubits_per_qccd)
    os.environ["MUSS_NUM_OPTICAL_ZONES"] = str(args.num_optical_zones)
    os.environ["MUSS_ENABLE_SWAP_INSERT"] = str(args.enable_swap_insert)

    # 先导入 muss_schedule7，再打 monkey patch
    import muss_schedule7

    target_cls = muss_schedule7.MUSSSchedule

    target_functions = [
        "_ensure_space_on_trap_large",
        "_schedule_cross_qccd_gate",
        "_execute_cross_qccd_swap",
        "_maybe_insert_swap_after_cross_qccd_gate",
    ]

    optional_functions = [
        "_copy_planning_state",   # 若还残留候选复制，会自动统计
    ]

    patched = []
    for fn in target_functions + optional_functions:
        ok = make_timed_wrapper(target_cls, fn)
        if ok:
            patched.append(fn)

    print("=" * 100)
    print("[TEST] SQRT299 + EML 定点插桩开始")
    print(f"[TEST] qasm={qasm_path}")
    print(f"[TEST] patched={patched}")
    print(f"[TEST] trap_capacity={os.environ['MUSS_TRAP_CAPACITY']}")
    print(f"[TEST] lookahead_k={os.environ['MUSS_SWAP_LOOKAHEAD_K']}")
    print(f"[TEST] swap_threshold={os.environ['MUSS_SWAP_THRESHOLD']}")
    print(f"[TEST] num_optical_zones={os.environ['MUSS_NUM_OPTICAL_ZONES']}")
    print(f"[TEST] enable_swap_insert={os.environ['MUSS_ENABLE_SWAP_INSERT']}")
    print("=" * 100)

    json_out = Path(args.json_out).resolve()
    atexit.register(dump_report, json_out)

    # 构造 run.py 兼容参数
    run_argv = [
        "run.py",
        str(qasm_path),
        args.machine,
        args.ions_per_region,
        args.mapper,
        args.reorder,
        args.serial_trap_ops,
        args.serial_comm,
        args.serial_all,
        args.gate_type,
        args.swap_type,
        args.sched_family,
        args.sched_version,
        args.analyzer_mode,
        args.architecture_scale,
    ]

    # 直接走现有 run.py 的真实入口
    old_argv = sys.argv[:]
    try:
        sys.argv = run_argv
        runpy.run_path("run.py", run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
