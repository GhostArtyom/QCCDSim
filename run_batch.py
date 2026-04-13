import os
import csv
import subprocess as sp
from pathlib import Path

# ============================================================
# 批量运行脚本（适配当前 run.py 参数）
# - 支持 MUSS V2/V3/V4 与 EJF
# - 支持 PAPER / EXTENDED analyzer
# - 默认配置为更贴近论文 Table 2 的路线：SABRE + MUSS V2 + PAPER
# ============================================================

PATH = Path("./programs")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 选择电路
# ============================================================
PROG = ["BV32", "GHZ32", "ADDER32", "QAOA32", "qft32_swap", "SQRT30"]
#PROG = ["QAOA32H"]

# ============================================================
# 选择架构与容量（对应论文 Table 2）
# ============================================================
MACHINE = {
    "G2x2": "12",
    "G2x3": "8",
}

# ============================================================
# 映射方式
# SABRE mapper 与 MUSS 调度器配合最佳
# ============================================================
MAPPER = "SABRE"

# ============================================================
# 链内重排序
# 对 faithful reproduction，建议先用 Naive；
# 在进行文献55的复现的时候，可以改用 Fidelity做链排序，这会导致文献55的效果很好，与论文表2结果不符。
# ============================================================
REORDER = "Naive"

# ============================================================
# 调度器组合
# Table 2 faithful reproduction 建议优先跑 ("MUSS", "V2")
# 
# ============================================================
SCHEDULERS = [
    ("MUSS", "V2"),   # 论文复现版本
    # ("MUSS", "V3"),   # 创新版本，当前在qft，sqrt，qaoa表现不佳
    # ("MUSS", "V4"),  # v3的优化版本 减轻加热的影响
    # ("MUSS", "V5"),  # v2的基础上，去掉rebalance的版本
    ("MUSS", "V6"),  # v2的基础上，去掉rebalance的版本,更收紧论文版本
    # ("EJF",  ""),   # 文献55版本 (EJF 与 PO mapper 有兼容性问题，暂时禁用)
]

# ============================================================
# 物理门 / swap 模型
# ============================================================
GATE_TYPE = "FM"
SWAP_TYPE = "PaperSwapDirect"

# ============================================================
# Analyzer 模式
# PAPER: 更贴论文 Table 2 口径
# EXTENDED: 增加了bi以及平均热链等操作的拟合版本
# ============================================================
ANALYZER_MODE = "PAPER"

# ============================================================
# 资源串行化开关
# 说明：run.py 的三个参数分别是
#   serial_trap_ops, serial_comm, serial_all
# 1均为串行，0为并行，根据论文理解，这里先都设置为串行，可以修改串并行组合来看对结果的影响
# 若要进一步贴近论文，可单独再做一组灵敏度对比。
# ============================================================
SERIAL_TRAP_OPS = "1"
SERIAL_COMM = "1"
SERIAL_ALL = "1"


# ============================================================
# 可选：是否在日志名中加入 analyzer/reorder，方便区分
# ============================================================
def build_log_name(prog: str, machine: str, ions: str, family: str, version: str) -> Path:
    parts = [
        prog,
        machine,
        ions,
        MAPPER,
        REORDER,
        family,
        version if version else "BASE",
        GATE_TYPE,
        SWAP_TYPE,
        ANALYZER_MODE,
    ]
    safe_name = "_".join(parts) + ".log"
    return OUTPUT_DIR / safe_name


# ============================================================
# 组装命令
# ============================================================
def build_args(prog: str, machine: str, ions: str, family: str, version: str) -> list[str]:
    qasm_path = PATH / f"{prog}.qasm"

    args = [
        "python",
        "run.py",
        str(qasm_path),
        machine,
        ions,
        MAPPER,
        REORDER,
        SERIAL_TRAP_OPS,
        SERIAL_COMM,
        SERIAL_ALL,
        GATE_TYPE,
        SWAP_TYPE,
    ]

    if family.upper() == "MUSS":
        args.extend([family, version if version else "V2", ANALYZER_MODE])
    else:
        # EJF 等非 MUSS 路径：run.py 仍接受 family，version 可省，
        # analyzer_mode 放在第 13 个参数位，因此补一个空版本占位更稳妥。
        args.extend([family, "", ANALYZER_MODE])

    return args



def parse_summary_line_from_log(log_path: Path) -> dict | None:
    """
    从单个日志文件中提取 run.py 末尾打印的 SUMMARY 行。

    期望格式：
      SUMMARY|program=BV32|machine=G2x2|version=V6|mapper=SABRE2|total_shuttle=3|execution_time_us=1831|fidelity=0.827...

    返回：
      {
        "程序": ...,
        "架构": ...,
        "版本": ...,
        "映射": ...,
        "穿梭次数": ...,
        "执行时间 (μs)": ...,
        "保真度": ...,
      }

    若日志中不存在 SUMMARY 行，则返回 None。
    """
    if not log_path.exists():
        return None

    summary_line = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SUMMARY|"):
                summary_line = line

    if summary_line is None:
        return None

    parts = summary_line.split("|")[1:]
    kv = {}
    for item in parts:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        kv[k] = v

    return {
        "程序": kv.get("program", ""),
        "架构": kv.get("machine", ""),
        "版本": kv.get("version", ""),
        "映射": kv.get("mapper", ""),
        "穿梭次数": kv.get("total_shuttle", ""),
        "执行时间 (μs)": kv.get("execution_time_us", ""),
        "保真度": kv.get("fidelity", ""),
    }


def write_summary_markdown(rows: list[dict], out_path: Path) -> None:
    """
    将汇总结果写成 Markdown 表格，便于直接查看或粘贴到文档中。
    """
    headers = ["程序", "架构", "版本", "映射", "穿梭次数", "执行时间 (μs)", "保真度"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """
    将汇总结果写成 CSV，方便后续用 Excel / pandas 继续处理。
    """
    headers = ["程序", "架构", "版本", "映射", "穿梭次数", "执行时间 (μs)", "保真度"]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



# ============================================================
# 运行
# ============================================================
def main() -> None:
    summary_rows = []

    print("=" * 72)
    print("Batch run configuration")
    print(f"Programs       : {PROG}")
    print(f"Machines       : {MACHINE}")
    print(f"Mapper         : {MAPPER}")
    print(f"Reorder        : {REORDER}")
    print(f"Schedulers     : {SCHEDULERS}")
    print(f"GateType       : {GATE_TYPE}")
    print(f"SwapType       : {SWAP_TYPE}")
    print(f"AnalyzerMode   : {ANALYZER_MODE}")
    print(
        f"Serial flags   : trap={SERIAL_TRAP_OPS}, comm={SERIAL_COMM}, all={SERIAL_ALL}"
    )
    print("=" * 72)

    for prog in PROG:
        for machine, ions in MACHINE.items():
            for family, version in SCHEDULERS:
                args = build_args(prog, machine, ions, family, version)
                log_path = build_log_name(prog, machine, ions, family, version)

                print(f"Running: {' '.join(args)}")
                print(f"Log    : {log_path}")

                with open(log_path, "w", encoding="utf-8") as f:
                    ret = sp.call(args, stdout=f, stderr=sp.STDOUT)

                if ret != 0:
                    print(f"[WARN] Process exited with code {ret}: {log_path}")
                else:
                    print(f"[OK]   Finished: {log_path}")

                    row = parse_summary_line_from_log(log_path)
                    if row is None:
                        print(f"[WARN] No SUMMARY line found in: {log_path}")
                    else:
                        summary_rows.append(row)

    summary_md_path = OUTPUT_DIR / "summary.md"
    summary_csv_path = OUTPUT_DIR / "summary.csv"

    write_summary_markdown(summary_rows, summary_md_path)
    write_summary_csv(summary_rows, summary_csv_path)

    print(f"Summary markdown: {summary_md_path}")
    print(f"Summary csv     : {summary_csv_path}")
    print("All jobs finished.")


if __name__ == "__main__":
    main()
