import os
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
# 当前对标的是 SABRE2，因此这里固定走 SABRE，
# 再由 run.py 依据 MUSS V2 自动选择 QubitMapSABRE2。
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
    #("MUSS", "V2"),   #论文复现版本
    ("MUSS", "V3"),   #创新版本，当前在qft，sqrt，qaoa表现不加。
    # ("MUSS", "V4"),  # v3的优化版本
    # ("EJF",  ""),   #文献55版本
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


# ============================================================
# 运行
# ============================================================
def main() -> None:
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

    print("All jobs finished.")


if __name__ == "__main__":
    main()
