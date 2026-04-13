# -*- coding: utf-8 -*-
"""
strict_repro.py
============================================================
严格复现辅助模块。

设计目标：
1) 不改坏原项目默认入口；
2) 当用户显式开启严格复现模式时，统一关闭各类兼容回退；
3) 对论文主线所需的关键参数、拓扑与统计口径做集中校验；
4) 让 run.py / muss_schedule7.py 中的“严格复现判定”可复用、可测试、可维护。

开启方式：
    export MUSS_STRICT_REPRO=1

说明：
- 本模块不会凭空补造论文未公开的实现细节；
- 它做的是“严格守门”：若检测到代码正走兼容路径、缺少关键拓扑、
  缺少 shuttle_id 聚合信息，或者实验配置不满足论文主线要求，就立刻失败；
- 因而它的目标不是“尽量跑起来”，而是“只要条件不满足就拒绝声称严格复现”。
============================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


class StrictReproError(RuntimeError):
    """严格复现模式下的统一异常。"""


@dataclass(frozen=True)
class StrictReproConfig:
    """严格复现模式的配置快照。"""

    enabled: bool = False
    require_v7: bool = True
    require_analyzer_v7: bool = True
    require_aggregate_shuttle: bool = True
    require_explicit_topology: bool = True
    require_large_scale: bool = True
    require_paper_mode_default: bool = False


TRUE_SET = {"1", "true", "yes", "y", "on", "enable", "enabled"}
FALSE_SET = {"0", "false", "no", "n", "off", "disable", "disabled"}


def parse_bool_env(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值；取值不合法时直接报错。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    text = str(raw).strip().lower()
    if text in TRUE_SET:
        return True
    if text in FALSE_SET:
        return False
    raise StrictReproError(f"环境变量 {name} 的取值无法解析为布尔值：{raw!r}")


def load_strict_repro_config() -> StrictReproConfig:
    """从环境变量构造严格复现配置。"""
    enabled = parse_bool_env("MUSS_STRICT_REPRO", False)
    return StrictReproConfig(
        enabled=enabled,
        require_v7=True,
        require_analyzer_v7=True,
        require_aggregate_shuttle=True,
        require_explicit_topology=True,
        require_large_scale=True,
        require_paper_mode_default=parse_bool_env("MUSS_STRICT_REQUIRE_PAPER_MODE", False),
    )


def ensure(condition: bool, message: str) -> None:
    """严格复现模式下的断言辅助函数。"""
    if not condition:
        raise StrictReproError(message)


def ensure_no_fallback(strict_cfg: StrictReproConfig, *, actual: str, expected: str, context: str) -> None:
    """用于禁止“实际走了兼容回退路径”的情况。"""
    if not strict_cfg.enabled:
        return
    if actual != expected:
        raise StrictReproError(f"严格复现模式禁止回退：{context} 要求 {expected}，实际为 {actual}")


def validate_large_env_values(effective_env: Dict[str, object], strict_cfg: StrictReproConfig) -> None:
    """校验 large-scale 关键参数的取值范围。"""
    if not strict_cfg.enabled:
        return

    def _read_int(key: str, *, allow_none: bool = False, minimum: Optional[int] = None) -> Optional[int]:
        val = effective_env.get(key)
        if val is None:
            if allow_none:
                return None
            raise StrictReproError(f"严格复现模式要求提供参数 {key}")
        try:
            iv = int(val)
        except Exception as exc:
            raise StrictReproError(f"参数 {key} 无法解析为整数：{val!r}") from exc
        if minimum is not None and iv < minimum:
            raise StrictReproError(f"参数 {key} 取值过小：{iv} < {minimum}")
        return iv

    _read_int("max_qubits_per_qccd", minimum=1)
    _read_int("lookahead_k", minimum=1)
    _read_int("swap_threshold", minimum=0)
    _read_int("num_optical_zones", minimum=1)
    trap_capacity = _read_int("trap_capacity", allow_none=False, minimum=1)
    if trap_capacity is None:
        raise StrictReproError("严格复现模式要求明确 trap_capacity，不能依赖历史默认值")


def validate_runtime_mode(
    strict_cfg: StrictReproConfig,
    *,
    architecture_scale: str,
    sched_family: str,
    sched_version: str,
    analyzer_mode: str,
) -> None:
    """校验运行模式是否落在论文主线路径。"""
    if not strict_cfg.enabled:
        return

    arch = str(architecture_scale).upper()
    fam = str(sched_family).upper()
    ver = str(sched_version).upper()
    mode = str(analyzer_mode).upper()

    if strict_cfg.require_large_scale:
        ensure(arch in {"LARGE", "L", "EML", "EML-QCCD"}, f"严格复现要求 large-scale 路径，当前 architecture_scale={architecture_scale}")
    if strict_cfg.require_v7:
        ensure(fam in {"MUSS", "MUSS-TI", "MUSS_TI_MODE"}, f"严格复现要求 MUSS 调度族，当前 sched_family={sched_family}")
        ensure(ver in {"V7", "7", "MUSS_SCHEDULE7", "LARGE"}, f"严格复现要求 MUSS V7，当前 sched_version={sched_version}")
    if strict_cfg.require_paper_mode_default:
        ensure(mode in {"PAPER", "TABLE2", "P"}, f"严格复现要求论文分析口径，当前 analyzer_mode={analyzer_mode}")


def _qccd_nodes(machine) -> Iterable[int]:
    if hasattr(machine, "qccd_graph") and machine.qccd_graph is not None:
        try:
            return list(machine.qccd_graph.nodes())
        except Exception:
            pass
    if hasattr(machine, "module_count"):
        try:
            return list(range(int(machine.module_count)))
        except Exception:
            pass
    raise StrictReproError("严格复现要求机器显式暴露 qccd_graph 或 module_count")


def _trap_ids(machine):
    return [int(t.id) for t in getattr(machine, "traps", [])]


def _qccd_of_trap(machine, trap_id: int) -> int:
    if hasattr(machine, "get_qccd_id_by_trap"):
        return int(machine.get_qccd_id_by_trap(int(trap_id)))
    trap = machine.get_trap(int(trap_id))
    # 兼容当前仓库的 Trap 元数据字段命名：优先 qccd_id，其次 module_id。
    if hasattr(trap, "qccd_id"):
        return int(trap.qccd_id)
    if hasattr(trap, "module_id"):
        return int(trap.module_id)
    raise StrictReproError("严格复现要求 Machine 能通过 trap 唯一确定所属 QCCD")


def _zone_type_of_trap(machine, trap_id: int) -> str:
    if hasattr(machine, "get_trap_zone_type"):
        return str(machine.get_trap_zone_type(int(trap_id)))
    trap = machine.get_trap(int(trap_id))
    if hasattr(trap, "zone_type"):
        return str(trap.zone_type)
    raise StrictReproError("严格复现要求 Machine/Trap 显式暴露 zone_type")


def _optical_traps_of_qccd(machine, qccd_id: int):
    if hasattr(machine, "get_qccd_optical_traps"):
        return [int(t.id) for t in machine.get_qccd_optical_traps(int(qccd_id))]
    out = []
    for tid in _trap_ids(machine):
        if _qccd_of_trap(machine, tid) == int(qccd_id) and _zone_type_of_trap(machine, tid) == "optical":
            out.append(int(tid))
    return sorted(out)


def _has_registered_fiber_link(machine, qccd_a: int, qccd_b: int, trap_a: int, trap_b: int) -> bool:
    if hasattr(machine, "get_fiber_links_between"):
        try:
            for link in machine.get_fiber_links_between(int(qccd_a), int(qccd_b)):
                src_trap = int(getattr(link, "src_trap", getattr(link, "src_trap_id", -1)))
                dst_trap = int(getattr(link, "dst_trap", getattr(link, "dst_trap_id", -1)))
                if (src_trap == int(trap_a) and dst_trap == int(trap_b)) or (src_trap == int(trap_b) and dst_trap == int(trap_a)):
                    return True
        except Exception:
            pass
    if hasattr(machine, "get_fiber_link"):
        try:
            link = machine.get_fiber_link(int(qccd_a), int(qccd_b))
            if link is not None:
                src_trap = int(getattr(link, "src_trap", getattr(link, "src_trap_id", -1)))
                dst_trap = int(getattr(link, "dst_trap", getattr(link, "dst_trap_id", -1)))
                if (src_trap == int(trap_a) and dst_trap == int(trap_b)) or (src_trap == int(trap_b) and dst_trap == int(trap_a)):
                    return True
        except Exception:
            pass
    return False


def validate_machine_topology(machine, *, expected_optical_zones: Optional[int], strict_cfg: StrictReproConfig) -> None:
    """
    校验 EML/QCCD 拓扑是否完整。

    检查项：
    1) 每个 trap 都能追踪到 module/qccd；
    2) 每个 trap 都有明确 zone_type；
    3) 每个 qccd 的 optical zone 数量与期望一致；
    4) 任意两个 qccd 之间，任意一对 optical trap 都必须有显式登记的 fiber link。
    """
    if not strict_cfg.enabled or not strict_cfg.require_explicit_topology:
        return

    nodes = list(_qccd_nodes(machine))
    ensure(len(nodes) > 0, "严格复现要求机器至少包含一个 QCCD 模块")

    for tid in _trap_ids(machine):
        _ = _qccd_of_trap(machine, tid)
        zone = _zone_type_of_trap(machine, tid)
        ensure(zone in {"storage", "operation", "optical"}, f"trap {tid} 的 zone_type 非法：{zone}")

    for qccd_id in nodes:
        opticals = _optical_traps_of_qccd(machine, qccd_id)
        ensure(len(opticals) > 0, f"QCCD {qccd_id} 缺少 optical trap")
        if expected_optical_zones is not None:
            ensure(
                len(opticals) == int(expected_optical_zones),
                f"QCCD {qccd_id} 的 optical trap 数量为 {len(opticals)}，与期望 {expected_optical_zones} 不一致",
            )

    for i, qa in enumerate(nodes):
        opt_a = _optical_traps_of_qccd(machine, qa)
        for qb in nodes[i + 1 :]:
            opt_b = _optical_traps_of_qccd(machine, qb)
            for ta in opt_a:
                for tb in opt_b:
                    ensure(
                        _has_registered_fiber_link(machine, qa, qb, ta, tb),
                        f"模块 {qa} 与 {qb} 之间缺少显式 fiber link：trap {ta} <-> trap {tb}",
                    )


def validate_analyzer_requirements(strict_cfg: StrictReproConfig, *, use_aggregate: bool, using_v7_analyzer: bool) -> None:
    """确保 analyzer 没有退化到兼容口径。"""
    if not strict_cfg.enabled:
        return
    if strict_cfg.require_analyzer_v7:
        ensure(using_v7_analyzer, "严格复现要求使用 analyzer_v7，禁止回退到 legacy analyzer.py")
    if strict_cfg.require_aggregate_shuttle:
        ensure(use_aggregate, "严格复现要求所有 shuttle 都带 shuttle_id，并使用 aggregate 口径")


def validate_summary_payload(strict_cfg: StrictReproConfig, summary_fields: Dict[str, object]) -> None:
    """在输出 SUMMARY 前再次检查关键字段是否齐全。"""
    if not strict_cfg.enabled:
        return
    required = [
        "trap_capacity",
        "lookahead_k",
        "swap_threshold",
        "num_optical_zones",
        "max_qubits_per_qccd",
        "enable_swap_insert",
        "execution_time_us",
        "fidelity",
        "total_shuttle",
    ]
    for key in required:
        if key not in summary_fields or summary_fields.get(key) is None:
            raise StrictReproError(f"严格复现要求 SUMMARY 中包含字段 {key}")
