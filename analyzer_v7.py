import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from schedule import Schedule


@dataclass
class AnalyzerV7Knobs:
    """
    V7 专用论文口径分析配置。

    设计原则：
    1) 默认严格走论文主线；
    2) 同时兼容 small-scale 与 large-scale；
    3) 与旧 analyzer 隔离，不影响其他调度器；
    4) 针对 V7 额外统计 fiber gate / SWAP insertion / zone 行为。

    说明：
    - alpha_bg 默认 0.0，对应最严格论文主线，不保留长期背景热记忆；
    - shuttle fidelity 默认使用论文式聚合：F = exp(-t/T1) - k * nbar；
    - V7 当前 SWAP insertion 实现为 3 个 fiber/MS 类两比特门，分析器按此口径统计。
    """

    alpha_bg: Optional[float] = None
    debug_events: bool = False
    debug_summary: bool = True
    infer_logical_swap_insert_from_triplets: bool = True

    @classmethod
    def paper_mode(cls, debug_summary: bool = True, alpha_bg=None, shuttle_fidelity_mode=None):
        """论文主模式。

        shuttle_fidelity_mode 参数保留给 run.py 做兼容式传参；
        对 V7 来说默认始终走论文聚合口径，因此这里不额外分叉。
        """
        _ = shuttle_fidelity_mode
        return cls(alpha_bg=alpha_bg, debug_events=False, debug_summary=debug_summary)

    @classmethod
    def extended_mode(cls, debug_summary: bool = True, alpha_bg=None, shuttle_fidelity_mode=None):
        """扩展模式：仍保持论文主口径，但打开更多调试输出。"""
        _ = shuttle_fidelity_mode
        return cls(alpha_bg=alpha_bg, debug_events=True, debug_summary=debug_summary)


class AnalyzerV7:
    # ==========================================================
    # 论文/模型默认参数（运行时优先从 machine.mparams 接线）
    # ==========================================================
    T1_US = 600 * 1e6
    K_HEATING = 0.001
    EPSILON_2Q = 1.0 / 25600.0

    FID_1Q = 0.9999
    FID_FIBER = 0.99

    def __init__(self, scheduler_obj, machine_obj, init_mapping, knobs: AnalyzerV7Knobs = None):
        self.scheduler = scheduler_obj
        self.schedule = scheduler_obj.schedule
        self.machine = machine_obj
        self.init_map = getattr(scheduler_obj, "init_map", init_mapping)
        self.knobs = knobs if knobs is not None else AnalyzerV7Knobs.paper_mode()

        mparams = getattr(self.machine, "mparams", None)

        # ------------------------------------------------------
        # 参数接线：严格优先 machine.mparams，其次回退到论文默认值
        # ------------------------------------------------------
        if self.knobs.alpha_bg is None:
            self.knobs.alpha_bg = float(getattr(mparams, "alpha_bg", 0.0)) if mparams is not None else 0.0

        self.T1_US = float(getattr(mparams, "T1", self.T1_US))
        self.K_HEATING = float(getattr(mparams, "k_heating", self.K_HEATING))
        self.EPSILON_2Q = float(getattr(mparams, "epsilon", self.EPSILON_2Q))

        # Table 1 固定/默认参数
        self.SPLIT_MERGE_US = float(getattr(mparams, "split_merge_time", 80.0))
        self.ION_SWAP_US = float(getattr(mparams, "ion_swap_time", 40.0))
        self.MOVE_SPEED_UM_PER_US = float(getattr(mparams, "move_speed_um_per_us", 2.0))
        self.FIBER_LATENCY_US = float(getattr(mparams, "qccd_fiber_latency_us", 200.0))
        self.FID_FIBER = float(getattr(mparams, "qccd_fiber_fidelity", self.FID_FIBER))

        # 严格论文主线：alpha_bg=0 时，门 fidelity 不再承受长期 trap 热背景惩罚。
        self.strict_paper_mainline = float(self.knobs.alpha_bg) == 0.0

        print(
            "AnalyzerV7 parameter wiring: "
            f"alpha_bg={self.knobs.alpha_bg}, "
            f"T1={self.T1_US}, "
            f"k_heating={self.K_HEATING}, "
            f"epsilon_2q={self.EPSILON_2Q}, "
            f"split_merge={self.SPLIT_MERGE_US}, "
            f"ion_swap={self.ION_SWAP_US}, "
            f"fiber_latency={self.FIBER_LATENCY_US}, "
            f"fiber_fidelity={self.FID_FIBER}, "
            f"strict_paper_mainline={self.strict_paper_mainline}"
        )

        # ------------------------------------------------------
        # trap 背景 / 热状态
        # 说明：为兼容论文中热效应模型，这里保留 per-ion heating；
        # 当 strict_paper_mainline=True 时，不把 shuttle heating 写回长期状态，
        # 仅在单次 aggregate shuttle fidelity 结算时使用。
        # ------------------------------------------------------
        self.trap_heat_state = {t.id: 0.0 for t in self.machine.traps}
        self.trap_bg = {t.id: 1.0 for t in self.machine.traps}

        capacity = self.machine.traps[0].capacity if self.machine.traps else 20
        num_traps = len(self.machine.traps)
        max_ions = num_traps * capacity * 4
        self.ion_heating = {i: 0.0 for i in range(max_ions + 256)}

        # ------------------------------------------------------
        # 总体结果
        # ------------------------------------------------------
        self.final_fidelity = 1.0
        self.prog_fin_time = 0.0
        self.op_count: Dict[str, int] = {}
        self.gate_chain_lengths: List[int] = []

        # 论文主结果：门 fidelity 与 shuttle fidelity 分离统计，便于核查
        self._gate_mult = 1.0
        self._gate_cnt = 0
        self._gate_avg_n = []
        self._regular_gate_mult = 1.0
        self._swap_insert_gate_mult = 1.0
        self._operation_local_2q_mult = 1.0
        self._optical_local_2q_mult = 1.0
        self._shuttle_mult = 1.0
        self._shuttle_cnt = 0
        self._shuttle_min = 1.0

        # ------------------------------------------------------
        # aggregate shuttle 统计
        # ------------------------------------------------------
        self._shuttle_acc_time = {}
        self._shuttle_acc_heat = {}
        self._shuttle_info = {}
        self._completed_shuttles = set()

        # ------------------------------------------------------
        # physical shuttle leg 统计（论文里通常直接用 shuttle 次数）
        # 这里同时保留 logical shuttle 与 physical leg 口径。
        # ------------------------------------------------------
        self._ion_active_leg = {}
        self._physical_shuttle_leg_count = 0
        self._physical_shuttle_legs = []

        # ------------------------------------------------------
        # V7 专用统计：small / large 共用，large 下会出现 fiber / swap_insert
        # ------------------------------------------------------
        self.zone_gate_counts = {
            "storage": 0,
            "operation": 0,
            "optical": 0,
            "unknown": 0,
        }
        self.zone_2q_counts = {
            "storage": 0,
            "operation": 0,
            "optical": 0,
            "unknown": 0,
        }
        self.local_1q_count = 0
        self.local_2q_count = 0
        self.remote_2q_count = 0
        self.operation_local_2q_count = 0
        self.optical_local_2q_count = 0
        self.storage_local_2q_count = 0
        self.fiber_gate_count = 0
        self.swap_insert_gate_count = 0
        self.swap_insert_logical_count = 0
        self.swap_insert_gate_remainder = 0
        self.regular_fiber_gate_count = 0

        # 时间分解
        self.local_1q_time = 0.0
        self.local_2q_time = 0.0
        self.operation_local_2q_time = 0.0
        self.optical_local_2q_time = 0.0
        self.storage_local_2q_time = 0.0
        self.fiber_gate_time = 0.0
        self.swap_insert_time = 0.0
        self.split_time_total = 0.0
        self.move_time_total = 0.0
        self.merge_time_total = 0.0
        self.shuttle_time_total = 0.0

        # 导出计数
        self.split_count = 0
        self.merge_count = 0
        self.move_count = 0
        self.logical_shuttle_count = 0
        self.physical_shuttle_leg_count = 0

    # ==========================================================
    # 基础工具
    # ==========================================================
    def _seg_length_um(self, seg_id: int) -> float:
        if hasattr(self.machine, "get_segment_length_um"):
            try:
                return float(self.machine.get_segment_length_um(seg_id))
            except Exception:
                pass
        if hasattr(self.machine, "segments_by_id"):
            try:
                return float(self.machine.segments_by_id[seg_id].length)
            except Exception:
                pass
        try:
            return float(self.machine.segments[seg_id].length)
        except Exception:
            return float(getattr(self.machine.mparams, "segment_length_um", 80.0))

    def _avg_nbar(self, ions) -> float:
        if not ions:
            return 0.0
        return sum(self.ion_heating.get(i, 0.0) for i in ions) / float(len(ions))

    def _zone_key(self, zone_type) -> str:
        if zone_type in ("storage", "operation", "optical"):
            return zone_type
        return "unknown"

    def _is_swap_insert_gate(self, info: dict) -> bool:
        gate_type = str(info.get("gate_type", ""))
        if gate_type == "swap_fiber":
            return True
        if info.get("is_swap_insert", False):
            return True
        return False

    # ==========================================================
    # 背景项 B_i
    # ==========================================================
    def _refresh_bg(self, trap_id: int):
        if trap_id is None:
            return
        if self.strict_paper_mainline:
            self.trap_bg[trap_id] = 1.0
            return
        a = float(self.knobs.alpha_bg)
        if a == 0.0:
            self.trap_bg[trap_id] = 1.0
            return
        H = max(self.trap_heat_state.get(trap_id, 0.0), 0.0)
        self.trap_bg[trap_id] = math.exp(-a * H)
        self.trap_bg[trap_id] = min(max(self.trap_bg[trap_id], 0.0), 1.0)

    def _apply_bg(self, trap_id: int, delta_avg_nbar: float):
        if trap_id is None:
            return
        if self.strict_paper_mainline:
            self.trap_bg[trap_id] = 1.0
            return
        a = float(self.knobs.alpha_bg)
        if a == 0.0:
            self.trap_bg[trap_id] = 1.0
            return
        self.trap_heat_state[trap_id] = self.trap_heat_state.get(trap_id, 0.0) + float(delta_avg_nbar)
        self._refresh_bg(trap_id)

    # ==========================================================
    # fidelity 公式
    # ==========================================================
    def _env_fidelity(self, t_us: float, avg_nbar: float) -> float:
        """
        论文主线 shuttle fidelity：
            F = exp(-t/T1) - k * nbar
        """
        f = math.exp(-float(t_us) / self.T1_US) - self.K_HEATING * float(avg_nbar)
        if f > 1.0:
            return 1.0
        return max(f, 1e-15)

    def _gate_fidelity(self, chain_ions, is_2q: bool, is_fiber: bool, trap_id: int):
        """
        论文门 fidelity 口径：
          - 1Q: 0.9999
          - local 2Q: 1 - epsilon * N^2
          - fiber gate: 0.99（或 machine.mparams.qccd_fiber_fidelity）
        若 alpha_bg=0，则背景项 B_i 恒为 1。
        """
        N = len(chain_ions)
        if is_fiber:
            f_g = self.FID_FIBER
        elif is_2q:
            N_eff = max(int(N), 1)
            f_g = 1.0 - self.EPSILON_2Q * (N_eff ** 2)
        else:
            f_g = self.FID_1Q
        f_g = max(f_g, 1e-15)

        self._refresh_bg(trap_id)
        B = self.trap_bg.get(trap_id, 1.0)
        fid = f_g * B
        return max(fid, 1e-15), f_g, B

    # ==========================================================
    # shuttle 聚合
    # ==========================================================
    def _accumulate_shuttle(self, shuttle_id, dt_us: float, delta_heat: float, etype: str = None, swap_cnt: int = 0):
        if shuttle_id is None:
            return
        self._shuttle_acc_time[shuttle_id] = self._shuttle_acc_time.get(shuttle_id, 0.0) + float(dt_us)
        self._shuttle_acc_heat[shuttle_id] = self._shuttle_acc_heat.get(shuttle_id, 0.0) + float(delta_heat)
        rec = self._shuttle_info.setdefault(shuttle_id, {"split": 0, "move": 0, "merge": 0, "swap_cnt": 0})
        if etype == "split":
            rec["split"] += 1
            rec["swap_cnt"] += int(swap_cnt)
        elif etype == "move":
            rec["move"] += 1
        elif etype == "merge":
            rec["merge"] += 1

    def _finalize_shuttle(self, shuttle_id):
        if shuttle_id is None:
            return 1.0
        t_sh = float(self._shuttle_acc_time.pop(shuttle_id, 0.0))
        nbar_sh = float(self._shuttle_acc_heat.pop(shuttle_id, 0.0))
        f_sh = self._env_fidelity(t_sh, nbar_sh)
        self._shuttle_mult *= f_sh
        self._shuttle_cnt += 1
        self._shuttle_min = min(self._shuttle_min, f_sh)
        self._completed_shuttles.add(shuttle_id)
        return f_sh

    # ==========================================================
    # physical shuttle leg 统计
    # ==========================================================
    def _start_physical_leg(self, ion_id, src_trap, split_event_info, split_start, split_end):
        self._ion_active_leg[ion_id] = {
            "src_trap": src_trap,
            "split_info": split_event_info,
            "split_start": split_start,
            "split_end": split_end,
            "saw_move": False,
            "move_count": 0,
        }

    def _update_physical_leg_move(self, ion_id):
        if ion_id not in self._ion_active_leg:
            return
        self._ion_active_leg[ion_id]["saw_move"] = True
        self._ion_active_leg[ion_id]["move_count"] += 1

    def _finish_physical_leg(self, ion_id, dst_trap, merge_event_info, merge_start, merge_end):
        rec = self._ion_active_leg.pop(ion_id, None)
        if rec is None:
            return
        src_trap = rec["src_trap"]
        if src_trap is None or dst_trap is None:
            return
        if src_trap != dst_trap:
            self._physical_shuttle_leg_count += 1
            self._physical_shuttle_legs.append({
                "ion": ion_id,
                "src_trap": src_trap,
                "dst_trap": dst_trap,
                "split_start": rec["split_start"],
                "split_end": rec["split_end"],
                "merge_start": merge_start,
                "merge_end": merge_end,
                "move_count": rec["move_count"],
                "saw_move": rec["saw_move"],
                "split_info": rec["split_info"],
                "merge_info": merge_event_info,
            })

    # ==========================================================
    # heating 状态更新
    # ==========================================================
    def _record_split_heat(self, trap, chain, d_split, d_swap):
        if self.strict_paper_mainline:
            return
        L = len(chain)
        if L > 0 and d_split > 0:
            for ion in chain:
                self.ion_heating[ion] += d_split
            self._apply_bg(trap, d_split)
        if L > 0 and d_swap > 0:
            for ion in chain:
                self.ion_heating[ion] += d_swap
            self._apply_bg(trap, d_swap)

    def _record_move_heat(self, ions, heat):
        if self.strict_paper_mainline:
            return
        for ion in ions:
            self.ion_heating[ion] += heat

    def _record_merge_heat(self, trap, new_chain, d_merge):
        if self.strict_paper_mainline:
            return
        L = len(new_chain)
        if L > 0 and d_merge > 0:
            for ion in new_chain:
                self.ion_heating[ion] += d_merge
            self._apply_bg(trap, d_merge)

    # ==========================================================
    # 主 replay 逻辑
    # ==========================================================
    def move_check(self):
        self.op_count = {
            Schedule.Gate: 0,
            Schedule.Split: 0,
            Schedule.Move: 0,
            Schedule.Merge: 0,
        }

        replay_traps = {t.id: [] for t in self.machine.traps}
        for t_id, ions in self.init_map.items():
            if t_id in replay_traps:
                replay_traps[t_id] = ions[:]

        self.prog_fin_time = 0.0
        for ev in self.schedule.events:
            self.prog_fin_time = max(self.prog_fin_time, float(ev[3]))

        acc = 1.0

        for ev in self.schedule.events:
            etype = ev[1]
            st, ed = float(ev[2]), float(ev[3])
            dt = max(ed - st, 0.0)
            info = ev[4]

            if etype in self.op_count:
                self.op_count[etype] += 1

            # --------------------------------------------------
            # Gate
            # --------------------------------------------------
            if etype == Schedule.Gate:
                trap = info.get("trap", None)
                ions = info.get("ions", [])
                chain = replay_traps.get(trap, [])
                is_2q = (len(ions) == 2)
                is_fiber = bool(info.get("is_fiber", False))
                gate_type = str(info.get("gate_type", ""))
                zone_type = self._zone_key(info.get("zone_type", None))

                fid, f_g, B = self._gate_fidelity(
                    chain_ions=chain,
                    is_2q=is_2q,
                    is_fiber=is_fiber,
                    trap_id=trap,
                )
                acc *= fid
                self._gate_mult *= fid
                self._gate_cnt += 1
                self._gate_avg_n.append(0.0 if self.strict_paper_mainline else self._avg_nbar(chain))

                self.zone_gate_counts[zone_type] += 1
                if is_2q:
                    self.zone_2q_counts[zone_type] += 1
                    self.gate_chain_lengths.append(len(chain))

                # 论文小规模与大规模统一口径：
                # - 1Q 单独统计
                # - operation / optical 本地 2Q 分开统计
                # - fiber 2Q 与 SWAP insertion 单独统计
                if not is_2q:
                    self.local_1q_count += 1
                    self.local_1q_time += dt
                    self._regular_gate_mult *= fid
                elif is_fiber:
                    self.remote_2q_count += 1
                    self.fiber_gate_count += 1
                    self.fiber_gate_time += dt
                    if self._is_swap_insert_gate(info):
                        self.swap_insert_gate_count += 1
                        self.swap_insert_time += dt
                        self._swap_insert_gate_mult *= fid
                    else:
                        self._regular_gate_mult *= fid
                else:
                    self.local_2q_count += 1
                    self.local_2q_time += dt
                    self._regular_gate_mult *= fid
                    if zone_type == "operation":
                        self.operation_local_2q_count += 1
                        self.operation_local_2q_time += dt
                        self._operation_local_2q_mult *= fid
                    elif zone_type == "optical":
                        self.optical_local_2q_count += 1
                        self.optical_local_2q_time += dt
                        self._optical_local_2q_mult *= fid
                    elif zone_type == "storage":
                        self.storage_local_2q_count += 1
                        self.storage_local_2q_time += dt

                if self.knobs.debug_events:
                    print(
                        "[DBG GATE-V7]",
                        "trap", trap,
                        "zone", zone_type,
                        "gate_type", gate_type,
                        "is_fiber", is_fiber,
                        "L", len(chain),
                        "f_g", round(f_g, 6),
                        "B", round(B, 6),
                        "fid", round(fid, 6),
                    )

            # --------------------------------------------------
            # Split
            # --------------------------------------------------
            elif etype == Schedule.Split:
                trap = info["trap"]
                moving_ions = info.get("ions", [])
                swap_cnt = int(info.get("swap_cnt", 0))
                shuttle_id = info.get("shuttle_id", None)
                chain = replay_traps.get(trap, [])

                # 论文 Table 1：
                # split 固定 80us；若目标离子不在链端，则按 PaperSwapDirect
                # 只补 1 次阱内 direct swap，而不是按 hop 数累计。
                d_split = 1.0
                d_swap = 1.0 if swap_cnt > 0 else 0.0
                self._record_split_heat(trap, chain, d_split, d_swap)
                self._accumulate_shuttle(shuttle_id, dt, d_split + d_swap, etype="split", swap_cnt=swap_cnt)
                self.split_time_total += dt
                self.shuttle_time_total += dt

                for ion in moving_ions:
                    if ion in chain:
                        chain.remove(ion)
                    self._start_physical_leg(ion, trap, info, st, ed)

                if self.knobs.debug_events:
                    print(
                        "[DBG SPLIT-V7]",
                        "trap", trap,
                        "dt", dt,
                        "swap_cnt", swap_cnt,
                        "shuttle_id", shuttle_id,
                    )

            # --------------------------------------------------
            # Move
            # --------------------------------------------------
            elif etype == Schedule.Move:
                ions = info.get("ions", [])
                dst_seg = info.get("dest_seg", None)
                shuttle_id = info.get("shuttle_id", None)

                if dst_seg is not None:
                    dist_um = self._seg_length_um(dst_seg)
                else:
                    dist_um = float(getattr(self.machine.mparams, "segment_length_um", 80.0))
                heat = dist_um * (0.1 / 2.0)
                self._record_move_heat(ions, heat)
                self._accumulate_shuttle(shuttle_id, dt, heat, etype="move")
                self.move_time_total += dt
                self.shuttle_time_total += dt

                for ion in ions:
                    self._update_physical_leg_move(ion)

                if self.knobs.debug_events:
                    print(
                        "[DBG MOVE-V7]",
                        "dst_seg", dst_seg,
                        "dt", dt,
                        "heat", round(heat, 6),
                        "shuttle_id", shuttle_id,
                    )

            # --------------------------------------------------
            # Merge
            # --------------------------------------------------
            elif etype == Schedule.Merge:
                trap = info["trap"]
                incoming = info.get("ions", [])
                shuttle_id = info.get("shuttle_id", None)

                new_chain = replay_traps.get(trap, []) + incoming
                replay_traps[trap] = new_chain
                d_merge = 1.0
                self._record_merge_heat(trap, new_chain, d_merge)
                self._accumulate_shuttle(shuttle_id, dt, d_merge, etype="merge")
                f_sh = self._finalize_shuttle(shuttle_id)
                acc *= f_sh
                self.merge_time_total += dt
                self.shuttle_time_total += dt

                for ion in incoming:
                    self._finish_physical_leg(ion, trap, info, st, ed)

                if self.knobs.debug_events:
                    print(
                        "[DBG MERGE-V7]",
                        "trap", trap,
                        "dt", dt,
                        "shuttle_id", shuttle_id,
                        "f_sh", round(f_sh, 6),
                    )

        # 若存在未闭合 shuttle_id，仍按论文聚合口径做收尾。
        pending_ids = list(self._shuttle_acc_time.keys())
        for sid in pending_ids:
            f_sh = self._finalize_shuttle(sid)
            acc *= f_sh
            if self.knobs.debug_events:
                print("[DBG SHUTTLE-FINALIZE-LATE-V7]", "shuttle_id", sid, "f_sh", round(f_sh, 6))

        self.final_fidelity = acc

        self.split_count = int(self.op_count.get(Schedule.Split, 0))
        self.merge_count = int(self.op_count.get(Schedule.Merge, 0))
        self.move_count = int(self.op_count.get(Schedule.Move, 0))
        self.logical_shuttle_count = int(self._shuttle_cnt)
        self.physical_shuttle_leg_count = int(self._physical_shuttle_leg_count)

        self.regular_fiber_gate_count = int(self.fiber_gate_count - self.swap_insert_gate_count)
        if self.knobs.infer_logical_swap_insert_from_triplets:
            self.swap_insert_logical_count = int(self.swap_insert_gate_count // 3)
            self.swap_insert_gate_remainder = int(self.swap_insert_gate_count % 3)
        else:
            self.swap_insert_logical_count = int(self.swap_insert_gate_count)
            self.swap_insert_gate_remainder = 0

        self._print_stats()

    # ==========================================================
    # 输出统计
    # ==========================================================
    def _print_stats(self):
        print("AnalyzerV7 mode: paper-faithful V7")
        print(f"Program Finish Time: {self.prog_fin_time} us")
        print(
            "OPCOUNTS",
            "Gate:", self.op_count.get(Schedule.Gate, 0),
            "Split:", self.op_count.get(Schedule.Split, 0),
            "Move:", self.op_count.get(Schedule.Move, 0),
            "Merge:", self.op_count.get(Schedule.Merge, 0),
        )

        if self.gate_chain_lengths:
            lens = np.array(self.gate_chain_lengths, dtype=float)
            print("\nTwo-qubit gate chain statistics")
            print(f"Mean: {np.mean(lens)} Max: {np.max(lens)}")

        print(f"Fidelity: {self.final_fidelity}")

        print("\nPaper-facing shuttle summary")
        print(f"  logical_shuttle_count      = {self.logical_shuttle_count}")
        print(f"  physical_shuttle_leg_count = {self.physical_shuttle_leg_count}")
        print(f"  split_count                = {self.split_count}")
        print(f"  move_count                 = {self.move_count}")
        print(f"  merge_count                = {self.merge_count}")

        print("\nV7 gate summary")
        print(f"  local_1q_count             = {self.local_1q_count}")
        print(f"  local_2q_count             = {self.local_2q_count}")
        print(f"  operation_local_2q_count   = {self.operation_local_2q_count}")
        print(f"  optical_local_2q_count     = {self.optical_local_2q_count}")
        print(f"  storage_local_2q_count     = {self.storage_local_2q_count}")
        print(f"  remote_2q_count            = {self.remote_2q_count}")
        print(f"  fiber_gate_count           = {self.fiber_gate_count}")
        print(f"  regular_fiber_gate_count   = {self.regular_fiber_gate_count}")
        print(f"  swap_insert_gate_count     = {self.swap_insert_gate_count}")
        print(f"  swap_insert_logical_count  = {self.swap_insert_logical_count}")
        if self.swap_insert_gate_remainder:
            print(f"  swap_insert_gate_remainder = {self.swap_insert_gate_remainder}")

        print("\nZone gate summary")
        for z in ("storage", "operation", "optical", "unknown"):
            print(f"  zone_gate_count[{z}] = {self.zone_gate_counts[z]}")
        for z in ("storage", "operation", "optical", "unknown"):
            print(f"  zone_2q_count[{z}]   = {self.zone_2q_counts[z]}")

        print("\nTime breakdown (paper-facing)")
        print(f"  local_1q_time        = {self.local_1q_time} us")
        print(f"  local_2q_time        = {self.local_2q_time} us")
        print(f"  operation_local_2q_time = {self.operation_local_2q_time} us")
        print(f"  optical_local_2q_time   = {self.optical_local_2q_time} us")
        print(f"  storage_local_2q_time   = {self.storage_local_2q_time} us")
        print(f"  fiber_gate_time      = {self.fiber_gate_time} us")
        print(f"  swap_insert_time     = {self.swap_insert_time} us")
        print(f"  split_time_total     = {self.split_time_total} us")
        print(f"  move_time_total      = {self.move_time_total} us")
        print(f"  merge_time_total     = {self.merge_time_total} us")
        print(f"  shuttle_time_total   = {self.shuttle_time_total} us")

        if self.knobs.debug_summary:
            if self._gate_cnt > 0:
                avg_gate_n = float(np.mean(self._gate_avg_n)) if self._gate_avg_n else 0.0
                print(f"[DBG SUMMARY] gates={self._gate_cnt}  gate_mult={self._gate_mult:.6g}  avg_gate_nbar={avg_gate_n:.4f}")
            if self._shuttle_cnt > 0:
                print(f"[DBG SUMMARY] logical_shuttles={self._shuttle_cnt}  shuttle_mult={self._shuttle_mult:.6g}  min_shuttle={self._shuttle_min:.6g}")
            if self.trap_bg:
                worst = min(self.trap_bg.items(), key=lambda x: x[1])
                worst_h = self.trap_heat_state.get(worst[0], 0.0)
                print(
                    f"[DBG SUMMARY] worst_B: Trap {worst[0]} -> {worst[1]:.6f} "
                    f"(heat_state={worst_h:.6f}, alpha_bg={self.knobs.alpha_bg}, "
                    f"strict_paper_mainline={self.strict_paper_mainline})"
                )

    # ==========================================================
    # 对外接口
    # ==========================================================
    def analyze_and_return(self):
        self.move_check()
        paper_shuttle_count = int(self.physical_shuttle_leg_count)
        return {
            # 兼容旧接口字段
            "fidelity": self.final_fidelity,
            "total_shuttle": paper_shuttle_count,
            "logical_shuttle_count": int(self.logical_shuttle_count),
            "physical_shuttle_leg_count": int(self.physical_shuttle_leg_count),
            "split_count": int(self.split_count),
            "merge_count": int(self.merge_count),
            "move_count": int(self.move_count),
            "time": self.prog_fin_time,

            # V7 专用字段
            "local_1q_count": int(self.local_1q_count),
            "local_2q_count": int(self.local_2q_count),
            "operation_local_2q_count": int(self.operation_local_2q_count),
            "optical_local_2q_count": int(self.optical_local_2q_count),
            "storage_local_2q_count": int(self.storage_local_2q_count),
            "remote_2q_count": int(self.remote_2q_count),
            "fiber_gate_count": int(self.fiber_gate_count),
            "regular_fiber_gate_count": int(self.regular_fiber_gate_count),
            "swap_insert_gate_count": int(self.swap_insert_gate_count),
            "swap_insert_logical_count": int(self.swap_insert_logical_count),
            "swap_insert_gate_remainder": int(self.swap_insert_gate_remainder),
            "zone_gate_counts": dict(self.zone_gate_counts),
            "zone_2q_counts": dict(self.zone_2q_counts),
            "local_1q_time": float(self.local_1q_time),
            "local_2q_time": float(self.local_2q_time),
            "operation_local_2q_time": float(self.operation_local_2q_time),
            "optical_local_2q_time": float(self.optical_local_2q_time),
            "storage_local_2q_time": float(self.storage_local_2q_time),
            "fiber_gate_time": float(self.fiber_gate_time),
            "swap_insert_time": float(self.swap_insert_time),
            "split_time_total": float(self.split_time_total),
            "move_time_total": float(self.move_time_total),
            "merge_time_total": float(self.merge_time_total),
            "shuttle_time_total": float(self.shuttle_time_total),
            "gate_mult": float(self._gate_mult),
            "regular_gate_mult": float(self._regular_gate_mult),
            "swap_insert_gate_mult": float(self._swap_insert_gate_mult),
            "operation_local_2q_mult": float(self._operation_local_2q_mult),
            "optical_local_2q_mult": float(self._optical_local_2q_mult),
            "shuttle_mult": float(self._shuttle_mult),
        }
