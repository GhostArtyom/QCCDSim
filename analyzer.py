import math
import numpy as np

from schedule import Schedule


class AnalyzerKnobs:
    """
    Analyzer 配置。

    论文主线口径：
    1) shuttle fidelity:
           F = exp(-t / T1) - k * nbar
    2) gate fidelity:
           1Q gate    -> 0.9999
           2Q gate    -> 1 - epsilon * N^2
           Fiber gate -> 0.99
       若启用背景项，则再乘 B_i。
    3) 当前 paper_mode 的更收紧版本：
       - 当 alpha_bg == 0.0 时，严格令 B_i 恒为 1
       - split/move/merge 的 heating 只参与 shuttle fidelity 聚合
         不再写回长期 ion / trap 热状态
    """

    def __init__(
        self,
        bg_model: str = "exp",
        alpha_bg=None,
        inject_norm: str = "none",
        swap_norm: str = "none",
        move_heat_use_distance: bool = True,
        move_heat_const: float = 0.1,
        move_bg_fraction: float = 0.0,
        gate_env_time_mode: str = "duration",
        gate_use_env: bool = False,
        gate_use_bg: bool = True,
        shuttle_fidelity_mode: str = "aggregate",
        merge_equalize: bool = False,
        debug_events: bool = False,
        debug_summary: bool = True,
        mode_name: str = "custom",
    ):
        self.bg_model = bg_model
        self.alpha_bg = (None if alpha_bg is None else float(alpha_bg))

        self.inject_norm = inject_norm
        self.swap_norm = swap_norm
        self.move_heat_use_distance = move_heat_use_distance
        self.move_heat_const = float(move_heat_const)
        self.move_bg_fraction = float(move_bg_fraction)
        self.gate_env_time_mode = gate_env_time_mode
        self.gate_use_env = bool(gate_use_env)
        self.gate_use_bg = bool(gate_use_bg)
        self.shuttle_fidelity_mode = shuttle_fidelity_mode
        self.merge_equalize = bool(merge_equalize)
        self.debug_events = debug_events
        self.debug_summary = debug_summary
        self.mode_name = mode_name

    @classmethod
    def paper_mode(
        cls,
        shuttle_fidelity_mode: str = "aggregate",
        debug_summary: bool = True,
        alpha_bg=None,
    ):
        """
        更收紧到论文主线的模式：
        - shuttle fidelity 使用论文式(1)
        - gate fidelity 仅使用论文表中的固有 fidelity，再乘 B_i
        - 若 alpha_bg == 0，则 B_i 恒为 1
        - 不额外乘 gate env fidelity
        - 不做 merge 后 heating 均衡
        - 不将 move heating 注入 zone background
        """
        return cls(
            bg_model="exp",
            alpha_bg=alpha_bg,
            inject_norm="none",
            swap_norm="none",
            move_heat_use_distance=True,
            move_heat_const=0.1,
            move_bg_fraction=0.0,
            gate_env_time_mode="duration",
            gate_use_env=False,
            gate_use_bg=True,
            shuttle_fidelity_mode=shuttle_fidelity_mode,
            merge_equalize=False,
            debug_events=False,
            debug_summary=debug_summary,
            mode_name="paper",
        )

    @classmethod
    def extended_mode(
        cls,
        shuttle_fidelity_mode: str = "aggregate",
        debug_summary: bool = True,
        alpha_bg=None,
    ):
        """
        扩展模式：
        保留原有可扩展实现，用于后续研究。
        """
        return cls(
            bg_model="exp",
            alpha_bg=alpha_bg,
            inject_norm="none",
            swap_norm="none",
            move_heat_use_distance=True,
            move_heat_const=0.1,
            move_bg_fraction=0.175,
            gate_env_time_mode="duration",
            gate_use_env=False,
            gate_use_bg=True,
            shuttle_fidelity_mode=shuttle_fidelity_mode,
            merge_equalize=True,
            debug_events=False,
            debug_summary=debug_summary,
            mode_name="extended",
        )


class Analyzer:
    # ----------------------------------------------------------
    # 论文/模型默认参数
    # 运行时优先从 machine.mparams 接线。
    # ----------------------------------------------------------
    T1_US = 600 * 1e6
    K_HEATING = 0.001
    EPSILON_2Q = 1.0 / 25600.0

    HEAT_SPLIT = 1.0
    HEAT_MERGE = 1.0
    HEAT_SWAP = 0.3
    HEAT_MOVE_PER_UM = 0.1 / 2.0

    FID_1Q = 0.9999
    FID_FIBER = 0.99

    def __init__(self, scheduler_obj, machine_obj, init_mapping, knobs: AnalyzerKnobs = None):
        self.scheduler = scheduler_obj
        self.schedule = scheduler_obj.schedule
        self.machine = machine_obj
        self.init_map = getattr(scheduler_obj, "init_map", init_mapping)
        self.knobs = knobs if knobs is not None else AnalyzerKnobs.paper_mode()

        mparams = getattr(self.machine, "mparams", None)

        if self.knobs.alpha_bg is None:
            if mparams is not None and hasattr(mparams, "alpha_bg"):
                self.knobs.alpha_bg = float(mparams.alpha_bg)
            else:
                self.knobs.alpha_bg = 0.0

        self.T1_US = float(getattr(mparams, "T1", self.T1_US))
        self.K_HEATING = float(getattr(mparams, "k_heating", self.K_HEATING))
        self.EPSILON_2Q = float(getattr(mparams, "epsilon", self.EPSILON_2Q))

        # 该标志用于严格论文主线：
        # - paper_mode
        # - alpha_bg == 0
        # 在此模式下，不保留长期热状态对后续 gate 的附加影响。
        self.strict_paper_mainline = (
            self.knobs.mode_name == "paper" and float(self.knobs.alpha_bg) == 0.0
        )

        print(
            "Analyzer parameter wiring: "
            f"alpha_bg={self.knobs.alpha_bg}, "
            f"T1={self.T1_US}, "
            f"k_heating={self.K_HEATING}, "
            f"epsilon_2q={self.EPSILON_2Q}, "
            f"strict_paper_mainline={self.strict_paper_mainline}"
        )

        # ------------------------------------------------------
        # trap 背景 / 热状态
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
        self.op_count = {}
        self.gate_chain_lengths = []

        # gate 相关 debug
        self._gate_mult = 1.0
        self._gate_cnt = 0
        self._gate_avg_n = []

        # per-event 动态 fidelity 相关 debug
        self._dyn_mult = 1.0
        self._dyn_cnt = 0
        self._dyn_min = 1.0

        # ------------------------------------------------------
        # aggregate shuttle 统计
        # ------------------------------------------------------
        self._shuttle_acc_time = {}
        self._shuttle_acc_heat = {}
        self._shuttle_acc_move_heat = {}
        self._shuttle_mult = 1.0
        self._shuttle_cnt = 0
        self._shuttle_min = 1.0
        self._shuttle_info = {}
        self._completed_shuttles = set()

        # ------------------------------------------------------
        # physical shuttle leg 统计
        # ------------------------------------------------------
        self._ion_active_leg = {}
        self._physical_shuttle_leg_count = 0
        self._physical_shuttle_legs = []

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
        """
        尽量从 machine 中获取某个 segment 的长度。
        """
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
            return float(getattr(self.machine.mparams, "segment_length_um", 53.0))

    def _avg_nbar(self, ions) -> float:
        if not ions:
            return 0.0
        return sum(self.ion_heating.get(i, 0.0) for i in ions) / float(len(ions))

    # ==========================================================
    # 背景项 B_i
    # ==========================================================
    def _refresh_bg(self, trap_id: int):
        """
        更新 trap 的背景项 B_i。

        在严格论文主线下，B_i 恒为 1。
        """
        if trap_id is None:
            return

        if self.strict_paper_mainline:
            self.trap_bg[trap_id] = 1.0
            return

        a = float(self.knobs.alpha_bg)

        if self.knobs.bg_model == "none" or a == 0.0:
            self.trap_bg[trap_id] = 1.0
            return

        H = max(self.trap_heat_state.get(trap_id, 0.0), 0.0)

        if self.knobs.bg_model == "exp":
            self.trap_bg[trap_id] = math.exp(-a * H)
        elif self.knobs.bg_model == "linear":
            self.trap_bg[trap_id] = max(1.0 - a * H, 0.0)
        else:
            self.trap_bg[trap_id] = 1.0

        self.trap_bg[trap_id] = min(max(self.trap_bg[trap_id], 0.0), 1.0)

    def _apply_bg(self, trap_id: int, delta_avg_nbar: float):
        """
        将 shuttle 相关 heating 注入背景状态。

        在严格论文主线下，直接忽略，不保留长期背景热记忆。
        """
        if trap_id is None:
            return

        if self.strict_paper_mainline:
            self.trap_bg[trap_id] = 1.0
            return

        a = float(self.knobs.alpha_bg)
        if self.knobs.bg_model == "none" or a == 0.0:
            self.trap_bg[trap_id] = 1.0
            return

        self.trap_heat_state[trap_id] = self.trap_heat_state.get(trap_id, 0.0) + float(delta_avg_nbar)
        self._refresh_bg(trap_id)

    # ==========================================================
    # fidelity 公式
    # ==========================================================
    def _env_fidelity(self, t_us: float, avg_nbar: float) -> float:
        """
        论文式(1):
            F = exp(-t/T1) - k * nbar
        """
        f = math.exp(-float(t_us) / self.T1_US) - self.K_HEATING * float(avg_nbar)
        if f > 1.0:
            return 1.0
        return max(f, 1e-15)

    def _gate_fidelity(self, chain_ions, is_2q: bool, is_fiber: bool,
                       gate_start_us: float, gate_end_us: float, trap_id: int):
        """
        计算单个 gate 的 fidelity。

        更收紧到论文主线：
            F_gate = B_i * F_g
        且在 strict_paper_mainline 下，B_i 恒为 1。
        """
        N = len(chain_ions)
        if N <= 0:
            self._refresh_bg(trap_id)
            B = self.trap_bg.get(trap_id, 1.0)
            return max(B * 1e-15, 1e-15), 0.0, 1.0, 1e-15, B

        if is_fiber:
            f_g = self.FID_FIBER
        elif is_2q:
            f_g = 1.0 - self.EPSILON_2Q * (N ** 2)
        else:
            f_g = self.FID_1Q
        f_g = max(f_g, 1e-15)

        # 在严格论文主线下，gate 不再承受长期 heating 记忆带来的额外惩罚
        avg_n = 0.0 if self.strict_paper_mainline else self._avg_nbar(chain_ions)

        # 仅保留 debug 用，不乘进 gate fidelity
        if self.knobs.gate_env_time_mode == "duration":
            t_env = float(gate_end_us - gate_start_us)
        else:
            t_env = float(gate_end_us)
        f_env = self._env_fidelity(t_env, avg_n)

        self._refresh_bg(trap_id)
        B = self.trap_bg.get(trap_id, 1.0)

        fid = f_g
        if self.knobs.gate_use_bg:
            fid *= B

        return max(fid, 1e-15), avg_n, f_env, f_g, B

    def _dyn_event_mult(self, dt_us: float, delta_avg_nbar: float):
        """
        per-event fidelity 口径下的单事件乘子。
        当前论文主线默认不用该模式，接口保留。
        """
        f_dyn = self._env_fidelity(dt_us, delta_avg_nbar)
        self._dyn_mult *= f_dyn
        self._dyn_cnt += 1
        self._dyn_min = min(self._dyn_min, f_dyn)
        return f_dyn

    # ==========================================================
    # shuttle 聚合
    # ==========================================================
    def _accumulate_shuttle(self, shuttle_id, dt_us: float, delta_heat: float,
                            etype: str = None, swap_cnt: int = 0):
        if shuttle_id is None:
            return

        self._shuttle_acc_time[shuttle_id] = self._shuttle_acc_time.get(shuttle_id, 0.0) + float(dt_us)
        self._shuttle_acc_heat[shuttle_id] = self._shuttle_acc_heat.get(shuttle_id, 0.0) + float(delta_heat)

        if etype == "move":
            self._shuttle_acc_move_heat[shuttle_id] = self._shuttle_acc_move_heat.get(shuttle_id, 0.0) + float(delta_heat)

        rec = self._shuttle_info.setdefault(
            shuttle_id,
            {"split": 0, "move": 0, "merge": 0, "swap_cnt": 0}
        )

        if etype == "split":
            rec["split"] += 1
            rec["swap_cnt"] += int(swap_cnt)
        elif etype == "move":
            rec["move"] += 1
        elif etype == "merge":
            rec["merge"] += 1

    def _finalize_shuttle(self, shuttle_id):
        """
        完成一次 aggregate shuttle 的 fidelity 结算。
        """
        if shuttle_id is None:
            return 1.0

        t_sh = float(self._shuttle_acc_time.pop(shuttle_id, 0.0))
        nbar_sh = float(self._shuttle_acc_heat.pop(shuttle_id, 0.0))
        self._shuttle_acc_move_heat.pop(shuttle_id, 0.0)

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
        """
        split 阶段的 heating 记账。

        在严格论文主线下：
        - 只用于 shuttle fidelity 聚合
        - 不写回长期 ion / trap 状态
        """
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
        """
        move 阶段的 heating 记账。

        在严格论文主线下：
        - 只用于 shuttle fidelity 聚合
        - 不写回长期 ion 状态
        """
        if self.strict_paper_mainline:
            return

        for ion in ions:
            self.ion_heating[ion] += heat

    def _record_merge_heat(self, trap, new_chain, d_merge, move_bg):
        """
        merge 阶段的 heating / background 记账。

        在严格论文主线下：
        - 只用于 shuttle fidelity 聚合
        - 不写回长期 ion / trap 状态
        """
        if self.strict_paper_mainline:
            return

        L = len(new_chain)
        if L > 0 and d_merge > 0:
            for ion in new_chain:
                self.ion_heating[ion] += d_merge
            self._apply_bg(trap, d_merge)

        if move_bg > 0:
            self._apply_bg(trap, move_bg)

        if self.knobs.merge_equalize and new_chain:
            avg_h = self._avg_nbar(new_chain)
            for ion in new_chain:
                self.ion_heating[ion] = avg_h

    # ==========================================================
    # 主 replay 逻辑
    # ==========================================================
    def move_check(self):
        """
        replay schedule.events，计算：
        - 最终 fidelity
        - 程序总时间
        - 各类事件计数
        - logical shuttle count
        - physical shuttle leg count

        注意：
        - 不改执行时间统计
        - 不改 shuttle 计数统计
        """
        self.op_count = {
            Schedule.Gate: 0,
            Schedule.Split: 0,
            Schedule.Move: 0,
            Schedule.Merge: 0
        }

        replay_traps = {t.id: [] for t in self.machine.traps}
        for t_id, ions in self.init_map.items():
            if t_id in replay_traps:
                replay_traps[t_id] = ions[:]

        self.prog_fin_time = 0.0
        for ev in self.schedule.events:
            self.prog_fin_time = max(self.prog_fin_time, ev[3])

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

                fid, avg_n, f_env, f_g, B = self._gate_fidelity(
                    chain_ions=chain,
                    is_2q=(len(ions) == 2),
                    is_fiber=info.get("is_fiber", False),
                    gate_start_us=st,
                    gate_end_us=ed,
                    trap_id=trap,
                )

                acc *= fid
                self._gate_mult *= fid
                self._gate_cnt += 1
                self._gate_avg_n.append(avg_n)

                if len(ions) == 2:
                    self.gate_chain_lengths.append(len(chain))

                if self.knobs.debug_events:
                    print(
                        "[DBG GATE]",
                        "trap", trap, "L", len(chain),
                        "avg_n", round(avg_n, 6),
                        "B", round(B, 6),
                        "f_env", round(f_env, 6),
                        "f_g", round(f_g, 6),
                        "fid", round(fid, 6)
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
                L = len(chain)

                d_split = self.HEAT_SPLIT / float(L) if self.knobs.inject_norm == "chain" and L > 0 else self.HEAT_SPLIT
                d_swap = 0.0
                if swap_cnt > 0:
                    d_swap = (self.HEAT_SWAP * swap_cnt) / float(L) if self.knobs.swap_norm == "chain" and L > 0 else (self.HEAT_SWAP * swap_cnt)

                # 更收紧到论文主线：
                # 这些 heating 在 strict_paper_mainline 下只进入 shuttle fidelity 聚合，
                # 不再写回长期 ion / trap 状态。
                self._record_split_heat(trap, chain, d_split, d_swap)

                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(
                        shuttle_id,
                        dt,
                        d_split + d_swap,
                        etype="split",
                        swap_cnt=swap_cnt
                    )
                    f_dyn = 1.0
                else:
                    f_dyn = self._dyn_event_mult(dt, d_split + d_swap)
                    acc *= f_dyn

                for ion in moving_ions:
                    if ion in chain:
                        chain.remove(ion)

                for ion in moving_ions:
                    self._start_physical_leg(
                        ion_id=ion,
                        src_trap=trap,
                        split_event_info=info,
                        split_start=st,
                        split_end=ed
                    )

                if self.knobs.debug_events:
                    print(
                        "[DBG SPLIT]",
                        "trap", trap, "L", L, "dt", dt,
                        "swap_cnt", swap_cnt,
                        "shuttle_id", shuttle_id,
                        "d_split", round(d_split, 6),
                        "d_swap", round(d_swap, 6),
                        "f_dyn", round(f_dyn, 6)
                    )

            # --------------------------------------------------
            # Move
            # --------------------------------------------------
            elif etype == Schedule.Move:
                ions = info.get("ions", [])
                dst_seg = info.get("dest_seg", None)
                shuttle_id = info.get("shuttle_id", None)

                if self.knobs.move_heat_use_distance:
                    dist_um = self._seg_length_um(dst_seg) if dst_seg is not None else float(getattr(self.machine.mparams, "segment_length_um", 53.0))
                    heat = dist_um * self.HEAT_MOVE_PER_UM
                else:
                    heat = self.knobs.move_heat_const

                self._record_move_heat(ions, heat)

                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, heat, etype="move")
                    f_dyn = 1.0
                else:
                    f_dyn = self._dyn_event_mult(dt, heat)
                    acc *= f_dyn

                for ion in ions:
                    self._update_physical_leg_move(ion)

                if self.knobs.debug_events:
                    print(
                        "[DBG MOVE]",
                        "dst_seg", dst_seg,
                        "dt", dt,
                        "heat", round(heat, 6),
                        "shuttle_id", shuttle_id,
                        "f_dyn", round(f_dyn, 6)
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
                L = len(new_chain)

                d_merge = self.HEAT_MERGE / float(L) if self.knobs.inject_norm == "chain" and L > 0 else self.HEAT_MERGE
                move_bg = self.knobs.move_bg_fraction * self._shuttle_acc_move_heat.get(shuttle_id, 0.0)

                self._record_merge_heat(trap, new_chain, d_merge, move_bg)

                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, d_merge, etype="merge")
                    f_dyn = self._finalize_shuttle(shuttle_id)
                    acc *= f_dyn
                else:
                    f_dyn = self._dyn_event_mult(dt, d_merge)
                    acc *= f_dyn

                for ion in incoming:
                    self._finish_physical_leg(
                        ion_id=ion,
                        dst_trap=trap,
                        merge_event_info=info,
                        merge_start=st,
                        merge_end=ed
                    )

                if self.knobs.debug_events:
                    print(
                        "[DBG MERGE]",
                        "trap", trap, "L", L, "dt", dt,
                        "d_merge", round(d_merge, 6),
                        "move_bg", round(move_bg, 6),
                        "shuttle_id", shuttle_id,
                        "f_dyn", round(f_dyn, 6)
                    )

        if self.knobs.shuttle_fidelity_mode == "aggregate":
            pending_ids = list(self._shuttle_acc_time.keys())
            for sid in pending_ids:
                f_sh = self._finalize_shuttle(sid)
                acc *= f_sh
                if self.knobs.debug_events:
                    print("[DBG SHUTTLE-FINALIZE-LATE]", "shuttle_id", sid, "f_sh", round(f_sh, 6))

        self.final_fidelity = acc

        self.split_count = int(self.op_count.get(Schedule.Split, 0))
        self.merge_count = int(self.op_count.get(Schedule.Merge, 0))
        self.move_count = int(self.op_count.get(Schedule.Move, 0))
        self.logical_shuttle_count = int(self._shuttle_cnt)
        self.physical_shuttle_leg_count = int(self._physical_shuttle_leg_count)

        self._print_stats()

    # ==========================================================
    # 输出统计
    # ==========================================================
    def _print_stats(self):
        print(f"Analyzer mode: {self.knobs.mode_name}")
        print(f"Program Finish Time: {self.prog_fin_time} us")

        print(
            "OPCOUNTS",
            "Gate:", self.op_count.get(Schedule.Gate, 0),
            "Split:", self.op_count.get(Schedule.Split, 0),
            "Move:", self.op_count.get(Schedule.Move, 0),
            "Merge:", self.op_count.get(Schedule.Merge, 0)
        )

        if self.gate_chain_lengths:
            lens = np.array(self.gate_chain_lengths, dtype=float)
            print("\nTwo-qubit gate chain statistics")
            print(f"Mean: {np.mean(lens)} Max: {np.max(lens)}")

        print(f"Fidelity: {self.final_fidelity}")

        print("\nShuttle count summary")
        print(f"  logical_shuttle_count      = {self.logical_shuttle_count}   # High-level shuttle count aggregated by shuttle_id")
        print(f"  physical_shuttle_leg_count = {self.physical_shuttle_leg_count}   # Paper-facing count: each complete trap-to-trap split+move+merge leg")
        print(f"  split_count                = {self.split_count}")
        print(f"  merge_count                = {self.merge_count}")
        print(f"  move_count                 = {self.move_count}")

        if self.knobs.debug_summary:
            if self._gate_cnt > 0:
                avg_gate_n = float(np.mean(self._gate_avg_n)) if self._gate_avg_n else 0.0
                print(f"[DBG SUMMARY] gates={self._gate_cnt}  gate_mult={self._gate_mult:.6g}  avg_gate_nbar={avg_gate_n:.4f}")

            if self._dyn_cnt > 0:
                print(f"[DBG SUMMARY] dyn_ops={self._dyn_cnt}  dyn_mult={self._dyn_mult:.6g}  min_dyn={self._dyn_min:.6g}")

            if self._shuttle_cnt > 0:
                print(f"[DBG SUMMARY] logical_shuttles={self._shuttle_cnt}  shuttle_mult={self._shuttle_mult:.6g}  min_shuttle={self._shuttle_min:.6g}")

            if self.trap_bg:
                worst = min(self.trap_bg.items(), key=lambda x: x[1])
                worst_h = self.trap_heat_state.get(worst[0], 0.0)
                print(
                    f"[DBG SUMMARY] worst_B: Trap {worst[0]} -> {worst[1]:.6f} "
                    f"(heat_state={worst_h:.6f}, alpha_bg={self.knobs.alpha_bg}, "
                    f"move_bg_fraction={self.knobs.move_bg_fraction}, model={self.knobs.bg_model}, "
                    f"strict_paper_mainline={self.strict_paper_mainline})"
                )

    # ==========================================================
    # 对外接口
    # ==========================================================
    def analyze_and_return(self):
        self.move_check()

        paper_shuttle_count = int(self.physical_shuttle_leg_count)

        return {
            "fidelity": self.final_fidelity,
            "total_shuttle": paper_shuttle_count,
            "logical_shuttle_count": int(self.logical_shuttle_count),
            "physical_shuttle_leg_count": int(self.physical_shuttle_leg_count),
            "split_count": int(self.split_count),
            "merge_count": int(self.merge_count),
            "move_count": int(self.move_count),
            "time": self.prog_fin_time,
        }
