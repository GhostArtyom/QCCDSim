import math
import numpy as np

from schedule import Schedule


class AnalyzerKnobs:
    """
    Analyzer 配置。

    这里把两类口径分开：
    1) paper_mode：尽量贴近论文 Table 2 / Table 1 的保真度口径。
       - 不启用自定义背景项 B_i
       - 不做 merge 后链内均衡
       - 门保真度直接使用论文给出的固有 fidelity
       - shuttle fidelity 用环境项 exp(-t/T1) - k*nbar
    2) extended_mode：保留你原来的扩展研究口径。
    """

    def __init__(
        self,
        bg_model: str = "exp",
        alpha_bg: float = 0.005,
        inject_norm: str = "none",
        swap_norm: str = "none",
        move_heat_use_distance: bool = True,
        move_heat_const: float = 0.1,
        move_bg_fraction: float = 0.25,
        gate_env_time_mode: str = "duration",
        gate_use_env: bool = False,
        gate_use_bg: bool = True,
        shuttle_fidelity_mode: str = "aggregate",
        merge_equalize: bool = True,
        debug_events: bool = False,
        debug_summary: bool = True,
        mode_name: str = "custom",
    ):
        self.bg_model = bg_model
        self.alpha_bg = float(alpha_bg)
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
    def paper_mode(cls, shuttle_fidelity_mode: str = "aggregate", debug_summary: bool = True):
        """
        更贴论文 Table 2 的建议口径：
        - 不启用自定义背景项 B_i（论文未给出可执行更新式）
        - 门保真度只用论文表中的固有门 fidelity
        - shuttle fidelity 采用环境项
        - 不做 merge 后链内加热均衡（避免引入额外实现假设）
        """
        return cls(
            bg_model="none",
            alpha_bg=0.0,
            inject_norm="none",
            swap_norm="none",
            move_heat_use_distance=True,
            move_heat_const=0.1,
            move_bg_fraction=0.0,
            gate_env_time_mode="duration",
            gate_use_env=False,
            gate_use_bg=False,
            shuttle_fidelity_mode=shuttle_fidelity_mode,
            merge_equalize=False,
            debug_events=False,
            debug_summary=debug_summary,
            mode_name="paper",
        )

    @classmethod
    def extended_mode(cls, shuttle_fidelity_mode: str = "aggregate", debug_summary: bool = True):
        return cls(
            bg_model="exp",
            alpha_bg=0.001,
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

        if self.knobs.alpha_bg == 0.0:
            if hasattr(self.machine, "mparams") and hasattr(self.machine.mparams, "alpha_bg"):
                self.knobs.alpha_bg = float(self.machine.mparams.alpha_bg)

        self.trap_heat_state = {t.id: 0.0 for t in self.machine.traps}
        self.trap_bg = {t.id: 1.0 for t in self.machine.traps}

        capacity = self.machine.traps[0].capacity if self.machine.traps else 20
        num_traps = len(self.machine.traps)
        max_ions = num_traps * capacity * 4
        self.ion_heating = {i: 0.0 for i in range(max_ions + 256)}

        self.final_fidelity = 1.0
        self.prog_fin_time = 0.0
        self.op_count = {}
        self.gate_chain_lengths = []
        self._gate_mult = 1.0
        self._gate_cnt = 0
        self._gate_avg_n = []
        self._dyn_mult = 1.0
        self._dyn_cnt = 0
        self._dyn_min = 1.0
        self._shuttle_acc_time = {}
        self._shuttle_acc_heat = {}
        self._shuttle_acc_move_heat = {}
        self._shuttle_mult = 1.0
        self._shuttle_cnt = 0
        self._shuttle_min = 1.0
        self._shuttle_info = {}

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
            return float(getattr(self.machine.mparams, "segment_length_um", 53.0))

    def _avg_nbar(self, ions) -> float:
        if not ions:
            return 0.0
        return sum(self.ion_heating.get(i, 0.0) for i in ions) / float(len(ions))

    def _refresh_bg(self, trap_id: int):
        if trap_id is None:
            return
        if self.knobs.bg_model == "none":
            self.trap_bg[trap_id] = 1.0
            return
        H = max(self.trap_heat_state.get(trap_id, 0.0), 0.0)
        a = self.knobs.alpha_bg
        if self.knobs.bg_model == "exp":
            self.trap_bg[trap_id] = math.exp(-a * H)
        elif self.knobs.bg_model == "linear":
            self.trap_bg[trap_id] = max(1.0 - a * H, 0.0)
        else:
            self.trap_bg[trap_id] = 1.0
        self.trap_bg[trap_id] = min(max(self.trap_bg[trap_id], 0.0), 1.0)

    def _apply_bg(self, trap_id: int, delta_avg_nbar: float):
        if trap_id is None:
            return
        if self.knobs.bg_model == "none":
            self.trap_bg[trap_id] = 1.0
            return
        self.trap_heat_state[trap_id] = self.trap_heat_state.get(trap_id, 0.0) + float(delta_avg_nbar)
        self._refresh_bg(trap_id)

    def _env_fidelity(self, t_us: float, avg_nbar: float) -> float:
        f = math.exp(-float(t_us) / self.T1_US) - self.K_HEATING * float(avg_nbar)
        return max(f, 1e-15)

    def _gate_fidelity(self, chain_ions, is_2q: bool, is_fiber: bool, gate_start_us: float, gate_end_us: float, trap_id: int):
        N = len(chain_ions)
        if N <= 0:
            avg_n = 0.0
            f_env = 1.0
            self._refresh_bg(trap_id)
            B = self.trap_bg.get(trap_id, 1.0)
            f_g = 1e-15
            return 1e-15, avg_n, f_env, f_g, B

        if is_fiber:
            f_g = self.FID_FIBER
        elif is_2q:
            f_g = 1.0 - self.EPSILON_2Q * (N ** 2)
        else:
            f_g = self.FID_1Q
        f_g = max(f_g, 1e-15)

        avg_n = self._avg_nbar(chain_ions)
        if self.knobs.gate_env_time_mode == "duration":
            t_env = float(gate_end_us - gate_start_us)
        else:
            t_env = float(gate_end_us)
        f_env = self._env_fidelity(t_env, avg_n)

        self._refresh_bg(trap_id)
        B = self.trap_bg.get(trap_id, 1.0)

        fid = f_g
        if self.knobs.gate_use_env:
            fid *= f_env
        if self.knobs.gate_use_bg:
            fid *= B
        return max(fid, 1e-15), avg_n, f_env, f_g, B

    def _dyn_event_mult(self, dt_us: float, delta_avg_nbar: float):
        f_dyn = self._env_fidelity(dt_us, delta_avg_nbar)
        self._dyn_mult *= f_dyn
        self._dyn_cnt += 1
        self._dyn_min = min(self._dyn_min, f_dyn)
        return f_dyn

    def _accumulate_shuttle(self, shuttle_id, dt_us: float, delta_heat: float, etype: str = None, swap_cnt: int = 0):
        if shuttle_id is None:
            return
        self._shuttle_acc_time[shuttle_id] = self._shuttle_acc_time.get(shuttle_id, 0.0) + float(dt_us)
        self._shuttle_acc_heat[shuttle_id] = self._shuttle_acc_heat.get(shuttle_id, 0.0) + float(delta_heat)
        if etype == "move":
            self._shuttle_acc_move_heat[shuttle_id] = self._shuttle_acc_move_heat.get(shuttle_id, 0.0) + float(delta_heat)
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
        self._shuttle_acc_move_heat.pop(shuttle_id, 0.0)
        f_sh = self._env_fidelity(t_sh, nbar_sh)
        self._shuttle_mult *= f_sh
        self._shuttle_cnt += 1
        self._shuttle_min = min(self._shuttle_min, f_sh)
        return f_sh

    def move_check(self):
        self.op_count = {Schedule.Gate: 0, Schedule.Split: 0, Schedule.Move: 0, Schedule.Merge: 0}
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
                    print("[DBG GATE]", "trap", trap, "L", len(chain), "avg_n", round(avg_n, 6), "B", round(B, 6), "f_env", round(f_env, 6), "f_g", round(f_g, 6), "fid", round(fid, 6))

            elif etype == Schedule.Split:
                trap = info["trap"]
                moving_ions = info.get("ions", [])
                swap_cnt = int(info.get("swap_cnt", 0))
                shuttle_id = info.get("shuttle_id", None)
                chain = replay_traps.get(trap, [])
                L = len(chain)
                d_split = self.HEAT_SPLIT / float(L) if self.knobs.inject_norm == "chain" and L > 0 else self.HEAT_SPLIT
                if L > 0:
                    for ion in chain:
                        self.ion_heating[ion] += d_split
                    self._apply_bg(trap, d_split)
                if swap_cnt > 0:
                    d_swap = (self.HEAT_SWAP * swap_cnt) / float(L) if self.knobs.swap_norm == "chain" and L > 0 else (self.HEAT_SWAP * swap_cnt)
                    if L > 0:
                        for ion in chain:
                            self.ion_heating[ion] += d_swap
                    self._apply_bg(trap, d_swap)
                else:
                    d_swap = 0.0
                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, d_split + d_swap, etype="split", swap_cnt=swap_cnt)
                    f_dyn = 1.0
                else:
                    f_dyn = self._dyn_event_mult(dt, d_split + d_swap)
                    acc *= f_dyn
                for ion in moving_ions:
                    if ion in chain:
                        chain.remove(ion)
                if self.knobs.debug_events:
                    print("[DBG SPLIT]", "trap", trap, "L", L, "dt", dt, "swap_cnt", swap_cnt, "shuttle_id", shuttle_id, "d_split", round(d_split, 6), "d_swap", round(d_swap, 6), "f_dyn", round(f_dyn, 6))

            elif etype == Schedule.Move:
                ions = info.get("ions", [])
                dst_seg = info.get("dest_seg", None)
                shuttle_id = info.get("shuttle_id", None)
                if self.knobs.move_heat_use_distance:
                    dist_um = self._seg_length_um(dst_seg) if dst_seg is not None else float(getattr(self.machine.mparams, "segment_length_um", 53.0))
                    heat = dist_um * self.HEAT_MOVE_PER_UM
                else:
                    heat = self.knobs.move_heat_const
                for ion in ions:
                    self.ion_heating[ion] += heat
                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, heat, etype="move")
                    f_dyn = 1.0
                else:
                    f_dyn = self._dyn_event_mult(dt, heat)
                    acc *= f_dyn
                if self.knobs.debug_events:
                    print("[DBG MOVE]", "dst_seg", dst_seg, "dt", dt, "heat", round(heat, 6), "shuttle_id", shuttle_id, "f_dyn", round(f_dyn, 6))

            elif etype == Schedule.Merge:
                trap = info["trap"]
                incoming = info.get("ions", [])
                shuttle_id = info.get("shuttle_id", None)
                new_chain = replay_traps.get(trap, []) + incoming
                replay_traps[trap] = new_chain
                L = len(new_chain)
                d_merge = self.HEAT_MERGE / float(L) if self.knobs.inject_norm == "chain" and L > 0 else self.HEAT_MERGE
                if L > 0:
                    for ion in new_chain:
                        self.ion_heating[ion] += d_merge
                    self._apply_bg(trap, d_merge)
                move_bg = self.knobs.move_bg_fraction * self._shuttle_acc_move_heat.get(shuttle_id, 0.0)
                if move_bg > 0:
                    self._apply_bg(trap, move_bg)
                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, d_merge, etype="merge")
                    f_dyn = self._finalize_shuttle(shuttle_id)
                    acc *= f_dyn
                else:
                    f_dyn = self._dyn_event_mult(dt, d_merge)
                    acc *= f_dyn
                if self.knobs.merge_equalize and new_chain:
                    avg_h = self._avg_nbar(new_chain)
                    for ion in new_chain:
                        self.ion_heating[ion] = avg_h
                if self.knobs.debug_events:
                    print("[DBG MERGE]", "trap", trap, "L", L, "dt", dt, "d_merge", round(d_merge, 6), "move_bg", round(move_bg, 6), "shuttle_id", shuttle_id, "f_dyn", round(f_dyn, 6))

        if self.knobs.shuttle_fidelity_mode == "aggregate":
            pending_ids = list(self._shuttle_acc_time.keys())
            for sid in pending_ids:
                f_sh = self._finalize_shuttle(sid)
                acc *= f_sh
                if self.knobs.debug_events:
                    print("[DBG SHUTTLE-FINALIZE-LATE]", "shuttle_id", sid, "f_sh", round(f_sh, 6))

        self.final_fidelity = acc
        self._print_stats()

    def _print_stats(self):
        print(f"Analyzer mode: {self.knobs.mode_name}")
        print(f"Program Finish Time: {self.prog_fin_time} us")
        print("OPCOUNTS", "Gate:", self.op_count.get(Schedule.Gate, 0), "Split:", self.op_count.get(Schedule.Split, 0), "Move:", self.op_count.get(Schedule.Move, 0), "Merge:", self.op_count.get(Schedule.Merge, 0))
        if self.gate_chain_lengths:
            lens = np.array(self.gate_chain_lengths, dtype=float)
            print("\nTwo-qubit gate chain statistics")
            print(f"Mean: {np.mean(lens)} Max: {np.max(lens)}")
        print(f"Fidelity: {self.final_fidelity}")
        if self.knobs.debug_summary:
            if self._gate_cnt > 0:
                avg_gate_n = float(np.mean(self._gate_avg_n)) if self._gate_avg_n else 0.0
                print(f"[DBG SUMMARY] gates={self._gate_cnt}  gate_mult={self._gate_mult:.6g}  avg_gate_nbar={avg_gate_n:.4f}")
            if self._dyn_cnt > 0:
                print(f"[DBG SUMMARY] dyn_ops={self._dyn_cnt}  dyn_mult={self._dyn_mult:.6g}  min_dyn={self._dyn_min:.6g}")
            if self._shuttle_cnt > 0:
                print(f"[DBG SUMMARY] shuttles={self._shuttle_cnt}  shuttle_mult={self._shuttle_mult:.6g}  min_shuttle={self._shuttle_min:.6g}")
            if self.trap_bg:
                worst = min(self.trap_bg.items(), key=lambda x: x[1])
                worst_h = self.trap_heat_state.get(worst[0], 0.0)
                print(f"[DBG SUMMARY] worst_B: Trap {worst[0]} -> {worst[1]:.6f} (heat_state={worst_h:.6f}, alpha_bg={self.knobs.alpha_bg}, move_bg_fraction={self.knobs.move_bg_fraction}, model={self.knobs.bg_model})")

    def analyze_and_return(self):
        self.move_check()
        if hasattr(self.scheduler, "shuttle_counter"):
            shuttle_count = int(getattr(self.scheduler, "shuttle_counter"))
        else:
            shuttle_count = int(self.op_count.get(Schedule.Split, 0))
        return {"fidelity": self.final_fidelity, "total_shuttle": shuttle_count, "time": self.prog_fin_time}
