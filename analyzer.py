import math
import numpy as np

from schedule import Schedule


class AnalyzerKnobs:
    """
    ===== 这个类是“可调旋钮/实现细节开关（knobs）” =====

    目标：把论文（MUSS-TI）没有明确写死、工程实现必须自己决定的部分集中管理。
    注意：论文里表格/公式给死的常量，放在 Analyzer 里，不要在这里调。

    主要包含：
    1) 背景因子 Bi 的模型形式（论文没明确给出具体函数，只说有背景项/背景损耗）
    2) 分裂/合并/交换的“加热注入”是按“整条链”还是按“每个离子”计
    3) MOVE 的加热是按距离（需要段长度）还是按常数
    4) 门操作是否乘环境项 / 背景项
    5) shuttling fidelity 按“整次 shuttle 聚合”还是“每个微操作都乘一次”
    6) 调试开关

    当前版本的背景模型推荐：
      - 先维护每个 trap 的 heat-state: H_i
      - 再映射到背景因子：
            B_i = exp(-alpha_bg * H_i)
      - Move 的热默认不全部直接写入背景，而是只把 move heat 的一部分
        （move_bg_fraction）在 merge 时注入目标 trap 的背景热状态。
    """

    def __init__(
        self,
        # -------- 背景 Bi 模型 --------
        bg_model: str = "exp",           # "none" | "linear" | "exp"
        alpha_bg: float = 0.001,         # exp / linear Bi 模型强度

        # -------- 加热注入口径：按链长归一化还是不归一化 --------
        inject_norm: str = "none",       # "none" | "chain"
        swap_norm: str = "none",         # "none" | "chain"

        # -------- MOVE 加热模型：按移动距离 or 常数 --------
        move_heat_use_distance: bool = True,
        move_heat_const: float = 0.1,    # 若不按距离，用这个常数作为每次 Move 的加热增量

        # -------- Move heat 注入背景热状态的比例 --------
        # 推荐 0.2 ~ 0.35；0 表示 Move 只计入 shuttle fidelity，不影响 B_i
        move_bg_fraction: float = 0.25,

        # -------- 门操作环境项的时间口径 --------
        gate_env_time_mode: str = "duration",  # "global" | "duration"

        # -------- gate fidelity 组成 --------
        gate_use_env: bool = False,      # gate fidelity 是否乘环境项
        gate_use_bg: bool = True,        # gate fidelity 是否乘 Bi

        # -------- shuttle fidelity 口径 --------
        shuttle_fidelity_mode: str = "aggregate",   # "aggregate" | "per_event"

        # -------- merge 后是否均衡整条链加热 --------
        merge_equalize: bool = True,

        # -------- 调试输出 --------
        debug_events: bool = False,      # True：逐事件打印
        debug_summary: bool = True,      # True：末尾打印汇总
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


class Analyzer:
    """
    ===== 这个类是“回放(schedule replay) + 计算保真度/加热/背景项”的分析器 =====

    用法：
      Analyzer(scheduler_obj, machine_obj, init_mapping, knobs).analyze_and_return()

    核心思路：
    1) 从 scheduler 里拿到 schedule.events（一串事件：Gate/Split/Move/Merge）
    2) 用 replay_traps 维护“每个 trap 当前有哪些离子（链）”
    3) 用 ion_heating 维护“每个离子的平均热激发数 n̄（累积）”
    4) 每个事件会：
       - 更新离子加热（n̄）
       - 更新陷阱背景热状态 H_i，并映射成 B_i（如果启用）
       - 给总保真度 acc 乘上对应的门保真度/动态惩罚

    与论文对齐：
    【论文固定、不应在 knobs 调】
      - 环境项形式：exp(-t/T1) - k*n̄
      - 门固有保真度常量：1Q=0.9999, fiber=0.99, 2Q=1-eps*N^2
      - 加热增量：Split=1, Merge=1, Swap=0.3, Move=0.1 per 2um(=0.05/um)

    【实现细节/可调】
      - Bi 背景模型形式/强度
      - “加热注入”按链/按离子口径
      - Move 的距离模型如何取段长度
      - 门环境项用全局时间还是门持续时间
      - shuttling fidelity 是按整次 shuttle 聚合，还是按 Split/Move/Merge 每步乘一次

    当前版本的 B_i 实现：
      - 维护 trap_heat_state（每个 trap 的累计背景热状态）
      - 再由当前热状态通过指数形式映射：
            B_i = exp(-alpha_bg * trap_heat_state[i])
      - Move heat 不默认全部写入背景，而是在 merge 时把
            move_bg_fraction * (该 shuttle 的 total move heat)
        注入目标 trap 的背景热状态
    """

    # ========== 论文固定常量（不要在 knobs 里调） ==========
    T1_US = 600 * 1e6                 # T1（单位 us）
    K_HEATING = 0.001                 # 环境项中对 n̄ 的线性惩罚系数 k
    EPSILON_2Q = 1.0 / 25600.0        # 2Q 门误差参数 epsilon

    # 加热注入（论文表格）
    HEAT_SPLIT = 1.0
    HEAT_MERGE = 1.0
    HEAT_SWAP = 0.3
    HEAT_MOVE_PER_UM = 0.1 / 2.0      # 0.1 per 2um => 0.05 per um

    # 门固有保真度（论文表格）
    FID_1Q = 0.9999
    FID_FIBER = 0.99

    def __init__(self, scheduler_obj, machine_obj, init_mapping, knobs: AnalyzerKnobs = None):
        # scheduler/schedule/machine 是外部系统提供的对象
        self.scheduler = scheduler_obj
        self.schedule = scheduler_obj.schedule
        self.machine = machine_obj

        # init_map：初始 trap -> 离子列表（链）
        # scheduler_obj 里可能已经有 init_map，否则用传入 init_mapping
        self.init_map = getattr(scheduler_obj, "init_map", init_mapping)

        # knobs：可调实现开关
        self.knobs = knobs if knobs is not None else AnalyzerKnobs()

        # 若 knobs.alpha_bg == 0，则尝试从 machine.mparams 里读取默认 alpha_bg
        # （相当于：knobs 没设就用机器参数）
        if self.knobs.alpha_bg == 0.0:
            if hasattr(self.machine, "mparams") and hasattr(self.machine.mparams, "alpha_bg"):
                self.knobs.alpha_bg = float(self.machine.mparams.alpha_bg)

        # trap_heat_state：每个 trap 的累计背景热状态（初始为 0）
        self.trap_heat_state = {t.id: 0.0 for t in self.machine.traps}

        # trap_bg：每个 trap 当前的背景因子 Bi（初始由 heat_state 映射得到，为 1）
        self.trap_bg = {t.id: 1.0 for t in self.machine.traps}

        # ion_heating：每个 ion 的 n̄（平均热激发数）累计值
        # 这里用一个“宽松上界”给 ion id 分配空间，避免运行中 KeyError
        capacity = self.machine.traps[0].capacity if self.machine.traps else 20
        num_traps = len(self.machine.traps)
        max_ions = num_traps * capacity * 4
        self.ion_heating = {i: 0.0 for i in range(max_ions + 256)}

        # -------- 统计/输出用变量 --------
        self.final_fidelity = 1.0
        self.prog_fin_time = 0.0
        self.op_count = {}

        # 用于统计“两比特门所在链长”的分布
        self.gate_chain_lengths = []

        # 分拆统计：门相关乘积、门次数、门时的平均 n̄
        self._gate_mult = 1.0
        self._gate_cnt = 0
        self._gate_avg_n = []

        # 动力学（split/move/merge）相关统计
        self._dyn_mult = 1.0
        self._dyn_cnt = 0
        self._dyn_min = 1.0

        # shuttle 聚合统计（aggregate 模式）
        self._shuttle_acc_time = {}       # shuttle_id -> total dt
        self._shuttle_acc_heat = {}       # shuttle_id -> total heat
        self._shuttle_acc_move_heat = {}  # shuttle_id -> total move heat
        self._shuttle_mult = 1.0
        self._shuttle_cnt = 0
        self._shuttle_min = 1.0
        self._shuttle_info = {}           # shuttle_id -> {"split":..., "move":..., "merge":..., "swap_cnt":...}

    # -------------------------
    # 工具函数
    # -------------------------
    def _seg_length_um(self, seg_id: int) -> float:
        """
        获取某个“移动段/segment”的长度（um）
        - 优先调用 machine.get_segment_length_um(seg_id)（如果机器实现了）
        - 否则尝试 machine.segments[seg_id].length
        - 再不行用 machine.mparams.segment_length_um 或默认 53um
        """
        if hasattr(self.machine, "get_segment_length_um"):
            try:
                return float(self.machine.get_segment_length_um(seg_id))
            except Exception:
                pass

        # 优先使用 segments_by_id，避免 seg_id != list index 时错取
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
        """计算某条链（一组离子）的平均 n̄。"""
        if not ions:
            return 0.0
        return sum(self.ion_heating.get(i, 0.0) for i in ions) / float(len(ions))

    def _refresh_bg(self, trap_id: int):
        """
        根据当前 trap_heat_state 刷新 trap_bg。
        """
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
            # 兼容保留：若仍想试线性模型，可使用一次性映射而非连乘递推
            self.trap_bg[trap_id] = max(1.0 - a * H, 0.0)
        else:
            # 未知模型，保守回退
            self.trap_bg[trap_id] = 1.0

        self.trap_bg[trap_id] = min(max(self.trap_bg[trap_id], 0.0), 1.0)

    def _apply_bg(self, trap_id: int, delta_avg_nbar: float):
        """
        更新某个 trap 的背景热状态，再映射为背景因子 Bi。

        当前实现：
          trap_heat_state[trap_id] += delta_avg_nbar
          B_i = exp(-alpha_bg * trap_heat_state[trap_id])

        说明：
        - 论文说明了存在 zone background fidelity B_i，但未给出显式更新公式
        - 当前采用 heat-state + exponential B_i，更适合统一解释 GHZ / BV / Adder
        - 仍保留 bg_model="linear" 兼容实验对比
        """
        if trap_id is None:
            return
        if self.knobs.bg_model == "none":
            self.trap_bg[trap_id] = 1.0
            return

        self.trap_heat_state[trap_id] = self.trap_heat_state.get(trap_id, 0.0) + float(delta_avg_nbar)
        self._refresh_bg(trap_id)

    def _env_fidelity(self, t_us: float, avg_nbar: float) -> float:
        """
        论文环境项（Eq.1 形式）：
            f_env = exp(-t/T1) - k*n̄
        并做下界截断，避免出现负数或 0 导致整体乘积归零。
        """
        f = math.exp(-float(t_us) / self.T1_US) - self.K_HEATING * float(avg_nbar)
        return max(f, 1e-15)

    def _gate_fidelity(self, chain_ions, is_2q: bool, is_fiber: bool,
                       gate_start_us: float, gate_end_us: float, trap_id: int):
        """
        计算一次 Gate 事件的保真度分解。

        默认（更贴近论文完整模型）：
            fidelity = B_i * f_g

        可选增强模式：
            fidelity = f_g * f_env * B_i
        由 knobs.gate_use_env / knobs.gate_use_bg 控制。

        其中：
          - f_g：门固有保真度（论文表格）
              fiber: 0.99
              2Q: 1 - eps * N^2   （N 为链长）
              1Q: 0.9999
          - f_env：环境项（论文公式），时间口径由 knobs.gate_env_time_mode 决定
          - Bi：该 trap 的背景因子（由 heat-state 映射得到）
        """
        N = len(chain_ions)

        # 对空链做保守保护，避免崩溃；理论上正常 gate 不应在空链上发生
        if N <= 0:
            avg_n = 0.0
            f_env = 1.0
            self._refresh_bg(trap_id)
            B = self.trap_bg.get(trap_id, 1.0)
            f_g = 1e-15
            return 1e-15, avg_n, f_env, f_g, B

        # ---- 门固有保真度 f_g（论文固定）----
        if is_fiber:
            f_g = self.FID_FIBER
        elif is_2q:
            f_g = 1.0 - self.EPSILON_2Q * (N ** 2)
        else:
            f_g = self.FID_1Q
        f_g = max(f_g, 1e-15)

        # ---- 环境项 f_env（论文公式，时间口径可调）----
        avg_n = self._avg_nbar(chain_ions)
        if self.knobs.gate_env_time_mode == "duration":
            t_env = float(gate_end_us - gate_start_us)  # 用门时长
        else:
            t_env = float(gate_end_us)                  # 用全局累计时间

        f_env = self._env_fidelity(t_env, avg_n)

        # ---- 背景因子 Bi（trap 粒度）----
        self._refresh_bg(trap_id)
        B = self.trap_bg.get(trap_id, 1.0)

        # 推荐论文完整模式：gate_use_bg=True, gate_use_env=False
        fid = f_g
        if self.knobs.gate_use_env:
            fid *= f_env
        if self.knobs.gate_use_bg:
            fid *= B

        return max(fid, 1e-15), avg_n, f_env, f_g, B

    def _dyn_event_mult(self, dt_us: float, delta_avg_nbar: float):
        """
        动力学事件（Split/Move/Merge）额外引入的乘性惩罚（旧实现兼容）：
            f_dyn = exp(-dt/T1) - k*Δn̄

        该逻辑只在 shuttle_fidelity_mode == "per_event" 时使用。
        """
        f_dyn = self._env_fidelity(dt_us, delta_avg_nbar)
        self._dyn_mult *= f_dyn
        self._dyn_cnt += 1
        self._dyn_min = min(self._dyn_min, f_dyn)
        return f_dyn

    def _accumulate_shuttle(self, shuttle_id, dt_us: float, delta_heat: float, etype: str = None, swap_cnt: int = 0):
        """aggregate 模式下，累计某次 shuttle 的总时间与总加热。"""
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
        aggregate 模式下，在 merge 时把整次 shuttle 的时间/加热统一映射成一次 fidelity。
        若没有 shuttle_id，则返回 1。
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

        return f_sh

    # -------------------------
    # 主逻辑：回放 schedule
    # -------------------------
    def move_check(self):
        """
        回放 schedule.events，逐事件更新：
          - replay_traps（trap -> 当前离子链）
          - ion_heating（每个离子的 n̄）
          - trap_bg（每个 trap 的 Bi）
          - acc（总保真度乘积）
        并记录统计信息。
        """
        # 统计各类操作次数
        self.op_count = {
            Schedule.Gate: 0,
            Schedule.Split: 0,
            Schedule.Move: 0,
            Schedule.Merge: 0
        }

        # replay_traps：当前每个 trap 的离子链（从 init_map 初始化）
        replay_traps = {t.id: [] for t in self.machine.traps}
        for t_id, ions in self.init_map.items():
            if t_id in replay_traps:
                replay_traps[t_id] = ions[:]

        # 计算程序完工时间（schedule 中事件的最大 end time）
        self.prog_fin_time = 0.0
        for ev in self.schedule.events:
            self.prog_fin_time = max(self.prog_fin_time, ev[3])

        # 总保真度累乘
        acc = 1.0

        # ========== 逐事件回放 ==========
        for ev in self.schedule.events:
            etype = ev[1]                     # 事件类型：Gate/Split/Move/Merge
            st, ed = float(ev[2]), float(ev[3])
            dt = max(ed - st, 0.0)           # 事件持续时间（us）
            info = ev[4]                     # 事件附加信息字典

            # 计数
            if etype in self.op_count:
                self.op_count[etype] += 1

            # ---- 1) Gate 事件：只乘门保真度，不直接改加热 ----
            if etype == Schedule.Gate:
                trap = info.get("trap", None)
                ions = info.get("ions", [])
                chain = replay_traps.get(trap, [])

                # 根据当前 trap 链长、门类型等计算 gate fidelity
                fid, avg_n, f_env, f_g, B = self._gate_fidelity(
                    chain_ions=chain,
                    is_2q=(len(ions) == 2),             # 用 ions 长度判断是否 2Q 门
                    is_fiber=info.get("is_fiber", False),
                    gate_start_us=st,
                    gate_end_us=ed,
                    trap_id=trap,
                )

                # 总乘积
                acc *= fid

                # 统计项
                self._gate_mult *= fid
                self._gate_cnt += 1
                self._gate_avg_n.append(avg_n)

                # 仅对 2Q 门记录链长分布
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

            # ---- 2) Split 事件：链内所有离子加热 + 可能的 swap 加热；并移除 moving ions ----
            elif etype == Schedule.Split:
                trap = info["trap"]
                moving_ions = info.get("ions", [])
                swap_cnt = int(info.get("swap_cnt", 0))
                shuttle_id = info.get("shuttle_id", None)

                chain = replay_traps.get(trap, [])
                L = len(chain)

                # (a) Split 加热注入：HEAT_SPLIT
                if self.knobs.inject_norm == "chain" and L > 0:
                    d_split = self.HEAT_SPLIT / float(L)
                else:
                    d_split = self.HEAT_SPLIT

                # 将 split 加热加到 trap 当前链上的每个离子
                if L > 0:
                    for ion in chain:
                        self.ion_heating[ion] += d_split
                    # 同时把这次“平均加热增量口径”用于更新 Bi
                    self._apply_bg(trap, d_split)

                # (b) swap 加热：HEAT_SWAP * swap_cnt
                if swap_cnt > 0:
                    if self.knobs.swap_norm == "chain" and L > 0:
                        d_swap = (self.HEAT_SWAP * swap_cnt) / float(L)
                    else:
                        d_swap = (self.HEAT_SWAP * swap_cnt)
                    if L > 0:
                        for ion in chain:
                            self.ion_heating[ion] += d_swap
                    # swap 发生在该 trap/zone 内，应同时计入背景热状态
                    self._apply_bg(trap, d_swap)
                else:
                    d_swap = 0.0

                # (c) 动力学惩罚
                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(
                        shuttle_id,
                        dt,
                        d_split + d_swap,
                        etype="split",
                        swap_cnt=swap_cnt,
                    )
                    f_dyn = 1.0
                else:
                    f_dyn = self._dyn_event_mult(dt, d_split + d_swap)
                    acc *= f_dyn

                # (d) 真正的“分裂动作”：把 moving_ions 从该 trap 的链中移除
                for ion in moving_ions:
                    if ion in chain:
                        chain.remove(ion)

                if self.knobs.debug_events:
                    print(
                        "[DBG SPLIT]",
                        "trap", trap, "L", L,
                        "dt", dt,
                        "swap_cnt", swap_cnt,
                        "shuttle_id", shuttle_id,
                        "d_split", round(d_split, 6),
                        "d_swap", round(d_swap, 6),
                        "f_dyn", round(f_dyn, 6)
                    )

            # ---- 3) Move 事件：移动的离子加热（按距离或常数）+ 动力学惩罚 ----
            elif etype == Schedule.Move:
                ions = info.get("ions", [])
                dst_seg = info.get("dest_seg", None)
                shuttle_id = info.get("shuttle_id", None)

                # (a) 计算 Move 的加热量
                # 默认按移动段长度 dist_um * HEAT_MOVE_PER_UM
                if self.knobs.move_heat_use_distance:
                    dist_um = self._seg_length_um(dst_seg) if dst_seg is not None else float(
                        getattr(self.machine.mparams, "segment_length_um", 53.0)
                    )
                    heat = dist_um * self.HEAT_MOVE_PER_UM
                else:
                    heat = self.knobs.move_heat_const

                # (b) 对参与移动的离子加热
                for ion in ions:
                    self.ion_heating[ion] += heat

                # (c) 动力学惩罚
                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, heat, etype="move")
                    f_dyn = 1.0
                else:
                    f_dyn = self._dyn_event_mult(dt, heat)
                    acc *= f_dyn

                if self.knobs.debug_events:
                    print(
                        "[DBG MOVE]",
                        "dst_seg", dst_seg,
                        "dt", dt,
                        "heat", round(heat, 6),
                        "shuttle_id", shuttle_id,
                        "f_dyn", round(f_dyn, 6)
                    )

            # ---- 4) Merge 事件：合并链，加热注入，动力学惩罚，并做“merge equalization” ----
            elif etype == Schedule.Merge:
                trap = info["trap"]
                incoming = info.get("ions", [])
                shuttle_id = info.get("shuttle_id", None)

                # (a) 合并：trap 原链 + incoming 离子
                new_chain = replay_traps.get(trap, []) + incoming
                replay_traps[trap] = new_chain
                L = len(new_chain)

                # (b) Merge 加热注入：HEAT_MERGE
                if self.knobs.inject_norm == "chain" and L > 0:
                    d_merge = self.HEAT_MERGE / float(L)
                else:
                    d_merge = self.HEAT_MERGE

                # 将 merge 加热加到合并后的整条链上
                if L > 0:
                    for ion in new_chain:
                        self.ion_heating[ion] += d_merge
                    # 更新 Bi
                    self._apply_bg(trap, d_merge)

                # (b.1) 将本次 shuttle 的 move heat 的一部分注入目标 trap 的背景热状态
                move_bg = self.knobs.move_bg_fraction * self._shuttle_acc_move_heat.get(shuttle_id, 0.0)
                if move_bg > 0:
                    self._apply_bg(trap, move_bg)

                # (c) 动力学惩罚
                if self.knobs.shuttle_fidelity_mode == "aggregate":
                    self._accumulate_shuttle(shuttle_id, dt, d_merge, etype="merge")
                    f_dyn = self._finalize_shuttle(shuttle_id)
                    acc *= f_dyn
                else:
                    f_dyn = self._dyn_event_mult(dt, d_merge)
                    acc *= f_dyn

                # (d) 你确认过的行为：merge 后做“链内加热均衡”
                if self.knobs.merge_equalize and new_chain:
                    avg_h = self._avg_nbar(new_chain)
                    for ion in new_chain:
                        self.ion_heating[ion] = avg_h

                if self.knobs.debug_events:
                    print(
                        "[DBG MERGE]",
                        "trap", trap, "L", L,
                        "dt", dt,
                        "d_merge", round(d_merge, 6),
                        "move_bg", round(move_bg, 6),
                        "shuttle_id", shuttle_id,
                        "f_dyn", round(f_dyn, 6)
                    )

        # 若 aggregate 模式下有未 finalize 的 shuttle（异常/尾部不完整），这里保守补结算
        if self.knobs.shuttle_fidelity_mode == "aggregate":
            pending_ids = list(self._shuttle_acc_time.keys())
            for sid in pending_ids:
                f_sh = self._finalize_shuttle(sid)
                acc *= f_sh
                if self.knobs.debug_events:
                    print("[DBG SHUTTLE-FINALIZE-LATE]", "shuttle_id", sid, "f_sh", round(f_sh, 6))

        # 回放结束：记录最终结果并打印统计
        self.final_fidelity = acc
        self._print_stats()

    def _print_stats(self):
        """
        输出汇总信息：程序时间、操作计数、2Q 门链长统计、最终 fidelity，
        以及 debug_summary 下的额外分解统计。
        """
        print(f"Program Finish Time: {self.prog_fin_time} us")
        print(
            "OPCOUNTS",
            "Gate:", self.op_count.get(Schedule.Gate, 0),
            "Split:", self.op_count.get(Schedule.Split, 0),
            "Move:", self.op_count.get(Schedule.Move, 0),
            "Merge:", self.op_count.get(Schedule.Merge, 0)
        )

        # 2Q 门所在链长统计
        if self.gate_chain_lengths:
            lens = np.array(self.gate_chain_lengths, dtype=float)
            print("\nTwo-qubit gate chain statistics")
            print(f"Mean: {np.mean(lens)} Max: {np.max(lens)}")

        print(f"Fidelity: {self.final_fidelity}")

        # 额外 debug 汇总
        if self.knobs.debug_summary:
            if self._gate_cnt > 0:
                avg_gate_n = float(np.mean(self._gate_avg_n)) if self._gate_avg_n else 0.0
                print(
                    f"[DBG SUMMARY] gates={self._gate_cnt}  "
                    f"gate_mult={self._gate_mult:.6g}  avg_gate_nbar={avg_gate_n:.4f}"
                )

            if self._dyn_cnt > 0:
                print(
                    f"[DBG SUMMARY] dyn_ops={self._dyn_cnt}  "
                    f"dyn_mult={self._dyn_mult:.6g}  min_dyn={self._dyn_min:.6g}"
                )

            if self._shuttle_cnt > 0:
                print(
                    f"[DBG SUMMARY] shuttles={self._shuttle_cnt}  "
                    f"shuttle_mult={self._shuttle_mult:.6g}  min_shuttle={self._shuttle_min:.6g}"
                )

            # 找到最差（最小）的 Bi
            if self.trap_bg:
                worst = min(self.trap_bg.items(), key=lambda x: x[1])
                worst_h = self.trap_heat_state.get(worst[0], 0.0)
                print(
                    f"[DBG SUMMARY] worst_B: Trap {worst[0]} -> {worst[1]:.6f} "
                    f"(heat_state={worst_h:.6f}, alpha_bg={self.knobs.alpha_bg}, "
                    f"move_bg_fraction={self.knobs.move_bg_fraction}, model={self.knobs.bg_model})"
                )

    def analyze_and_return(self):
        """
        外部调用接口：
          1) 先回放并计算
          2) 返回一个 dict，包含 fidelity / total_shuttle / time

        total_shuttle 的口径：
          - 如果 scheduler 有 shuttle_counter，用它
          - 否则用 Split 的次数近似
        """
        self.move_check()
        if hasattr(self.scheduler, "shuttle_counter"):
            shuttle_count = int(getattr(self.scheduler, "shuttle_counter"))
        else:
            shuttle_count = int(self.op_count.get(Schedule.Split, 0))
        return {
            "fidelity": self.final_fidelity,
            "total_shuttle": shuttle_count,
            "time": self.prog_fin_time
        }
