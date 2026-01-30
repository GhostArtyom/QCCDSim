import math
from utils import *
from machine_state import *
from schedule import *


class Analyzer:
    def __init__(self, scheduler_obj, machine_obj, init_mapping):
        self.scheduler = scheduler_obj
        self.schedule = scheduler_obj.schedule
        self.machine = machine_obj

        # 合法化初始映射
        if hasattr(scheduler_obj, "init_map"):
            self.init_map = scheduler_obj.init_map
        else:
            self.init_map = init_mapping

        # === MUSS-TI Paper Parameters (Table 1 & Eq 1) ===
        # T1 Coherence Time: 600 seconds = 600 * 10^6 us
        self.T1 = 600 * 1e6

        # Heating Rate Coefficient (k)
        self.k = 0.001

        # Gate Fidelity Decay Coefficient (epsilon)
        # Formula: F = 1 - epsilon * N^2
        self.epsilon = 1 / 25600.0

        # Heating Quanta per Operation (Table 1)
        # "Average motional quanta increase"
        self.HEAT_SPLIT = 1.0
        self.HEAT_MERGE = 1.0
        self.HEAT_MOVE = 0.1
        self.HEAT_SWAP = 0.3

        # Base Fidelities
        self.FID_FIBER = 0.99

        # === [严谨修正] 离子级热量追踪 ===
        # 记录每个离子的当前热量 (n_bar)。
        # 初始时假设冷却到基态 (n=0)。
        capacity = self.machine.traps[0].capacity if self.machine.traps else 20
        num_traps = len(self.machine.traps)
        max_ions = num_traps * capacity * 2

        # key: ion_id, value: accumulated n_bar
        self.ion_heating = {i: 0.0 for i in range(max_ions + 50)}

    def compute_gate_fidelity(self, chain_ions, is_fiber=False):
        """
        Calculates fidelity exactly based on MUSS-TI Section 4.
        """
        N = len(chain_ions)
        if N == 0:
            return 0.0

        # 1. Chain Length Penalty (epsilon * N^2)
        if is_fiber:
            f_gate = self.FID_FIBER
        else:
            # Table 1: 2-qubit gate fidelity
            f_gate = 1.0 - self.epsilon * (N**2)

        # 2. Heating Penalty (k * n_bar)
        # 计算当前链的平均热量
        total_n = sum(self.ion_heating[ion] for ion in chain_ions)
        avg_n_bar = total_n / N

        heating_penalty = self.k * avg_n_bar
        b_i = math.exp(-heating_penalty)

        # Total
        return f_gate * b_i

    def move_check(self):
        """
        Strict Replay: Tracks Ion positions and Ion heating.
        """
        op_count = {Schedule.Gate: 0, Schedule.Split: 0, Schedule.Move: 0, Schedule.Merge: 0}
        op_times = {Schedule.Gate: 0.0, Schedule.Split: 0.0, Schedule.Move: 0.0, Schedule.Merge: 0.0}

        # === 1. 重建初始物理分布 ===
        # replay_traps[trap_id] = [ion_list]
        replay_traps = {t.id: [] for t in self.machine.traps}
        for t_id, ions in self.init_map.items():
            if t_id in replay_traps:
                replay_traps[t_id] = ions[:]

        prog_fin_time = 0.0
        log_fidelity = 0.0

        for event in self.schedule.events:
            prog_fin_time = max(prog_fin_time, event[3])

        print(f"Program Finish Time: {prog_fin_time} us")

        # === 2. 事件重演 ===
        for event in self.schedule.events:
            etype = event[1]
            start_t, end_t = event[2], event[3]
            duration = end_t - start_t
            info = event[4]

            if etype in op_count:
                op_count[etype] += 1
                op_times[etype] += duration

            # --- GATE ---
            if etype == Schedule.Gate:
                ions = info["ions"]
                trap = info["trap"]

                if len(ions) == 2:
                    # 获取当前 Trap 内所有离子（不仅仅是参与门的离子，因为链长N影响所有人）
                    chain_ions = replay_traps.get(trap, [])

                    is_fiber = info.get("is_fiber", False)

                    fid = self.compute_gate_fidelity(chain_ions, is_fiber)

                    if fid > 0:
                        log_fidelity += math.log(fid)
                    else:
                        log_fidelity += -999.0  # Fidelity Collapse

            # --- SPLIT ---
            elif etype == Schedule.Split:
                trap = info["trap"]
                moving_ions = info["ions"]
                staying_ions = [i for i in replay_traps[trap] if i not in moving_ions]

                # [物理逻辑] Split 操作作用于整个晶体，增加热量
                # 无论是离开的还是留下的，都经历了 Split 过程
                # Table 1: Split heating = 1.0
                for ion in replay_traps[trap]:
                    self.ion_heating[ion] += self.HEAT_SPLIT

                # Swap Penalty
                # 如果 Split 需要交换顺序，热量加在 Trap 内所有离子上
                total_swaps = info.get("ion_hops", 0) + info.get("swap_hops", 0)
                if total_swaps > 0:
                    for ion in replay_traps[trap]:
                        self.ion_heating[ion] += self.HEAT_SWAP * total_swaps

                # 更新位置状态
                for ion in moving_ions:
                    if ion in replay_traps[trap]:
                        replay_traps[trap].remove(ion)

            # --- MOVE ---
            elif etype == Schedule.Move:
                ions = info["ions"]
                # [物理逻辑] 只有在飞行的离子会因为电场噪声加热
                # Table 1: Move heating = 0.1
                for ion in ions:
                    self.ion_heating[ion] += self.HEAT_MOVE

            # --- MERGE ---
            elif etype == Schedule.Merge:
                trap = info["trap"]
                incoming_ions = info["ions"]
                existing_ions = replay_traps[trap]

                # 1. 更新位置状态（先合并列表）
                # 注意：这里简单的 append 模拟合并，具体顺序依赖 physics engine，但对 N 计数无影响
                new_chain = existing_ions + incoming_ions
                replay_traps[trap] = new_chain

                # 2. [物理逻辑] Merge 操作加热
                # Table 1: Merge heating = 1.0
                # 所有参与合并的离子都会变热
                for ion in new_chain:
                    self.ion_heating[ion] += self.HEAT_MERGE

                # 3. [物理逻辑] 热化 (Thermalization)
                # 当热离子和冷离子合并，它们会通过库伦相互作用交换能量
                # 我们假设它们瞬间达到热平衡，平均分摊热量
                total_heat = sum(self.ion_heating[i] for i in new_chain)
                avg_heat = total_heat / len(new_chain) if len(new_chain) > 0 else 0
                for ion in new_chain:
                    self.ion_heating[ion] = avg_heat

        # === 3. 最终计算 ===

        # T1 Decay (Global)
        t1_decay = math.exp(-prog_fin_time / self.T1)

        final_fidelity = math.exp(log_fidelity) * t1_decay

        # 统计总系统热量 (Sum of all ions)
        total_system_heating = sum(self.ion_heating.values())

        print("OPCOUNTS Gate:", op_count[Schedule.Gate], "Split:", op_count[Schedule.Split], "Move:", op_count[Schedule.Move], "Merge:", op_count[Schedule.Merge])

        print(f"Fidelity: {final_fidelity}")
        print(f"Total System Heating (quanta): {int(total_system_heating)}")
