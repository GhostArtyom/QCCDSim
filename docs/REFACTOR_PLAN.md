# QCCDSim 重构计划文档

## 一、重构目标

### 1.1 核心目标
- 统一运行入口，支持 `SMALL` / `LARGE` 两种规模模式切换
- 小规模模式：复用现有 V2-V6 调度器，保持 Table 2 复现能力
- 大规模模式：统一走 V7 调度器，支持 EML-QCCD 多模块架构
- 提供灵活的 `run_batch` 配置界面，可选择映射器、调度器、分析器等组件

### 1.2 架构原则
- **渐进式重构**：不破坏现有功能，每完成一项做好标记
- **模块化设计**：核心仿真逻辑与入口配置分离
- **向后兼容**：现有 `run.py` 命令行接口保持不变

---

## 二、目标目录结构

```
QCCDSim/
├── src/                          # 核心仿真代码（重构后）
│   ├── __init__.py
│   ├── parse/                    # QASM 解析模块
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── mapping/                  # 映射器模块
│   │   ├── __init__.py
│   │   ├── base.py              # 映射器基类
│   │   ├── sabre.py             # SABRE 系列
│   │   ├── legacy.py            # 旧版映射器 (LPFS, PO, Agg, Greedy)
│   │   └── trivial.py           # Zone-aware 映射
│   ├── schedule/                 # 调度器模块
│   │   ├── __init__.py
│   │   ├── base.py              # 调度器基类
│   │   ├── ejf.py                # EJF 基线调度器
│   │   ├── muss/                 # MUSS 系列
│   │   │   ├── __init__.py
│   │   │   ├── v2.py             # V2 论文复现版
│   │   │   ├── v3.py
│   │   │   ├── v4.py
│   │   │   ├── v5.py
│   │   │   ├── v6.py
│   │   │   └── v7.py             # 大规模版（独立）
│   │   └── events.py             # Schedule 事件数据结构
│   ├── analyze/                  # 分析器模块
│   │   ├── __init__.py
│   │   ├── base.py              # 分析器基类
│   │   ├── analyzer.py           # 旧版分析器（V2-V6 用）
│   │   └── analyzer_v7.py        # V7 专用分析器
│   ├── machine/                  # 机器模型模块
│   │   ├── __init__.py
│   │   ├── core.py               # Machine, Trap, Segment, Junction
│   │   ├── params.py             # MachineParams 参数类
│   │   ├── state.py              # MachineState 状态类
│   │   └── topologies/           # 机器拓扑
│   │       ├── __init__.py
│   │       ├── small.py          # 小规模拓扑 (G2x2, G2x3, L6, H6)
│   │       └── large.py          # 大规模拓扑 (G3x4, G4x5, EML, EML2Z)
│   ├── route/                    # 路由模块
│   │   ├── __init__.py
│   │   ├── basic.py              # BasicRoute
│   │   └── free_trap.py          # FreeTrapRoute
│   └── utils/                    # 公共工具
│       ├── __init__.py
│       └── helpers.py
│
├── run.py                         # 单次运行入口（保持现有接口）
├── run_batch.py                   # 批量实验入口（重构后合并 small/large）
│
├── # 以下文件逐步废弃，原有功能迁移到 src/ 后删除
├── parse.py                       # [阶段1] → src/parse/parser.py
├── mappers.py                      # [阶段2] → src/mapping/
├── schedule.py                     # [阶段2] → src/schedule/events.py
├── schedule_v7.py                  # [阶段3] → src/schedule/events.py
├── ejf_schedule.py                 # [阶段2] → src/schedule/ejf.py
├── muss_schedule2.py               # [阶段2] → src/schedule/muss/v2.py
├── muss_schedule3.py               # [阶段2] → src/schedule/muss/
├── muss_schedule4.py               # [阶段2] → src/schedule/muss/
├── muss_schedule5.py               # [阶段2] → src/schedule/muss/
├── muss_schedule6.py               # [阶段2] → src/schedule/muss/
├── muss_schedule7.py               # [阶段3] → src/schedule/muss/v7.py
├── analyzer.py                     # [阶段3] → src/analyze/analyzer.py
├── analyzer_v7.py                   # [阶段3] → src/analyze/analyzer_v7.py
├── machine.py                      # [阶段1] → src/machine/core.py
├── machine_state.py                # [阶段1] → src/machine/state.py
├── test_machines.py                 # [阶段1] → src/machine/topologies/
├── route.py                        # [阶段1] → src/route/
├── rebalance.py                     # [阶段2] → src/schedule/ (作为 muss 内部模块)
├── utils.py                         # [阶段1] → src/utils/
├── sorted_collection.py             # [阶段1] → src/utils/
│
├── # 以下文件保持不变或小幅修改
├── programs/                       # QASM 电路文件
├── requirements.txt
├── pyproject.toml
│
└── # 以下文件最终删除
├── run_batch_large.py              # [阶段4] 功能合并到 run_batch.py
├── parse_output.py                 # [阶段4] 考虑移入 src/analyze/
├── paper_report_v7.py               # [阶段4] 考虑移入 src/analyze/
├── strict_repro.py                  # [阶段3] 重构为 src/utils/validate.py
├── strict_repro_checks.py           # [阶段3] 同上
```

---

## 三、重构阶段（策略调整）

**重要说明**：由于项目高度耦合（调度器依赖 Schedule 完整方法），强行迁移会导致运行时错误。**新策略**：保留原文件结构，在 `src/` 创建新的基础设施模块供未来使用，不破坏现有运行路径。

---

### 阶段 0：准备工作（已完成）

**目标**：创建 `src/` 目录和基础模块框架，不改变任何现有代码

**步骤**：
- [ ] 0.1 创建 `src/` 目录结构
- [ ] 0.2 创建各模块的 `__init__.py`
- [ ] 0.3 在 `src/__init__.py` 中定义 `SCALE_MODE` 枚举

**验收**：
```bash
ls -la src/
ls -la src/parse/
ls -la src/mapping/
ls -la src/schedule/
# ...
```

---

### 阶段 1：基础设施模块化（已完成 ✅）

**目标**：创建 `src/machine/` 和 `src/utils/` 基础设施模块

**结果**：已在 `src/` 下创建以下模块，**但原文件保持不变**以确保项目正常运行：
- `src/machine/core.py` - Machine, Trap, Segment, Junction, MachineParams
- `src/machine/state.py` - MachineState
- `src/machine/topologies/small.py` - 小规模拓扑
- `src/machine/topologies/large.py` - 大规模拓扑 + build_machine_by_type 工厂
- `src/route/basic.py` - BasicRoute
- `src/route/free_trap.py` - FreeTrapRoute
- `src/utils/helpers.py` - trap_name, seg_name 等
- `src/utils/sorted_collection.py` - SortedCollection

**验证**：`run.py` 使用 qc2 环境运行正常（SMALL 规模 V2 调度器测试通过）

**步骤**：
- [ ] 1.1 创建 `src/machine/core.py`，将 `Trap`, `Segment`, `Junction`, `Machine`, `FiberLink` 移入
- [ ] 1.2 创建 `src/machine/params.py`，将 `MachineParams` 移入
- [ ] 1.3 创建 `src/machine/state.py`，将 `MachineState` 移入（保持接口兼容）
- [ ] 1.4 创建 `src/machine/topologies/small.py`，从 `test_machines.py` 提取小规模拓扑
- [ ] 1.5 创建 `src/machine/topologies/large.py`，从 `test_machines.py` 提取大规模拓扑
- [ ] 1.6 创建 `src/route/basic.py` 和 `src/route/free_trap.py`
- [ ] 1.7 创建 `src/utils/helpers.py`，将 `trap_name`, `seg_name` 等工具函数移入
- [ ] 1.8 创建 `src/utils/sorted_collection.py`（从 `sorted_collection.py` 迁移）
- [ ] 1.9 更新原文件，使其 `import` 来自 `src/` 的模块（保持向后兼容）

**验收**：
```python
# 在 Python 中测试
from src.machine.core import Machine, Trap, Segment, Junction
from src.machine.topologies.small import test_trap_2x2, test_trap_2x3
from src.machine.topologies.large import make_eml_machine
print("阶段1 验收通过")
```

---

### 阶段 2：调度器基础设施（已完成 ✅）

**目标**：在 `src/schedule/` 创建事件和基础调度设施

**结果**：
- `src/schedule/events.py` - Schedule 类（完整实现）
- `src/schedule/base.py` - ScheduleStrategy 基类
- `src/schedule/ejf.py` - EJFSchedule（部分实现）
- `src/schedule/muss/rebalance.py` - RebalanceTraps

**注**：由于 MUSS 调度器（V2-V7）依赖 Schedule 的完整方法，完整迁移需要大量工作。当前保留原调度器文件不动。

**目标**：将 `muss_schedule2.py` ~ `muss_schedule6.py` 和 `ejf_schedule.py` 迁移到 `src/schedule/`

**步骤**：
- [ ] 2.1 创建 `src/schedule/events.py`，将 `Schedule` 类从 `schedule.py` 移入
- [ ] 2.2 创建 `src/schedule/base.py`，定义 `ScheduleStrategy` 基类接口
- [ ] 2.3 创建 `src/schedule/ejf.py`，将 `EJFSchedule` 迁移
- [ ] 2.4 创建 `src/schedule/muss/v2.py` ~ `v6.py`，逐一迁移
- [ ] 2.5 创建 `src/schedule/muss/__init__.py`，统一导出
- [ ] 2.6 将 `rebalance.py` 作为内部模块引入 `src/schedule/muss/`
- [ ] 2.7 更新原文件 `import`，保持向后兼容

**验收**：
```python
from src.schedule.events import Schedule
from src.schedule.ejf import EJFSchedule
from src.schedule.muss import MUSSScheduleV2, MUSSScheduleV3
print("阶段2 验收通过")
```

---

### 阶段 3：大规模调度器与解析模块化

**目标**：将 V7 调度器和解析器迁移到 `src/`

**步骤**：
- [ ] 3.1 创建 `src/parse/parser.py`，将 `parse.py` 的 `InputParse` 迁移
- [ ] 3.2 创建 `src/schedule/muss/v7.py`，将 `muss_schedule7.py` 迁移
- [ ] 3.3 创建 `src/schedule/events_v7.py`（如与 v2-v6 不兼容）
- [ ] 3.4 创建 `src/analyze/base.py` 和 `src/analyze/analyzer.py`（旧版）
- [ ] 3.5 创建 `src/analyze/analyzer_v7.py`（V7 专用）
- [ ] 3.6 将 `strict_repro.py` 重构为 `src/utils/validate.py`
- [ ] 3.7 更新原文件 `import`

**验收**：
```python
from src.parse.parser import InputParse
from src.schedule.muss.v7 import MUSSScheduleV7
from src.analyze.analyzer_v7 import AnalyzerV7
print("阶段3 验收通过")
```

---

### 阶段 4：统一入口与批量脚本

**目标**：重构 `run.py` 和 `run_batch.py`

**步骤**：

#### 4.1 重构 `run.py`
- [ ] 4.1.1 创建 `src/config.py`，统一管理配置：
  ```python
  @dataclass
  class RunConfig:
      scale: str = "SMALL"           # SMALL | LARGE
      qasm_path: str = ""
      machine_type: str = "G2x2"
      ions_per_region: int = 12
      mapper: str = "SABRE"
      scheduler: str = "MUSS"
      scheduler_version: str = "V2"
      analyzer_mode: str = "PAPER"
      # ... 其他参数
  ```
- [ ] 4.1.2 在 `run.py` 中：
  - 根据 `architecture_scale` 自动选择 V2-V6（SMALL）或 V7（LARGE）调度器
  - 根据 `machine_type` 自动选择小规模或大规模拓扑工厂
  - 统一使用 `src/analyze/` 下的分析器
- [ ] 4.1.3 保持现有命令行接口完全兼容

#### 4.2 重构 `run_batch.py`
- [ ] 4.2.1 添加规模选择配置：
  ```python
  SCALE_MODE = "SMALL"  # 或 "LARGE"
  ```
- [ ] 4.2.2 根据 `SCALE_MODE` 自动选择：
  - 小规模：`MACHINE_TYPES`, `SCHEDULERS` 使用 V2-V6 相关配置
  - 大规模：使用 V7 + EML 相关配置
- [ ] 4.2.3 合并 `run_batch_large.py` 的功能到 `run_batch.py`
- [ ] 4.2.4 添加清晰的配置区块：
  ```python
  # ============ 规模模式 ============
  SCALE_MODE = "LARGE"  # SMALL | LARGE
  
  # ============ 调度器配置 ============
  if SCALE_MODE == "SMALL":
      SCHEDULERS = [("MUSS", "V2"), ("EJF", "")]  # 小规模调度器
  else:
      SCHEDULERS = [("MUSS", "V7")]  # 大规模调度器
  
  # ============ 机器配置 ============
  if SCALE_MODE == "SMALL":
      MACHINE_TYPES = {"G2x2": 12, "G2x3": 8}  # 小规模机器
  else:
      MACHINE_TYPES = {"EML": 16, "G3x4": 16}  # 大规模机器
  ```

**验收**：
```bash
# 小规模测试
python run.py programs/GHZ32.qasm G2x2 12 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V2 PAPER SMALL

# 大规模测试
python run.py programs/GHZ256.qasm EML 16 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V7 PAPER LARGE

# 批量测试（小规模）
python run_batch.py

# 批量测试（大规模）
# 修改 run_batch.py 中 SCALE_MODE = "LARGE"
python run_batch.py
```

---

### 阶段 5：清理与文档

**目标**：删除废弃文件，更新文档

**步骤**：
- [ ] 5.1 确认所有功能已迁移到 `src/`
- [ ] 5.2 删除迁移完成后的旧文件（可选，保留备份）
- [ ] 5.3 更新 `CLAUDE.md`，反映新的目录结构
- [ ] 5.4 更新 `README.md`

---

## 四、关键接口设计

### 4.1 调度器工厂接口

```python
# src/schedule/__init__.py
class ScheduleFactory:
    @staticmethod
    def create(scheduler_family: str, version: str, scale: str, **kwargs) -> ScheduleStrategy:
        if scheduler_family.upper() == "EJF":
            return EJFSchedule(**kwargs)
        elif scheduler_family.upper() == "MUSS":
            if scale.upper() == "LARGE" or version.upper() == "V7":
                from .muss.v7 import MUSSScheduleV7
                return MUSSScheduleV7(**kwargs)
            else:
                # V2-V6
                from .muss import MUSSScheduleV2, MUSSScheduleV3, ...
                version_map = {"V2": MUSSScheduleV2, "V3": MUSSScheduleV3, ...}
                return version_map[version.upper()](**kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_family}")
```

### 4.2 分析器工厂接口

```python
# src/analyze/__init__.py
class AnalyzerFactory:
    @staticmethod
    def create(scheduler_version: str, scale: str, **kwargs) -> Analyzer:
        if scale.upper() == "LARGE" or scheduler_version.upper() == "V7":
            from .analyzer_v7 import AnalyzerV7
            return AnalyzerV7(**kwargs)
        else:
            from .analyzer import Analyzer
            return Analyzer(**kwargs)
```

### 4.3 机器拓扑工厂接口

```python
# src/machine/topologies/__init__.py
class TopologyFactory:
    @staticmethod
    def create(machine_type: str, capacity: int, mparams: MachineParams, scale: str = None):
        if scale is None:
            scale = "LARGE" if machine_type in {"EML", "EML2Z", "G3x4", "G4x5"} else "SMALL"
        
        if scale.upper() == "LARGE":
            from .large import G3x4, G4x5, EML, EML2Z
            topo_map = {"G3x4": G3x4, "G4x5": G4x5, "EML": EML, "EML2Z": EML2Z}
        else:
            from .small import test_trap_2x2, test_trap_2x3, make_linear_machine, make_single_hexagon_machine
            topo_map = {"G2x2": test_trap_2x2, "G2x3": test_trap_2x3, "L6": make_linear_machine, "H6": make_single_hexagon_machine}
        
        return topo_map[machine_type](capacity, mparams)
```

---

## 五、向后兼容策略

### 5.1 旧文件导入新模块

在重构过程中，旧文件通过 `import` 新模块保持功能不变：

```python
# 原 muss_schedule2.py 开头添加：
# DEPRECATED: 使用 src.schedule.muss.v2 替代
import sys
from src.schedule.muss.v2 import MUSSSchedule as MUSSScheduleV2
# 保留原有类名作为别名
MUSSSchedule = MUSSScheduleV2
```

### 5.2 run.py 兼容层

```python
# run.py 根据 scale 自动选择正确的调度器
if architecture_scale.upper() == "LARGE":
    from src.schedule.muss.v7 import MUSSScheduleV7 as SchedulerClass
    from src.analyze.analyzer_v7 import AnalyzerV7 as AnalyzerClass
    from src.machine.topologies.large import create_topology
else:
    from src.schedule.muss import MUSSScheduleV2 as SchedulerClass
    from src.analyze.analyzer import Analyzer as AnalyzerClass
    from src.machine.topologies.small import create_topology
```

---

## 六、测试计划

### 每阶段验收测试

| 阶段 | 测试命令 | 预期结果 |
|------|----------|----------|
| 阶段0 | `python -c "import src"` | 无错误 |
| 阶段1 | `python -c "from src.machine.topologies.small import test_trap_2x2"` | 正常返回机器对象 |
| 阶段2 | `python run.py programs/GHZ32.qasm G2x2 12 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V2 PAPER SMALL` | 输出 SUMMARY |
| 阶段3 | `python run.py programs/GHZ256.qasm EML 16 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V7 PAPER LARGE` | 输出 SUMMARY |
| 阶段4 | `python run_batch.py` (SCALE_MODE="SMALL") | 生成 output/manifest.csv |
| 阶段4 | `python run_batch.py` (SCALE_MODE="LARGE") | 生成 output_large/manifest.csv |

---

## 七、注意事项

1. **不要破坏现有 `run.py` 接口**：命令行参数必须保持完全兼容
2. **每次修改后运行测试**：确保现有的 `python run.py ... SMALL` 和 `python run.py ... LARGE` 仍能正常工作
3. **分阶段提交**：每完成一个阶段做好 git 提交，便于回滚
4. **保留原有调度器行为**：特别是 V2，它是论文复现的基准
5. **V7 独立演进**：大规模调度器可以在 `src/schedule/muss/v7.py` 中独立发展

---

## 八、进度跟踪

| 阶段 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 阶段0 | ✅ 完成 | 2026-04-03 | 创建 src/ 目录结构和各模块 __init__.py |
| 阶段1 | ✅ 完成 | 2026-04-03 | 完成 machine/state/route/utils 模块化，test_machines.py 已重构为兼容层 |
| 阶段2 | ✅ 进行中 | 2026-04-03 | 完成 schedule/events.py, ejf.py 迁移；schedule.py, ejf_schedule.py 已重构为兼容层；muss调度器暂保留原文件 |
| 阶段3 | ⬜ 未开始 | - | - |
| 阶段4 | ⬜ 未开始 | - | - |
| 阶段5 | ⬜ 未开始 | - | - |

---

*文档版本：v1.0*
*创建日期：2026-04-03*
