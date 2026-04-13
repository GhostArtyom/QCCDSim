# CLAUDE.md

## 语言与输出约定

- 默认使用中文回复。
- 所有项目分析、代码解释、修改建议、调试说明、任务计划和总结报告都使用中文。
- 代码、命令、路径、配置键名、环境变量名、类名、函数名和报错原文保留英文。
- 除非我明确要求英文，否则不要使用英文回答正文内容。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

QCCDSim - QCCD (Quantum Charge-Coupled Device) Compiler and Simulator based on the MUSS-TI paper: https://ieeexplore.ieee.org/document/9138945

Simulates ion-trap quantum computer architectures with shuttling-based qubit movement. The codebase has two main tracks:
- **Small-scale** (Table 2 reproduction): Single QCCD with 6-12 traps
- **Large-scale** (EML-QCCD): Multi-module architecture with fiber interconnects

## Setup & Execution

```bash
cd QCCDSim
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

**Main entry point:**
```bash
python run.py <qasm> <machine_type> <ions_per_region> <mapper> <reorder> <serial_trap> <serial_comm> <serial_all> <gate_type> <swap_type> [sched_family] [sched_version] [analyzer_mode] [architecture_scale]
```

**Key Parameters:**
- `machine_type`: G2x2, G2x3, G3x4 (grid), L6 (linear), H6 (hexagon), EML (multi-module)
- `mapper`: LPFS, Agg, Random, PO, Greedy, SABRE, SABRELarge, Trivial
- `reorder`: Naive, Fidelity (qubit ordering within trap)
- `gate_type`: FM (40us fixed), PM, Duan, Trout
- `swap_type`: PaperSwapDirect (ion directly swaps with chain end), GateSwap, IonSwap
- `sched_family/version`: MUSS V2-V7, EJF (baseline)
- `analyzer_mode`: PAPER (Table 2 faithful), EXTENDED (with background heating)
- `architecture_scale`: SMALL (default for G2x2/G2x3/L6/H6), LARGE (for EML/G3x4/G4x5)

**Example commands:**
```bash
# Small-scale Table 2 reproduction
python run.py ghz32.qasm G2x2 12 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V2 PAPER SMALL

# Large-scale EML-QCCD with V7 scheduler
python run.py qft256.qasm EML 16 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V7 PAPER LARGE
```

**Batch execution:**
```bash
python run_batch.py      # Small-scale parameter sweeps
python run_batch_large.py  # Large-scale architecture sweeps
python parse_output.py   # Parse SUMMARY lines into CSV
```

## Architecture

### Core Pipeline (data flow)

```
OpenQASM → src/parse/ → src/mapping/ → scheduler → src/analyze/ → SUMMARY output
              ↓              ↓            ↓              ↓
         gate_graph   q→trap      events[]      fidelity,
         cx_graph     mapping                   shuttle_count
```

### 1. src/parse/parser.py - QASM Parsing
- `InputParse.parse_ir()` - Parses OpenQASM subset (cx, 1Q gates, barrier)
- Outputs:
  - `cx_gate_map`: gate_id → [q1, q2] for 2Q gates only
  - `gate_graph`: Full DAG including 1Q gates
  - `twoq_gate_graph`: 2Q-only DAG for SABRE/scheduler
  - `all_gate_map`: gate_id → {type, qubits}

### 2. src/mapping/mapper.py - Initial Qubit Mapping
- **Legacy mappers**: LPFS (longest path), PO (program order), Agg (agglomerative), Random, Greedy
- **SABRE family** (recommended for Table 2):
  - `QubitMapSABRE2`: Small-scale, returns `{layout: q→trap, trap_to_qubits: trap→[q...]}`
  - `QubitMapSABRELarge`: Large-scale EML-QCCD, module-aware placement
- **Trivial**: Zone-aware sequential placement (optical > operation > storage)

### 3. Schedulers (muss_schedule*.py, ejf_schedule.py)
- **EJFSchedule** (src/schedule/ejf.py): Baseline Earliest Job First with shuttling
- **MUSSSchedule V2-V7** (muss_schedule*.py):
  - V2: Paper-faithful for Table 2
  - V6: Stricter paper reproduction (no rebalance)
  - V7: Large-scale scheduler with swap insertion, cross-QCCD gates

### 4. src/schedule/events.py - Event Timeline
Event types (sorted by finish time):
- `Gate (1)`: 2Q gate execution
- `Split (2)`: Ion chain split for shuttling
- `Merge (3)`: Ion chain merge after shuttling
- `Move (4)`: Ion movement through segments

### 5. src/analyze/ - Fidelity Model
- `analyzer.py`: Legacy model with per-event or aggregate shuttle mode
- `analyzer_v7.py`: Paper-faithful model for V7
- **paper_mode**: F = B_i × F_gate × F_shuttle
  - F_gate: 1Q=0.9999, 2Q=1-εN², Fiber=0.99
  - F_shuttle: exp(-t/T1) - k×nbar (per aggregate shuttle)
- **extended_mode**: Adds background heating, merge equalization

### Machine Model (src/machine/)
```python
MachineParams:
  # MUSS-TI Table 1 fixed values
  split_merge_time = 80us
  ion_swap_time = 40us
  move_speed_um_per_us = 2.0
  junction_cross_time = 5us
  
  # Implementation knobs
  segment_length_um = 28.0 (small) / 80.0 (large)
  gate_type = "FM"|"PM"|"Duan"|"Trout"
  swap_type = "PaperSwapDirect"|"GateSwap"|"IonSwap"
```

## File Organization

**Core simulation (src/):**
- `src/parse/parser.py` - QASM parser
- `src/mapping/mapper.py` - Initial mapping (17 mapper classes including SABRE variants)
- `src/schedule/events.py` - Event data structure
- `src/schedule/ejf.py` - EJF baseline scheduler
- `src/schedule/base.py` - Scheduler abstract base class
- `src/analyze/analyzer.py` - Legacy fidelity analysis
- `src/analyze/analyzer_v7.py` - V7 fidelity analysis
- `src/machine/core.py` - Machine model and parameters
- `src/machine/state.py` - Runtime ion positions
- `src/route/` - Pathfinding (BasicRoute, FreeTrapRoute)
- `src/utils/` - Helper functions (sorted_collection.py)

**Entry points (root):**
- `run.py` - Main entry point with scheduler/analyzer selection
- `run_batch.py` - Small-scale parameter sweeps
- `run_batch_large.py` - Large-scale architecture sweeps
- `test_machines.py` - Machine topology definitions (retained for compatibility)

**Utilities (root):**
- `route.py` - Legacy pathfinding (retained for compatibility)
- `utils.py` - Legacy helpers (retained for compatibility)
- `rebalance.py` - Network simplex for trap rebalancing
- `sorted_collection.py` - Legacy sorted event list

**Analysis (root):**
- `parse_output.py` - Parse SUMMARY lines to CSV
- `paper_report_v7.py` - Generate paper-style reports
- `strict_repro.py` - Strict reproduction mode validation
- `strict_repro_checks.py` - Validation helpers

**Output directories:**
- `output/` - Small-scale results
- `output_large/` - Large-scale results with manifest.md

## Module Structure (src/)

```
src/
├── __init__.py           # ScaleMode enum, main class exports
├── analyze/              # Fidelity analysis
│   ├── __init__.py
│   ├── analyzer.py
│   └── analyzer_v7.py
├── machine/              # Machine models
│   ├── __init__.py
│   ├── core.py           # Machine, MachineParams
│   └── state.py          # MachineState
├── mapping/             # Qubit mapping
│   ├── __init__.py
│   └── mapper.py         # 17 mapper classes
├── parse/               # QASM parsing
│   ├── __init__.py
│   └── parser.py
├── route/               # Pathfinding
│   ├── __init__.py
│   ├── basic.py
│   └── free_trap.py
├── schedule/            # Scheduling strategies
│   ├── __init__.py
│   ├── base.py
│   ├── events.py
│   ├── ejf.py
│   └── muss/            # MUSS schedulers (future)
└── utils/              # Utilities
    ├── __init__.py
    └── sorted_collection.py
```

## Version History (git log)

Key commits:
- `e057f80`: Complete MUSS paper implementation (V2-V7, SABRE2/SABRELarge, analyzer_v7)
- `de11299`: SABRE2 returns ordered trap_to_qubits; V6 strict reproduction
- `bd4b071`: Trivial/SABRE skip qubit ordering; parser error on unsupported gates
- `6df92d6`: Large/small scale split; zone-aware partitioning

## Common Development Tasks

### Running a single circuit
```bash
python run.py programs/ghz32.qasm G2x2 12 SABRE Naive 1 1 1 FM PaperSwapDirect MUSS V2 PAPER SMALL
```

### Comparing scheduler versions
Edit `run_batch.py` SCHEDULERS list, then:
```bash
python run_batch.py
python parse_output.py output  # Generates CSV from SUMMARY lines
```

### Adding a new scheduler version
1. Create `muss_schedule8.py` inheriting from MUSSSchedule base pattern
2. Add import to `run.py` lines 100-103
3. Add version selection in `build_scheduler()` around line 654
4. Update this CLAUDE.md version table

### Modifying fidelity model
- `analyzer.py`: Legacy model with per-event or aggregate shuttle mode
- `analyzer_v7.py`: Paper-faithful model for V7
- Key methods: `_gate_fidelity()`, `_env_fidelity()`, `_finalize_shuttle()`

### Environment variables for large-scale sweeps
```bash
MUSS_TRAP_CAPACITY=16
MUSS_SWAP_LOOKAHEAD_K=8
MUSS_SWAP_THRESHOLD=4
MUSS_MAX_QUBITS_PER_QCCD=32
MUSS_NUM_OPTICAL_ZONES=1
MUSS_ENABLE_SWAP_INSERT=1
```

## Parameter Recommendations

**For Table 2 faithful reproduction:**
- Scheduler: MUSS V2 or V6
- Mapper: SABRE (auto-selects QubitMapSABRE2)
- Analyzer: PAPER mode
- Gate/Swap: FM, PaperSwapDirect
- serial_trap/comm/all: 1 (fully serial)

**For large-scale exploration:**
- Scheduler: MUSS V7
- Mapper: SABRELarge (QubitMapSABRELarge)
- Architecture: EML with 2+ optical zones
- Enable swap insertion for cross-QCCD gates
