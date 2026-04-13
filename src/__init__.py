"""
QCCDSim - QCCD Quantum Compiler and Simulator

A comprehensive simulator for ion-trap quantum computers with shuttling-based qubit movement.
Supports both small-scale (Table 2 reproduction) and large-scale (EML-QCCD) architectures.

Submodules:
- analyze: Fidelity analysis (Analyzer, AnalyzerV7)
- machine: Machine models and parameters (Machine, MachineParams, MachineState)
- mapping: Qubit-to-trap mapping algorithms (SABRE, LPFS, Trivial, etc.)
- parse: QASM circuit parsing (InputParse)
- route: Pathfinding for ion shuttling
- schedule: Scheduling strategies (EJF, MUSS V2-V7)
- utils: Utility functions
"""

from enum import Enum


class ScaleMode(Enum):
    """运行规模模式枚举"""
    SMALL = "SMALL"   # 小规模：单 QCCD，V2-V6 调度器
    LARGE = "LARGE"   # 大规模：EML-QCCD 多模块，V7 调度器


# 当前版本
__version__ = "1.0.0"

# Re-export commonly used classes for convenient access
from src.analyze import Analyzer, AnalyzerKnobs, AnalyzerV7, AnalyzerV7Knobs
from src.machine import Machine, MachineParams, MachineState
from src.mapping import (
    QubitMapGreedy,
    QubitMapLPFS,
    QubitMapRandom,
    QubitMapPO,
    QubitMapMetis,
    QubitMapAgg,
    QubitOrdering,
    QubitMapTrivial,
    QubitMapSABRE1,
    QubitMapSABRE2,
    QubitMapSABRE3,
    QubitMapSABRE4,
    QubitMapSABRE5,
    QubitMapSABRE6,
    QubitMapSABRE7,
    QubitMapSABRELarge,
)
from src.parse import InputParse
from src.schedule import Schedule, EJFSchedule, ScheduleStrategy
