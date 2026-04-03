"""
Machine module - ion trap machine models and topologies

Classes:
- Machine: Ion trap quantum computer model
- MachineParams: Configuration parameters for machine
- MachineState: Runtime state of ions in the machine

Migration Notes:
- Machine and MachineParams moved from machine.py -> src/machine/core.py
- MachineState moved from machine_state.py -> src/machine/state.py
"""

from src.machine.core import Machine, MachineParams
from src.machine.state import MachineState

__all__ = [
    "Machine",
    "MachineParams",
    "MachineState",
]
