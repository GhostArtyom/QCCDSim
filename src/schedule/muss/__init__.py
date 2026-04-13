"""
MUSS scheduler family - V6 (small-scale) and V7 (large-scale)

Classes:
- MUSSScheduleV6: Paper-faithful strict small-scale scheduler (Section 3.2)
- MUSSScheduleV7: Large-scale scheduler with swap insertion, cross-QCCD gates
- ScheduleV7: Event timeline with O(1) caching for V7

Migration Notes:
- Original muss_schedule2-5.py → unified in v6.py (same class, different config)
- Original muss_schedule6.py → src/schedule/muss/v6.py
- Original muss_schedule7.py → src/schedule/muss/v7.py
- Original schedule_v7.py → src/schedule/muss/schedule_v7.py
"""

from src.schedule.muss.v6 import MUSSSchedule as MUSSScheduleV6
from src.schedule.muss.v7 import MUSSSchedule as MUSSScheduleV7
from src.schedule.muss.schedule_v7 import ScheduleV7

# V2-V5 are unified with V6 (same class, different initialization modes)
# Alias for backward compatibility
MUSSScheduleV2 = MUSSScheduleV6
MUSSScheduleV3 = MUSSScheduleV6
MUSSScheduleV4 = MUSSScheduleV6
MUSSScheduleV5 = MUSSScheduleV6

__all__ = [
    "MUSSScheduleV2",
    "MUSSScheduleV3",
    "MUSSScheduleV4",
    "MUSSScheduleV5",
    "MUSSScheduleV6",
    "MUSSScheduleV7",
    "ScheduleV7",
]
