"""
Schedule module - scheduling strategies for ion shuttling

Classes:
- Schedule: Event timeline data structure
- EJFSchedule: Earliest Job First scheduling strategy
- ScheduleStrategy: Abstract base class for schedulers

Migration Notes:
- EJFSchedule moved from ejf_schedule.py -> src/schedule/ejf.py
- Schedule moved from schedule.py -> src/schedule/events.py
"""

from src.schedule.events import Schedule
from src.schedule.ejf import EJFSchedule
from src.schedule.base import ScheduleStrategy

__all__ = [
    "Schedule",
    "EJFSchedule",
    "ScheduleStrategy",
]
