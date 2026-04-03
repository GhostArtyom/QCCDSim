"""
Route module - pathfinding for ion shuttling

Classes:
- BasicRoute: Basic shortest-path routing
- FreeTrapRoute: Routing with free trap selection

Migration Notes:
- BasicRoute moved from route.py -> src/route/basic.py
- FreeTrapRoute moved from route.py -> src/route/free_trap.py
"""

from src.route.basic import BasicRoute
from src.route.free_trap import FreeTrapRoute

__all__ = [
    "BasicRoute",
    "FreeTrapRoute",
]
