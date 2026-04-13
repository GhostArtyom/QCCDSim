"""
Mapping module - qubit mapping algorithms for QCCD/MUSS experiments.

Classes:
- QubitMapGreedy: Greedy edge-weighted mapping
- QubitMapLPFS: Linear Program Flow Scheduling inspired mapping
- QubitMapRandom: Random mapping for baseline
- QubitMapPO: Program Order based mapping
- QubitMapMetis: Metis-style clustering (experimental)
- QubitMapAgg: Agglomerative clustering based mapping
- QubitOrdering: Qubit ordering within traps
- QubitMapTrivial: Paper-faithful trivial mapping
- QubitMapSABRE1: Iterative SABRE with greedy capacity
- QubitMapSABRE2: Paper-faithful SABRE2 with full heuristics
- QubitMapSABRE3: Strict capacity-limited SABRE
- QubitMapSABRE4: Greedy clustering + legalization
- QubitMapSABRE5: Overload + greedy legalization
- QubitMapSABRE6: Overload + Min-Cost Max-Flow legalization
- QubitMapSABRE7: Overload mapping for muss scheduler V4
- QubitMapSABRELarge: Large/EML-QCCD aware SABRE mapping

Migration Notes:
- Original mappers.py classes migrated to src/mapping/mapper.py
- Imports updated to use src.* paths
"""

from src.mapping.mapper import (
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

__all__ = [
    # Basic mappers
    "QubitMapGreedy",
    "QubitMapLPFS",
    "QubitMapRandom",
    "QubitMapPO",
    "QubitMapMetis",
    "QubitMapAgg",
    # Qubit ordering
    "QubitOrdering",
    # Trivial mapper
    "QubitMapTrivial",
    # SABRE variants
    "QubitMapSABRE1",
    "QubitMapSABRE2",
    "QubitMapSABRE3",
    "QubitMapSABRE4",
    "QubitMapSABRE5",
    "QubitMapSABRE6",
    "QubitMapSABRE7",
    "QubitMapSABRELarge",
]
