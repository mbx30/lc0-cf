"""4sight — four sights; chess proving ground uses three.

Product sights: OPTIMAL | HUMAN | ACTUAL | WORST (adversarial-coalition worst
outcome, 3+ actors only). Two-actor chess implements the first three — WORST
collapses into optimal adversarial play and is not a separate sight.

Leela-CF supplies OPTIMAL, Maia-3 HUMAN, and real games ACTUAL.
:mod:`foursight.compare` emits the three pairwise gaps for chess; the market
engine will add WORST and six gaps when actor count ≥ 3.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
