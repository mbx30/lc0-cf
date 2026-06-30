"""4sight — three-axis (optimal | human | actual) outcome comparison.

The chess proving ground for the market product: Leela-CF supplies the OPTIMAL
axis, Maia-3 the HUMAN axis, and real games the ACTUAL axis. The three pairwise
gaps emitted by :mod:`foursight.compare` are the same signals the market engine
will later compute.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
