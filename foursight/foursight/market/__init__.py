"""Prediction-market four-sight engine (Phase 4).

Chess uses :mod:`foursight.compare`; markets use :mod:`foursight.market.compare`
with :class:`~foursight.market.records.MarketRecord` ingest and synthesis.
"""

from foursight.market.compare import FourWayComparator, FourWayComparison
from foursight.market.records import MarketRecord, PricePoint

__all__ = [
    "FourWayComparison",
    "FourWayComparator",
    "MarketRecord",
    "PricePoint",
]
