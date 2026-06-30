"""Tests for Polymarket and Kalshi CSV ingest adapters."""

from __future__ import annotations

import io

from foursight.market.ingest.kalshi import load_kalshi_csv
from foursight.market.ingest.polymarket import load_polymarket_csv

_POLY_MARKETS = """market_id,question,category,resolution
pm-1,Will it rain?,weather,yes
"""

_POLY_PRICES = """market_id,timestamp,implied_prob
pm-1,2025-01-01T12:00:00+00:00,0.45
pm-1,2025-01-01T12:15:00+00:00,0.50
"""

_KALSHI_MARKETS = """ticker,title,category,result
KXTEST-1,GDP up?,economics,no
"""

_KALSHI_PRICES = """ticker,timestamp,yes_price
KXTEST-1,2025-02-01T09:00:00+00:00,55
KXTEST-1,2025-02-01T09:15:00+00:00,48
"""


def test_polymarket_lightweight_only() -> None:
    recs = load_polymarket_csv(io.StringIO(_POLY_MARKETS))
    assert len(recs) == 1
    assert recs[0].platform == "polymarket"
    assert recs[0].resolution == 1.0
    assert recs[0].prices == []


def test_polymarket_medium_tier() -> None:
    recs = load_polymarket_csv(
        io.StringIO(_POLY_MARKETS),
        prices_path=io.StringIO(_POLY_PRICES),
    )
    assert len(recs[0].prices) == 2
    assert recs[0].prices[0].implied_prob == 0.45


def test_kalshi_medium_tier() -> None:
    recs = load_kalshi_csv(
        io.StringIO(_KALSHI_MARKETS),
        prices_path=io.StringIO(_KALSHI_PRICES),
    )
    assert recs[0].platform == "kalshi"
    assert recs[0].resolution == 0.0
    assert abs(recs[0].prices[1].implied_prob - 0.48) < 1e-9
