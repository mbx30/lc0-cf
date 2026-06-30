"""Tests for multi-platform synthesis pipeline."""

from __future__ import annotations

import io
from datetime import UTC, datetime

from foursight.market.ingest.kalshi import load_kalshi_csv
from foursight.market.ingest.polymarket import load_polymarket_csv
from foursight.market.records import MarketRecord, PricePoint
from foursight.market.synthesis import dedupe_records, synthesize, walk_forward_split


def _rec(
    mid: str,
    platform: str,
    question: str,
    prob: float,
    *,
    ts: datetime | None = None,
) -> MarketRecord:
    t = ts or datetime(2025, 1, 1, tzinfo=UTC)
    return MarketRecord(
        market_id=mid,
        platform=platform,  # type: ignore[arg-type]
        question=question,
        prices=[PricePoint(t, prob)],
    )


def test_dedupe_prefers_richer_history() -> None:
    a = _rec("a", "polymarket", "Will GDP rise?", 0.5)
    b = _rec("b", "kalshi", "Will GDP rise?", 0.5)
    b.prices.append(PricePoint(datetime(2025, 1, 2, tzinfo=UTC), 0.6))
    out = dedupe_records([a, b])
    assert len(out) == 1
    assert len(out[0].prices) == 2


def test_synthesize_merges_platforms() -> None:
    poly = load_polymarket_csv(
        io.StringIO("market_id,question\np1,Unique event?\n"),
        prices_path=io.StringIO(
            "market_id,timestamp,implied_prob\n"
            "p1,2025-01-01T00:00:00+00:00,0.4\n"
            "p1,2025-01-01T00:07:00+00:00,0.42\n"
        ),
    )
    kalshi = load_kalshi_csv(
        io.StringIO("ticker,title\nk1,Other event?\n"),
        prices_path=io.StringIO(
            "ticker,timestamp,yes_price\n"
            "k1,2025-01-01T00:00:00+00:00,60\n"
        ),
    )
    merged = synthesize(poly, kalshi, resample=True, dedupe=True)
    assert len(merged) == 2
    assert all(r.actor_count == 3 for r in merged)


def test_walk_forward_is_chronological() -> None:
    early = _rec("e", "polymarket", "Early", 0.3, ts=datetime(2024, 1, 1, tzinfo=UTC))
    late = _rec("l", "kalshi", "Late", 0.7, ts=datetime(2025, 6, 1, tzinfo=UTC))
    train, test = walk_forward_split([late, early], train_frac=0.5)
    assert train[0].market_id == "e"
    assert test[0].market_id == "l"
