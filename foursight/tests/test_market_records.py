"""Tests for MarketRecord schema and Parquet round-trip."""

from __future__ import annotations

from datetime import UTC, datetime

from foursight.market.records import (
    MarketRecord,
    PricePoint,
    from_dataframe,
    read_parquet,
    to_dataframe,
    write_parquet,
)


def test_market_record_realized_score_and_label() -> None:
    rec = MarketRecord(
        market_id="m1",
        platform="polymarket",
        question="Will X happen?",
        resolution=1.0,
        prices=[
            PricePoint(datetime(2025, 1, 1, tzinfo=UTC), 0.4),
            PricePoint(datetime(2025, 1, 2, tzinfo=UTC), 0.9),
        ],
    )
    assert rec.realized_score() == 1.0
    assert rec.outcome_label() == "up"


def test_parquet_round_trip(tmp_path) -> None:
    rec = MarketRecord(
        market_id="k1",
        platform="kalshi",
        question="Fed cuts?",
        category="economics",
        resolution=0.0,
        prices=[PricePoint(datetime(2025, 6, 1, 12, 0, tzinfo=UTC), 0.62)],
        actor_count=3,
    )
    path = tmp_path / "markets.parquet"
    write_parquet([rec], path)
    loaded = read_parquet(path)
    assert len(loaded) == 1
    assert loaded[0].market_id == "k1"
    assert loaded[0].platform == "kalshi"
    assert len(loaded[0].prices) == 1
    assert abs(loaded[0].prices[0].implied_prob - 0.62) < 1e-9

    df = to_dataframe([rec])
    rebuilt = from_dataframe(df)
    assert rebuilt[0].question == "Fed cuts?"
