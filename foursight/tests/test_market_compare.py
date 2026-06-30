"""Tests for trend net and four-way market comparison."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from foursight.market.compare import FourWayComparator
from foursight.market.records import MarketRecord, PricePoint
from foursight.market.trend_net import TrendNet, label_from_delta


def _sample_record() -> MarketRecord:
    base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    prices = [
        PricePoint(base + timedelta(minutes=15 * i), 0.40 + 0.02 * i)
        for i in range(8)
    ]
    return MarketRecord(
        market_id="demo",
        platform="polymarket",
        question="Demo market?",
        resolution=1.0,
        prices=prices,
        actor_count=3,
    )


def test_trend_net_predicts_policy() -> None:
    net = TrendNet()
    rec = _sample_record()
    pred = net.predict(rec.prices)
    assert pred is not None
    assert abs(sum(pred.policy.values()) - 1.0) < 1e-6
    assert pred.best_label in ("down", "flat", "up")


def test_label_from_delta() -> None:
    assert label_from_delta(0.05) == "up"
    assert label_from_delta(-0.05) == "down"
    assert label_from_delta(0.0) == "flat"


def test_four_way_comparison_has_six_gaps() -> None:
    comp = FourWayComparator().compare(_sample_record())
    assert comp.worst_applicable is True
    assert comp.sights.worst_payoff is not None
    assert comp.gaps.opt_vs_human.delta_prob is not None
    assert comp.gaps.worst_vs_actual.delta_prob is not None
    blob = json.dumps(comp.to_dict())
    parsed = json.loads(blob)
    assert "worst_vs_opt" in parsed["gaps"]
    assert parsed["actor_count"] == 3
