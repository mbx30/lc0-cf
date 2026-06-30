"""Multi-platform market synthesis — normalize, resample, dedupe, walk-forward."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from foursight.market.records import MarketRecord, PricePoint

_FREQ = timedelta(minutes=15)


def _normalize_question(question: str) -> str:
    """Crude dedupe key: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def resample_prices(
    prices: list[PricePoint],
    *,
    freq: timedelta = _FREQ,
) -> list[PricePoint]:
    """Bucket observations onto a fixed UTC grid (last observation carried)."""
    if not prices:
        return []
    ordered = sorted(prices, key=lambda p: p.ts)
    start = ordered[0].ts.replace(second=0, microsecond=0)
    # align to 15-minute boundary
    minute = (start.minute // 15) * 15
    start = start.replace(minute=minute)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)

    end = ordered[-1].ts
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    buckets: list[PricePoint] = []
    idx = 0
    current = start
    last_prob = ordered[0].implied_prob
    while current <= end:
        while idx < len(ordered):
            ts = ordered[idx].ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if ts <= current:
                last_prob = ordered[idx].implied_prob
                idx += 1
            else:
                break
        buckets.append(PricePoint(ts=current, implied_prob=last_prob))
        current += freq
    return buckets


def dedupe_records(records: Sequence[MarketRecord]) -> list[MarketRecord]:
    """Keep one record per normalized question, preferring more price history."""
    best: dict[str, MarketRecord] = {}
    for rec in records:
        key = _normalize_question(rec.question)
        if not key:
            key = f"{rec.platform}:{rec.market_id}"
        existing = best.get(key)
        if existing is None or len(rec.prices) > len(existing.prices):
            best[key] = rec
    return list(best.values())


def synthesize(
    *batches: Sequence[MarketRecord],
    resample: bool = True,
    dedupe: bool = True,
    actor_count: int = 3,
) -> list[MarketRecord]:
    """Merge platform-specific records into a unified corpus."""
    merged: list[MarketRecord] = []
    for batch in batches:
        merged.extend(batch)
    if dedupe:
        merged = dedupe_records(merged)
    out: list[MarketRecord] = []
    for rec in merged:
        prices = resample_prices(rec.prices) if resample and rec.prices else list(rec.prices)
        out.append(
            MarketRecord(
                market_id=rec.market_id,
                platform=rec.platform,
                question=rec.question,
                category=rec.category,
                resolution=rec.resolution,
                prices=prices,
                actor_count=actor_count,
            )
        )
    out.sort(key=lambda r: (r.platform, r.market_id))
    return out


def walk_forward_split(
    records: Sequence[MarketRecord],
    *,
    train_frac: float = 0.7,
) -> tuple[list[MarketRecord], list[MarketRecord]]:
    """Chronological split by each market's last price timestamp — never shuffle."""
    if not records:
        return [], []

    def last_ts(rec: MarketRecord) -> datetime:
        if rec.prices:
            return rec.prices[-1].ts
        return datetime.min.replace(tzinfo=UTC)

    ordered = sorted(records, key=last_ts)
    cut = max(1, int(len(ordered) * train_frac))
    if cut >= len(ordered):
        cut = len(ordered) - 1
    return list(ordered[:cut]), list(ordered[cut:])
