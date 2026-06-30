"""Polymarket ingest — Lightweight (markets.csv) and Medium (prices.csv) tiers.

Expected Lightweight columns (markets metadata):
  market_id, question, category, resolution  (resolution optional)

Expected Medium columns (15-min snapshots):
  market_id, timestamp, implied_prob  (or price / mid as 0–1 float)

Heavy (on-chain trades, order books) is intentionally unsupported.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TextIO

from foursight.market.records import MarketRecord, PricePoint

_RESOLUTION_MAP = {
    "yes": 1.0,
    "no": 0.0,
    "1": 1.0,
    "0": 0.0,
    "true": 1.0,
    "false": 0.0,
}


def _parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def _parse_resolution(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    key = raw.strip().lower()
    if key in _RESOLUTION_MAP:
        return _RESOLUTION_MAP[key]
    try:
        val = float(raw)
        if 0.0 <= val <= 1.0:
            return val
    except ValueError:
        pass
    return None


def _parse_prob(raw: str) -> float:
    val = float(raw)
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def _read_rows(source: str | Path | TextIO) -> Iterator[dict[str, str]]:
    if isinstance(source, (str, Path)):
        with open(source, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)
    else:
        yield from csv.DictReader(source)


def load_polymarket_csv(
    markets_path: str | Path | None = None,
    *,
    prices_path: str | Path | None = None,
) -> list[MarketRecord]:
    """Load Polymarket records from metadata and/or price CSV exports.

    Lightweight-only: pass ``markets_path`` alone.
    Medium: pass both ``markets_path`` and ``prices_path``.
    """
    meta: dict[str, MarketRecord] = {}

    if markets_path is not None:
        for row in _read_rows(markets_path):
            mid = row.get("market_id") or row.get("id") or row.get("condition_id")
            if not mid:
                continue
            meta[mid] = MarketRecord(
                market_id=mid,
                platform="polymarket",
                question=row.get("question") or row.get("title") or "",
                category=row.get("category") or row.get("group"),
                resolution=_parse_resolution(row.get("resolution") or row.get("outcome")),
            )

    prices_by_market: dict[str, list[PricePoint]] = defaultdict(list)
    if prices_path is not None:
        for row in _read_rows(prices_path):
            mid = row.get("market_id") or row.get("condition_id")
            if not mid:
                continue
            ts_raw = row.get("timestamp") or row.get("ts") or row.get("time")
            prob_raw = row.get("implied_prob") or row.get("price") or row.get("mid")
            if not ts_raw or prob_raw is None:
                continue
            prices_by_market[mid].append(
                PricePoint(ts=_parse_ts(ts_raw), implied_prob=_parse_prob(prob_raw))
            )
            if mid not in meta:
                meta[mid] = MarketRecord(
                    market_id=mid,
                    platform="polymarket",
                    question=row.get("question") or "",
                )

    records: list[MarketRecord] = []
    for mid, rec in meta.items():
        rec.prices = sorted(prices_by_market.get(mid, []), key=lambda p: p.ts)
        records.append(rec)
    return records
