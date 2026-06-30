"""Kalshi ingest — Lightweight (markets) and Medium (price history) tiers.

Expected Lightweight columns:
  ticker, title, category, result  (result: yes/no or 1/0)

Expected Medium columns:
  ticker, timestamp, yes_price  (cents 1–99 or probability 0–1)

Heavy (full trade archive, order book) is intentionally unsupported.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TextIO

from foursight.market.records import MarketRecord, PricePoint

_RESULT_MAP = {"yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0}


def _parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def _parse_result(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    key = raw.strip().lower()
    if key in _RESULT_MAP:
        return _RESULT_MAP[key]
    try:
        cents = float(raw)
        if cents > 1.0:
            return cents / 100.0
        return cents
    except ValueError:
        return None


def _yes_prob(raw: str) -> float:
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


def load_kalshi_csv(
    markets_path: str | Path | None = None,
    *,
    prices_path: str | Path | None = None,
) -> list[MarketRecord]:
    """Load Kalshi records from metadata and/or price CSV exports."""
    meta: dict[str, MarketRecord] = {}

    if markets_path is not None:
        for row in _read_rows(markets_path):
            ticker = row.get("ticker") or row.get("market_id")
            if not ticker:
                continue
            meta[ticker] = MarketRecord(
                market_id=ticker,
                platform="kalshi",
                question=row.get("title") or row.get("question") or "",
                category=row.get("category") or row.get("event_ticker"),
                resolution=_parse_result(row.get("result") or row.get("resolution")),
            )

    prices_by_market: dict[str, list[PricePoint]] = defaultdict(list)
    if prices_path is not None:
        for row in _read_rows(prices_path):
            ticker = row.get("ticker") or row.get("market_id")
            if not ticker:
                continue
            ts_raw = row.get("timestamp") or row.get("created_time") or row.get("ts")
            prob_raw = row.get("yes_price") or row.get("implied_prob") or row.get("price")
            if not ts_raw or prob_raw is None:
                continue
            prices_by_market[ticker].append(
                PricePoint(ts=_parse_ts(ts_raw), implied_prob=_yes_prob(prob_raw))
            )
            if ticker not in meta:
                meta[ticker] = MarketRecord(
                    market_id=ticker,
                    platform="kalshi",
                    question=row.get("title") or "",
                )

    records: list[MarketRecord] = []
    for ticker, rec in meta.items():
        rec.prices = sorted(prices_by_market.get(ticker, []), key=lambda p: p.ts)
        records.append(rec)
    return records
