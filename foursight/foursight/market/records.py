"""Unified market record schema for multi-platform synthesis.

Mirrors the :mod:`foursight.ingest.actual` pattern: a small dataclass, helpers
to build from tabular sources, and a Parquet seam for bulk storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

Platform = Literal["polymarket", "kalshi", "other"]
OutcomeLabel = Literal["down", "flat", "up"]


@dataclass(frozen=True)
class PricePoint:
    """One implied-probability observation (typically 15-minute snapshot)."""

    ts: datetime
    implied_prob: float  # YES probability in [0, 1]


@dataclass
class MarketRecord:
    """One prediction market across its lifetime — metadata + price series."""

    market_id: str
    platform: Platform
    question: str
    category: str | None = None
    resolution: float | None = None  # 1.0 = YES, 0.0 = NO, 0.5 = tie/void
    prices: list[PricePoint] = field(default_factory=list)
    actor_count: int = 3  # ≥3 activates WORST sight in four-way compare

    def realized_score(self) -> float | None:
        """Final outcome as a scalar (1 / 0.5 / 0), analogous to chess ACTUAL."""
        return self.resolution

    def outcome_label(self, *, flat_band: float = 0.01) -> OutcomeLabel | None:
        """Bucket terminal move into DOWN / FLAT / UP for trend-net targets."""
        if self.resolution is None or len(self.prices) < 2:
            return None
        start = self.prices[0].implied_prob
        end = self.resolution
        delta = end - start
        if delta > flat_band:
            return "up"
        if delta < -flat_band:
            return "down"
        return "flat"


def to_dataframe(records: list[MarketRecord]) -> pd.DataFrame:
    """Flatten records for Parquet export (one row per price point)."""
    import pandas as pd

    rows: list[dict] = []
    for rec in records:
        for pt in rec.prices:
            rows.append(
                {
                    "market_id": rec.market_id,
                    "platform": rec.platform,
                    "question": rec.question,
                    "category": rec.category,
                    "resolution": rec.resolution,
                    "actor_count": rec.actor_count,
                    "ts": pt.ts,
                    "implied_prob": pt.implied_prob,
                }
            )
        if not rec.prices:
            rows.append(
                {
                    "market_id": rec.market_id,
                    "platform": rec.platform,
                    "question": rec.question,
                    "category": rec.category,
                    "resolution": rec.resolution,
                    "actor_count": rec.actor_count,
                    "ts": pd.NaT,
                    "implied_prob": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def from_dataframe(df: pd.DataFrame) -> list[MarketRecord]:
    """Reconstruct records from a flattened Parquet/DataFrame export."""
    import pandas as pd

    if df.empty:
        return []
    grouped: dict[tuple[str, str], MarketRecord] = {}
    for _, row in df.iterrows():
        key = (str(row["market_id"]), str(row["platform"]))
        if key not in grouped:
            grouped[key] = MarketRecord(
                market_id=str(row["market_id"]),
                platform=row["platform"],  # type: ignore[arg-type]
                question=str(row.get("question", "")),
                category=None if pd.isna(row.get("category")) else str(row["category"]),
                resolution=None if pd.isna(row.get("resolution")) else float(row["resolution"]),
                actor_count=int(row.get("actor_count", 3)),
            )
        ts = row.get("ts")
        prob = row.get("implied_prob")
        if ts is not None and not pd.isna(ts) and prob is not None and not pd.isna(prob):
            grouped[key].prices.append(
                PricePoint(ts=pd.Timestamp(ts).to_pydatetime(), implied_prob=float(prob))
            )
    for rec in grouped.values():
        rec.prices.sort(key=lambda p: p.ts)
    return list(grouped.values())


def write_parquet(records: list[MarketRecord], path: str | Path) -> None:
    """Persist records as parquet."""
    to_dataframe(records).to_parquet(path, index=False)


def read_parquet(path: str | Path) -> list[MarketRecord]:
    """Load records from parquet written by :func:`write_parquet`."""
    import pandas as pd

    return from_dataframe(pd.read_parquet(path))
