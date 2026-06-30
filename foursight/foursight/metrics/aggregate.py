"""Phase-1 calibration metrics over three-way chess comparisons.

The comparison layer emits one :class:`~foursight.compare.ThreeWayComparison`
per position. This module flattens those objects into tabular rows and computes
aggregate calibration summaries by Elo band and game phase.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chess

from foursight.compare import ThreeWayComparison

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


@dataclass(frozen=True)
class ComparisonRow:
    """One tabular row derived from a three-way comparison."""

    fen: str
    ply: int
    phase: str
    elo: int
    elo_band: str
    opt_human_agree_at_1: bool | None
    human_actual_agree_at_1: bool | None
    opt_actual_agree_at_1: bool | None
    js_divergence: float | None
    kl_divergence: float | None
    delta_exp_human: float | None
    delta_exp_actual: float | None
    p_actual_given_human: float | None
    realized_score: float | None


@dataclass(frozen=True)
class CalibrationSummary:
    """Aggregate metrics for one grouping key (e.g. Elo band or phase)."""

    bucket: str
    n_positions: int
    human_agree_at_1: float | None
    optimal_agree_at_1: float | None
    mean_js: float | None
    mean_kl: float | None
    mean_delta_exp_human: float | None
    mean_delta_exp_actual: float | None
    mean_p_actual_given_human: float | None
    mean_realized_score: float | None


def ply_from_fen(fen: str) -> int:
    """Compute ply index from FEN (0 at the initial white move)."""
    board = chess.Board(fen)
    return (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)


def phase_for_ply(ply: int) -> str:
    """Simple phase bucketing for Phase-1 dashboards."""
    if ply <= 20:
        return "opening"
    if ply <= 50:
        return "middlegame"
    return "endgame"


def elo_band(elo: int | None, *, width: int = 200) -> str:
    """Map Elo into fixed-width bands."""
    if elo is None:
        return "unknown"
    lo = (elo // width) * width
    hi = lo + width - 1
    return f"{lo}-{hi}"


def comparison_to_row(comparison: ThreeWayComparison) -> ComparisonRow:
    """Flatten one comparison into a row suited for aggregation/dataframes."""
    ply = ply_from_fen(comparison.fen)
    return ComparisonRow(
        fen=comparison.fen,
        ply=ply,
        phase=phase_for_ply(ply),
        elo=comparison.elo,
        elo_band=elo_band(comparison.elo),
        opt_human_agree_at_1=comparison.gaps.opt_vs_human.agree_at_1,
        human_actual_agree_at_1=comparison.gaps.human_vs_actual.agree_at_1,
        opt_actual_agree_at_1=comparison.gaps.opt_vs_actual.agree_at_1,
        js_divergence=comparison.gaps.opt_vs_human.js_divergence,
        kl_divergence=comparison.gaps.opt_vs_human.kl_divergence,
        delta_exp_human=comparison.gaps.opt_vs_human.delta_expectation,
        delta_exp_actual=comparison.gaps.opt_vs_actual.delta_expectation,
        p_actual_given_human=comparison.gaps.human_vs_actual.target_prob,
        realized_score=comparison.gaps.opt_vs_actual.realized_score,
    )


def comparisons_to_rows(comparisons: Iterable[ThreeWayComparison]) -> list[ComparisonRow]:
    """Flatten an iterable of comparisons to rows."""
    return [comparison_to_row(c) for c in comparisons]


def rows_to_dataframe(rows: list[ComparisonRow]) -> pd.DataFrame:
    """Convert flattened rows to a DataFrame (Phase-2 parquet seam)."""
    import pandas as pd

    return pd.DataFrame([asdict(row) for row in rows])


def write_rows_parquet(rows: list[ComparisonRow], path: str | Path) -> None:
    """Persist flattened rows for downstream bulk ingest."""
    rows_to_dataframe(rows).to_parquet(path, index=False)


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _rate(values: Iterable[bool | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    true_count = sum(1 for v in filtered if v)
    return true_count / len(filtered)


def _sort_bucket(value: str) -> tuple[int, str]:
    if value == "unknown":
        return (10_000, value)
    lo_text = value.split("-", 1)[0]
    try:
        return (int(lo_text), value)
    except ValueError:
        return (10_000, value)


def _aggregate(rows: list[ComparisonRow], *, key: str) -> list[CalibrationSummary]:
    grouped: dict[str, list[ComparisonRow]] = {}
    for row in rows:
        bucket = getattr(row, key)
        grouped.setdefault(bucket, []).append(row)

    summaries: list[CalibrationSummary] = []
    for bucket, items in grouped.items():
        summaries.append(
            CalibrationSummary(
                bucket=bucket,
                n_positions=len(items),
                human_agree_at_1=_rate(r.human_actual_agree_at_1 for r in items),
                optimal_agree_at_1=_rate(r.opt_actual_agree_at_1 for r in items),
                mean_js=_mean(r.js_divergence for r in items),
                mean_kl=_mean(r.kl_divergence for r in items),
                mean_delta_exp_human=_mean(r.delta_exp_human for r in items),
                mean_delta_exp_actual=_mean(r.delta_exp_actual for r in items),
                mean_p_actual_given_human=_mean(r.p_actual_given_human for r in items),
                mean_realized_score=_mean(r.realized_score for r in items),
            )
        )
    return sorted(summaries, key=lambda s: _sort_bucket(s.bucket))


def aggregate_by_elo_band(rows: list[ComparisonRow]) -> list[CalibrationSummary]:
    """Compute summary metrics grouped by Elo band."""
    return _aggregate(rows, key="elo_band")


def aggregate_by_phase(rows: list[ComparisonRow]) -> list[CalibrationSummary]:
    """Compute summary metrics grouped by game phase."""
    order = {"opening": 0, "middlegame": 1, "endgame": 2}
    summaries = _aggregate(rows, key="phase")
    return sorted(summaries, key=lambda s: order.get(s.bucket, 99))
