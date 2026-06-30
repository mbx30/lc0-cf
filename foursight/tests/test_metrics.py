"""Phase-1 calibration metrics tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from foursight.cli import app
from foursight.compare import ThreeWayComparator
from foursight.config import Settings
from foursight.engines.registry import build_engines
from foursight.ingest.actual import records_from_pgn
from foursight.metrics import (
    aggregate_by_elo_band,
    aggregate_by_phase,
    comparisons_to_rows,
    phase_for_ply,
    ply_from_fen,
)

_RUNNER = CliRunner()
_FIXTURE = Path(__file__).parent / "fixtures" / "two_games.pgn"


def _comparisons():
    records = list(records_from_pgn(_FIXTURE))
    with build_engines(Settings(force_mock=True)) as pair:
        comparator = ThreeWayComparator(pair.optimal, pair.human)
        return [
            comparator.compare(record.board, elo=record.player_elo or 1500, actual=record)
            for record in records
        ]


def test_rows_are_flattened_from_comparisons() -> None:
    rows = comparisons_to_rows(_comparisons())
    assert len(rows) == 15
    first = rows[0]
    assert first.fen
    assert first.ply == ply_from_fen(first.fen)
    assert first.phase == phase_for_ply(first.ply)
    assert first.elo_band == "1600-1799"
    assert first.js_divergence is not None and first.js_divergence >= 0.0


def test_aggregate_by_elo_band_counts_rows() -> None:
    rows = comparisons_to_rows(_comparisons())
    summary = {item.bucket: item.n_positions for item in aggregate_by_elo_band(rows)}
    assert summary["1400-1599"] == 3
    assert summary["1600-1799"] == 8
    assert summary["1800-1999"] == 4


def test_aggregate_by_phase_uses_opening_bucket() -> None:
    rows = comparisons_to_rows(_comparisons())
    by_phase = aggregate_by_phase(rows)
    assert len(by_phase) == 1
    assert by_phase[0].bucket == "opening"
    assert by_phase[0].n_positions == 15


def test_calibrate_cli_writes_json_report(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    result = _RUNNER.invoke(
        app,
        [
            "calibrate",
            str(_FIXTURE),
            "--elo",
            "auto",
            "--json-out",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert "Calibration report" in result.stdout
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 15
    assert len(payload["by_elo_band"]) == 3
