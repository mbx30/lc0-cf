"""Phase-2 Track A Lichess bulk-ingest tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import zstandard
from typer.testing import CliRunner

from foursight.cli import app
from foursight.ingest.lichess import (
    GameFilter,
    ingest_lichess_to_parquet,
    iter_lichess_records,
    write_manifest,
    write_records_parquet_streaming,
)

_RUNNER = CliRunner()

# Game 1: Standard rated, both Elos present -> accepted (7 plies).
_GAME_RATED = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abc"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1500"]
[Variant "Standard"]
[TimeControl "300+3"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

# Game 2: Standard rated but missing Elo headers -> rejected when require_elo.
_GAME_NO_ELO = """[Event "Rated Bullet game"]
[Site "https://lichess.org/def"]
[White "Carol"]
[Black "Dave"]
[Result "1/2-1/2"]
[Variant "Standard"]
[TimeControl "60+0"]

1. d4 d5 2. c4 e6 1/2-1/2
"""

# Game 3: non-standard variant -> rejected.
_GAME_VARIANT = """[Event "Rated Crazyhouse game"]
[Site "https://lichess.org/ghi"]
[White "Erin"]
[Black "Frank"]
[Result "0-1"]
[WhiteElo "2000"]
[BlackElo "2100"]
[Variant "Crazyhouse"]
[TimeControl "180+0"]

1. e4 e5 2. Nf3 Nc6 0-1
"""

# Game 4: Standard rated, low Elo -> rejected by min_elo (3 plies).
_GAME_LOW_ELO = """[Event "Rated Blitz game"]
[Site "https://lichess.org/jkl"]
[White "Gina"]
[Black "Hank"]
[Result "1-0"]
[WhiteElo "1100"]
[BlackElo "1050"]
[Variant "Standard"]
[TimeControl "300+0"]

1. d4 d5 2. c4 1-0
"""

_ALL_GAMES = _GAME_RATED + "\n" + _GAME_NO_ELO + "\n" + _GAME_VARIANT + "\n" + _GAME_LOW_ELO


def _write_zst(tmp_path: Path, text: str = _ALL_GAMES) -> Path:
    path = tmp_path / "games.pgn.zst"
    cctx = zstandard.ZstdCompressor()
    path.write_bytes(cctx.compress(text.encode("utf-8")))
    return path


def test_iter_yields_records_from_accepted_games_only(tmp_path: Path) -> None:
    source = _write_zst(tmp_path)
    # Standard + rated + both Elos present.
    # Game 1 (7 plies) and game 4 (3 plies) qualify; no-elo & variant rejected.
    records = list(iter_lichess_records(source, game_filter=GameFilter()))
    assert len(records) == 10
    assert records[0].player_elo == 1600


def test_filter_rejects_missing_elo_variant_and_below_min(tmp_path: Path) -> None:
    no_elo = {"Event": "Rated Blitz game", "Variant": "Standard"}
    assert GameFilter().accepts(no_elo) is False
    assert GameFilter(require_elo=False).accepts(no_elo) is True

    crazyhouse = {
        "Event": "Rated Crazyhouse game",
        "Variant": "Crazyhouse",
        "WhiteElo": "2000",
        "BlackElo": "2100",
    }
    assert GameFilter().accepts(crazyhouse) is False

    low = {
        "Event": "Rated Blitz game",
        "Variant": "Standard",
        "WhiteElo": "1100",
        "BlackElo": "1050",
    }
    assert GameFilter(min_elo=1500).accepts(low) is False
    assert GameFilter(min_elo=1000).accepts(low) is True

    unrated = {"Event": "Casual Blitz game", "WhiteElo": "1600", "BlackElo": "1600"}
    assert GameFilter().accepts(unrated) is False
    assert GameFilter(rated_only=False).accepts(unrated) is True

    fast_tc = {
        "Event": "Rated Bullet game",
        "Variant": "Standard",
        "WhiteElo": "1600",
        "BlackElo": "1600",
        "TimeControl": "60+0",
    }
    assert GameFilter(time_control_min_seconds=180).accepts(fast_tc) is False
    assert GameFilter(time_control_min_seconds=30).accepts(fast_tc) is True


def test_min_elo_keeps_high_elo_game(tmp_path: Path) -> None:
    source = _write_zst(tmp_path)
    # With min_elo=1400 only game 1 qualifies (game 4 is below, others rejected).
    records = list(iter_lichess_records(source, game_filter=GameFilter(min_elo=1400)))
    assert len(records) == 7


def test_streaming_parquet_and_manifest(tmp_path: Path) -> None:
    source = _write_zst(tmp_path)
    out = tmp_path / "out.parquet"
    game_filter = GameFilter()
    records = iter_lichess_records(source, game_filter=game_filter)
    rows = write_records_parquet_streaming(records, out, batch_size=2)
    assert rows == 10
    assert out.exists()

    df = pd.read_parquet(out)
    assert list(df.columns) == [
        "fen",
        "played_move",
        "player_elo",
        "game_result",
        "realized_score",
    ]
    assert len(df) == 10
    assert df.iloc[0]["played_move"] == "e2e4"

    manifest_path = write_manifest(
        out,
        source=source,
        game_filter=game_filter,
        games_read=4,
        games_accepted=2,
        rows_written=rows,
    )
    assert manifest_path == Path(f"{out}.manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["games_read"] == 4
    assert payload["games_accepted"] == 2
    assert payload["rows_written"] == 10
    assert payload["filter"]["variant"] == "Standard"


def test_max_games_and_max_positions_caps(tmp_path: Path) -> None:
    # Two accepted Standard rated games with Elo.
    two_accepted = _GAME_RATED + "\n" + _GAME_LOW_ELO
    source = _write_zst(tmp_path, two_accepted)

    capped_games = list(iter_lichess_records(source, max_games=1))
    assert len(capped_games) == 7  # only the first accepted game

    capped_positions = list(iter_lichess_records(source, max_positions=3))
    assert len(capped_positions) == 3


def test_end_to_end_ingest_helper(tmp_path: Path) -> None:
    source = _write_zst(tmp_path)
    out = tmp_path / "e2e.parquet"
    result = ingest_lichess_to_parquet(source, out)
    assert result.games_read == 4
    assert result.games_accepted == 2
    assert result.rows_written == 10
    assert result.output.exists()
    assert result.manifest.exists()


def test_cli_ingest_lichess_smoke(tmp_path: Path) -> None:
    source = _write_zst(tmp_path)
    out = tmp_path / "cli.parquet"
    result = _RUNNER.invoke(
        app,
        [
            "ingest",
            "lichess",
            str(source),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "rows written=10" in result.stdout
    assert out.exists()
    assert Path(f"{out}.manifest.json").exists()
