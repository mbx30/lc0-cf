"""Ingest of the ACTUAL axis from PGN and FEN+move lists."""

from __future__ import annotations

import chess

from foursight.ingest.actual import (
    records_from_fen_moves,
    records_from_pgn,
    to_dataframe,
)

_PGN = """[Event "Test"]
[Site "?"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1500"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""


def test_pgn_expands_to_per_move_records() -> None:
    records = list(records_from_pgn(_PGN))
    # 7 plies played in the mainline above.
    assert len(records) == 7
    first = records[0]
    assert first.fen == chess.STARTING_FEN
    assert first.played_move == chess.Move.from_uci("e2e4")
    assert first.game_result == "1-0"
    # White to move first -> WhiteElo applies.
    assert first.player_elo == 1600
    # Second ply is Black to move -> BlackElo.
    assert records[1].player_elo == 1500


def test_realized_score_is_from_movers_pov() -> None:
    records = list(records_from_pgn(_PGN))
    # White won. White-to-move positions score 1.0, black-to-move positions 0.0.
    assert records[0].realized_score() == 1.0
    assert records[1].realized_score() == 0.0


def test_records_from_fen_moves() -> None:
    records = records_from_fen_moves(
        chess.STARTING_FEN, ["e2e4", "c7c5"], player_elo=2000, game_result="1/2-1/2"
    )
    assert len(records) == 2
    assert records[0].played_move == chess.Move.from_uci("e2e4")
    assert records[1].played_move == chess.Move.from_uci("c7c5")
    assert records[0].player_elo == 2000
    assert records[0].realized_score() == 0.5


def test_to_dataframe_columns() -> None:
    records = records_from_fen_moves(chess.STARTING_FEN, ["e2e4"], game_result="0-1")
    df = to_dataframe(records)
    assert list(df.columns) == [
        "fen",
        "played_move",
        "player_elo",
        "game_result",
        "realized_score",
    ]
    assert df.iloc[0]["played_move"] == "e2e4"
    # White to move, black won -> realized score 0.0 for the mover.
    assert df.iloc[0]["realized_score"] == 0.0
