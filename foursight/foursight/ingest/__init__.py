"""Ingest of the ACTUAL axis — what was really played, and how the game ended."""

from __future__ import annotations

from foursight.ingest.actual import (
    ActualRecord,
    records_from_fen_moves,
    records_from_game,
    records_from_pgn,
    to_dataframe,
    write_parquet,
)

__all__ = [
    "ActualRecord",
    "records_from_fen_moves",
    "records_from_game",
    "records_from_pgn",
    "to_dataframe",
    "write_parquet",
]
