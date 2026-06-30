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
from foursight.ingest.lichess import (
    GameFilter,
    IngestResult,
    ingest_lichess_to_parquet,
    iter_lichess_records,
    open_pgn_stream,
    write_manifest,
    write_records_parquet_streaming,
)

__all__ = [
    "ActualRecord",
    "GameFilter",
    "IngestResult",
    "ingest_lichess_to_parquet",
    "iter_lichess_records",
    "open_pgn_stream",
    "records_from_fen_moves",
    "records_from_game",
    "records_from_pgn",
    "to_dataframe",
    "write_manifest",
    "write_parquet",
    "write_records_parquet_streaming",
]
