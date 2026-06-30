"""Bulk ACTUAL ingest from Lichess ``.pgn.zst`` dumps.

Phase 2 Track A: stream games out of a (possibly zstd-compressed) Lichess
database export, filter on the PGN headers, and persist per-position
:class:`~foursight.ingest.actual.ActualRecord` rows to parquet in batches. The
whole pipeline is streaming — at no point is the full dump held in memory.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import chess.pgn

from foursight.ingest.actual import ActualRecord, records_from_game

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

# Manifest format version; bump when the schema below changes.
SCHEMA_VERSION = 1

# Column schema shared with ``actual.to_dataframe``.
_COLUMNS = ("fen", "played_move", "player_elo", "game_result", "realized_score")


@contextmanager
def open_pgn_stream(source: str | Path) -> Iterator[TextIO]:
    """Open a ``.pgn`` or ``.pgn.zst`` file as a streaming utf-8 text handle.

    ``.pgn.zst`` files are decompressed on the fly with a streaming zstd reader
    so the dump is never fully materialised in memory.
    """
    path = Path(source)
    if path.suffix.lower() == ".zst":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with open(path, "rb") as raw:
            reader = dctx.stream_reader(raw)
            text = io.TextIOWrapper(reader, encoding="utf-8")
            try:
                yield text
            finally:
                text.close()
    else:
        with open(path, encoding="utf-8") as handle:
            yield handle


def _parse_elo(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _time_control_base(value: str | None) -> int | None:
    """Parse a ``base+inc`` ``TimeControl`` header into the base seconds.

    Returns ``None`` for missing, unlimited (``-``), or unparseable values.
    """
    if not value or value == "-":
        return None
    base = value.split("+", 1)[0].strip()
    try:
        return int(base)
    except ValueError:
        return None


@dataclass(frozen=True)
class GameFilter:
    """Header-level acceptance rules for Lichess games."""

    variant: str = "Standard"
    rated_only: bool = True
    min_elo: int | None = None
    require_elo: bool = True
    time_control_min_seconds: int | None = None

    def accepts(self, headers: Mapping[str, str]) -> bool:
        """Decide whether a game's PGN headers pass this filter."""
        variant = headers.get("Variant") or "Standard"
        if variant.strip().lower() != self.variant.strip().lower():
            return False

        if self.rated_only:
            event = (headers.get("Event") or "").lower()
            if "rated" not in event or "unrated" in event:
                return False

        white_elo = _parse_elo(headers.get("WhiteElo"))
        black_elo = _parse_elo(headers.get("BlackElo"))
        if self.min_elo is not None:
            if white_elo is None or black_elo is None:
                return False
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False
        elif self.require_elo and (white_elo is None or black_elo is None):
            return False

        if self.time_control_min_seconds is not None:
            base = _time_control_base(headers.get("TimeControl"))
            if base is None or base < self.time_control_min_seconds:
                return False

        return True


def _stream_records(
    source: str | Path,
    game_filter: GameFilter | None,
    max_games: int | None,
    max_positions: int | None,
    counts: dict[str, int] | None,
) -> Iterator[ActualRecord]:
    accepted = 0
    positions = 0
    with open_pgn_stream(source) as handle:
        while True:
            if max_games is not None and accepted >= max_games:
                break
            if max_positions is not None and positions >= max_positions:
                break
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if counts is not None:
                counts["read"] += 1
            if game_filter is not None and not game_filter.accepts(game.headers):
                continue
            accepted += 1
            if counts is not None:
                counts["accepted"] += 1
            for record in records_from_game(game):
                if max_positions is not None and positions >= max_positions:
                    return
                yield record
                positions += 1


def iter_lichess_records(
    source: str | Path,
    *,
    game_filter: GameFilter | None = None,
    max_games: int | None = None,
    max_positions: int | None = None,
) -> Iterator[ActualRecord]:
    """Stream :class:`ActualRecord` rows from accepted games in ``source``.

    Reads one game at a time, applies ``game_filter`` to the headers, and yields
    records from accepted games via :func:`records_from_game`. ``max_games``
    counts accepted games; ``max_positions`` caps the total records yielded.
    """
    yield from _stream_records(
        source, game_filter, max_games, max_positions, counts=None
    )


def write_records_parquet_streaming(
    records: Iterable[ActualRecord],
    path: str | Path,
    *,
    batch_size: int = 50_000,
) -> int:
    """Write records to parquet incrementally, one row group per ``batch_size``.

    Returns the total number of rows written. Unlike ``actual.write_parquet``
    this never accumulates the whole record set in memory.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("fen", pa.string()),
            ("played_move", pa.string()),
            ("player_elo", pa.int64()),
            ("game_result", pa.string()),
            ("realized_score", pa.float64()),
        ]
    )

    batch: list[ActualRecord] = []
    total = 0
    writer: pq.ParquetWriter | None = None

    def _flush(rows: list[ActualRecord]) -> None:
        nonlocal writer
        if not rows:
            return
        table = pa.table(
            {
                "fen": [r.fen for r in rows],
                "played_move": [r.played_move.uci() for r in rows],
                "player_elo": [r.player_elo for r in rows],
                "game_result": [r.game_result for r in rows],
                "realized_score": [r.realized_score() for r in rows],
            },
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(str(path), schema)
        writer.write_table(table)

    try:
        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                _flush(batch)
                total += len(batch)
                batch = []
        if batch:
            _flush(batch)
            total += len(batch)
        if writer is None:
            # No rows seen: still emit an empty parquet with the right schema.
            writer = pq.ParquetWriter(str(path), schema)
    finally:
        if writer is not None:
            writer.close()

    return total


def write_manifest(
    path: str | Path,
    *,
    source: str | Path,
    game_filter: GameFilter,
    games_read: int,
    games_accepted: int,
    rows_written: int,
) -> Path:
    """Emit a ``<path>.manifest.json`` describing one ingest run."""
    manifest_path = Path(f"{path}.manifest.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "output": str(path),
        "filter": {
            "variant": game_filter.variant,
            "rated_only": game_filter.rated_only,
            "min_elo": game_filter.min_elo,
            "require_elo": game_filter.require_elo,
            "time_control_min_seconds": game_filter.time_control_min_seconds,
        },
        "games_read": games_read,
        "games_accepted": games_accepted,
        "rows_written": rows_written,
        "columns": list(_COLUMNS),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


@dataclass
class IngestResult:
    """Summary returned by :func:`ingest_lichess_to_parquet`."""

    games_read: int
    games_accepted: int
    rows_written: int
    output: Path
    manifest: Path


def ingest_lichess_to_parquet(
    source: str | Path,
    out: str | Path,
    *,
    game_filter: GameFilter | None = None,
    max_games: int | None = None,
    max_positions: int | None = None,
    batch_size: int = 50_000,
) -> IngestResult:
    """Run the full streaming ingest: read, filter, write parquet + manifest.

    Counts of games read/accepted are tracked while streaming so the manifest
    reflects exactly what the iterator consumed.
    """
    used_filter = game_filter if game_filter is not None else GameFilter()
    counts = {"read": 0, "accepted": 0}

    records = _stream_records(
        source, used_filter, max_games, max_positions, counts=counts
    )
    rows_written = write_records_parquet_streaming(records, out, batch_size=batch_size)
    manifest = write_manifest(
        out,
        source=source,
        game_filter=used_filter,
        games_read=counts["read"],
        games_accepted=counts["accepted"],
        rows_written=rows_written,
    )
    return IngestResult(
        games_read=counts["read"],
        games_accepted=counts["accepted"],
        rows_written=rows_written,
        output=Path(out),
        manifest=manifest,
    )
