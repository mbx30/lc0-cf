"""Parse real games into per-position ACTUAL records.

The ACTUAL axis is first-class: for every position we capture the move a human
actually played, the mover's rating, and the realized game result. Phase 0 reads
a single PGN (or a FEN + move list); bulk Lichess ingestion is deferred.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import chess
import chess.pgn

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Result strings as they appear in PGN headers.
_WHITE_WIN = "1-0"
_BLACK_WIN = "0-1"
_DRAW = "1/2-1/2"


@dataclass
class ActualRecord:
    """One real-world decision: the position, the move played, who and how it ended."""

    fen: str
    played_move: chess.Move
    player_elo: int | None = None
    game_result: str | None = None

    @property
    def board(self) -> chess.Board:
        return chess.Board(self.fen)

    @property
    def white_to_move(self) -> bool:
        # FEN active-colour field: 'w' or 'b'.
        return self.fen.split()[1] == "w"

    def realized_score(self) -> float | None:
        """Game result from the moving side's point of view (1 / 0.5 / 0)."""
        if self.game_result == _DRAW:
            return 0.5
        if self.game_result == _WHITE_WIN:
            return 1.0 if self.white_to_move else 0.0
        if self.game_result == _BLACK_WIN:
            return 0.0 if self.white_to_move else 1.0
        return None


def _elo_for(headers: chess.pgn.Headers, white_to_move: bool) -> int | None:
    key = "WhiteElo" if white_to_move else "BlackElo"
    value = headers.get(key)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def records_from_game(game: chess.pgn.Game) -> list[ActualRecord]:
    """Expand one parsed game into per-move :class:`ActualRecord` rows."""
    result = game.headers.get("Result")
    board = game.board()
    records: list[ActualRecord] = []
    for move in game.mainline_moves():
        records.append(
            ActualRecord(
                fen=board.fen(),
                played_move=move,
                player_elo=_elo_for(game.headers, board.turn == chess.WHITE),
                game_result=result,
            )
        )
        board.push(move)
    return records


def records_from_pgn(source: str | Path | TextIO) -> Iterator[ActualRecord]:
    """Yield records from every game in a PGN path, string, or open handle."""
    handle: TextIO
    close = False
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            handle = open(source, encoding="utf-8")
            close = True
        elif isinstance(source, str):
            handle = io.StringIO(source)  # treat a bare string as PGN content
        else:
            raise FileNotFoundError(source)
    else:
        handle = source  # already a file-like object
    try:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            yield from records_from_game(game)
    finally:
        if close:
            handle.close()


def records_from_fen_moves(
    fen: str,
    moves: list[str | chess.Move],
    *,
    player_elo: int | None = None,
    game_result: str | None = None,
) -> list[ActualRecord]:
    """Build records by replaying ``moves`` (UCI strings or Moves) from ``fen``."""
    board = chess.Board(fen)
    records: list[ActualRecord] = []
    for mv in moves:
        move = board.parse_uci(mv) if isinstance(mv, str) else mv
        records.append(
            ActualRecord(
                fen=board.fen(),
                played_move=move,
                player_elo=player_elo,
                game_result=game_result,
            )
        )
        board.push(move)
    return records


def to_dataframe(records: list[ActualRecord]) -> pd.DataFrame:
    """Flatten records into a DataFrame (the seam toward bulk/parquet storage)."""
    import pandas as pd

    return pd.DataFrame(
        {
            "fen": [r.fen for r in records],
            "played_move": [r.played_move.uci() for r in records],
            "player_elo": [r.player_elo for r in records],
            "game_result": [r.game_result for r in records],
            "realized_score": [r.realized_score() for r in records],
        }
    )


def write_parquet(records: list[ActualRecord], path: str | Path) -> None:
    """Persist records as parquet (deferred bulk ingest will reuse this schema)."""
    to_dataframe(records).to_parquet(path, index=False)
