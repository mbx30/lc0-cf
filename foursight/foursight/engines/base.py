"""Engine abstraction shared by the optimal (Leela-CF) and human (Maia-3) axes.

Both engines expose the *same* surface so the comparison layer can treat them
symmetrically: a policy distribution over legal moves, a best move, and — where
meaningful — a native win/draw/loss value. Reproducibility favours ``nodes``
budgets over wall-clock limits.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal

import chess

#: Which axis an engine fills. "optimal" = best-EV play, "human" = human-like.
EngineKind = Literal["optimal", "human"]


@dataclass(frozen=True)
class EngineMove:
    """A single candidate move and the probability the engine assigns it."""

    move: chess.Move
    prob: float


@dataclass
class EngineResult:
    """Everything one engine reports about one position.

    ``wdl`` is a ``(win, draw, loss)`` triple from the side-to-move's point of
    view that sums to 1; ``expectation`` is the derived expected score in [0, 1]
    (``win + 0.5 * draw``). The human axis usually leaves both ``None``.
    """

    kind: EngineKind
    bestmove: chess.Move | None
    policy: dict[chess.Move, float] = field(default_factory=dict)
    wdl: tuple[float, float, float] | None = None
    expectation: float | None = None
    elo: int | None = None
    raw: dict = field(default_factory=dict)

    def top(self, n: int = 1) -> list[EngineMove]:
        """Return the ``n`` highest-probability moves, descending."""
        ordered = sorted(self.policy.items(), key=lambda kv: kv[1], reverse=True)
        return [EngineMove(move=m, prob=p) for m, p in ordered[:n]]

    def prob_of(self, move: chess.Move) -> float:
        """Probability assigned to ``move`` (0.0 if unseen)."""
        return self.policy.get(move, 0.0)


class ChessEngine(abc.ABC):
    """Common interface for a side-by-side engine handle.

    Concrete engines set :attr:`kind`, :attr:`name`, :attr:`backend` and
    :attr:`is_mock` in their constructor.
    """

    kind: EngineKind
    name: str
    backend: str
    is_mock: bool

    @abc.abstractmethod
    def analyse(
        self,
        board: chess.Board,
        *,
        nodes: int | None = None,
        multipv: int = 1,
    ) -> EngineResult:
        """Evaluate ``board`` and return a populated :class:`EngineResult`."""

    def policy(self, board: chess.Board, *, elo: int | None = None) -> dict[chess.Move, float]:
        """Return the move-probability distribution for ``board``.

        The base implementation ignores ``elo``; the human engine overrides this
        to condition on rating.
        """
        return self.analyse(board).policy

    def expectation_after(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        nodes: int | None = None,
    ) -> float | None:
        """Side-to-move expected score *after* playing ``move``.

        Terminal positions are scored exactly; otherwise the child position is
        evaluated and its expectation flipped (the opponent is then to move).
        """
        child = board.copy(stack=False)
        child.push(move)
        if child.is_checkmate():
            # The side that just moved delivered mate -> a win for the mover.
            return 1.0
        if child.is_game_over(claim_draw=False):
            return 0.5
        result = self.analyse(child, nodes=nodes)
        if result.expectation is None:
            return None
        return 1.0 - result.expectation

    @abc.abstractmethod
    def close(self) -> None:
        """Release any underlying process/resources."""

    def __enter__(self) -> ChessEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def normalize_policy(scores: dict[chess.Move, float]) -> dict[chess.Move, float]:
    """Normalize non-negative move scores into a probability distribution."""
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        if n == 0:
            return {}
        return dict.fromkeys(scores, 1.0 / n)
    return {m: s / total for m, s in scores.items()}
