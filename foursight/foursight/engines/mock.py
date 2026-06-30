"""Dependency-free CPU mock that fills either axis.

The mock makes the Phase-0 gate pass with no GPU, no nets and no subprocess. It
is fully deterministic: scores derive from a hash of ``(position, move)`` so the
same board always yields the same distribution. The optimal and human roles
share those base scores but differ in temperature — the optimal policy is
peaked, the human policy is flatter and warms/cools with Elo — so the pairwise
gaps the comparison layer computes are non-trivial and stable across runs.
"""

from __future__ import annotations

import hashlib
import math

import chess

from foursight.engines.base import ChessEngine, EngineKind, EngineResult, normalize_policy

# Temperatures: lower => more peaked (closer to a single best move).
_OPTIMAL_TEMP = 0.45
_HUMAN_BASE_TEMP = 1.6


def _move_score(board: chess.Board, move: chess.Move) -> float:
    """Deterministic pseudo-utility in [0, 1) for a move, plus light chess sense."""
    key = f"{board.board_fen()} {board.turn} {move.uci()}".encode()
    digest = hashlib.sha256(key).digest()
    base = int.from_bytes(digest[:6], "big") / float(1 << 48)
    bonus = 0.0
    if board.is_capture(move):
        bonus += 0.20
    if board.gives_check(move):
        bonus += 0.10
    return base + bonus


def _softmax(scores: dict[chess.Move, float], temp: float) -> dict[chess.Move, float]:
    if not scores:
        return {}
    hi = max(scores.values())
    exps = {m: math.exp((s - hi) / temp) for m, s in scores.items()}
    return normalize_policy(exps)


def _human_temp(elo: int) -> float:
    """Higher Elo => cooler (more optimal-looking) human policy."""
    temp = _HUMAN_BASE_TEMP - (elo - 1000) / 1000.0
    return max(0.5, min(2.5, temp))


class MockEngine(ChessEngine):
    """A deterministic stand-in for Leela-CF or Maia-3."""

    def __init__(self, kind: EngineKind, *, default_elo: int = 1500) -> None:
        self.kind = kind
        self.name = f"mock-{kind}"
        self.backend = "mock"
        self.is_mock = True
        self._default_elo = default_elo

    def _scores(self, board: chess.Board) -> dict[chess.Move, float]:
        return {mv: _move_score(board, mv) for mv in board.legal_moves}

    def _temp(self, elo: int | None) -> float:
        if self.kind == "optimal":
            return _OPTIMAL_TEMP
        return _human_temp(elo if elo is not None else self._default_elo)

    def analyse(
        self,
        board: chess.Board,
        *,
        nodes: int | None = None,
        multipv: int = 1,
    ) -> EngineResult:
        elo = self._default_elo
        scores = self._scores(board)
        policy = _softmax(scores, self._temp(elo))
        if not policy:
            return EngineResult(
                kind=self.kind,
                bestmove=None,
                policy={},
                wdl=(0.0, 1.0, 0.0) if self.kind == "optimal" else None,
                expectation=0.5 if self.kind == "optimal" else None,
                elo=elo if self.kind == "human" else None,
                raw={"terminal": board.is_game_over()},
            )

        bestmove = max(policy, key=lambda m: policy[m])

        wdl: tuple[float, float, float] | None = None
        expectation: float | None = None
        if self.kind == "optimal":
            # Expectation tracks the policy-weighted score of this position.
            expectation = sum(policy[m] * scores[m] for m in policy)
            expectation = max(0.0, min(1.0, expectation))
            draw = 0.2 * (1.0 - abs(2.0 * expectation - 1.0))
            win = max(0.0, expectation - draw / 2.0)
            loss = max(0.0, 1.0 - expectation - draw / 2.0)
            total = win + draw + loss
            wdl = (win / total, draw / total, loss / total)
            expectation = wdl[0] + 0.5 * wdl[1]

        return EngineResult(
            kind=self.kind,
            bestmove=bestmove,
            policy=policy,
            wdl=wdl,
            expectation=expectation,
            elo=elo if self.kind == "human" else None,
            raw={"temp": self._temp(elo), "nodes": nodes, "multipv": multipv},
        )

    def policy(self, board: chess.Board, *, elo: int | None = None) -> dict[chess.Move, float]:
        scores = self._scores(board)
        return _softmax(scores, self._temp(elo))

    def close(self) -> None:  # nothing to release
        return None
