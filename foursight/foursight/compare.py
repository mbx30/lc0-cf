"""The deliverable: one structured three-way comparison per position.

For a position we line up the three axes — OPTIMAL (Leela-CF), HUMAN (Maia-3) and
ACTUAL (what was really played) — and compute the three pairwise gaps that are
exactly the signals the market engine will later compute:

* **optimal vs human**  — where humans are *predictably* suboptimal
  (policy divergence + the EV the human's top move gives up).
* **human vs actual**   — how well the human model predicts reality
  (the probability it assigned to the move actually played).
* **optimal vs actual**  — how often reality tracked the optimal move
  (agreement + the EV reality left on the table).

The optimal engine's value head is the EV yardstick: for a move ``m`` we score
``loss(m) = q* - q(m)``, where ``q*`` is the position's optimal expectation and
``q(m)`` is the side-to-move expectation after playing ``m``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import chess
import numpy as np

from foursight.engines.base import ChessEngine, EngineResult
from foursight.ingest.actual import ActualRecord

_EPS = 1e-12


def _vectorize(
    p: dict[chess.Move, float], q: dict[chess.Move, float]
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(p) | set(q), key=lambda m: m.uci())
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    pv = pv / pv.sum() if pv.sum() > 0 else pv
    qv = qv / qv.sum() if qv.sum() > 0 else qv
    return pv, qv


def kl_divergence(p: dict[chess.Move, float], q: dict[chess.Move, float]) -> float:
    """KL(p || q) in bits, with epsilon smoothing on the support of ``q``."""
    pv, qv = _vectorize(p, q)
    pv = pv + _EPS
    qv = qv + _EPS
    return float(np.sum(pv * np.log2(pv / qv)))


def js_divergence(p: dict[chess.Move, float], q: dict[chess.Move, float]) -> float:
    """Jensen-Shannon divergence in bits — symmetric and bounded in [0, 1]."""
    pv, qv = _vectorize(p, q)
    m = 0.5 * (pv + qv)
    pv = pv + _EPS
    qv = qv + _EPS
    m = m + _EPS
    kl_pm = float(np.sum(pv * np.log2(pv / m)))
    kl_qm = float(np.sum(qv * np.log2(qv / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


@dataclass
class Gap:
    """Pairwise comparison metrics. Only the fields relevant to a pair are set."""

    agree_at_1: bool | None = None
    js_divergence: float | None = None
    kl_divergence: float | None = None
    delta_expectation: float | None = None
    target_prob: float | None = None
    realized_score: float | None = None


@dataclass
class Gaps:
    opt_vs_human: Gap
    human_vs_actual: Gap
    opt_vs_actual: Gap


@dataclass
class ThreeWayComparison:
    """optimal | human | actual for one position, plus the three pairwise gaps."""

    fen: str
    elo: int
    optimal: EngineResult
    human: EngineResult
    actual: ActualRecord | None
    gaps: Gaps
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable view (moves rendered as UCI)."""

        def render_result(r: EngineResult) -> dict:
            return {
                "kind": r.kind,
                "bestmove": r.bestmove.uci() if r.bestmove else None,
                "wdl": list(r.wdl) if r.wdl else None,
                "expectation": r.expectation,
                "elo": r.elo,
                "policy": {em.move.uci(): round(em.prob, 6) for em in r.top(5)},
            }

        return {
            "fen": self.fen,
            "elo": self.elo,
            "optimal": render_result(self.optimal),
            "human": render_result(self.human),
            "actual": (
                {
                    "played_move": self.actual.played_move.uci(),
                    "player_elo": self.actual.player_elo,
                    "game_result": self.actual.game_result,
                    "realized_score": self.actual.realized_score(),
                }
                if self.actual
                else None
            ),
            "gaps": {
                "opt_vs_human": asdict(self.gaps.opt_vs_human),
                "human_vs_actual": asdict(self.gaps.human_vs_actual),
                "opt_vs_actual": asdict(self.gaps.opt_vs_actual),
            },
        }


class ThreeWayComparator:
    """Holds the engine pair and emits a :class:`ThreeWayComparison` per board."""

    def __init__(self, optimal: ChessEngine, human: ChessEngine) -> None:
        self.optimal = optimal
        self.human = human

    def _loss(
        self, board: chess.Board, move: chess.Move | None, q_star: float | None, nodes: int | None
    ) -> float | None:
        """Optimal-EV regret of ``move``: q* - q(move). Non-negative up to noise."""
        if move is None or q_star is None:
            return None
        q = self.optimal.expectation_after(board, move, nodes=nodes)
        if q is None:
            return None
        return q_star - q

    def compare(
        self,
        board: chess.Board,
        *,
        elo: int,
        actual: ActualRecord | None = None,
        nodes: int | None = None,
    ) -> ThreeWayComparison:
        optimal = self.optimal.analyse(board, nodes=nodes)
        human = self.human.analyse(board, nodes=nodes)
        # Condition the human policy on the requested Elo.
        human.policy = self.human.policy(board, elo=elo)
        if human.policy:
            human.bestmove = max(human.policy, key=lambda m: human.policy[m])
        human.elo = elo

        q_star = optimal.expectation

        # --- optimal vs human -------------------------------------------------
        opt_vs_human = Gap(
            agree_at_1=(
                optimal.bestmove == human.bestmove
                if optimal.bestmove and human.bestmove
                else None
            ),
            js_divergence=js_divergence(optimal.policy, human.policy),
            kl_divergence=kl_divergence(human.policy, optimal.policy),
            delta_expectation=self._loss(board, human.bestmove, q_star, nodes),
            target_prob=optimal.prob_of(human.bestmove) if human.bestmove else None,
        )

        # --- human vs actual --------------------------------------------------
        human_vs_actual = Gap()
        opt_vs_actual = Gap()
        if actual is not None:
            played = actual.played_move
            human_vs_actual = Gap(
                agree_at_1=(human.bestmove == played if human.bestmove else None),
                target_prob=human.prob_of(played),
                delta_expectation=(
                    self._signed_diff(
                        self.optimal.expectation_after(board, human.bestmove, nodes=nodes)
                        if human.bestmove
                        else None,
                        self.optimal.expectation_after(board, played, nodes=nodes),
                    )
                ),
            )
            opt_vs_actual = Gap(
                agree_at_1=(optimal.bestmove == played if optimal.bestmove else None),
                target_prob=optimal.prob_of(played),
                delta_expectation=self._loss(board, played, q_star, nodes),
                realized_score=actual.realized_score(),
            )

        gaps = Gaps(
            opt_vs_human=opt_vs_human,
            human_vs_actual=human_vs_actual,
            opt_vs_actual=opt_vs_actual,
        )
        return ThreeWayComparison(
            fen=board.fen(),
            elo=elo,
            optimal=optimal,
            human=human,
            actual=actual,
            gaps=gaps,
        )

    @staticmethod
    def _signed_diff(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b


def compare(
    board: chess.Board,
    *,
    optimal: ChessEngine,
    human: ChessEngine,
    elo: int,
    actual: ActualRecord | None = None,
    nodes: int | None = None,
) -> ThreeWayComparison:
    """Convenience wrapper around :class:`ThreeWayComparator`."""
    return ThreeWayComparator(optimal, human).compare(
        board, elo=elo, actual=actual, nodes=nodes
    )
