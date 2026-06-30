"""Maia-3 adapter — the HUMAN axis.

Drives the ``maia3`` submodule's Python UCI predictor (``maia3-5m`` / ``-23m`` /
``-79m``). Maia-3 is Elo-conditioned and does a single forward pass — no search —
so we set its ``SelfElo``/``OppoElo`` options to the requested rating and read a
human move-probability distribution out of a ``multipv`` enumeration.

Exercised only on the opt-in live path; the CPU gate uses the mock.
"""

from __future__ import annotations

import math

import chess
import chess.engine

from foursight.engines.base import ChessEngine, EngineResult, normalize_policy


def _entropy(policy: dict[chess.Move, float]) -> float:
    return -sum(p * math.log2(p) for p in policy.values() if p > 0.0)


class Maia3Engine(ChessEngine):
    """HUMAN engine backed by the Maia-3 UCI predictor."""

    kind = "human"

    def __init__(self, command: list[str], *, default_elo: int = 1500) -> None:
        if not command:
            raise ValueError("Maia3Engine requires a launch command")
        self.name = "maia3"
        self.backend = "maia3"
        self.is_mock = False
        self._default_elo = default_elo
        self._engine = chess.engine.SimpleEngine.popen_uci(list(command))

    def _set_elo(self, elo: int) -> None:
        for opt in ({"SelfElo": elo}, {"OppoElo": elo}):
            try:
                self._engine.configure(opt)
            except Exception:  # noqa: BLE001 - tolerate naming differences
                pass

    def analyse(
        self,
        board: chess.Board,
        *,
        nodes: int | None = None,
        multipv: int = 1,
    ) -> EngineResult:
        elo = self._default_elo
        self._set_elo(elo)
        legal_count = board.legal_moves.count()
        want = max(multipv, legal_count)
        infos = self._engine.analyse(
            board, chess.engine.Limit(nodes=nodes or 1), multipv=want
        )
        if isinstance(infos, dict):
            infos = [infos]

        scores: dict[chess.Move, float] = {}
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            move = pv[0]
            score = info.get("score")
            if score is not None:
                cp = score.pov(board.turn).score(mate_score=100000)
                scores[move] = float(cp) / 100.0 if cp is not None else 0.0
            else:
                scores[move] = 0.0

        policy = _softmax_scores(scores)
        bestmove = max(policy, key=lambda m: policy[m]) if policy else None
        return EngineResult(
            kind="human",
            bestmove=bestmove,
            policy=policy,
            wdl=None,
            expectation=None,
            elo=elo,
            raw={"entropy": _entropy(policy), "elo": elo},
        )

    def policy(self, board: chess.Board, *, elo: int | None = None) -> dict[chess.Move, float]:
        if elo is not None:
            self._default_elo = elo
        return self.analyse(board).policy

    def elo_sweep(
        self, board: chess.Board, elos: list[int]
    ) -> dict[int, dict[chess.Move, float]]:
        """Return the human policy at each Elo band — the predictable-error curve."""
        out: dict[int, dict[chess.Move, float]] = {}
        original = self._default_elo
        try:
            for elo in elos:
                self._default_elo = elo
                out[elo] = self.analyse(board).policy
        finally:
            self._default_elo = original
        return out

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:  # noqa: BLE001
            pass


def _softmax_scores(scores: dict[chess.Move, float]) -> dict[chess.Move, float]:
    if not scores:
        return {}
    hi = max(scores.values())
    exps = {m: math.exp(s - hi) for m, s in scores.items()}
    return normalize_policy(exps)
