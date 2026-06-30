"""Leela-CF adapter — the OPTIMAL axis.

Drives the repository's built Lc0 binary loaded with a Chessformer *strength*
net (Leela-CF). Unlike Stockfish there is no centipawn->win% sigmoid: Lc0
reports a native win/draw/loss value, which we read directly. A move-prior
policy is recovered from ``VerboseMoveStats`` on a one-node search.

This adapter is exercised only on the opt-in live path (``FOURSIGHT_LIVE=1`` plus
a real binary + net); the CPU gate uses :class:`~foursight.engines.mock.MockEngine`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import chess
import chess.engine

from foursight.engines.base import ChessEngine, EngineResult, normalize_policy

# Matches Lc0 verbose move stats lines, e.g.
#   "d2d4  (322 ) N:   45 (+0) ... (P:  8.45%) (Q:  0.123) ..."
_VERBOSE_RE = re.compile(r"^\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b.*\(P:\s*([\d.]+)%\)")


def _backend_for(device: str) -> str | None:
    device = (device or "").strip().lower()
    if device in {"", "cpu"}:
        return "blas"
    return device


class LeelaCFEngine(ChessEngine):
    """OPTIMAL engine backed by Lc0 + a Leela-CF strength net."""

    kind = "optimal"

    def __init__(
        self,
        command: list[str],
        *,
        net: str | None = None,
        device: str = "cuda",
        default_nodes: int | None = 800,
    ) -> None:
        if not command:
            raise ValueError("LeelaCFEngine requires a launch command")
        argv = list(command)
        if net:
            argv.append(f"--weights={net}")
        backend = _backend_for(device)
        if backend:
            argv.append(f"--backend={backend}")

        self.name = "leela-cf"
        self.backend = "leela-cf"
        self.is_mock = False
        self._default_nodes = default_nodes
        self._engine = chess.engine.SimpleEngine.popen_uci(argv)
        # Best-effort: ask Lc0 to surface WDL and per-move priors.
        for opt in ({"UCI_ShowWDL": True}, {"VerboseMoveStats": True}):
            try:
                self._engine.configure(opt)
            except Exception:  # noqa: BLE001 - unknown option is non-fatal
                pass

    def _limit(self, nodes: int | None) -> chess.engine.Limit:
        n = nodes if nodes is not None else self._default_nodes
        return chess.engine.Limit(nodes=n) if n else chess.engine.Limit(time=0.1)

    def analyse(
        self,
        board: chess.Board,
        *,
        nodes: int | None = None,
        multipv: int = 1,
    ) -> EngineResult:
        strings: list[str] = []
        with self._engine.analysis(
            board, self._limit(nodes), multipv=multipv, info=chess.engine.INFO_ALL
        ) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str):
                    strings.append(s)
            multipv_info = list(analysis.multipv)

        top: Mapping[str, Any] = multipv_info[0] if multipv_info else {}

        wdl: tuple[float, float, float] | None = None
        expectation: float | None = None
        pov_wdl = top.get("wdl")
        if pov_wdl is not None:
            rel = pov_wdl.pov(board.turn)
            w, d, ll = rel.wins, rel.draws, rel.losses
            total = w + d + ll
            if total > 0:
                wdl = (w / total, d / total, ll / total)
                expectation = wdl[0] + 0.5 * wdl[1]

        policy = self._policy_from_strings(board, strings)
        bestmove = self._bestmove(board, top, policy)
        if not policy and bestmove is not None:
            policy = {bestmove: 1.0}

        return EngineResult(
            kind="optimal",
            bestmove=bestmove,
            policy=policy,
            wdl=wdl,
            expectation=expectation,
            elo=None,
            raw={"multipv": len(multipv_info), "nodes": nodes},
        )

    @staticmethod
    def _policy_from_strings(board: chess.Board, strings: list[str]) -> dict[chess.Move, float]:
        scores: dict[chess.Move, float] = {}
        legal = {m.uci(): m for m in board.legal_moves}
        for line in strings:
            match = _VERBOSE_RE.match(line)
            if not match:
                continue
            uci, pct = match.group(1), match.group(2)
            move = legal.get(uci.lower())
            if move is None:
                continue
            try:
                scores[move] = float(pct)
            except ValueError:
                continue
        return normalize_policy(scores)

    @staticmethod
    def _bestmove(
        board: chess.Board, top: Mapping[str, Any], policy: dict[chess.Move, float]
    ) -> chess.Move | None:
        pv = top.get("pv")
        if pv:
            return pv[0]
        if policy:
            return max(policy, key=lambda m: policy[m])
        return None

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:  # noqa: BLE001 - already-dead process is fine
            pass
