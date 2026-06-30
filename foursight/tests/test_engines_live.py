"""Opt-in live checks against the real engines.

These only run with ``FOURSIGHT_LIVE=1`` and real binaries/nets configured; the
CPU gate skips them entirely. They sanity-check that Leela-CF returns a coherent
WDL and that Maia-3 reproduces a published-ish move-match on a tiny fixed set.
"""

from __future__ import annotations

import math
import os

import chess
import pytest

from foursight.config import load_settings
from foursight.engines.registry import build_engines

pytestmark = pytest.mark.skipif(
    os.environ.get("FOURSIGHT_LIVE") != "1",
    reason="set FOURSIGHT_LIVE=1 with real engines to run",
)

# A few opening positions and the human-typical reply, for a loose move-match.
_HUMAN_FIXTURES = [
    (chess.STARTING_FEN, {"e2e4", "d2d4", "g1f3", "c2c4"}),
    (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        {"c7c5", "e7e5", "e7e6", "c7c6", "g8f6"},
    ),
]


def test_leela_cf_returns_sane_wdl() -> None:
    settings = load_settings()
    with build_engines(settings) as pair:
        if pair.optimal.is_mock:
            pytest.skip("optimal axis resolved to mock; configure Leela-CF")
        result = pair.optimal.analyse(chess.Board(), nodes=100)
        assert result.wdl is not None
        assert math.isclose(sum(result.wdl), 1.0, abs_tol=1e-6)
        assert result.expectation is not None
        assert 0.0 <= result.expectation <= 1.0
        assert result.bestmove in chess.Board().legal_moves


def test_maia3_reproduces_human_moves() -> None:
    settings = load_settings()
    with build_engines(settings) as pair:
        if pair.human.is_mock:
            pytest.skip("human axis resolved to mock; configure Maia-3")
        hits = 0
        for fen, plausible in _HUMAN_FIXTURES:
            board = chess.Board(fen)
            result = pair.human.analyse(board, nodes=1)
            if result.bestmove and result.bestmove.uci() in plausible:
                hits += 1
        # Expect the human net to pick a human-typical move most of the time.
        assert hits >= len(_HUMAN_FIXTURES) - 1
