"""Three-way compare on the mock pair: the structured deliverable."""

from __future__ import annotations

import json

import chess

from foursight.compare import (
    ThreeWayComparator,
    js_divergence,
    kl_divergence,
)
from foursight.config import Settings
from foursight.engines.registry import build_engines
from foursight.ingest.actual import records_from_fen_moves


def _comparator() -> ThreeWayComparator:
    pair = build_engines(Settings(force_mock=True))
    return ThreeWayComparator(pair.optimal, pair.human)


def test_identical_policies_have_zero_divergence() -> None:
    board = chess.Board()
    p = {mv: 1.0 / board.legal_moves.count() for mv in board.legal_moves}
    assert js_divergence(p, p) < 1e-9
    assert abs(kl_divergence(p, p)) < 1e-9


def test_divergence_is_non_negative_and_bounded() -> None:
    board = chess.Board()
    moves = list(board.legal_moves)
    p = {moves[0]: 0.9, moves[1]: 0.1}
    q = {moves[1]: 0.9, moves[2]: 0.1}
    js = js_divergence(p, q)
    assert 0.0 <= js <= 1.0 + 1e-9
    assert kl_divergence(p, q) >= -1e-9


def test_compare_without_actual_fills_optimal_human_gap() -> None:
    comparator = _comparator()
    board = chess.Board()
    c = comparator.compare(board, elo=1500)
    assert c.optimal.bestmove in board.legal_moves
    assert c.human.bestmove in board.legal_moves
    assert c.human.elo == 1500
    g = c.gaps.opt_vs_human
    assert isinstance(g.agree_at_1, bool)
    assert g.js_divergence is not None and g.js_divergence >= 0.0
    assert g.kl_divergence is not None
    assert g.delta_expectation is not None
    # actual-dependent gaps are empty without an actual move
    assert c.gaps.opt_vs_actual.agree_at_1 is None


def test_compare_with_actual_fills_all_three_gaps() -> None:
    comparator = _comparator()
    fen = chess.STARTING_FEN
    actual = records_from_fen_moves(fen, ["e2e4"], game_result="1-0")[0]
    board = chess.Board(fen)
    c = comparator.compare(board, elo=1500, actual=actual)

    assert c.actual is not None
    hva = c.gaps.human_vs_actual
    ova = c.gaps.opt_vs_actual
    assert isinstance(hva.agree_at_1, bool)
    assert 0.0 <= hva.target_prob <= 1.0
    assert isinstance(ova.agree_at_1, bool)
    assert ova.delta_expectation is not None
    # White won and white is to move at the start -> realized score 1.0
    assert ova.realized_score == 1.0


def test_comparison_to_dict_is_json_serializable() -> None:
    comparator = _comparator()
    actual = records_from_fen_moves(chess.STARTING_FEN, ["d2d4"])[0]
    c = comparator.compare(chess.Board(), elo=1800, actual=actual)
    blob = json.dumps(c.to_dict())
    parsed = json.loads(blob)
    assert parsed["elo"] == 1800
    assert parsed["actual"]["played_move"] == "d2d4"
    assert "opt_vs_human" in parsed["gaps"]
