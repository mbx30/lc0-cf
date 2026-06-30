"""The CPU gate: registry -> mock engines behave like real symmetric adapters."""

from __future__ import annotations

import math

import chess

from foursight.config import Settings
from foursight.engines.base import ChessEngine
from foursight.engines.registry import build_engines, doctor


def _entropy(policy: dict[chess.Move, float]) -> float:
    return -sum(p * math.log2(p) for p in policy.values() if p > 0)


def _mock_pair() -> tuple[ChessEngine, ChessEngine]:
    pair = build_engines(Settings(force_mock=True))
    return pair.optimal, pair.human


def test_force_mock_builds_two_mocks() -> None:
    optimal, human = _mock_pair()
    assert optimal.is_mock and human.is_mock
    assert optimal.kind == "optimal"
    assert human.kind == "human"
    optimal.close()
    human.close()


def test_doctor_reports_mock_under_force() -> None:
    rows = doctor(Settings(force_mock=True))
    assert {r["role"] for r in rows} == {"optimal", "human"}
    assert all(r["is_mock"] for r in rows)


def test_optimal_policy_is_a_distribution() -> None:
    optimal, _ = _mock_pair()
    board = chess.Board()
    result = optimal.analyse(board)
    assert result.policy
    assert all(mv in board.legal_moves for mv in result.policy)
    assert math.isclose(sum(result.policy.values()), 1.0, abs_tol=1e-9)
    assert result.bestmove in board.legal_moves
    assert result.wdl is not None
    assert math.isclose(sum(result.wdl), 1.0, abs_tol=1e-9)
    assert result.expectation is not None
    assert 0.0 <= result.expectation <= 1.0


def test_mock_is_deterministic() -> None:
    optimal, _ = _mock_pair()
    board = chess.Board()
    a = optimal.analyse(board).policy
    b = optimal.analyse(board).policy
    assert a == b


def test_human_policy_warms_with_lower_elo() -> None:
    _, human = _mock_pair()
    board = chess.Board()
    low = human.policy(board, elo=1100)
    high = human.policy(board, elo=2200)
    # Stronger players are more decisive -> lower-entropy policy.
    assert _entropy(high) < _entropy(low)


def test_human_policy_is_flatter_than_optimal() -> None:
    optimal, human = _mock_pair()
    board = chess.Board()
    assert _entropy(human.policy(board, elo=1500)) > _entropy(optimal.analyse(board).policy)


def test_expectation_after_terminal_mate() -> None:
    optimal, _ = _mock_pair()
    # Fool's mate: white to play Qh5xf7# is not set up; use a clean mate-in-1.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    mate = chess.Move.from_uci("a1a8")
    assert board.is_legal(mate)
    after = board.copy()
    after.push(mate)
    assert after.is_checkmate()
    assert optimal.expectation_after(board, mate) == 1.0
