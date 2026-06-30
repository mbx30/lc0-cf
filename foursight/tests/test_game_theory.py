"""Tests for game_theory WORST sight."""

from __future__ import annotations

import pytest

from foursight.game_theory import GameSpec, worst_applicable, worst_outcome


def test_worst_applicable_threshold() -> None:
    assert not worst_applicable(2)
    assert worst_applicable(3)


def test_worst_outcome_adversarial_coalition() -> None:
    game = GameSpec(
        actors=["focal", "b", "c"],
        action_sets={"focal": ["a1", "a2"], "b": ["x", "y"], "c": ["x", "y"]},
        payoffs={},
        focal="focal",
    )
    # Populate payoffs: focal hurts most when b=y and c=y
    for fa in ["a1", "a2"]:
        for j in [("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")]:
            profile = tuple(sorted([("focal", fa), ("b", j[0]), ("c", j[1])]))
            key = tuple(a for _, a in profile)
            base = 0.6 if fa == "a1" else 0.55
            penalty = 0.2 if j == ("y", "y") else 0.0
            game.payoffs[("focal", key)] = base - penalty

    w = worst_outcome(game, "a1")
    assert w < 0.6
    assert abs(w - 0.4) < 1e-9


def test_worst_raises_for_two_actors() -> None:
    game = GameSpec(
        actors=["a", "b"],
        action_sets={"a": ["x"], "b": ["y"]},
        payoffs={("a", ("x", "y")): 0.5},
        focal="a",
    )
    with pytest.raises(ValueError):
        worst_outcome(game, "x")
