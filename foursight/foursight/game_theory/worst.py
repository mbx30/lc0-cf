"""WORST sight: adversarial-coalition minimum payoff."""

from __future__ import annotations

from foursight.game_theory.spec import Action, GameSpec


def worst_applicable(actor_count: int) -> bool:
    """WORST is defined only when there are three or more actors."""
    return actor_count >= 3


def worst_outcome(game: GameSpec, focal_action: Action) -> float:
    """Minimum focal payoff assuming all other actors coordinate against focal.

    WORST(a_i) = min over joint opponent strategies of u_i(a_i, a_{-i})
    """
    if not worst_applicable(game.actor_count()):
        raise ValueError("worst_outcome requires actor_count >= 3")

    opponents = game.joint_actions_for(focal_action)
    if not opponents:
        return game.payoff(focal_action, ())

    return min(game.payoff(focal_action, joint) for joint in opponents)
