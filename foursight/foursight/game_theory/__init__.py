"""Adversarial-coalition game theory for the WORST sight (3+ actors)."""

from foursight.game_theory.spec import Action, ActorId, GameSpec, PayoffTable
from foursight.game_theory.worst import worst_applicable, worst_outcome

__all__ = [
    "Action",
    "ActorId",
    "GameSpec",
    "PayoffTable",
    "worst_applicable",
    "worst_outcome",
]
