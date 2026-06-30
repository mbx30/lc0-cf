"""Game specification for N-actor payoff evaluation."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

ActorId = str
Action = Hashable

PayoffTable = dict[tuple[ActorId, tuple[Action, ...]], float]


@dataclass
class GameSpec:
    """Discrete N-actor game: each actor picks one action; payoffs are tabular."""

    actors: list[ActorId]
    action_sets: dict[ActorId, list[Action]]
    payoffs: PayoffTable
    focal: ActorId
    meta: dict = field(default_factory=dict)

    def actor_count(self) -> int:
        return len(self.actors)

    def opponents(self) -> list[ActorId]:
        return [a for a in self.actors if a != self.focal]

    def joint_actions_for(self, focal_action: Action) -> list[tuple[Action, ...]]:
        """Enumerate opponent joint action tuples (brute force; small games only)."""
        opps = self.opponents()
        if not opps:
            return [()]

        sets = [self.action_sets[o] for o in opps]
        joints: list[tuple[Action, ...]] = [()]

        for action_set in sets:
            joints = [j + (a,) for j in joints for a in action_set]
        return joints

    def payoff(self, focal_action: Action, opponent_joint: tuple[Action, ...]) -> float:
        """Payoff to ``focal`` under a full action profile."""
        profile: list[tuple[ActorId, Action]] = [(self.focal, focal_action)]
        for opp, act in zip(self.opponents(), opponent_joint, strict=True):
            profile.append((opp, act))
        profile.sort(key=lambda x: x[0])
        key_actions = tuple(a for _, a in profile)
        return self.payoffs[(self.focal, key_actions)]
