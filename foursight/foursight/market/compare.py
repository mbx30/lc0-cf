"""Four-way market comparison — OPTIMAL | HUMAN | ACTUAL | WORST + six gaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import cast

from foursight.game_theory import GameSpec, worst_applicable, worst_outcome
from foursight.game_theory.spec import Action, PayoffTable
from foursight.market.records import MarketRecord, OutcomeLabel
from foursight.market.trend_net import LABELS, TrendNet


@dataclass
class MarketGap:
    """Pairwise gap metrics between two market sights."""

    delta_prob: float | None = None
    target_prob: float | None = None
    agree_label: bool | None = None
    realized_score: float | None = None


@dataclass
class MarketGaps:
    opt_vs_human: MarketGap
    human_vs_actual: MarketGap
    opt_vs_actual: MarketGap
    worst_vs_opt: MarketGap
    worst_vs_human: MarketGap
    worst_vs_actual: MarketGap


@dataclass
class MarketSights:
    """The four sight values at one evaluation point."""

    optimal_prob: float
    human_prob: float
    actual_score: float | None
    worst_payoff: float | None
    human_policy: dict[OutcomeLabel, float] | None = None
    optimal_label: OutcomeLabel | None = None
    human_label: OutcomeLabel | None = None
    actual_label: OutcomeLabel | None = None


@dataclass
class FourWayComparison:
    """Four sights for one market evaluation, plus six pairwise gaps."""

    market_id: str
    platform: str
    question: str
    actor_count: int
    sights: MarketSights
    gaps: MarketGaps
    worst_applicable: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "platform": self.platform,
            "question": self.question,
            "actor_count": self.actor_count,
            "worst_applicable": self.worst_applicable,
            "sights": asdict(self.sights),
            "gaps": {
                "opt_vs_human": asdict(self.gaps.opt_vs_human),
                "human_vs_actual": asdict(self.gaps.human_vs_actual),
                "opt_vs_actual": asdict(self.gaps.opt_vs_actual),
                "worst_vs_opt": asdict(self.gaps.worst_vs_opt),
                "worst_vs_human": asdict(self.gaps.worst_vs_human),
                "worst_vs_actual": asdict(self.gaps.worst_vs_actual),
            },
            "extra": self.extra,
        }


def _label_from_prob_delta(delta: float, *, flat_band: float = 0.01) -> OutcomeLabel:
    if delta > flat_band:
        return "up"
    if delta < -flat_band:
        return "down"
    return "flat"


def _market_game(actor_count: int, focal_action: str = "trade") -> GameSpec:
    """Toy 3-actor game: focal vs market_maker vs informed_flow."""
    actors = ["focal", "market_maker", "informed_flow"][: max(3, actor_count)]
    while len(actors) < 3:
        actors.append(f"actor_{len(actors)}")
    actions_focal = cast(list[Action], ["hold", "trade"])
    actions_opp = cast(list[Action], ["passive", "aggressive"])
    action_sets: dict[str, list[Action]] = {actors[0]: actions_focal}
    for a in actors[1:]:
        action_sets[a] = actions_opp

    payoffs: PayoffTable = {}
    for fa in actions_focal:
        for j1 in actions_opp:
            for j2 in actions_opp:
                joint = (j1, j2) if len(actors) >= 3 else (j1,)
                # Coalition hurts focal when both opponents are aggressive and focal trades.
                agg = sum(1 for x in joint if x == "aggressive")
                base = 0.55 if fa == "trade" else 0.5
                payoff = base - 0.12 * agg - (0.05 if fa == "trade" and agg == len(joint) else 0.0)
                profile = tuple(sorted([(actors[0], fa), (actors[1], j1), (actors[2], j2)]))
                key_actions = tuple(a for _, a in profile)
                payoffs[(actors[0], key_actions)] = payoff
    return GameSpec(actors=actors[:3], action_sets=action_sets, payoffs=payoffs, focal=actors[0])


class FourWayComparator:
    """Emit :class:`FourWayComparison` for a :class:`MarketRecord`."""

    def __init__(self, trend_net: TrendNet | None = None) -> None:
        self.trend_net = trend_net or TrendNet()

    def compare(self, record: MarketRecord) -> FourWayComparison:
        if not record.prices:
            raise ValueError("MarketRecord needs at least one price point")

        start_prob = record.prices[0].implied_prob
        end_prob = record.prices[-1].implied_prob
        pred = self.trend_net.predict(record.prices)

        human_prob = pred.policy["up"] - pred.policy["down"] + 0.5 if pred else end_prob
        human_prob = max(0.0, min(1.0, human_prob))
        optimal_prob = max(0.0, min(1.0, 0.5 + 0.5 * (end_prob - start_prob)))

        actual_score = record.realized_score()
        actual_label = record.outcome_label()
        opt_label = _label_from_prob_delta(optimal_prob - start_prob)
        hum_label = pred.best_label if pred else _label_from_prob_delta(end_prob - start_prob)

        worst_payoff: float | None = None
        w_app = worst_applicable(record.actor_count)
        if w_app:
            game = _market_game(record.actor_count)
            worst_payoff = worst_outcome(game, "trade")

        sights = MarketSights(
            optimal_prob=optimal_prob,
            human_prob=human_prob,
            actual_score=actual_score,
            worst_payoff=worst_payoff,
            human_policy=pred.policy if pred else None,
            optimal_label=opt_label,
            human_label=hum_label,
            actual_label=actual_label,
        )

        def gap(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return a - b

        opt_vs_human = MarketGap(
            delta_prob=gap(optimal_prob, human_prob),
            target_prob=human_prob,
            agree_label=(opt_label == hum_label),
        )
        human_vs_actual = MarketGap(
            delta_prob=gap(human_prob, actual_score),
            target_prob=human_prob,
            agree_label=(hum_label == actual_label) if actual_label else None,
            realized_score=actual_score,
        )
        opt_vs_actual = MarketGap(
            delta_prob=gap(optimal_prob, actual_score),
            target_prob=optimal_prob,
            agree_label=(opt_label == actual_label) if actual_label else None,
            realized_score=actual_score,
        )
        worst_vs_opt = MarketGap(
            delta_prob=gap(worst_payoff, optimal_prob),
            target_prob=worst_payoff,
        )
        worst_vs_human = MarketGap(
            delta_prob=gap(worst_payoff, human_prob),
            target_prob=worst_payoff,
        )
        worst_vs_actual = MarketGap(
            delta_prob=gap(worst_payoff, actual_score),
            target_prob=worst_payoff,
            realized_score=actual_score,
        )

        gaps = MarketGaps(
            opt_vs_human=opt_vs_human,
            human_vs_actual=human_vs_actual,
            opt_vs_actual=opt_vs_actual,
            worst_vs_opt=worst_vs_opt,
            worst_vs_human=worst_vs_human,
            worst_vs_actual=worst_vs_actual,
        )

        return FourWayComparison(
            market_id=record.market_id,
            platform=record.platform,
            question=record.question,
            actor_count=record.actor_count,
            sights=sights,
            gaps=gaps,
            worst_applicable=w_app,
            extra={"n_prices": len(record.prices), "labels": list(LABELS)},
        )
