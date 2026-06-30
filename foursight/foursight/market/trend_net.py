"""Tiny trend net — DOWN / FLAT / UP policy + value head (numpy, CPU-only).

Mirrors the Maia v2 blueprint trend predictor: rolling-window features over
implied-probability series, three-way softmax policy, scalar value output.
No PyTorch dependency — suitable for laptop inference and CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foursight.market.records import OutcomeLabel, PricePoint

LABELS: tuple[OutcomeLabel, ...] = ("down", "flat", "up")
_FLAT_BAND = 0.01


def _features(window: list[float]) -> np.ndarray:
    """Build feature vector from a window of implied probabilities."""
    arr = np.asarray(window, dtype=np.float64)
    if len(arr) < 2:
        return np.zeros(4, dtype=np.float64)
    ret_1 = arr[-1] - arr[-2]
    ret_w = arr[-1] - arr[0]
    vol = float(np.std(np.diff(arr))) if len(arr) > 2 else 0.0
    ma_ratio = arr[-1] / (float(np.mean(arr)) + 1e-9)
    return np.array([ret_1, ret_w, vol, ma_ratio], dtype=np.float64)


def label_from_delta(delta: float, *, flat_band: float = _FLAT_BAND) -> OutcomeLabel:
    if delta > flat_band:
        return "up"
    if delta < -flat_band:
        return "down"
    return "flat"


@dataclass
class TrendNetWeights:
    """Small MLP weights: 4 features -> hidden -> 3 policy logits + 1 value."""

    w1: np.ndarray  # (hidden, 4)
    b1: np.ndarray  # (hidden,)
    w_policy: np.ndarray  # (3, hidden)
    b_policy: np.ndarray  # (3,)
    w_value: np.ndarray  # (1, hidden)
    b_value: np.ndarray  # (1,)

    @classmethod
    def random(cls, hidden: int = 8, seed: int = 0) -> TrendNetWeights:
        rng = np.random.default_rng(seed)
        return cls(
            w1=rng.normal(0, 0.1, size=(hidden, 4)),
            b1=np.zeros(hidden),
            w_policy=rng.normal(0, 0.1, size=(3, hidden)),
            b_policy=np.zeros(3),
            w_value=rng.normal(0, 0.1, size=(1, hidden)),
            b_value=np.zeros(1),
        )


@dataclass
class TrendPrediction:
    """One forward-pass output at the end of a price window."""

    policy: dict[OutcomeLabel, float]
    value: float
    best_label: OutcomeLabel


class TrendNet:
    """Single forward-pass predictor over implied-probability windows."""

    def __init__(self, weights: TrendNetWeights | None = None, *, window: int = 8) -> None:
        self.weights = weights or TrendNetWeights.random()
        self.window = window

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        w = self.weights
        h = np.tanh(w.w1 @ x + w.b1)
        logits = w.w_policy @ h + w.b_policy
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        value = float((w.w_value @ h + w.b_value).item())
        return probs, value

    def predict(self, prices: list[PricePoint]) -> TrendPrediction | None:
        """Predict from the trailing ``window`` implied probabilities."""
        if len(prices) < 2:
            return None
        window_vals = [p.implied_prob for p in prices[-self.window :]]
        probs, value = self._forward(_features(window_vals))
        policy = {label: float(probs[i]) for i, label in enumerate(LABELS)}
        best = max(policy, key=lambda k: policy[k])
        return TrendPrediction(policy=policy, value=value, best_label=best)

    def fit_gd(
        self,
        examples: list[tuple[list[float], OutcomeLabel, float]],
        *,
        epochs: int = 50,
        lr: float = 0.05,
    ) -> None:
        """Minimal batch GD trainer for tests and small corpora."""
        w = self.weights
        label_to_idx = {label: i for i, label in enumerate(LABELS)}

        for _ in range(epochs):
            for window, label, target_value in examples:
                x = _features(window)
                h = np.tanh(w.w1 @ x + w.b1)
                logits = w.w_policy @ h + w.b_policy
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / probs.sum()
                y = label_to_idx[label]
                grad_logits = probs.copy()
                grad_logits[y] -= 1.0

                w.w_policy -= lr * np.outer(grad_logits, h)
                w.b_policy -= lr * grad_logits
                dh = w.w_policy.T @ grad_logits
                dh *= 1.0 - h**2
                w.w1 -= lr * np.outer(dh, x)
                w.b1 -= lr * dh

                pred_v = float((w.w_value @ h + w.b_value).item())
                grad_v = 2.0 * (pred_v - target_value)
                w.w_value -= lr * grad_v * h.reshape(1, -1)
                w.b_value -= lr * np.array([grad_v])
