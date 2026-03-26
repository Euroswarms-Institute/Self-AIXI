"""Phase 3 — function approximation for Self-AIXI policies and Q (IMPLEMENTATION_PLAN §6)."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from random import Random

import numpy as np

from aixi.aixi.planning.self_aixi.agent import Policy


@dataclass
class SoftmaxParamPolicy(Policy):
    """Stationary π(a) ∝ exp(θ_a / T) over the actions that appear in ``action_logits``."""

    action_logits: dict[int, float]
    temperature: float = 1.0

    def _tem(self) -> float:
        t = self.temperature
        if t <= 0:
            raise ValueError("temperature must be > 0")
        return t

    def prob(self, action: int, valid_actions: tuple[int, ...]) -> float:
        if not valid_actions:
            return 0.0
        t = self._tem()
        logits = np.array(
            [float(self.action_logits.get(int(a), -1.0e9)) for a in valid_actions],
            dtype=float,
        )
        logits = logits / t
        logits -= float(np.max(logits))
        w = np.exp(logits)
        s = float(np.sum(w))
        if s <= 0.0:
            return 1.0 / len(valid_actions)
        try:
            idx = valid_actions.index(int(action))
        except ValueError:
            return 0.0
        return float(w[idx] / s)

    def sample_action(self, rng: Random, valid_actions: tuple[int, ...]) -> int:
        if not valid_actions:
            raise ValueError("valid_actions must be non-empty")
        probs = [self.prob(int(a), valid_actions) for a in valid_actions]
        s = float(sum(probs))
        if s <= 0.0:
            return int(rng.choice(valid_actions))
        weights = [p / s for p in probs]
        return int(rng.choices(valid_actions, weights=weights, k=1)[0])


def fit_softmax_logits_to_policy(
    reference: Policy,
    valid_actions: tuple[int, ...],
    *,
    rng: Random,
    steps: int = 3000,
    lr: float = 0.2,
    temperature: float = 1.0,
) -> SoftmaxParamPolicy:
    """
    Minimize cross-entropy H(p_ref || p_model) with gradient descent on logits.
    """
    theta = {int(a): float(rng.gauss(0.0, 0.01)) for a in valid_actions}
    p_ref = np.array(
        [max(reference.prob(int(a), valid_actions), 1e-12) for a in valid_actions],
        dtype=float,
    )
    p_ref = p_ref / float(np.sum(p_ref))
    for _ in range(steps):
        logits = np.array([theta[int(a)] for a in valid_actions], dtype=float)
        logits -= float(np.max(logits))
        p_model = np.exp(logits)
        p_model = p_model / float(np.sum(p_model))
        grad = p_model - p_ref
        for i, a in enumerate(valid_actions):
            theta[int(a)] -= lr * float(grad[i])
    return SoftmaxParamPolicy(action_logits=theta, temperature=temperature)


@dataclass
class HistoryKeyedQFA:
    """
    Q̂(h, ·) stored at a discrete history key (e.g. ξ bit-stream prefix).

    This is the constructive Phase-3 baseline: a finite-capacity FA that **interpolates
    nowhere**—it exactly reproduces Phase-2 tabular rollout targets on visited keys and
    surfaces the hook where smooth / neural Q heads plug in later.
    """

    _table: dict[Hashable, dict[int, float]] = field(default_factory=dict)

    def update(self, key: Hashable, q_by_action: Mapping[int, float]) -> None:
        self._table[key] = {int(a): float(q_by_action[a]) for a in q_by_action}

    def predict(self, key: Hashable, valid_actions: tuple[int, ...]) -> dict[int, float]:
        if key not in self._table:
            raise KeyError("history key has no stored Q snapshot")
        row = self._table[key]
        return {int(a): float(row[int(a)]) for a in valid_actions}


def greedy_action_from_q(q: Mapping[int, float], valid_actions: tuple[int, ...]) -> int:
    return max(valid_actions, key=lambda x: (float(q[int(x)]), -int(x)))
