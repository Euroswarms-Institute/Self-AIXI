"""ε-greedy action shell (exploration rate τ in config)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random


def epsilon_greedy_action(
    q_hat: Mapping[int, float],
    valid_actions: Sequence[int],
    exploration_tau: float,
    rng: Random,
) -> int:
    """With probability ``exploration_tau``, sample uniformly from ``valid_actions``; else argmax Q̂."""

    acts = tuple(valid_actions)
    if not acts:
        raise ValueError("valid_actions must be non-empty")
    if exploration_tau < 0.0 or exploration_tau > 1.0:
        raise ValueError("exploration_tau must be in [0, 1]")
    if rng.random() < exploration_tau:
        return int(rng.choice(acts))
    best = max((q_hat.get(a, float("-inf")), a) for a in acts)
    return int(best[1])
