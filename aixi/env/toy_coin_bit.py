"""Minimal toy env implementing ``Environment`` (Family C smoke tests, no pyaixi)."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from aixi.aixi.env.protocol import FiniteSpace, Percept


@dataclass
class ToyBiasedCoinEnv:
    """Single-step-style bandit: two actions (ignored for dynamics), observation ~ Bernoulli(p).

    Rewards match the observation bit as float in ``[0, 1]`` — smallest signal for AIQI plumbing.
    """

    p_heads: float = 0.6
    rng: Random = field(default_factory=Random)
    action_space: FiniteSpace = field(default_factory=lambda: FiniteSpace(2))
    observation_space: FiniteSpace = field(default_factory=lambda: FiniteSpace(2))
    reward_range: tuple[float, float] = (0.0, 1.0)
    _observation: int = 0

    def reset(self) -> Percept:
        self._observation = 1 if self.rng.random() < self.p_heads else 0
        return Percept(self._observation, float(self._observation))

    def step(self, action: int) -> Percept:
        if action < 0 or action >= self.action_space.size:
            raise ValueError(f"action {action} out of range for action_space.size={self.action_space.size}")
        self._observation = 1 if self.rng.random() < self.p_heads else 0
        return Percept(self._observation, float(self._observation))
