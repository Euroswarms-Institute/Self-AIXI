"""Wrap a ``pyaixi.environment.Environment`` as ``aixi.aixi.env.protocol.Environment``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aixi.aixi.env.protocol import FiniteSpace, Percept


def _space_size(values: list[int]) -> int:
    return int(max(values)) + 1 if values else 0


@dataclass
class PyaixiEnvAdapter:
    """Thin protocol view over a pyaixi env (shared object — same instance as ``MC_AIXI_CTW_Agent`` uses)."""

    inner: Any
    action_space: FiniteSpace = field(init=False)
    observation_space: FiniteSpace = field(init=False)
    reward_range: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_space", FiniteSpace(_space_size(list(self.inner.valid_actions))))
        object.__setattr__(
            self,
            "observation_space",
            FiniteSpace(_space_size(list(self.inner.valid_observations))),
        )
        rewards = [int(r) for r in self.inner.valid_rewards]
        object.__setattr__(self, "reward_range", (float(min(rewards)), float(max(rewards))))

    def reset(self) -> Percept:
        """Re-run the pyaixi constructor in place so episode state matches a fresh ``__init__``."""
        opts = dict(getattr(self.inner, "options", {}) or {})
        self.inner.__init__(opts)
        return Percept(int(self.inner.observation), float(self.inner.reward))

    def step(self, action: int) -> Percept:
        if not self.inner.is_valid_action(action):
            raise ValueError(f"invalid action {action!r} for {type(self.inner).__name__}")
        self.inner.perform_action(action)
        return Percept(int(self.inner.observation), float(self.inner.reward))
