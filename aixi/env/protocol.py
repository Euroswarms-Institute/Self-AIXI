from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class FiniteSpace:
    """Finite outcome set (IMPLEMENTATION_PLAN §2)."""

    size: int


@dataclass(frozen=True)
class Percept:
    """One percept e_t = (observation, reward) after action."""

    observation: int
    reward: float


@runtime_checkable
class Environment(Protocol):
    """Minimal environment contract aligned with IMPLEMENTATION_PLAN §2."""

    action_space: FiniteSpace
    observation_space: FiniteSpace
    reward_range: Tuple[float, float]

    def reset(self) -> Percept: ...

    def step(self, action: int) -> Percept: ...
