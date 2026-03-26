"""AIQI prototype skeleton: augmentation schedule + ψ hook + τ-greedy loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from aixi.aixi.env.protocol import Environment, Percept

from aixi.aixi.planning.aiqi.config import AIQIConfig
from aixi.aixi.planning.aiqi.predictor import (
    ReturnPredictor,
    mean_return_from_probs,
    total_return_to_bin,
)
from aixi.aixi.planning.aiqi.schedule import augmentation_insert_at_timestep
from aixi.aixi.planning.aiqi.shell import epsilon_greedy_action

if TYPE_CHECKING:
    pass


@dataclass
class AIQISkeletonAgent:
    """On-policy skeleton: Q̂(h,a) from ψ_phase; no environment model in the decision rule."""

    env: Environment
    config: AIQIConfig
    predictor: ReturnPredictor
    rng: Random = field(default_factory=Random)
    timestep: int = 0
    last_percept: Percept | None = None
    last_action: int | None = field(default=None, repr=False)
    _obs_before_action: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rewards: deque[float] = deque(maxlen=self.config.horizon_H)

    def valid_actions(self) -> tuple[int, ...]:
        return tuple(range(self.env.action_space.size))

    def augmentation_cycle_slot(self) -> int:
        """``timestep mod N`` — which slot in the length-N cycle (not ``floor(t/N)``; see ``aixi.formal_ci``)."""
        return self.timestep % self.config.augmentation_period_N

    def estimated_q(self, action: int) -> float:
        if self.last_percept is None:
            raise RuntimeError("observe a percept before querying Q̂")
        probs = self.predictor.predicted_return_probs(
            timestep=self.timestep,
            history_len=self.timestep,
            observation=self.last_percept.observation,
            action=action,
            grid_bins_M=self.config.grid_bins_M,
            phase_n=self.augmentation_cycle_slot(),
        )
        return mean_return_from_probs(probs, self.config.grid_bins_M)

    def act(self) -> int:
        if self.last_percept is None:
            raise RuntimeError("observe a percept before act")
        self._obs_before_action = self.last_percept.observation
        acts = self.valid_actions()
        q_map = {a: self.estimated_q(a) for a in acts}
        chosen = epsilon_greedy_action(
            q_map,
            acts,
            self.config.exploration_tau,
            self.rng,
        )
        self.last_action = chosen
        return chosen

    def observe(self, percept: Percept) -> None:
        if (
            self.last_action is not None
            and self._obs_before_action is not None
            and self.timestep > 0
        ):
            self._rewards.append(percept.reward)
            t_idx = self.timestep - 1
            r_lo, r_hi = self.env.reward_range
            for n in range(self.config.augmentation_period_N):
                if not augmentation_insert_at_timestep(t_idx, n, self.config.augmentation_period_N):
                    continue
                h = min(len(self._rewards), self.config.horizon_H)
                total = sum(list(self._rewards)[-h:])
                b = total_return_to_bin(
                    total,
                    horizon_H=self.config.horizon_H,
                    grid_bins_M=self.config.grid_bins_M,
                    reward_min=r_lo,
                    reward_max=r_hi,
                )
                updater = getattr(self.predictor, "observe_augmented_return", None)
                if callable(updater):
                    updater(
                        phase_n=n,
                        observation=self._obs_before_action,
                        action=self.last_action,
                        return_bin=b,
                        grid_bins_M=self.config.grid_bins_M,
                    )
        self.last_percept = percept
        self.timestep += 1
        self.last_action = None

    def phase_inserts_return_symbol(self, phase_n: int) -> bool:
        """Whether the paper's augmentation slot fires for ``phase_n`` at the *previous* step index."""

        if self.timestep == 0:
            return False
        return augmentation_insert_at_timestep(self.timestep - 1, phase_n, self.config.augmentation_period_N)
