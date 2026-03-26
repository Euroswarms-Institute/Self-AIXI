"""Tabular Self-AIXI v0: ω over finitely many policies, greedy action on Q_{ωξ}."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aixi.aixi.models.mixture import MixtureEnvModel

from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
from aixi.aixi.planning.xi_rollouts import (
    imagined_trajectory_discounted_return,
    restore_mixture_after_imagination,
)


class Policy:
    """Stationary policy over a finite action set (toy Self-AIXI v0)."""

    def prob(self, action: int, valid_actions: tuple[int, ...]) -> float:
        """Likelihood π(a | valid_actions) for ω Bayes updates."""
        raise NotImplementedError

    def sample_action(self, rng: Random, valid_actions: tuple[int, ...]) -> int:
        """Imagination-time action choice."""
        raise NotImplementedError


@dataclass
class SoftDeterministicPolicy(Policy):
    """Nearly deterministic preferred action (smooth likelihoods for ω updates)."""

    preferred: int
    slip: float = 0.05

    def prob(self, action: int, valid_actions: tuple[int, ...]) -> float:
        if action == self.preferred:
            return 1.0 - self.slip
        n = len(valid_actions)
        if n <= 1:
            return 1.0
        return self.slip / (n - 1)

    def sample_action(self, rng: Random, valid_actions: tuple[int, ...]) -> int:
        if not valid_actions:
            raise ValueError("valid_actions must be non-empty")
        p_pref = 1.0 - self.slip
        if rng.random() < p_pref and self.preferred in valid_actions:
            return int(self.preferred)
        others = [a for a in valid_actions if a != self.preferred]
        return int(rng.choice(others) if others else valid_actions[0])


@dataclass
class UniformRandomPolicy(Policy):
    """Uniform over legal actions (stationary)."""

    def prob(self, action: int, valid_actions: tuple[int, ...]) -> float:
        if not valid_actions or action not in valid_actions:
            return 0.0
        return 1.0 / len(valid_actions)

    def sample_action(self, rng: Random, valid_actions: tuple[int, ...]) -> int:
        return int(rng.choice(valid_actions))


@dataclass
class SelfAIXIV0Agent:
    """
    Greedy agent on Q_{ωξ}(h,a) ≈ Σ_π ω(π|h) · G_{π,ξ}(h,a), with G from a single ξ rollout each.

    ω is updated from observed actions via Bayes rule with π(a) as likelihood (§4.2).
    This targets the double-mixture Q (notation table: Q_{ζξ}); it is not Bayes-optimal Q_ξ^*.
    """

    xi: MixtureEnvModel
    encode_action: Callable[[int], Sequence[int]]
    encode_percept: Callable[..., Sequence[int]]
    decode_percept: Callable[[list[int]], tuple[object, float]]
    valid_actions: tuple[int, ...]
    n_action_bits: int
    n_percept_bits: int
    policies: tuple[Policy, ...]
    imagination_cycles: int
    discount_gamma: float
    rng: Random
    omega: list[float] = field(init=False)
    last_q_by_action: dict[int, float] = field(default_factory=dict)
    last_q_pi: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("policies must be non-empty")
        n = len(self.policies)
        self.omega = [1.0 / n] * n

    def _decode_reward_only(self, symbols: Sequence[int]) -> float:
        _obs, rew = self.decode_percept(list(symbols))
        return float(rew)

    def update_omega(self, observed_action: int) -> None:
        lik = [max(p.prob(observed_action, self.valid_actions), 1e-12) for p in self.policies]
        s = sum(w * ell for w, ell in zip(self.omega, lik, strict=True))
        if s <= 0.0:
            return
        for i in range(len(self.omega)):
            self.omega[i] = self.omega[i] * lik[i] / s

    def act(self) -> int:
        """Return argmax_a Σ_k ω_k · G_{π_k,ξ}(a) with one imagined rollout per (a, k)."""
        ref_lp = self.xi.root_log_probability()
        xi_snap = (
            self.xi.snapshot_symbol_history() if isinstance(self.xi, PyAixiCTWBitMixture) else None
        )

        q_by_action: dict[int, float] = {}
        q_pi_rows: list[list[float]] = []

        for a in self.valid_actions:
            mix_q = 0.0
            row: list[float] = []
            for k, pol in enumerate(self.policies):
                g = imagined_trajectory_discounted_return(
                    self.xi,
                    first_action=a,
                    encode_action=self.encode_action,
                    n_action_bits=self.n_action_bits,
                    n_percept_bits=self.n_percept_bits,
                    decode_reward=self._decode_reward_only,
                    valid_actions=self.valid_actions,
                    n_cycles=self.imagination_cycles,
                    gamma=self.discount_gamma,
                    rng=self.rng,
                    subsequent_action=lambda r, acts, p=pol: p.sample_action(r, acts),
                )
                restore_mixture_after_imagination(
                    self.xi,
                    xi_snap=xi_snap,
                    ref_log_p=ref_lp,
                    n_action_bits=self.n_action_bits,
                    n_percept_bits=self.n_percept_bits,
                    n_cycles=self.imagination_cycles,
                )
                row.append(g)
                mix_q += self.omega[k] * g
            q_pi_rows.append(row)
            q_by_action[a] = mix_q

        self.last_q_by_action = q_by_action
        self.last_q_pi = q_pi_rows

        return max(self.valid_actions, key=lambda x: (q_by_action[x], -x))

    def append_real_action(self, action: int) -> None:
        self.xi.append_history_symbols(self.encode_action(action))

    def learn_real_percept(self, observation: object, reward: float) -> None:
        self.xi.learn_symbols(self.encode_percept(observation, reward))
