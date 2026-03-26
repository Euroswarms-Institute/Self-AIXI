"""
Family A — minimal root UCT over ξ with explicit revert (IMPLEMENTATION_PLAN §4.1).

Search uses **finite** simulation count and **finite** planning horizon only; returns
are accumulated over a bounded number of imagined (action → percept) cycles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aixi.aixi.models.mixture import MixtureEnvModel

from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
from aixi.aixi.planning.xi_rollouts import (
    imagined_trajectory_discounted_return,
    restore_mixture_after_imagination,
)


@dataclass(frozen=True)
class MCTSSearchBudget:
    """
    **A1 — documented search budget** (finite truncation; no infinite-horizon semantics).

    - ``mc_simulations``: UCT iterations at the root (each ends by restoring ξ to the
      pre-rollout interaction stream — history replay for ``PyAixiCTWBitMixture``, else paired revert).
    - ``planning_horizon``: imagined (action → percept) cycles per rollout after the
      UCT-selected first action.
    - ``uct_exploration_c``: UCT exploration constant (typical ~sqrt(2)–2).
    - ``discount_gamma``: per-step discount on decoded scalar rewards.
    """

    mc_simulations: int
    planning_horizon: int
    uct_exploration_c: float = 1.4
    discount_gamma: float = 0.99

    def __post_init__(self) -> None:
        if self.mc_simulations < 1:
            raise ValueError("mc_simulations must be >= 1")
        if self.planning_horizon < 1:
            raise ValueError("planning_horizon must be >= 1")
        if self.uct_exploration_c < 0.0:
            raise ValueError("uct_exploration_c must be non-negative")
        if not (0.0 < self.discount_gamma <= 1.0):
            raise ValueError("discount_gamma must be in (0, 1]")


def root_uct_action(
    xi: MixtureEnvModel,
    budget: MCTSSearchBudget,
    *,
    valid_actions: Sequence[int],
    encode_action: Callable[[int], Sequence[int]],
    n_action_bits: int,
    n_percept_bits: int,
    decode_reward: Callable[[Sequence[int]], float],
    rng: Random,
) -> int:
    """
    Root-only predictive UCT: each simulation picks an arm (legal action), runs a
    bounded-length rollout under ξ (sampled percepts), backs up the return, then
    **fully reverts** ξ to the pre-rollout state.

    **Exploration term:** arms with zero visits use ``c·sqrt(log n + ε)`` with small ``ε``,
    not the unbounded ``+∞`` bonus of textbook UCT — a finite-root variant appropriate
    for fixed ``mc_simulations``.

    Preconditions: same as MC-AIXI ``search()`` — ξ must be ready for an **action**
    (real history should end with a learned percept).
    """
    actions = tuple(valid_actions)
    if not actions:
        raise ValueError("valid_actions must be non-empty")

    for a in actions:
        if len(encode_action(a)) != n_action_bits:
            raise ValueError(
                f"encode_action({a!r}) length must equal n_action_bits={n_action_bits}"
            )

    ref_log_p = xi.root_log_probability()
    # pyaixi: incremental revert after predict-heavy rollouts can desync internal KT weights from
    # ``history`` (extra nodes / wrong root log). Replay from a snapshot instead.
    xi_snap: tuple[int, ...] | None = (
        xi.snapshot_symbol_history() if isinstance(xi, PyAixiCTWBitMixture) else None
    )

    visits: dict[int, int] = defaultdict(int)
    sum_returns: dict[int, float] = defaultdict(float)

    n_sims = budget.mc_simulations
    h = budget.planning_horizon
    c = budget.uct_exploration_c

    for _ in range(n_sims):
        total = sum(visits[a] for a in actions)
        if total == 0:
            chosen = rng.choice(actions)
        else:
            best_a = actions[0]
            best_score = float("-inf")
            log_n = math.log(total + 1e-9)
            for a in actions:
                na = visits[a]
                mean = (sum_returns[a] / na) if na else 0.0
                bonus = c * math.sqrt(log_n / (na + 1e-9)) if na else c * math.sqrt(log_n + 1e-9)
                score = mean + bonus
                if score > best_score:
                    best_score = score
                    best_a = a
            chosen = best_a

        ret = imagined_trajectory_discounted_return(
            xi,
            first_action=chosen,
            encode_action=encode_action,
            n_action_bits=n_action_bits,
            n_percept_bits=n_percept_bits,
            decode_reward=decode_reward,
            valid_actions=actions,
            n_cycles=h,
            gamma=budget.discount_gamma,
            rng=rng,
            subsequent_action=None,
        )
        restore_mixture_after_imagination(
            xi,
            xi_snap=xi_snap,
            ref_log_p=ref_log_p,
            n_action_bits=n_action_bits,
            n_percept_bits=n_percept_bits,
            n_cycles=h,
        )

        visits[chosen] += 1
        sum_returns[chosen] += ret

    # Pick most visited (tie-break toward higher mean, then lower action id).
    best = max(
        actions,
        key=lambda a: (visits[a], sum_returns[a] / visits[a] if visits[a] else 0.0, -a),
    )
    return int(best)
