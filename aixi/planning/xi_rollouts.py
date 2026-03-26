"""
Shared ξ (MixtureEnvModel) imagination paths for Family A (MCTS) and Family B (Self-AIXI).

Both planners must call the same predict / append / learn / revert sequence so rollouts
stay aligned with IMPLEMENTATION_PLAN §6 Phase 1.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aixi.aixi.models.mixture import MixtureEnvModel

from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture


def sample_next_bit(xi: MixtureEnvModel, rng: Random) -> int:
    p1 = float(xi.predict_bit_probability(1))
    p1 = min(1.0, max(0.0, p1))
    return 1 if rng.random() < p1 else 0


def sample_percept_and_learn(
    xi: MixtureEnvModel,
    n_percept_bits: int,
    rng: Random,
) -> list[int]:
    symbols: list[int] = []
    for _ in range(n_percept_bits):
        b = sample_next_bit(xi, rng)
        symbols.append(b)
        xi.learn_symbols([b])
    return symbols


def revert_imagined_trajectory(
    xi: MixtureEnvModel,
    *,
    n_action_bits: int,
    n_percept_bits: int,
    n_cycles: int,
) -> None:
    """Undo ``n_cycles`` (action → percept) pairs from the model (MC-AIXI undo order)."""
    for _ in range(n_cycles):
        xi.revert_learned_symbols(n_percept_bits)
        xi.revert_history_symbols(n_action_bits)


def imagined_trajectory_discounted_return(
    xi: MixtureEnvModel,
    *,
    first_action: int,
    encode_action: Callable[[int], Sequence[int]],
    n_action_bits: int,
    n_percept_bits: int,
    decode_reward: Callable[[Sequence[int]], float],
    valid_actions: Sequence[int],
    n_cycles: int,
    gamma: float,
    rng: Random,
    subsequent_action: Callable[[Random, tuple[int, ...]], int] | None = None,
) -> float:
    """
    ``n_cycles`` imagined (action → percept) steps. The first uses ``first_action``;
    later steps use ``subsequent_action(rng, valid_actions_tuple)`` when provided,
    otherwise uniform random over ``valid_actions``.

    Mutates ``xi``; caller must ``restore_mixture_after_imagination`` (or equivalent).
    """
    actions = tuple(valid_actions)
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if not actions:
        raise ValueError("valid_actions must be non-empty")

    def pick_next(r: Random) -> int:
        if subsequent_action is not None:
            return int(subsequent_action(r, actions))
        return int(r.choice(actions))

    g = 0.0
    discount = 1.0

    a_syms = list(encode_action(first_action))
    if len(a_syms) != n_action_bits:
        raise ValueError(
            f"encode_action({first_action!r}) length must equal n_action_bits={n_action_bits}"
        )
    xi.append_history_symbols(a_syms)
    percept_syms = sample_percept_and_learn(xi, n_percept_bits, rng)
    r = float(decode_reward(percept_syms))
    g += discount * r
    discount *= gamma

    for _ in range(n_cycles - 1):
        a = pick_next(rng)
        a_syms = list(encode_action(a))
        if len(a_syms) != n_action_bits:
            raise ValueError(
                f"encode_action({a!r}) length must equal n_action_bits={n_action_bits}"
            )
        xi.append_history_symbols(a_syms)
        percept_syms = sample_percept_and_learn(xi, n_percept_bits, rng)
        r = float(decode_reward(percept_syms))
        g += discount * r
        discount *= gamma

    return g


def restore_mixture_after_imagination(
    xi: MixtureEnvModel,
    *,
    xi_snap: tuple[int, ...] | None,
    ref_log_p: float,
    n_action_bits: int,
    n_percept_bits: int,
    n_cycles: int,
) -> None:
    """Restore ξ after ``imagined_trajectory_discounted_return``; assert log P unchanged."""
    if xi_snap is not None:
        assert isinstance(xi, PyAixiCTWBitMixture)
        xi.replay_symbol_history(
            xi_snap,
            n_action_bits=n_action_bits,
            n_percept_bits=n_percept_bits,
        )
    else:
        revert_imagined_trajectory(
            xi,
            n_action_bits=n_action_bits,
            n_percept_bits=n_percept_bits,
            n_cycles=n_cycles,
        )
    if not math.isclose(xi.root_log_probability(), ref_log_p, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("ξ root log P drifted after imagination revert")
