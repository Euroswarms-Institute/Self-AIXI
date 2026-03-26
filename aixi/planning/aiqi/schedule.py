"""Periodic return augmentation schedule (AIQI: insert z only when i ≡ n (mod N)).

Cycle-slot semantics ``t mod N`` match :func:`aixi.formal_ci.augmentation_cycle_slot`;
coarse block index ``t // N`` is :func:`aixi.formal_ci.augmentation_epoch_index`.
"""

from __future__ import annotations


def augmentation_insert_at_timestep(timestep: int, phase_n: int, period_N: int) -> bool:
    """Return True iff a discretized return symbol may be recorded for phase ``phase_n`` at time ``timestep``.

    Matches the construction in arXiv:2602.23242 — past augmented returns only use rewards already observed.
    """

    if period_N <= 0:
        raise ValueError("period_N must be positive")
    return (timestep % period_N) == (phase_n % period_N)
