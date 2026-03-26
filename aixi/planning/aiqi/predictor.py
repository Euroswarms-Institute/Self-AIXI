"""Return-predictor interface (ψ / mixture placeholder; no grain-of-truth claims)."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Protocol, Sequence, Tuple, runtime_checkable


def grid_bin_value(bin_index: int, grid_bins_M: int) -> float:
    """Return value z ∈ {0, 1/M, …, (M-1)/M} for ``bin_index`` in ``0 .. M-1``."""

    if grid_bins_M < 1:
        raise ValueError("grid_bins_M must be >= 1")
    if not 0 <= bin_index < grid_bins_M:
        raise ValueError("bin_index out of range")
    return bin_index / grid_bins_M


def mean_return_from_probs(probs: Sequence[float], grid_bins_M: int) -> float:
    """Posterior mean Σ_k (k/M) · p_k (IMPLEMENTATION_PLAN: Q̂ from ψ)."""

    if len(probs) != grid_bins_M:
        raise ValueError("probs length must equal grid_bins_M")
    s = sum(probs)
    if not s > 0:
        return 0.0
    norm = [float(p) / s for p in probs]
    return sum(grid_bin_value(k, grid_bins_M) * norm[k] for k in range(grid_bins_M))


def total_return_to_bin(
    total_return: float,
    *,
    horizon_H: int,
    grid_bins_M: int,
    reward_min: float,
    reward_max: float,
) -> int:
    """Map an (unnormalized) H-step return sum onto ``0 .. grid_bins_M-1``."""

    if reward_max <= reward_min:
        raise ValueError("reward_max must exceed reward_min")
    span = (reward_max - reward_min) * horizon_H
    if span <= 0:
        return 0
    x = (total_return - horizon_H * reward_min) / span
    x = max(0.0, min(1.0, x))
    idx = int(x * grid_bins_M)
    if idx >= grid_bins_M:
        idx = grid_bins_M - 1
    return idx


@runtime_checkable
class ReturnPredictor(Protocol):
    """Pluggable ψ: predicts a distribution over discretized H-step returns."""

    def predicted_return_probs(
        self,
        *,
        timestep: int,
        history_len: int,
        observation: int,
        action: int,
        grid_bins_M: int,
        phase_n: int,
    ) -> Sequence[float]: ...


class UniformReturnPredictor:
    """Placeholder ψ: uniform over return bins (skeleton only)."""

    def predicted_return_probs(
        self,
        *,
        timestep: int,
        history_len: int,
        observation: int,
        action: int,
        grid_bins_M: int,
        phase_n: int,
    ) -> list[float]:
        _ = (timestep, history_len, observation, action, phase_n)
        return [1.0 / grid_bins_M] * grid_bins_M


ContextKey = Tuple[int, int]


class TabularPhaseReturnPredictor:
    """On-policy tabular ψ per augmentation phase (heuristic; not a grain-of-truth construction)."""

    def __init__(self, *, pseudocount: float = 1.0) -> None:
        if pseudocount < 0.0:
            raise ValueError("pseudocount must be non-negative")
        self._pseudocount = float(pseudocount)
        self._counts: DefaultDict[int, DefaultDict[ContextKey, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [])
        )

    def predicted_return_probs(
        self,
        *,
        timestep: int,
        history_len: int,
        observation: int,
        action: int,
        grid_bins_M: int,
        phase_n: int,
    ) -> list[float]:
        _ = (timestep, history_len)
        key = (observation, action)
        row = self._counts[phase_n].get(key)
        if row is None or len(row) != grid_bins_M:
            base = self._pseudocount
            return [base] * grid_bins_M
        return [self._pseudocount + float(c) for c in row]

    def observe_augmented_return(
        self,
        *,
        phase_n: int,
        observation: int,
        action: int,
        return_bin: int,
        grid_bins_M: int,
    ) -> None:
        key = (observation, action)
        row = self._counts[phase_n][key]
        if len(row) != grid_bins_M:
            row.clear()
            row.extend([0.0] * grid_bins_M)
        if not 0 <= return_bin < grid_bins_M:
            raise ValueError("return_bin out of range")
        row[return_bin] += 1.0
