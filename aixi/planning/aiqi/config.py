"""AIQI configuration (IMPLEMENTATION_PLAN §4.3, §3 table)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIQIConfig:
    """Finite hyperparameters for return truncation grid and periodic augmentation."""

    horizon_H: int
    grid_bins_M: int
    augmentation_period_N: int
    exploration_tau: float

    def __post_init__(self) -> None:
        if self.horizon_H < 1:
            raise ValueError("horizon_H must be >= 1")
        if self.grid_bins_M < 1:
            raise ValueError("grid_bins_M must be >= 1")
        if self.augmentation_period_N < self.horizon_H:
            raise ValueError("augmentation_period_N must be >= horizon_H (paper construction)")
        if not 0.0 <= self.exploration_tau <= 1.0:
            raise ValueError("exploration_tau must be in [0, 1]")
