"""Single-qubit CPTP step + projective measurement → classical ``Percept`` (analysis `06`)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aixi.aixi.env.protocol import FiniteSpace, Percept

_ID2 = np.eye(2, dtype=np.complex128)
_P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def _depolarize(rho: np.ndarray, p: float) -> np.ndarray:
    """ρ ↦ (1−p)ρ + p·I/2 (CPTP)."""
    if p < 0.0 or p > 1.0:
        raise ValueError("depolarizing parameter p must be in [0, 1]")
    return (1.0 - p) * rho + p * 0.5 * _ID2


def _apply_pauli_x(rho: np.ndarray) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return x @ rho @ x.conj().T


@dataclass
class SingleQubitMeasureEnv:
    """
    One qubit, density matrix ``rho``; each step applies a CPTP map then measures Z.

    * Action 0: light depolarizing noise ``p0``.
    * Action 1: stronger depolarizing noise ``p1``.
    * Classical percept: measurement outcome bit; reward is float(bit).

    This is a **toy** for CPTP + Born probabilities, not a universal quantum agent.
    """

    p0: float = 0.05
    p1: float = 0.35
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    action_space: FiniteSpace = field(default_factory=lambda: FiniteSpace(2))
    observation_space: FiniteSpace = field(default_factory=lambda: FiniteSpace(2))
    reward_range: tuple[float, float] = (0.0, 1.0)
    _rho: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    def reset(self) -> Percept:
        self._rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        return Percept(0, 0.0)

    def step(self, action: int) -> Percept:
        if action < 0 or action >= self.action_space.size:
            raise ValueError(f"action {action} out of range")
        p = self.p0 if action == 0 else self.p1
        rho = _depolarize(self._rho, p)
        rho = _apply_pauli_x(rho)  # unitary CPTP — keeps trace and Hermiticity
        p0 = float(np.real(np.trace(_P0 @ rho)))
        p0 = min(1.0, max(0.0, p0))
        bit = 0 if self.rng.random() < p0 else 1
        if bit == 0:
            self._rho = (_P0 @ rho @ _P0) / max(p0, 1e-15)
        else:
            p1 = 1.0 - p0
            self._rho = (_P1 @ rho @ _P1) / max(p1, 1e-15)
        return Percept(observation=bit, reward=float(bit))
