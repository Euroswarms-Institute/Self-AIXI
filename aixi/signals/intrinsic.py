"""Utilities aligned with variational empowerment / FEP narrative (arXiv 2502.15820).

**Safety (plan §8):** intrinsic terms can encourage power-seeking and optionality-seeking.
All hooks default to **off** at call sites; enable only after explicit scope + review.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence


def softmax_probs(
    values_by_action: Mapping[int, float],
    valid_actions: tuple[int, ...],
    *,
    temperature: float = 1.0,
) -> dict[int, float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not valid_actions:
        raise ValueError("valid_actions must be non-empty")
    mx = max(float(values_by_action[int(a)]) for a in valid_actions)
    w = {int(a): math.exp((float(values_by_action[int(a)]) - mx) / temperature) for a in valid_actions}
    s = sum(w.values())
    if s <= 0.0:
        u = 1.0 / len(valid_actions)
        return {int(a): u for a in valid_actions}
    return {int(a): w[int(a)] / s for a in valid_actions}


def discrete_kl(p: Mapping[int, float], q: Mapping[int, float], valid_actions: tuple[int, ...], *, eps: float = 1e-12) -> float:
    """D_KL(p || q) over ``valid_actions`` (natural logarithm)."""
    tot = 0.0
    for a in valid_actions:
        ia = int(a)
        pi = max(float(p[ia]), eps)
        qi = max(float(q[ia]), eps)
        tot += pi * math.log(pi / qi)
    return tot


def free_energy_decomposition_placeholder(surprise: float, kl_term: float) -> float:
    """
    Placeholder sum ``surprise + kl_term`` for FEP-style bookkeeping (paper Eq.~(26) area).

    Not a certified bound here — callers supply components from their model.
    """
    return float(surprise) + float(kl_term)


def mixture_zeta_from_policies(
    omega: Sequence[float],
    policy_prob: Callable[[int, int, tuple[int, ...]], float],
    valid_actions: tuple[int, ...],
) -> dict[int, float]:
    """
    ζ(a) ∝ Σ_k ω_k π_k(a | valid) for finite policy mixtures (Self-AIXI ω loop).
    ``policy_prob(k, a, valid_actions)`` returns π_k(a).
    """
    zeta: dict[int, float] = {int(a): 0.0 for a in valid_actions}
    for k, w in enumerate(omega):
        if w <= 0.0:
            continue
        for a in valid_actions:
            ia = int(a)
            zeta[ia] += float(w) * max(float(policy_prob(k, ia, valid_actions)), 0.0)
    s = sum(zeta.values())
    if s <= 0.0:
        u = 1.0 / len(valid_actions)
        return {int(a): u for a in valid_actions}
    return {int(a): zeta[int(a)] / s for a in valid_actions}


def adjust_q_by_log_ratio(
    q_by_action: dict[int, float],
    valid_actions: tuple[int, ...],
    *,
    pi_star: Mapping[int, float],
    zeta: Mapping[int, float],
    lam: float,
    eps: float = 1e-12,
) -> None:
    """
    In-place: ``q(a) -= lam * log(pi*(a) / zeta(a))`` (Hayashi–Takahashi KL(π*||ζ) local shaping).

    When ``lam == 0``, no-op. Intended for **research** toggles on small envs only.
    """
    if lam == 0.0:
        return
    for a in valid_actions:
        ia = int(a)
        num = max(float(pi_star[ia]), eps)
        den = max(float(zeta[ia]), eps)
        q_by_action[ia] -= lam * math.log(num / den)


def regularize_mixture_q_values(
    q_by_action: dict[int, float],
    valid_actions: tuple[int, ...],
    *,
    omega: Sequence[float],
    policy_prob: Callable[[int, int, tuple[int, ...]], float],
    lam: float,
    temperature: float = 1.0,
) -> None:
    """
    Apply ``adjust_q_by_log_ratio`` using π* = softmax(Q) and mixture ζ from ``omega`` / policies.

    **Default:** call with ``lam=0.0`` at integration sites until an explicit safety review.
    """
    if lam == 0.0:
        return
    zeta = mixture_zeta_from_policies(omega, policy_prob, valid_actions)
    pi_star = softmax_probs(q_by_action, valid_actions, temperature=temperature)
    adjust_q_by_log_ratio(q_by_action, valid_actions, pi_star=pi_star, zeta=zeta, lam=lam)
