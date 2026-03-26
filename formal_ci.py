"""
WP-Formal-CI hooks aligned with SWA-21 plan §10 (P1–P7).

These are **contracts + toy backends** so CI can run before full MixtureEnvModel /
Self-AIXI / AIQI land. Real implementations should satisfy the same invariants.

References: ``aixi/IMPLEMENTATION_PLAN.md`` §1.1, §3; SWA-21 plan §4, §10.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

# --- P5: Family C — augmentation indices (pure functions of (t, N)) ---


def augmentation_epoch_index(step_index: int, period_n: int) -> int:
    """
    Index of the length-``period_n`` block containing ``step_index``: ``step_index // period_n``.

    Use this when you need a **coarse time bucket** (which “epoch” of length N).

    This is **not** the AIQI augmentation *cycle slot* ``t mod N``; for that see
    :func:`augmentation_cycle_slot` and ``aixi.aixi.planning.aiqi.schedule``.

    Invariant **P5:** same ``(step_index, period_n)`` always yields the same value; no hidden state.
    """
    if period_n <= 0:
        raise ValueError("period_n must be positive")
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    return step_index // period_n


def augmentation_phase_index(step_index: int, period_n: int) -> int:
    """Backward-compatible alias for :func:`augmentation_epoch_index` (older name; easily confused with AIQI slot)."""
    return augmentation_epoch_index(step_index, period_n)


def augmentation_cycle_slot(step_index: int, period_n: int) -> int:
    """
    Position within one augmentation period: ``step_index % period_n`` (in ``0 .. period_n - 1``).

    Aligns with AIQI (:meth:`aixi.aixi.planning.aiqi.agent.AIQISkeletonAgent.augmentation_cycle_slot`)
    and ``augmentation_insert_at_timestep`` modulo semantics. Distinct from
    :func:`augmentation_epoch_index` (``floor(t/N)``).
    """
    if period_n <= 0:
        raise ValueError("period_n must be positive")
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    return step_index % period_n


# --- P6: Family C — documented ε schedule (deterministic given step) ---


def epsilon_greedy_epsilon(
    step: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    decay_steps: int,
) -> float:
    """
    Linear ε decay from ``epsilon_start`` to ``epsilon_end`` over ``decay_steps``.

    After ``decay_steps - 1`` (inclusive), ε stays at ``epsilon_end``.
    Used as a **smoke** contract for P6; swap for cosine/exponential schedules
    behind the same function name once specified in docs.
    """
    if decay_steps < 1:
        raise ValueError("decay_steps must be >= 1")
    if not (0.0 <= epsilon_start <= 1.0 and 0.0 <= epsilon_end <= 1.0):
        raise ValueError("epsilon values must be in [0, 1]")
    if step < 0:
        raise ValueError("step must be non-negative")
    if step >= decay_steps:
        return float(epsilon_end)
    t = step / max(decay_steps - 1, 1)
    return float(epsilon_start + (epsilon_end - epsilon_start) * t)


# --- P3 / P4: Family B — finite policy posterior ω(π|h) ---


def omega_sum(omega: Mapping[int, float]) -> float:
    """Sum of mixture masses (linear domain, not log)."""
    return float(sum(omega.values()))


@dataclass
class ToyPolicyMixture:
    """
    Minimal finite-support policy posterior for **P3** (normalization) and **P4**
    (no update on counterfactual-only branches).

    Naming: use ``policy_posterior`` in product code where possible to avoid
    confusion with ordinal ω (supertask literature); see SWA-21 §5.
    """

    log_weights: MutableMapping[int, float] = field(default_factory=dict)

    def _normalized_probs(self) -> Dict[int, float]:
        if not self.log_weights:
            return {}
        m = max(self.log_weights.values())
        raw = {k: math.exp(v - m) for k, v in self.log_weights.items()}
        s = sum(raw.values()) or 1.0
        return {k: v / s for k, v in raw.items()}

    def omega(self) -> Dict[int, float]:
        """Finite-support posterior ω(π|h); sums to 1 when non-empty."""
        return self._normalized_probs()

    def update_from_observed_action(self, policy_id: int, learning_rate: float) -> None:
        """On-policy style nudge: only call for **executed** trajectories (P4)."""
        lw = float(self.log_weights.get(policy_id, 0.0))
        self.log_weights[policy_id] = lw + learning_rate

    def simulate_counterfactual_branch(self, policy_id: int, learning_rate: float) -> None:
        """
        Rollout-only hook: mutates a disposable clone, never the live posterior.

        **P4:** counterfactual search must not push ω updates for unrealized branches.
        """
        raise RuntimeError(
            "ToyPolicyMixture: use branch_counterfactual_clone() for search; "
            "never call update on branches that were not executed."
        )

    def branch_counterfactual_clone(self) -> ToyPolicyMixture:
        """Return a deep copy for MCTS / imagined branches."""
        other = ToyPolicyMixture(log_weights=dict(self.log_weights))
        return other


# --- P1 / P2 / P7: shared ξ toy — revert stack + bounded calls ---


@dataclass
class ToyRevertibleMixture:
    """
    Toy mixture with explicit **revert stack** (M3-style) for Families A/B.

    - **P1:** ``revert`` restores weights to pre-``update`` state.
    - **P2:** draining the stack returns to the snapshot taken at construction
      (pre-search baseline) if every ``update`` was paired with ``revert`` or
      fully unwound.
    """

    log_weights: MutableMapping[str, float] = field(default_factory=dict)
    _undo_stack: List[Dict[str, float]] = field(default_factory=list)
    _baseline: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self._baseline = copy.deepcopy(dict(self.log_weights))

    def predict(self, history_key: str, action: int) -> float:
        """Cheap surrogate for per-call budget tests (P7)."""
        _ = (history_key, action)
        return sum(math.exp(w) for w in self.log_weights.values()) or 0.0

    def update(self, percept: Any) -> None:
        _ = percept
        self._undo_stack.append(copy.deepcopy(dict(self.log_weights)))
        if not self.log_weights:
            self.log_weights["m0"] = 0.0
        # Nudge first model key (stable ordering) for deterministic tests.
        k = sorted(self.log_weights.keys())[0]
        self.log_weights[k] = float(self.log_weights[k]) + 0.01

    def revert(self) -> None:
        if not self._undo_stack:
            raise RuntimeError("revert stack empty")
        prev = self._undo_stack.pop()
        self.log_weights.clear()
        self.log_weights.update(prev)

    def revert_all(self) -> None:
        """Drain the revert stack (e.g. end of search)."""
        while self._undo_stack:
            self.revert()

    def baseline_snapshot(self) -> Dict[str, float]:
        """Pre-search weights for P2 comparison."""
        return copy.deepcopy(self._baseline)

    def current_weights_copy(self) -> Dict[str, float]:
        return copy.deepcopy(dict(self.log_weights))


def assert_predict_update_under_budget(
    model: ToyRevertibleMixture,
    budget_s: float,
    *,
    predict_calls: int = 3,
    update_calls: int = 2,
) -> None:
    """
    **P7 (smoke):** ``predict`` / ``update`` complete within wall-clock budget.

    Full stress tests against adversarial slow envs stay behind ``@pytest.mark.slow``.
    """
    t0 = time.monotonic()
    for _ in range(predict_calls):
        model.predict("h", 0)
    for _ in range(update_calls):
        model.update((0, 1.0))
    for _ in range(update_calls):
        model.revert()
    elapsed = time.monotonic() - t0
    if elapsed > budget_s:
        raise AssertionError(
            f"P7 budget exceeded: {elapsed:.4f}s > {budget_s}s "
            f"({predict_calls} predict, {update_calls} update/revert pairs)"
        )


def run_under_wall_clock(seconds: float, fn: Callable[[], None]) -> float:
    """Run ``fn``; return elapsed seconds; raises if over ``seconds`` (+ small slack)."""
    t0 = time.monotonic()
    fn()
    elapsed = time.monotonic() - t0
    slack = 0.1 + 0.05 * seconds
    if elapsed > seconds + slack:
        raise AssertionError(f"deadline exceeded: {elapsed:.4f}s > {seconds}s")
    return elapsed
