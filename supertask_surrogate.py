"""
Finite **supertask surrogate** (SWA-25): bounded schedules only.

This is not a supertask solver. It encodes the engineering contract: explicit
step envelopes, halting transitions, and no APIs that imply completion of an
actual infinite schedule. Aligns with blueprint “shipped v1” and formal CI
P7-style wall-clock/step budgets (``aixi.formal_ci``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Final, Literal, Optional

Phase = Literal["running", "halted"]


class FiniteBudgetExceeded(RuntimeError):
    """Raised when a caller requests work outside the declared schedule envelope."""


# Sentinel: modules must not expose “run forever” / hypercomputer-style entry points.
_HYPERCOMPUTER_FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "run_unbounded",
        "complete_supertask",
        "execute_transfinite_schedule",
        "hypercompute",
        "solve_halting_problem",
    }
)


def assert_no_hypercomputer_api_on_module(module: object) -> None:
    """
    Guardrail for tests: refuse public callables whose names imply unbounded
    or impossible completion. Intentionally shallow (name-based).
    """
    for name in dir(module):
        if name.startswith("_"):
            continue
        if name in _HYPERCOMPUTER_FORBIDDEN_NAMES:
            raise AssertionError(f"forbidden hypercomputer-style API: {name!r}")
        attr = getattr(module, name, None)
        if callable(attr) and name in _HYPERCOMPUTER_FORBIDDEN_NAMES:
            raise AssertionError(f"forbidden hypercomputer-style API: {name!r}")


@dataclass(frozen=True)
class ScheduleEnvelope:
    """
    Finite cap on all work: macro rounds × micro-steps per macro (+ optional slack).

    ``max_macro_steps`` is the truncated “depth” of a Zeno-style story;
    ``micro_steps_per_macro`` is explicit sub-work per round. Total micro-steps
    allowed is ``max_macro_steps * micro_steps_per_macro`` (see ``max_micro_steps``).
    """

    max_macro_steps: int
    micro_steps_per_macro: int

    def __post_init__(self) -> None:
        if self.max_macro_steps < 0:
            raise ValueError("max_macro_steps must be non-negative")
        if self.micro_steps_per_macro < 1:
            raise ValueError("micro_steps_per_macro must be >= 1")

    @property
    def max_micro_steps(self) -> int:
        return self.max_macro_steps * self.micro_steps_per_macro


@dataclass
class SupertaskSurrogateState:
    """Mutable runner state; transitions always advance ``micro_steps_used`` by at most one per ``step_micro``."""

    envelope: ScheduleEnvelope
    macro_index: int = 0
    micro_within_macro: int = 0
    micro_steps_used: int = 0
    phase: Phase = "running"
    lamp_on: bool = False

    def halted(self) -> bool:
        return self.phase == "halted"

    def step_micro(self) -> None:
        """
        One finite micro-transition. Halts when the envelope is exhausted.
        Never schedules “remaining infinitely many” substeps.
        """
        if self.phase == "halted":
            return
        if self.micro_steps_used >= self.envelope.max_micro_steps:
            self.phase = "halted"
            return

        self.lamp_on = not self.lamp_on
        self.micro_within_macro += 1
        self.micro_steps_used += 1

        if self.micro_within_macro >= self.envelope.micro_steps_per_macro:
            self.micro_within_macro = 0
            self.macro_index += 1

        if self.micro_steps_used >= self.envelope.max_micro_steps:
            self.phase = "halted"


def run_bounded(
    state: SupertaskSurrogateState,
    *,
    max_steps: int,
    wall_clock_budget_s: Optional[float] = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """
    Run at most ``max_steps`` micro-steps. Returns steps actually executed this call.

    Raises ``FiniteBudgetExceeded`` if ``max_steps`` exceeds the envelope’s remaining
    capacity (strict surrogate: no silent extension of the contract).
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    remaining_envelope = state.envelope.max_micro_steps - state.micro_steps_used
    if state.phase == "halted":
        remaining_envelope = 0

    if max_steps > remaining_envelope:
        raise FiniteBudgetExceeded(
            f"requested {max_steps} steps but only {remaining_envelope} remain in envelope"
        )

    t0 = clock()
    executed = 0
    while executed < max_steps and state.phase == "running":
        state.step_micro()
        executed += 1
        if wall_clock_budget_s is not None:
            if clock() - t0 > wall_clock_budget_s:
                raise AssertionError(
                    f"wall-clock budget exceeded during run_bounded ({wall_clock_budget_s}s)"
                )
    return executed


def drain_envelope(state: SupertaskSurrogateState) -> int:
    """Run until halted; returns total micro-steps executed in this call."""
    if state.halted():
        return 0
    remaining = state.envelope.max_micro_steps - state.micro_steps_used
    if remaining == 0:
        state.phase = "halted"
        return 0
    return run_bounded(state, max_steps=remaining)
