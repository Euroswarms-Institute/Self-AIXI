"""Phase 2: compare Self-AIXI v0 to MC-AIXI on the same ξ (shared CTW)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
from aixi.aixi.planning.self_aixi.agent import SelfAIXIV0Agent


@dataclass(frozen=True)
class McVersusSelfSnapshot:
    """One decision-time snapshot: MC-AIXI search vs Self-AIXI greedy on Q_{ωξ}."""

    action_mc_aixi: int
    action_self_aixi: int
    q_zeta_xi_by_action: dict[int, float]

    @property
    def actions_match(self) -> bool:
        return self.action_mc_aixi == self.action_self_aixi

    @property
    def q_gap_argmax_vs_mc_choice(self) -> float:
        """How much higher is max_a Q_{ωξ}(a) than Q_{ωξ}(a_mc) (Self-AIXI's own estimates)."""
        q = self.q_zeta_xi_by_action
        return max(q.values()) - q[self.action_mc_aixi]


def snapshot_mc_vs_self_identical_xi(*, mc: Any, self_agent: SelfAIXIV0Agent) -> McVersusSelfSnapshot:
    """
    Run ``MC_AIXI_CTW_Agent.search()`` then ``SelfAIXIV0Agent.act()`` on the same mixture state.

    For ``PyAixiCTWBitMixture``, replays symbol history after ``search()`` (pyaixi undo quirk),
    then checks ``SelfAIXIV0Agent.act()`` restores ξ via the usual imagination path.
    """
    xi = self_agent.xi
    ref_lp = xi.root_log_probability()
    hist_snap = (
        xi.snapshot_symbol_history() if isinstance(xi, PyAixiCTWBitMixture) else None
    )
    a_mc = int(mc.search())
    # pyaixi search undo can leave CTW weights inconsistent with history (SWA-26); replay repairs.
    if isinstance(xi, PyAixiCTWBitMixture) and hist_snap is not None:
        xi.replay_symbol_history(
            hist_snap,
            n_action_bits=self_agent.n_action_bits,
            n_percept_bits=self_agent.n_percept_bits,
        )
    if not math.isclose(xi.root_log_probability(), ref_lp, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("ξ root log P not restored after MC-AIXI search() + replay")
    a_self = int(self_agent.act())
    if not math.isclose(xi.root_log_probability(), ref_lp, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("ξ root log P changed after Self-AIXI act()")
    return McVersusSelfSnapshot(
        action_mc_aixi=a_mc,
        action_self_aixi=a_self,
        q_zeta_xi_by_action=dict(self_agent.last_q_by_action),
    )
