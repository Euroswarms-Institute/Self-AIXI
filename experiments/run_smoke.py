"""Smoke-check local AIXI paths: Phase 0 coin-flip parity and a short MCTS rollout."""

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import Sequence, Tuple


def _run_parity(*, seed: int, cycles: int) -> Tuple[bool, str]:
    from aixi.aixi.parity.coin_flip import run_coin_flip_symbol_parity

    rows = run_coin_flip_symbol_parity(seed=seed, cycles=cycles)
    for i, (agent_lp, manual_lp) in enumerate(rows):
        if not math.isclose(agent_lp, manual_lp, rel_tol=0.0, abs_tol=1e-9):
            return False, f"parity row {i}: agent={agent_lp} manual={manual_lp}"
    return True, f"parity ok ({len(rows)} rows, seed={seed})"


def _run_mcts_short_episode(*, seed: int, horizon: int) -> Tuple[bool, str]:
    from pyaixi.agents.mc_aixi_ctw import MC_AIXI_CTW_Agent
    from pyaixi.environments.coin_flip import CoinFlip
    from pyaixi.prediction import ctw_context_tree

    from aixi.aixi.models.ctw_pyaixi import PyAixiCTWBitMixture
    from aixi.aixi.parity.coin_flip import coin_flip_options_toy
    from aixi.aixi.planning.mcts import MCTSSearchBudget, root_uct_action

    erng = random.Random(seed)
    env = CoinFlip({"coin-flip-p": 0.55})
    agent = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
    tree = ctw_context_tree.CTWContextTree(depth=10)
    xi = PyAixiCTWBitMixture(tree)
    xi.learn_symbols(agent.encode_percept(env.observation, env.reward))

    budget = MCTSSearchBudget(mc_simulations=8, planning_horizon=2, uct_exploration_c=1.2)

    def decode_reward(symbols: list[int] | tuple[int, ...]) -> float:
        _obs, rew = agent.decode_percept(list(symbols))
        return float(rew)

    total_reward = 0.0
    for _ in range(horizon):
        ref = xi.root_log_probability()
        action = root_uct_action(
            xi,
            budget,
            valid_actions=env.valid_actions,
            encode_action=agent.encode_action,
            n_action_bits=env.action_bits(),
            n_percept_bits=env.percept_bits(),
            decode_reward=decode_reward,
            rng=erng,
        )
        if not math.isclose(xi.root_log_probability(), ref, rel_tol=0.0, abs_tol=1e-8):
            return False, "mcts: ξ root log_p changed after root_uct_action (revert contract)"
        xi.append_history_symbols(agent.encode_action(action))
        env.perform_action(action)
        xi.learn_symbols(agent.encode_percept(env.observation, env.reward))
        total_reward += float(env.reward)

    return True, f"mcts ok ({horizon} steps, seed={seed}, return_sum={total_reward:.4f})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run quick local checks: CoinFlip symbol parity (pyaixi) and/or short MCTS episode.",
    )
    parser.add_argument("--parity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mcts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for both checks.")
    parser.add_argument(
        "--parity-cycles",
        type=int,
        default=20,
        metavar="N",
        help="CoinFlip interaction cycles for parity (after initial percept).",
    )
    parser.add_argument(
        "--mcts-horizon",
        type=int,
        default=6,
        metavar="H",
        help="Real environment steps in the MCTS smoke episode.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.parity and not args.mcts:
        parser.error("enable at least one of --parity or --mcts")

    ok_all = True
    lines: list[str] = []

    if args.parity:
        try:
            p_ok, p_msg = _run_parity(seed=args.seed, cycles=args.parity_cycles)
        except ImportError as e:
            p_ok, p_msg = False, f"parity skipped/failed: {e}"
        lines.append(("PASS" if p_ok else "FAIL") + " " + p_msg)
        ok_all = ok_all and p_ok

    if args.mcts:
        try:
            m_ok, m_msg = _run_mcts_short_episode(seed=args.seed, horizon=args.mcts_horizon)
        except ImportError as e:
            m_ok, m_msg = False, f"mcts skipped/failed: {e}"
        lines.append(("PASS" if m_ok else "FAIL") + " " + m_msg)
        ok_all = ok_all and m_ok

    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
