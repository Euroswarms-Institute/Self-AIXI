"""Phase 0: symbol-level parity vs ``MC_AIXI_CTW_Agent`` on ``CoinFlip``."""

from __future__ import annotations

import random
from typing import Any, List, Tuple

try:
    from pyaixi.agent import action_update, percept_update
    from pyaixi.agents.mc_aixi_ctw import MC_AIXI_CTW_Agent
    from pyaixi.environments.coin_flip import CoinFlip
    from pyaixi.prediction import ctw_context_tree

    _PYAIXI = True
except ImportError:
    _PYAIXI = False


def coin_flip_options_toy() -> dict[str, Any]:
    """Minimal MC-AIXI-CTW options suitable for fast tests (one toy conf)."""
    return {
        "agent-horizon": 4,
        "ct-depth": 8,
        "mc-simulations": 2,
        "learning-period": 0,
    }


def run_coin_flip_symbol_parity(
    *,
    seed: int,
    cycles: int,
) -> List[Tuple[float, float]]:
    """Return rows ``(agent_root_log_p, manual_root_log_p)`` after each percept update.

    Raises
    ------
    ImportError
        If ``pyaixi`` is not installed.
    """
    if not _PYAIXI:
        raise ImportError("pyaixi is required for parity runs; install optional extra `crca[aixi]`.")

    random.seed(seed)

    options = {"coin-flip-p": 0.6}
    env = CoinFlip(options)
    agent = MC_AIXI_CTW_Agent(environment=env, options=coin_flip_options_toy())
    agent.reset()

    depth = agent.depth
    manual = ctw_context_tree.CTWContextTree(depth)

    rows: List[Tuple[float, float]] = []

    assert agent.last_update == action_update
    agent.model_update_percept(env.observation, env.reward)
    manual.update(agent.encode_percept(env.observation, env.reward))
    rows.append((agent.context_tree.root.log_probability, manual.root.log_probability))

    for _ in range(cycles):
        assert agent.last_update == percept_update
        action = random.choice(env.valid_actions)
        agent.model_update_action(action)
        manual.update_history(agent.encode_action(action))

        env.perform_action(action)
        agent.model_update_percept(env.observation, env.reward)
        manual.update(agent.encode_percept(env.observation, env.reward))

        rows.append((agent.context_tree.root.log_probability, manual.root.log_probability))

    return rows
