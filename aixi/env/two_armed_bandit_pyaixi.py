"""Two-armed Bernoulli bandit as a ``pyaixi.environment.Environment`` (1 action bit, 2 percept bits)."""

from __future__ import annotations

import random

from pyaixi import environment, util

bandit_action_enum = util.enum("aArm0", "aArm1")
bandit_observation_enum = util.enum("oLow", "oHigh")
bandit_reward_enum = util.enum("rLose", "rWin")

aArm0 = bandit_action_enum.aArm0
aArm1 = bandit_action_enum.aArm1
oLow = bandit_observation_enum.oLow
oHigh = bandit_observation_enum.oHigh
rLose = bandit_reward_enum.rLose
rWin = bandit_reward_enum.rWin


class TwoArmedBandit(environment.Environment):
    """Pull one of two arms; each draw is high with arm-specific probability.

    Bit layout matches ``CoinFlip`` (``action_bits == 1``, ``percept_bits == 2``) so
    MC-AIXI + ξ snapshot replay stay aligned.
    """

    def __init__(self, options: dict | None = None) -> None:
        options = dict(options or {})
        environment.Environment.__init__(self, options=options)

        self.valid_actions = list(bandit_action_enum.keys())
        self.valid_observations = list(bandit_observation_enum.keys())
        self.valid_rewards = list(bandit_reward_enum.keys())

        self.p_high_arm0 = float(options.get("arm0-p-high", 0.28))
        self.p_high_arm1 = float(options.get("arm1-p-high", 0.82))
        assert 0.0 <= self.p_high_arm0 <= 1.0
        assert 0.0 <= self.p_high_arm1 <= 1.0

        self.observation = oHigh if random.random() < self.p_high_arm0 else oLow
        self.reward = 0

    def perform_action(self, action: int) -> tuple[int, int]:
        assert self.is_valid_action(action)

        self.action = action
        p = self.p_high_arm0 if action == aArm0 else self.p_high_arm1
        if random.random() < p:
            self.observation = oHigh
            self.reward = rWin
        else:
            self.observation = oLow
            self.reward = rLose

        return (self.observation, self.reward)

    def print(self) -> str:
        return (
            f"arm={self.action}, observation={'high' if self.observation == oHigh else 'low'}, "
            f"reward={self.reward}"
        )
