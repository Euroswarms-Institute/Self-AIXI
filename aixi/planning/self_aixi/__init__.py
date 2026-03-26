"""Family B — Self-AIXI v0: finite policy mixture ω + ξ rollouts (IMPLEMENTATION_PLAN §4.2)."""

from aixi.aixi.planning.self_aixi.agent import SelfAIXIV0Agent, SoftDeterministicPolicy, UniformRandomPolicy
from aixi.aixi.planning.self_aixi.fa import (
    HistoryKeyedQFA,
    SoftmaxParamPolicy,
    fit_softmax_logits_to_policy,
    greedy_action_from_q,
)

__all__ = [
    "SelfAIXIV0Agent",
    "SoftDeterministicPolicy",
    "UniformRandomPolicy",
    "HistoryKeyedQFA",
    "SoftmaxParamPolicy",
    "fit_softmax_logits_to_policy",
    "greedy_action_from_q",
]
