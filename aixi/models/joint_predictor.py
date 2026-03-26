"""Joint action–percept prediction hook (IMPLEMENTATION_PLAN §6 Phase 5, analysis `05` / MUPI spirit).

Full MUPI / RUI is not computable in this repo; this module only fixes an **interface**
for prospective models that emit probabilities over **paired** (action, percept) futures.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JointSequencePredictor(Protocol):
    """
    Embedded / joint predictor: conditions on history and scores **next** (action, percept) pairs.

    Multi-agent non-stationarity and infinite-order ToM are out of scope; see ``05`` analysis.
    """

    def log_prob_next_pair(self, history_key: object, action: int, percept_key: object) -> float:
        """Unnormalized is allowed if documented; return should be comparable across pairs at fixed history."""
        ...
