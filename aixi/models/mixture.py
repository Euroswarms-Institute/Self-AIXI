from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class MixtureEnvModel(Protocol):
    """Tractable ξ over the next binary symbol (pyaixi bit stream).

    Maps to IMPLEMENTATION_PLAN §3.1: ``predict`` / ``update`` / ``revert`` on
    the interaction stream. Concrete agents compose action + percept encodings
    before calling these methods.
    """

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Finite index set for the implemented mixture (constructive ξ)."""
        ...

    def root_log_probability(self) -> float:
        """log P_w(history) at the mixture root — parity / diagnostics."""
        ...

    def predict_bit_probability(self, symbol: int) -> float:
        """Conditional probability of the next bit ``symbol`` ∈ {0, 1} given history."""
        ...

    def learn_symbols(self, symbols: Sequence[int]) -> None:
        """Apply a full CTW ``update`` (percept path in MC-AIXI)."""
        ...

    def append_history_symbols(self, symbols: Sequence[int]) -> None:
        """Append symbols without leaf learning (action path: ``update_history``)."""
        ...

    def revert_learned_symbols(self, n_symbols: int) -> None:
        """Undo the last ``n_symbols`` learned updates (``CTWContextTree.revert``)."""
        ...

    def revert_history_symbols(self, n_symbols: int) -> None:
        """Shrink history without touching KT state (``revert_history``)."""
        ...
