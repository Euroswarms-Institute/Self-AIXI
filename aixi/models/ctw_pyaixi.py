from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from pyaixi.prediction.ctw_context_tree import CTWContextTree


class PyAixiCTWBitMixture:
    """Thin ξ adapter over pyaixi's ``CTWContextTree`` (Phase 1 scaffold)."""

    def __init__(self, context_tree: CTWContextTree, *, model_id: str = "pyaixi-ctw") -> None:
        self._tree = context_tree
        self._model_id = model_id

    @property
    def model_ids(self) -> tuple[str, ...]:
        return (self._model_id,)

    def root_log_probability(self) -> float:
        return self._tree.root.log_probability

    def predict_bit_probability(self, symbol: int) -> float:
        return self._tree.predict(int(symbol))

    def learn_symbols(self, symbols: Sequence[int]) -> None:
        self._tree.update(list(symbols))

    def append_history_symbols(self, symbols: Sequence[int]) -> None:
        self._tree.update_history(list(symbols))

    def revert_learned_symbols(self, n_symbols: int) -> None:
        self._tree.revert(int(n_symbols))

    def revert_history_symbols(self, n_symbols: int) -> None:
        self._tree.revert_history(int(n_symbols))

    def snapshot_symbol_history(self) -> tuple[int, ...]:
        """Immutable copy of the CTW bit stream (reward/observation + action symbols)."""
        return tuple(self._tree.history)

    def replay_symbol_history(
        self,
        symbols: Sequence[int],
        *,
        n_action_bits: int,
        n_percept_bits: int,
    ) -> None:
        """Rebuild the context tree from a flat stream: initial percept, then (action, percept)* pairs.

        Used after imagined rollouts: pyaixi's incremental ``revert`` can match ``history`` while
        leaving node weights / ``tree_size`` inconsistent when ``predict`` touched transient paths.
        """
        flat = list(symbols)
        self._tree.clear()
        i = 0
        if len(flat) < n_percept_bits:
            raise ValueError("symbol history shorter than one percept")
        self._tree.update(flat[i : i + n_percept_bits])
        i += n_percept_bits
        while i < len(flat):
            need = n_action_bits + n_percept_bits
            if i + need > len(flat):
                raise ValueError("incomplete action/percept tail in symbol history")
            self._tree.update_history(flat[i : i + n_action_bits])
            i += n_action_bits
            self._tree.update(flat[i : i + n_percept_bits])
            i += n_percept_bits
        if i != len(flat):
            raise ValueError("leftover symbols after replay")
