//! Environment models — the ξ side of MC-AIXI.
//!
//! `EnvModel` is the Rust rendering of the repo's `MixtureEnvModel` Protocol
//! (`aixi/models/mixture.py`, IMPLEMENTATION_PLAN.md §3): a sequential
//! probability model over the interleaved action/percept **bit** stream with
//! the FAC action-conditional split — percept bits are *learned* (they update
//! the model and carry probability), action bits are only *appended* (they
//! condition all later predictions but are never predicted themselves,
//! JAIR §5.3).
//!
//! Contract (the load-bearing part, from `aixi/planning/xi_rollouts.py`):
//! `learn`/`append` during imagined rollouts must be undone by
//! `revert_learned_symbols`/`revert_history_symbols` in strict LIFO order,
//! after which `root_log_probability()` must be restored — the Python
//! contract tolerates 1e-8 drift, the implementations here restore state
//! **bit-exactly** by recording previous values rather than replaying deltas.

pub mod ctw;
pub mod fac_ctw;
pub mod kt;
pub mod mixture;
pub mod uniform;

/// A revertible, budgeted sequential model over bits (§1 naming ledger:
/// realizes `MixtureEnvModel`; every method terminates in bounded time, §1.1).
pub trait EnvModel {
    /// ln ρ(learned bits so far ‖ appended context) — the model's running
    /// log-marginal of everything it was asked to learn.
    fn root_log_probability(&self) -> f64;

    /// P(next bit = `bit` | full history). Takes `&mut self` because some
    /// models (the LLM) lazily advance internal state, but must be
    /// observationally pure: calling it must not change any probability.
    fn predict_bit_probability(&mut self, bit: u8) -> f64;

    /// Percept path: condition on and learn each bit in order.
    fn learn_symbols(&mut self, bits: &[u8]);

    /// Action path: condition on each bit without learning (FAC split).
    fn append_history_symbols(&mut self, bits: &[u8]);

    /// Undo the most recent `n` learned bits (strict LIFO w.r.t. the
    /// interleaving of learn/append operations).
    fn revert_learned_symbols(&mut self, n: usize);

    /// Undo the most recent `n` appended bits (strict LIFO).
    fn revert_history_symbols(&mut self, n: usize);

    fn model_id(&self) -> String;
}

impl EnvModel for Box<dyn EnvModel> {
    fn root_log_probability(&self) -> f64 {
        (**self).root_log_probability()
    }
    fn predict_bit_probability(&mut self, bit: u8) -> f64 {
        (**self).predict_bit_probability(bit)
    }
    fn learn_symbols(&mut self, bits: &[u8]) {
        (**self).learn_symbols(bits)
    }
    fn append_history_symbols(&mut self, bits: &[u8]) {
        (**self).append_history_symbols(bits)
    }
    fn revert_learned_symbols(&mut self, n: usize) {
        (**self).revert_learned_symbols(n)
    }
    fn revert_history_symbols(&mut self, n: usize) {
        (**self).revert_history_symbols(n)
    }
    fn model_id(&self) -> String {
        (**self).model_id()
    }
}
