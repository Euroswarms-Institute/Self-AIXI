//! Uniform (coin-toss) model: predicts ½ for every bit, forever.
//!
//! Serves as the mixture's noise floor — it lower-bounds ξ's per-bit
//! probability at w⁰·½, keeping the agent's surprise finite on any stream —
//! and as an analytically trivial component for mixture unit tests.

use super::EnvModel;
use crate::logspace::LOG_HALF;

#[derive(Default)]
pub struct UniformModel {
    /// LIFO record of operations: true = learned bit, false = appended bit.
    kinds: Vec<bool>,
    learned: usize,
}

impl EnvModel for UniformModel {
    fn root_log_probability(&self) -> f64 {
        self.learned as f64 * LOG_HALF
    }

    fn predict_bit_probability(&mut self, _bit: u8) -> f64 {
        0.5
    }

    fn learn_symbols(&mut self, bits: &[u8]) {
        self.learned += bits.len();
        self.kinds.extend(std::iter::repeat_n(true, bits.len()));
    }

    fn append_history_symbols(&mut self, bits: &[u8]) {
        self.kinds.extend(std::iter::repeat_n(false, bits.len()));
    }

    fn revert_learned_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(true),
                "revert_learned out of LIFO order"
            );
            self.learned -= 1;
        }
    }

    fn revert_history_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(false),
                "revert_history out of LIFO order"
            );
        }
    }

    fn model_id(&self) -> String {
        "uniform".to_string()
    }
}
