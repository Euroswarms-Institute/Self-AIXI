//! `LlmModel` — the dissected base model as an MC-AIXI environment-model
//! component (IMPLEMENTATION_PLAN.md §3 contract, same as CTW).
//!
//! The interleaved action/percept bit stream maps to a token stream: one
//! stream-prime token at position 0 (this family has no BOS), then one bit
//! token per symbol. Frozen weights mean `learn` and `append` both merely
//! condition — the Bayes mixture provides the online adaptation *across*
//! components — but `LlmModel` still accumulates ln ρ(learned bits) so the
//! mixture can weigh it against CTW.
//!
//! Laziness and revert: tokens are pushed eagerly, advanced through the
//! network lazily (on the next `predict`), and per-position 2-logit outputs
//! are logged so predict-after-revert is O(1). Reverts truncate the token
//! log and roll the network state back via the checkpoint stack (bit-exact);
//! if a revert ever outruns the retained checkpoints, the state is rebuilt
//! by deterministic replay — exactness is never negotiated away.

use super::config::TokenProbe;
use super::model::Qwen35Model;
use super::state::LlmState;
use crate::models::EnvModel;

pub struct LlmModel {
    model: Qwen35Model,
    state: LlmState,
    /// Full token stream, prime included at index 0.
    tokens: Vec<u32>,
    /// One entry per post-prime symbol: was it learned (vs appended)?
    kinds: Vec<bool>,
    /// Cumulative ln ρ per learned symbol.
    loglik_prefix: Vec<f64>,
    /// [logit₀, logit₁] after each advanced position.
    logits2: Vec<[f32; 2]>,
}

impl LlmModel {
    pub fn new(model: Qwen35Model) -> Self {
        let state = model.new_state();
        let probe = model.probe;
        let mut m = LlmModel {
            model,
            state,
            tokens: vec![probe.prime],
            kinds: Vec::new(),
            loglik_prefix: Vec::new(),
            logits2: Vec::new(),
        };
        m.ensure_advanced();
        m
    }

    pub fn probe(&self) -> TokenProbe {
        self.model.probe
    }

    pub fn model(&self) -> &Qwen35Model {
        &self.model
    }

    /// Tokens currently conditioning the model (prime included).
    pub fn context_len(&self) -> usize {
        self.tokens.len()
    }

    pub fn deep_replays(&self) -> u64 {
        self.state.deep_replays
    }

    fn ensure_advanced(&mut self) {
        assert!(
            self.tokens.len() <= self.model.cfg.ctx_len,
            "history of {} tokens exceeds the model context window {}",
            self.tokens.len(),
            self.model.cfg.ctx_len
        );
        while self.state.pos < self.tokens.len() {
            let t = self.tokens[self.state.pos];
            let logits = self.model.advance(&mut self.state, t);
            self.logits2.push(logits);
        }
    }

    /// P(bit = 1) from the logged logits at the stream head.
    fn p_one(&mut self) -> f64 {
        self.ensure_advanced();
        let [l0, l1] = self.logits2[self.state.pos - 1];
        1.0 / (1.0 + ((l0 - l1) as f64).exp())
    }

    fn rollback_state(&mut self) {
        if self.state.pos <= self.tokens.len() {
            return;
        }
        let target = self.tokens.len();
        if self.state.revert_to(target) {
            self.logits2.truncate(target);
        } else {
            // Checkpoint evicted: deterministic full replay (exact, slow).
            self.state = self.model.new_state();
            self.state.deep_replays += 1;
            self.logits2.clear();
            let tokens = std::mem::take(&mut self.tokens);
            for &t in &tokens {
                let logits = self.model.advance(&mut self.state, t);
                self.logits2.push(logits);
            }
            self.tokens = tokens;
        }
    }

    fn push_symbol(&mut self, bit: u8, learned: bool) {
        debug_assert!(bit <= 1);
        self.tokens.push(self.model.probe.token_for_bit(bit));
        self.kinds.push(learned);
    }
}

impl EnvModel for LlmModel {
    fn root_log_probability(&self) -> f64 {
        self.loglik_prefix.last().copied().unwrap_or(0.0)
    }

    fn predict_bit_probability(&mut self, bit: u8) -> f64 {
        let p1 = self.p_one();
        if bit == 1 {
            p1
        } else {
            1.0 - p1
        }
    }

    fn learn_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            let p = self.predict_bit_probability(b).max(1e-300);
            self.loglik_prefix
                .push(self.root_log_probability() + p.ln());
            self.push_symbol(b, true);
        }
    }

    fn append_history_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            self.push_symbol(b, false);
        }
    }

    fn revert_learned_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(true),
                "revert_learned out of LIFO order"
            );
            self.tokens.pop();
            self.loglik_prefix.pop();
        }
        self.rollback_state();
    }

    fn revert_history_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(false),
                "revert_history out of LIFO order"
            );
            self.tokens.pop();
        }
        self.rollback_state();
    }

    fn model_id(&self) -> String {
        "llm".to_string()
    }
}
