//! The Bayesian mixture ξ over a finite model catalog
//! (IMPLEMENTATION_PLAN.md §3; JAIR §2, eq. 5; §1 ledger: `log_weights`).
//!
//! ```text
//! ξ(e₁:ₜ ‖ a₁:ₜ) = Σ_ν w⁰_ν · ν(e₁:ₜ ‖ a₁:ₜ)
//! ```
//!
//! maintained in log space as `log_w[ν] = ln w⁰_ν + ln ν(·)`, updated by
//! Bayes' rule on **learned** (percept) bits only — appended action bits
//! condition every component but carry no probability (FAC, JAIR §5.3).
//!
//! Dominance, the property that makes the mixture the right home for a
//! speculative component like the dissected LLM: for every component ν,
//! ln ξ ≥ ln w⁰_ν + ln ν, i.e. the mixture's log-loss exceeds the best
//! component's by at most its prior cost ln(1/w⁰_ν) — asserted in tests.
//!
//! Reverts restore the exact previous `log_w` vectors (recorded values, not
//! replayed deltas) and forward the undo to every component, preserving the
//! bit-exact revert contract of `aixi/planning/xi_rollouts.py`.

use super::EnvModel;
use crate::logspace::{log_sum_exp_slice, softmax};

pub struct BayesMixture {
    components: Vec<Box<dyn EnvModel>>,
    log_w: Vec<f64>,
    /// Previous `log_w` snapshots, one K-sized chunk per learned bit.
    weight_records: Vec<f64>,
    /// LIFO record of operations: true = learned bit, false = appended bit.
    kinds: Vec<bool>,
}

impl BayesMixture {
    /// Uniform prior over the catalog.
    pub fn uniform(components: Vec<Box<dyn EnvModel>>) -> Self {
        let k = components.len();
        Self::with_log_priors(components, vec![-(k as f64).ln(); k])
    }

    /// Explicit prior; normalized here so `root_log_probability` starts at 0.
    pub fn with_log_priors(components: Vec<Box<dyn EnvModel>>, log_priors: Vec<f64>) -> Self {
        assert!(
            !components.is_empty(),
            "mixture needs at least one component"
        );
        assert_eq!(components.len(), log_priors.len());
        let z = log_sum_exp_slice(&log_priors);
        assert!(z.is_finite(), "prior has no mass");
        let log_w = log_priors.iter().map(|&p| p - z).collect();
        BayesMixture {
            components,
            log_w,
            weight_records: Vec::new(),
            kinds: Vec::new(),
        }
    }

    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Posterior P(ν | learned bits so far), aligned with `component_ids`.
    pub fn posterior_weights(&self) -> Vec<f64> {
        softmax(&self.log_w)
    }

    pub fn component_ids(&self) -> Vec<String> {
        self.components.iter().map(|c| c.model_id()).collect()
    }

    /// ln w⁰_ν + ln ν(·) for component `i` (diagnostics/tests).
    pub fn component_log_weight(&self, i: usize) -> f64 {
        self.log_w[i]
    }

    fn learn_one(&mut self, bit: u8) {
        self.weight_records.extend_from_slice(&self.log_w);
        for (c, lw) in self.components.iter_mut().zip(self.log_w.iter_mut()) {
            let p = c.predict_bit_probability(bit);
            debug_assert!((0.0..=1.0 + 1e-12).contains(&p), "component prob {p}");
            *lw += p.ln();
            c.learn_symbols(&[bit]);
        }
        self.kinds.push(true);
    }
}

impl EnvModel for BayesMixture {
    fn root_log_probability(&self) -> f64 {
        log_sum_exp_slice(&self.log_w)
    }

    fn predict_bit_probability(&mut self, bit: u8) -> f64 {
        let posterior = softmax(&self.log_w);
        self.components
            .iter_mut()
            .zip(posterior)
            .map(|(c, w)| w * c.predict_bit_probability(bit))
            .sum()
    }

    fn learn_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            self.learn_one(b);
        }
    }

    fn append_history_symbols(&mut self, bits: &[u8]) {
        for c in &mut self.components {
            c.append_history_symbols(bits);
        }
        self.kinds.extend(std::iter::repeat_n(false, bits.len()));
    }

    fn revert_learned_symbols(&mut self, n: usize) {
        let k = self.components.len();
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(true),
                "revert_learned out of LIFO order"
            );
            for c in &mut self.components {
                c.revert_learned_symbols(1);
            }
            let start = self.weight_records.len() - k;
            self.log_w.copy_from_slice(&self.weight_records[start..]);
            self.weight_records.truncate(start);
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
        for c in &mut self.components {
            c.revert_history_symbols(n);
        }
    }

    fn model_id(&self) -> String {
        format!(
            "mixture[{}]",
            self.components
                .iter()
                .map(|c| c.model_id())
                .collect::<Vec<_>>()
                .join("+")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctw::CtwModel;
    use crate::models::uniform::UniformModel;
    use crate::rng::seeded;
    use rand::Rng;

    fn kt_uniform_mixture() -> BayesMixture {
        BayesMixture::uniform(vec![
            Box::new(CtwModel::new(0)), // order-0 KT
            Box::new(UniformModel::default()),
        ])
    }

    #[test]
    fn hand_computed_posterior_and_predictions() {
        // Learn "111". KT: ½·¾·⅚ = 5/16; uniform: 1/8.
        // ξ = ½(5/16 + 1/8) = 7/32; posterior_KT = 5/7.
        // Next-bit: KT gives 7/8, uniform ½ ⇒ P(1) = (5/7)(7/8) + (2/7)(½) = 43/56.
        let mut m = kt_uniform_mixture();
        m.learn_symbols(&[1, 1, 1]);
        assert!((m.root_log_probability() - (7.0f64 / 32.0).ln()).abs() < 1e-13);
        let w = m.posterior_weights();
        assert!((w[0] - 5.0 / 7.0).abs() < 1e-13);
        assert!((m.predict_bit_probability(1) - 43.0 / 56.0).abs() < 1e-13);
        let total = m.predict_bit_probability(0) + m.predict_bit_probability(1);
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dominance_over_every_component() {
        // ln ξ ≥ ln w⁰_ν + ln ν for components run standalone on the same
        // FAC stream (here w⁰ = ⅓ each).
        let mut mix = BayesMixture::uniform(vec![
            Box::new(CtwModel::new(0)),
            Box::new(CtwModel::new(4)),
            Box::new(UniformModel::default()),
        ]);
        let mut solo_kt = CtwModel::new(0);
        let mut solo_ctw = CtwModel::new(4);
        let mut rng = seeded(99);
        for _ in 0..80 {
            let a = u8::from(rng.random_bool(0.5));
            mix.append_history_symbols(&[a]);
            solo_kt.append_history_symbols(&[a]);
            solo_ctw.append_history_symbols(&[a]);
            let e = [u8::from(rng.random_bool(0.8)), a];
            mix.learn_symbols(&e);
            solo_kt.learn_symbols(&e);
            solo_ctw.learn_symbols(&e);
        }
        let prior_cost = (3.0f64).ln();
        let xi = mix.root_log_probability();
        assert!(xi >= solo_kt.root_log_probability() - prior_cost - 1e-12);
        assert!(xi >= solo_ctw.root_log_probability() - prior_cost - 1e-12);
        assert!(xi >= 160.0 * crate::logspace::LOG_HALF - prior_cost - 1e-12);
        // The stream is action-echoing, so the deep CTW must dominate KT and
        // the posterior must concentrate on it.
        let w = mix.posterior_weights();
        assert!(w[1] > 0.9, "posterior on ctw-d4 was {}", w[1]);
    }

    #[test]
    fn revert_restores_weights_and_components_exactly() {
        let mut m = BayesMixture::uniform(vec![
            Box::new(CtwModel::new(3)),
            Box::new(UniformModel::default()),
        ]);
        let mut rng = seeded(7);
        for _ in 0..30 {
            let b = u8::from(rng.random_bool(0.6));
            if rng.random_bool(0.3) {
                m.append_history_symbols(&[b]);
            } else {
                m.learn_symbols(&[b]);
            }
        }
        let root = m.root_log_probability();
        let w: Vec<u64> = m.log_w.iter().map(|x| x.to_bits()).collect();

        for _ in 0..100 {
            let mut ops: Vec<(bool, usize)> = Vec::new();
            for _ in 0..rng.random_range(1..10) {
                let learned = rng.random_bool(0.5);
                let count = rng.random_range(1..4);
                let bits: Vec<u8> = (0..count).map(|_| u8::from(rng.random_bool(0.5))).collect();
                if learned {
                    m.learn_symbols(&bits);
                } else {
                    m.append_history_symbols(&bits);
                }
                ops.push((learned, count));
            }
            for (learned, count) in ops.into_iter().rev() {
                if learned {
                    m.revert_learned_symbols(count);
                } else {
                    m.revert_history_symbols(count);
                }
            }
            assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
            let w2: Vec<u64> = m.log_w.iter().map(|x| x.to_bits()).collect();
            assert_eq!(w, w2);
        }
    }

    #[test]
    fn appended_bits_leave_weights_untouched() {
        let mut m = kt_uniform_mixture();
        m.learn_symbols(&[1, 0, 1]);
        let w = m.posterior_weights();
        let root = m.root_log_probability();
        m.append_history_symbols(&[1, 1, 0, 1]);
        assert_eq!(m.posterior_weights(), w);
        assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
    }
}
