//! Factored Action-Conditional CTW — FAC-CTW proper (JAIR §5.4).
//!
//! The single-tree action-conditional CTW (`CtwModel`, pyaixi's choice)
//! suffers *positional aliasing*: one tree's leaves accumulate statistics
//! from every percept bit position, so e.g. a deterministic reward bit and a
//! stochastic observation bit share KT estimators unless the tree spends
//! depth (and data) separating them. The JAIR experiments therefore factor
//! the percept:
//!
//! ```text
//! ρ(e | h) = Π_p CTW_p( bit_p | h, bits_<p )
//! ```
//!
//! one context tree per percept bit position p, each conditioning on the
//! full interleaved history. Realized here by composition: for a bit learned
//! at position p, tree p *learns* it while every other tree *appends* it
//! (context-only) — action bits are appended to all trees. Correctness and
//! bit-exact revert are inherited wholesale from the verified `CtwModel`.
//!
//! Position tracking needs no side channel: learned bits arrive in complete
//! percepts (the agent loop and ρUCT both sample whole percepts), so the
//! position of the next learned bit is `learned_total mod percept_bits`,
//! and reverts walk the same counter backwards.

use super::ctw::CtwModel;
use super::EnvModel;

pub struct FacCtwModel {
    percept_bits: usize,
    trees: Vec<CtwModel>,
    learned_total: usize,
}

impl FacCtwModel {
    pub fn new(depth: usize, percept_bits: usize) -> Self {
        assert!(percept_bits >= 1, "need at least one percept bit");
        FacCtwModel {
            percept_bits,
            trees: (0..percept_bits).map(|_| CtwModel::new(depth)).collect(),
            learned_total: 0,
        }
    }

    pub fn depth(&self) -> usize {
        self.trees[0].depth()
    }

    /// Total allocated nodes across the factored trees (§7 metric).
    pub fn node_count(&self) -> usize {
        self.trees.iter().map(|t| t.node_count()).sum()
    }

    pub fn state_digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for t in &self.trees {
            t.state_digest().hash(&mut h);
        }
        self.learned_total.hash(&mut h);
        h.finish()
    }
}

impl EnvModel for FacCtwModel {
    fn root_log_probability(&self) -> f64 {
        self.trees.iter().map(|t| t.root_log_probability()).sum()
    }

    fn predict_bit_probability(&mut self, bit: u8) -> f64 {
        let p = self.learned_total % self.percept_bits;
        self.trees[p].predict_bit_probability(bit)
    }

    fn learn_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            let p = self.learned_total % self.percept_bits;
            for (i, tree) in self.trees.iter_mut().enumerate() {
                if i == p {
                    tree.learn_symbols(&[b]);
                } else {
                    tree.append_history_symbols(&[b]);
                }
            }
            self.learned_total += 1;
        }
    }

    fn append_history_symbols(&mut self, bits: &[u8]) {
        for tree in &mut self.trees {
            tree.append_history_symbols(bits);
        }
    }

    fn revert_learned_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert!(self.learned_total > 0, "no learned bit to revert");
            self.learned_total -= 1;
            let p = self.learned_total % self.percept_bits;
            for (i, tree) in self.trees.iter_mut().enumerate() {
                if i == p {
                    tree.revert_learned_symbols(1);
                } else {
                    tree.revert_history_symbols(1);
                }
            }
        }
    }

    fn revert_history_symbols(&mut self, n: usize) {
        for tree in &mut self.trees {
            tree.revert_history_symbols(n);
        }
    }

    fn model_id(&self) -> String {
        format!("fac-ctw-d{}", self.depth())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;
    use rand::Rng;

    /// Drive FAC-CTW and a hand-wired bank of per-position CtwModels through
    /// the same FAC stream; factors must agree bit-exactly.
    #[test]
    fn equals_manual_tree_bank() {
        let (depth, pbits) = (3usize, 2usize);
        let mut fac = FacCtwModel::new(depth, pbits);
        let mut bank: Vec<CtwModel> = (0..pbits).map(|_| CtwModel::new(depth)).collect();
        let mut rng = seeded(11);
        for _ in 0..40 {
            let a = u8::from(rng.random_bool(0.5));
            fac.append_history_symbols(&[a]);
            for t in &mut bank {
                t.append_history_symbols(&[a]);
            }
            for p in 0..pbits {
                let b = u8::from(rng.random_bool(0.7));
                fac.learn_symbols(&[b]);
                for (i, t) in bank.iter_mut().enumerate() {
                    if i == p {
                        t.learn_symbols(&[b]);
                    } else {
                        t.append_history_symbols(&[b]);
                    }
                }
            }
        }
        let manual: f64 = bank.iter().map(|t| t.root_log_probability()).sum();
        assert_eq!(fac.root_log_probability().to_bits(), manual.to_bits());
    }

    #[test]
    fn revert_restores_state_bit_exactly() {
        let mut m = FacCtwModel::new(4, 3);
        let mut rng = seeded(5);
        for i in 0..60u64 {
            if i % 4 == 0 {
                m.append_history_symbols(&[u8::from(rng.random_bool(0.5))]);
            } else {
                m.learn_symbols(&[u8::from(rng.random_bool(0.6))]);
            }
        }
        let digest = m.state_digest();
        let root = m.root_log_probability();
        for _ in 0..100 {
            let mut ops = Vec::new();
            for _ in 0..rng.random_range(1..8) {
                let learned = rng.random_bool(0.6);
                let n = rng.random_range(1..4);
                let bits: Vec<u8> = (0..n).map(|_| u8::from(rng.random_bool(0.5))).collect();
                if learned {
                    m.learn_symbols(&bits);
                } else {
                    m.append_history_symbols(&bits);
                }
                ops.push((learned, n));
            }
            for (learned, n) in ops.into_iter().rev() {
                if learned {
                    m.revert_learned_symbols(n);
                } else {
                    m.revert_history_symbols(n);
                }
            }
            assert_eq!(m.state_digest(), digest);
            assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
        }
    }

    /// The pathology FAC fixes: a deterministic reward bit next to a random
    /// observation bit. The factored model must nail the reward bit fast.
    #[test]
    fn separates_deterministic_reward_from_random_observation() {
        let mut m = FacCtwModel::new(2, 2);
        let mut rng = seeded(3);
        for _ in 0..150 {
            let a = u8::from(rng.random_bool(0.5));
            let obs = u8::from(rng.random_bool(0.6));
            let rew = u8::from(obs == a);
            m.append_history_symbols(&[a]);
            m.learn_symbols(&[obs, rew]);
        }
        // Probe: after action 1 and observation 1, reward must be ~1.
        m.append_history_symbols(&[1]);
        let p_obs1 = m.predict_bit_probability(1);
        m.learn_symbols(&[1]);
        let p_rew1 = m.predict_bit_probability(1);
        m.revert_learned_symbols(1);
        m.revert_history_symbols(1);
        assert!((p_obs1 - 0.6).abs() < 0.1, "obs marginal {p_obs1}");
        assert!(p_rew1 > 0.9, "reward determinism not learned: {p_rew1}");
    }
}
