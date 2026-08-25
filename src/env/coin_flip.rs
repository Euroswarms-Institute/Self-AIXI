//! CoinFlip domain — the frozen Phase-0 parity environment
//! (`experiments/PHASE0_PYAIXI_PARITY_REPORT.md`: p = 0.6).
//!
//! Each cycle the agent predicts the outcome of a biased coin; the percept is
//! the actual flip plus reward 1 for a correct guess, else 0. The optimal
//! policy (always guess the biased side) earns 0.6 per cycle in expectation.

use super::{Environment, Percept};
use crate::rng::AgentRng;
use rand::Rng;

pub struct CoinFlip {
    /// Probability the coin comes up 1 ("heads").
    p_heads: f64,
}

impl CoinFlip {
    pub fn new(p_heads: f64) -> Self {
        assert!((0.0..=1.0).contains(&p_heads), "p_heads must be in [0,1]");
        CoinFlip { p_heads }
    }
}

impl Default for CoinFlip {
    /// The Phase-0 parity configuration (`coin-flip-p = 0.6`).
    fn default() -> Self {
        CoinFlip::new(0.6)
    }
}

impl Environment for CoinFlip {
    fn name(&self) -> &'static str {
        "coin_flip"
    }
    fn num_actions(&self) -> u64 {
        2
    }
    fn action_bits(&self) -> u32 {
        1
    }
    fn observation_bits(&self) -> u32 {
        1
    }
    fn reward_bits(&self) -> u32 {
        1
    }
    fn reward_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    fn reset(&mut self, _rng: &mut AgentRng) {}

    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept {
        assert!(action < self.num_actions());
        let flip = u64::from(rng.random_bool(self.p_heads));
        Percept {
            observation: flip,
            reward_code: u64::from(flip == action),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;

    #[test]
    fn heads_frequency_near_bias() {
        let mut env = CoinFlip::default();
        let mut rng = seeded(42);
        env.reset(&mut rng);
        let n = 20_000;
        let mut heads = 0u64;
        let mut reward = 0u64;
        for _ in 0..n {
            let p = env.step(1, &mut rng); // always guess heads
            heads += p.observation;
            reward += p.reward_code;
        }
        let f = heads as f64 / n as f64;
        assert!((f - 0.6).abs() < 0.02, "heads frequency {f}");
        assert_eq!(heads, reward); // guessing heads pays exactly when heads
    }

    #[test]
    fn deterministic_under_seed() {
        let run = || {
            let mut env = CoinFlip::default();
            let mut rng = seeded(7);
            env.reset(&mut rng);
            (0..64)
                .map(|i| env.step(i & 1, &mut rng).observation)
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
