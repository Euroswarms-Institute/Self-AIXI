//! Seeded randomness. Every stochastic component (environments, percept
//! sampling in ρUCT, rollout policies) draws from an explicitly seeded
//! generator so whole experiments replay deterministically
//! (IMPLEMENTATION_PLAN.md §7 requires reproducible evaluation runs).

use rand_chacha::rand_core::SeedableRng;

/// The one RNG type used across the crate.
pub type AgentRng = rand_chacha::ChaCha8Rng;

/// Construct the crate RNG from a u64 experiment seed.
pub fn seeded(seed: u64) -> AgentRng {
    AgentRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn same_seed_same_stream() {
        let mut a = seeded(42);
        let mut b = seeded(42);
        for _ in 0..64 {
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }
}
