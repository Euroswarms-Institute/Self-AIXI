//! Environment interface (JAIR §2).
//!
//! One interaction cycle: the agent emits an action `a_t`, the environment
//! replies with a percept `e_t = (o_t, r_t)`. All spaces are finite (§1.1) and
//! declared as bit widths. Rewards travel on the wire as non-negative *codes*
//! (offset encoding, as in JAIR §7's domains); `decode_reward` maps a code
//! back to its real value.

use crate::encoding::{decode_bits_msb, encode_bits_msb};
use crate::rng::AgentRng;

pub mod biased_rps;
pub mod cheese_maze;
pub mod coin_flip;
pub mod kuhn_poker;
pub mod text_bytes;
pub mod tiger;

/// A percept: observation symbol plus offset-encoded reward code
/// (§1 notation ledger: `Percept`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Percept {
    pub observation: u64,
    pub reward_code: u64,
}

impl Percept {
    /// Wire encoding: observation bits then reward bits, each MSB-first.
    pub fn encode_into(&self, obs_bits: u32, reward_bits: u32, out: &mut Vec<u8>) {
        encode_bits_msb(self.observation, obs_bits, out);
        encode_bits_msb(self.reward_code, reward_bits, out);
    }

    /// Inverse of [`Percept::encode_into`].
    pub fn decode(bits: &[u8], obs_bits: u32, reward_bits: u32) -> Percept {
        assert_eq!(bits.len(), (obs_bits + reward_bits) as usize);
        Percept {
            observation: decode_bits_msb(&bits[..obs_bits as usize]),
            reward_code: decode_bits_msb(&bits[obs_bits as usize..]),
        }
    }
}

/// A finite, resettable, stochastic environment.
///
/// Contract: `step` must only be called with `action < num_actions()`, and
/// every returned percept must satisfy the declared bit widths.
pub trait Environment {
    fn name(&self) -> &'static str;

    /// Number of legal actions (may be below `2^action_bits`, e.g. 3-action
    /// domains encoded in 2 bits — the planner only enumerates legal actions).
    fn num_actions(&self) -> u64;
    fn action_bits(&self) -> u32;
    fn observation_bits(&self) -> u32;
    fn reward_bits(&self) -> u32;

    /// Decoded reward bounds (α, β) used for ρUCT value normalization
    /// (JAIR §3.2).
    fn reward_range(&self) -> (f64, f64);

    /// Map an offset code back to the real reward. All bundled domains use
    /// integer rewards, hence the default `α + code`.
    fn decode_reward(&self, code: u64) -> f64 {
        self.reward_range().0 + code as f64
    }

    /// (Re)initialize internal state. No percept is produced: per the JAIR
    /// interaction protocol the agent moves first.
    fn reset(&mut self, rng: &mut AgentRng);

    /// Perform `action`, returning the environment's percept.
    fn step(&mut self, action: u64, rng: &mut AgentRng) -> Percept;

    fn percept_bits(&self) -> u32 {
        self.observation_bits() + self.reward_bits()
    }
}

/// The static facts a planner needs about a domain, detached from the live
/// environment (the search must never touch the real environment — it plans
/// against ξ only, JAIR §3).
#[derive(Clone, Copy, Debug)]
pub struct DomainSpec {
    pub num_actions: u64,
    pub action_bits: u32,
    pub observation_bits: u32,
    pub reward_bits: u32,
    pub reward_min: f64,
    pub reward_max: f64,
}

impl DomainSpec {
    pub fn from_env(env: &dyn Environment) -> Self {
        let (reward_min, reward_max) = env.reward_range();
        DomainSpec {
            num_actions: env.num_actions(),
            action_bits: env.action_bits(),
            observation_bits: env.observation_bits(),
            reward_bits: env.reward_bits(),
            reward_min,
            reward_max,
        }
    }

    pub fn percept_bits(&self) -> u32 {
        self.observation_bits + self.reward_bits
    }

    /// Decode a (possibly model-imagined) reward code. Imagined codes can
    /// exceed the environment's true span, so the decoded value is clamped to
    /// the declared range — both ρUCT and the exact expectimax use this same
    /// convention.
    pub fn decode_reward(&self, code: u64) -> f64 {
        (self.reward_min + code as f64).min(self.reward_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percept_roundtrip() {
        let p = Percept {
            observation: 0b1011,
            reward_code: 0b10110,
        };
        let mut bits = Vec::new();
        p.encode_into(4, 5, &mut bits);
        assert_eq!(bits.len(), 9);
        assert_eq!(Percept::decode(&bits, 4, 5), p);
    }
}
