//! Exact finite-horizon expectimax over ξ by full enumeration (JAIR eq. 10):
//!
//! ```text
//! V(h, m) = max_a Σ_e ξ(e | h, a) · [ r(e) + γ · V(hae, m−1) ]
//! ```
//!
//! Cost is (|A| · 2^percept_bits)^m — usable only on tiny domains, which is
//! the point: it is the ground truth that `rho_uct` converges to in tests and
//! the smoke suite. The model is mutated during enumeration and restored
//! exactly through the standard revert contract.

use crate::encoding::{decode_bits_msb, encode_bits_msb};
use crate::env::DomainSpec;
use crate::models::EnvModel;

/// Exact Q(h, a) for every action at the current model state.
pub fn exact_q_values(
    model: &mut dyn EnvModel,
    spec: &DomainSpec,
    gamma: f64,
    horizon: u32,
) -> Vec<f64> {
    (0..spec.num_actions)
        .map(|a| q_value(model, spec, gamma, horizon, a))
        .collect()
}

/// Exact expectimax decision: (argmax_a Q(h,a), V(h, m)); ties take the
/// lowest action id, matching the search's tie-breaking.
pub fn exact_expectimax(
    model: &mut dyn EnvModel,
    spec: &DomainSpec,
    gamma: f64,
    horizon: u32,
) -> (u64, f64) {
    let qs = exact_q_values(model, spec, gamma, horizon);
    let mut best = 0usize;
    for (i, q) in qs.iter().enumerate() {
        if *q > qs[best] {
            best = i;
        }
    }
    (best as u64, qs[best])
}

fn state_value(model: &mut dyn EnvModel, spec: &DomainSpec, gamma: f64, horizon: u32) -> f64 {
    if horizon == 0 {
        return 0.0;
    }
    (0..spec.num_actions)
        .map(|a| q_value(model, spec, gamma, horizon, a))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn q_value(
    model: &mut dyn EnvModel,
    spec: &DomainSpec,
    gamma: f64,
    horizon: u32,
    action: u64,
) -> f64 {
    let mut bits = Vec::with_capacity(spec.action_bits as usize);
    encode_bits_msb(action, spec.action_bits, &mut bits);
    model.append_history_symbols(&bits);
    let q = percept_expectation(model, spec, gamma, horizon, &mut Vec::new());
    model.revert_history_symbols(spec.action_bits as usize);
    q
}

/// Σ over completions of the current partial percept, weighted by ξ's
/// bit-conditionals, of r + γ·V.
fn percept_expectation(
    model: &mut dyn EnvModel,
    spec: &DomainSpec,
    gamma: f64,
    horizon: u32,
    prefix: &mut Vec<u8>,
) -> f64 {
    if prefix.len() == spec.percept_bits() as usize {
        let reward_code = decode_bits_msb(&prefix[spec.observation_bits as usize..]);
        let reward = spec.decode_reward(reward_code);
        return reward + gamma * state_value(model, spec, gamma, horizon - 1);
    }
    let mut acc = 0.0;
    for bit in [0u8, 1] {
        let p = model.predict_bit_probability(bit);
        if p > 0.0 {
            model.learn_symbols(&[bit]);
            prefix.push(bit);
            acc += p * percept_expectation(model, spec, gamma, horizon, prefix);
            prefix.pop();
            model.revert_learned_symbols(1);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctw::CtwModel;
    use crate::models::EnvModel;

    fn coin_spec() -> DomainSpec {
        DomainSpec {
            num_actions: 2,
            action_bits: 1,
            observation_bits: 1,
            reward_bits: 1,
            reward_min: 0.0,
            reward_max: 1.0,
        }
    }

    #[test]
    fn restores_model_and_matches_hand_horizon_one() {
        // Fresh KT model, horizon 1: percept bits are i.i.d. KT-uniform, so
        // every action has Q = E[reward bit] = ½ regardless of action.
        let mut model = CtwModel::new(0);
        let digest = model.state_digest();
        let qs = exact_q_values(&mut model, &coin_spec(), 1.0, 1);
        assert_eq!(model.state_digest(), digest);
        for q in qs {
            assert!((q - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn value_grows_with_horizon() {
        let mut model = CtwModel::new(2);
        model.learn_symbols(&[1, 1, 1, 1, 1, 1]);
        let (_, v1) = exact_expectimax(&mut model, &coin_spec(), 1.0, 1);
        let (_, v2) = exact_expectimax(&mut model, &coin_spec(), 1.0, 2);
        assert!(v2 > v1);
        assert!(v2 <= 2.0 + 1e-12);
    }
}
