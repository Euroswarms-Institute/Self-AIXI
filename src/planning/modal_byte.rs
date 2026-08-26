//! Exact one-step planning for byte-prediction domains.
//!
//! In a prediction domain (action byte = guess, observation byte = truth,
//! reward = correctness) the horizon-1 expectimax value of action a is
//!
//! ```text
//! Q(a) = Σ_{o,r} ξ(o, r | h, a) · r
//! ```
//!
//! With the bundled models' reward belief independent of which (a, o) pair
//! occurred (order-0 KT / short-context CTW early in training), the argmax
//! over a reduces to the modal byte of ξ's observation marginal conditioned
//! on the agent having guessed a — which this module computes *exactly* by
//! shared-prefix enumeration of the 8-bit observation tree, using the same
//! learn/revert conditioning as ρUCT's chance nodes. No sampling, no UCB:
//! at m = 1 the enumeration IS the expectimax, at a cost 256 sampled
//! simulations could not beat.
//!
//! The enumeration is cheap by design: CTW predicts are microsecond tree
//! walks, and the byte-carved LLM answers every query from its cached
//! 256-way distribution (completing an imagined byte does not advance the
//! network while it is reverted before the next byte is needed).

use crate::encoding::encode_bits_msb;
use crate::env::DomainSpec;
use crate::models::EnvModel;

/// ξ's exact marginal over the next observation byte given the committed
/// history (and any action bits the caller has appended), by depth-first
/// enumeration with learn/revert conditioning. Entries sum to the total
/// semimeasure mass on 8-bit observations (1 for proper measures).
pub fn byte_observation_marginal(model: &mut dyn EnvModel) -> [f64; 256] {
    let mut out = [0f64; 256];
    rec(model, 0, 0, 0.0, &mut out);
    out
}

fn rec(model: &mut dyn EnvModel, depth: usize, val: usize, logp: f64, out: &mut [f64; 256]) {
    if depth == 8 {
        out[val] = logp.exp();
        return;
    }
    let p1 = model.predict_bit_probability(1);
    for bit in 0..2u8 {
        let p = if bit == 1 { p1 } else { 1.0 - p1 };
        if p <= 0.0 {
            continue; // dead branch; its bytes keep probability 0
        }
        model.learn_symbols(&[bit]);
        rec(
            model,
            depth + 1,
            (val << 1) | bit as usize,
            logp + p.ln(),
            out,
        );
        model.revert_learned_symbols(1);
    }
}

/// The exact horizon-1 prediction policy: for every action byte a, condition
/// on having guessed a (appended, as the agent will), take ξ's observation
/// marginal, and score Q(a) = P_ξ(o = a | h, a). Returns the argmax action
/// and the full Q vector. Requires an 8-bit action / 8-bit observation
/// domain. ξ is restored exactly (append + revert per action).
pub fn plan_modal_byte(model: &mut dyn EnvModel, spec: &DomainSpec) -> (u64, [f64; 256]) {
    assert_eq!(
        (spec.action_bits, spec.observation_bits),
        (8, 8),
        "modal-byte planning requires a byte-prediction domain"
    );
    let mut q = [0f64; 256];
    let mut abits = Vec::with_capacity(8);
    for a in 0..spec.num_actions.min(256) {
        abits.clear();
        encode_bits_msb(a, 8, &mut abits);
        model.append_history_symbols(&abits);
        let marginal = byte_observation_marginal(model);
        q[a as usize] = marginal[a as usize];
        model.revert_history_symbols(8);
    }
    let best = q
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u64)
        .unwrap();
    (best, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctw::CtwModel;
    use crate::models::mixture::BayesMixture;
    use crate::models::uniform::UniformModel;
    use crate::models::EnvModel;

    #[test]
    fn marginal_is_a_distribution_and_restores_the_model() {
        let mut m = BayesMixture::uniform(vec![
            Box::new(CtwModel::new(4)) as Box<dyn EnvModel>,
            Box::new(UniformModel::default()),
        ]);
        // Bias the stream toward ASCII 'a' = 0x61 patterns.
        for _ in 0..12 {
            m.learn_symbols(&[0, 1, 1, 0, 0, 0, 0, 1]);
        }
        let root = m.root_log_probability();
        let marginal = byte_observation_marginal(&mut m);
        let total: f64 = marginal.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "marginal sums to {total}");
        assert_eq!(
            m.root_log_probability().to_bits(),
            root.to_bits(),
            "enumeration must restore the model bit-exactly"
        );
        // The trained pattern must be the modal byte.
        let modal = marginal
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(modal, 0x61, "modal byte should be the trained pattern");
    }

    #[test]
    fn marginal_matches_direct_chain_products() {
        // On an order-0-ish model, P(byte) from the enumeration must equal
        // the product of the eight conditional bit probabilities computed
        // directly — spot-check a few bytes.
        let mut m = BayesMixture::uniform(vec![
            Box::new(CtwModel::new(2)) as Box<dyn EnvModel>,
            Box::new(UniformModel::default()),
        ]);
        m.learn_symbols(&[1, 0, 1, 1, 0, 1, 0, 0, 1, 1]);
        let marginal = byte_observation_marginal(&mut m);
        for &byte in &[0x00u8, 0x61, 0xFF, 0x30] {
            let mut direct = 0.0f64;
            let mut learned = 0;
            for k in (0..8).rev() {
                let bit = (byte >> k) & 1;
                direct += m.predict_bit_probability(bit).ln();
                m.learn_symbols(&[bit]);
                learned += 1;
            }
            m.revert_learned_symbols(learned);
            assert!(
                (marginal[byte as usize] - direct.exp()).abs() < 1e-12,
                "byte {byte:#04x}: {} vs {}",
                marginal[byte as usize],
                direct.exp()
            );
        }
    }
}
