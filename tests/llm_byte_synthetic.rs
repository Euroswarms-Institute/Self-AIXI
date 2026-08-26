//! The byte carve exercised on the synthetic random-weight hybrid: probe
//! recovery, the EnvModel contract under the cycle layout (action byte +
//! observation byte + reward bit), chain-rule consistency of bit-level
//! observation predictions, bit-exact revert across byte boundaries, KT
//! reward-bit adaptation, and the checkpoint-eviction replay path.

mod common;

use common::{build_tiny_model, Vocab};
use mc_aixi::llm::byte_model::{load_byte_carved, LlmByteModel};
use mc_aixi::llm::config::{ByteProbe, TokenProbe};
use mc_aixi::llm::gguf::GgufFile;
use mc_aixi::llm::model::QGateLayout;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::modal_byte::byte_observation_marginal;
use mc_aixi::rng::seeded;
use rand::Rng;

fn load_tiny_bytes(seed: u64) -> LlmByteModel {
    let path = build_tiny_model(seed, Vocab::Bytes);
    let (model, probe, head) = load_byte_carved(&path, QGateLayout::PerHeadInterleaved).unwrap();
    LlmByteModel::new(model, probe, head, 8, 8, 1)
}

/// Encode a byte MSB-first.
fn byte_bits(b: u8) -> Vec<u8> {
    (0..8).rev().map(|k| (b >> k) & 1).collect()
}

#[test]
fn byte_probe_finds_all_256_tokens_and_agrees_with_the_bit_probe() {
    let path = build_tiny_model(11, Vocab::Bytes);
    let gguf = GgufFile::open(&path).unwrap();
    let bp = ByteProbe::from_gguf(&gguf).unwrap();
    let tp = TokenProbe::from_gguf(&gguf).unwrap();
    // The builder lays byte tokens out in byte order.
    for b in 0..=255u8 {
        assert_eq!(bp.byte_tokens[b as usize], b as u32);
    }
    // The "0"/"1" text tokens ARE the 0x30/0x31 byte tokens.
    assert_eq!(bp.byte_tokens[0x30], tp.bit0);
    assert_eq!(bp.byte_tokens[0x31], tp.bit1);
    assert_eq!(bp.prime, tp.prime);
}

#[test]
fn observation_marginal_is_a_distribution_and_chain_rule_holds() {
    let mut m = load_tiny_bytes(21);
    // Commit one full cycle so the next phase is an action boundary.
    m.append_history_symbols(&byte_bits(b'a'));
    m.learn_symbols(&byte_bits(b'b'));
    m.learn_symbols(&[1]);

    // Move to the observation phase (append the next action byte).
    m.append_history_symbols(&byte_bits(b'c'));
    let root = m.root_log_probability();
    let marginal = byte_observation_marginal(&mut m);
    let total: f64 = marginal.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "marginal sums to {total}");
    assert_eq!(m.root_log_probability().to_bits(), root.to_bits());

    // Chain rule: learning a byte adds exactly ln(marginal[byte]).
    for &b in &[b'x', 0x00, 0xFF] {
        let before = m.root_log_probability();
        m.learn_symbols(&byte_bits(b));
        let delta = m.root_log_probability() - before;
        m.revert_learned_symbols(8);
        assert!(
            (delta - marginal[b as usize].ln()).abs() < 1e-9,
            "byte {b:#04x}: bit chain {delta} vs byte marginal {}",
            marginal[b as usize].ln()
        );
    }
    assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
}

#[test]
fn cycle_stream_bit_exact_revert_including_byte_boundaries() {
    let mut m = load_tiny_bytes(31);
    let mut rng = seeded(7);
    // Commit a few full cycles.
    for _ in 0..3 {
        m.append_history_symbols(&byte_bits(rng.random::<u8>()));
        m.learn_symbols(&byte_bits(rng.random::<u8>()));
        m.learn_symbols(&[u8::from(rng.random_bool(0.5))]);
    }
    let root = m.root_log_probability();
    let p0 = m.predict_bit_probability(0);
    let ctx = m.context_len();

    // Random excursions of arbitrary bit counts (crossing byte boundaries
    // mid-cycle), unwound LIFO — everything must restore bit-exactly.
    for _ in 0..40 {
        let mut ops = Vec::new();
        for _ in 0..rng.random_range(1..5) {
            let learned = rng.random_bool(0.5);
            let count = rng.random_range(1..14);
            let bits: Vec<u8> = (0..count).map(|_| u8::from(rng.random_bool(0.5))).collect();
            if learned {
                m.learn_symbols(&bits);
            } else {
                m.append_history_symbols(&bits);
            }
            ops.push((learned, count));
        }
        let _ = m.predict_bit_probability(1); // may force advances mid-excursion
        for (learned, count) in ops.into_iter().rev() {
            if learned {
                m.revert_learned_symbols(count);
            } else {
                m.revert_history_symbols(count);
            }
        }
        assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
        assert_eq!(m.predict_bit_probability(0).to_bits(), p0.to_bits());
        assert_eq!(m.context_len(), ctx);
    }
    assert_eq!(m.deep_replays(), 0, "shallow excursions must not replay");
}

#[test]
fn reward_bits_adapt_via_kt_and_revert_exactly() {
    let mut m = load_tiny_bytes(41);
    let cycle = |m: &mut LlmByteModel, a: u8, o: u8, r: u8| {
        m.append_history_symbols(&byte_bits(a));
        m.learn_symbols(&byte_bits(o));
        m.learn_symbols(&[r]);
    };
    // Fresh KT: P(r=1) = 1/2.
    m.append_history_symbols(&byte_bits(b'q'));
    m.learn_symbols(&byte_bits(b'q'));
    assert!((m.predict_bit_probability(1) - 0.5).abs() < 1e-12);
    m.learn_symbols(&[1]);
    // After one observed 1: P(r=1) = (1 + 1/2) / 2 = 3/4.
    m.append_history_symbols(&byte_bits(b'q'));
    m.learn_symbols(&byte_bits(b'q'));
    assert!((m.predict_bit_probability(1) - 0.75).abs() < 1e-12);
    let root = m.root_log_probability();
    // A learned reward reverts its count: prediction is restored exactly.
    m.learn_symbols(&[0]);
    m.revert_learned_symbols(1);
    assert!((m.predict_bit_probability(1) - 0.75).abs() < 1e-12);
    assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
    m.learn_symbols(&[1]);
    // Two 1s of 2: P(r=1) = (2 + 1/2) / 3 = 5/6.
    cycle(&mut m, b'q', b'q', 1);
    m.append_history_symbols(&byte_bits(b'q'));
    m.learn_symbols(&byte_bits(b'q'));
    let p = m.predict_bit_probability(1);
    assert!((p - 3.5 / 4.0).abs() < 1e-12, "KT(3 ones/3) = 7/8, got {p}");
}

#[test]
fn checkpoint_eviction_falls_back_to_exact_replay() {
    let mut m = load_tiny_bytes(51);
    let mut rng = seeded(3);
    m.append_history_symbols(&byte_bits(b'a'));
    m.learn_symbols(&byte_bits(b'b'));
    m.learn_symbols(&[1]);
    let root = m.root_log_probability();
    let p = m.predict_bit_probability(1);

    // Learn far more bytes than CHECKPOINT_CAP tokens, then unwind.
    let n_cycles = mc_aixi::llm::state::CHECKPOINT_CAP + 16;
    for _ in 0..n_cycles {
        m.append_history_symbols(&byte_bits(rng.random::<u8>()));
        m.learn_symbols(&byte_bits(rng.random::<u8>()));
        m.learn_symbols(&[u8::from(rng.random_bool(0.5))]);
    }
    let _ = m.predict_bit_probability(1); // force the advances
    for _ in 0..n_cycles {
        m.revert_learned_symbols(9);
        m.revert_history_symbols(8);
    }
    assert!(m.deep_replays() >= 1, "eviction must have forced a replay");
    assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
    assert_eq!(m.predict_bit_probability(1).to_bits(), p.to_bits());
}
