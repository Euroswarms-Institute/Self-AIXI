//! The dissected forward pass exercised on a synthetic random-weight qwen35
//! hybrid (1 DeltaNet + 1 attention block) written through our own GGUF
//! writer — proving the *state machinery* (checkpointed recurrence, KV
//! truncation, deterministic replay, EnvModel contract) independently of the
//! real checkpoint. Absolute numeric fidelity of the real 2B forward is the
//! oracle script's job.

mod common;

use common::{build_tiny_model, Vocab};
use mc_aixi::llm::config::{Qwen35Config, TokenProbe};
use mc_aixi::llm::env_model::LlmModel;
use mc_aixi::llm::gguf::GgufFile;
use mc_aixi::llm::model::{QGateLayout, Qwen35Model};
use mc_aixi::models::EnvModel;
use mc_aixi::rng::seeded;

fn load_tiny(seed: u64) -> Qwen35Model {
    let path = build_tiny_model(seed, Vocab::Bits);
    let gguf = GgufFile::open(&path).unwrap();
    let cfg = Qwen35Config::from_gguf(&gguf).unwrap();
    assert_eq!((cfg.n_layers, cfg.hidden, cfg.vocab), (2, 16, 7));
    let probe = TokenProbe::from_gguf(&gguf).unwrap();
    assert_eq!((probe.bit0, probe.bit1, probe.prime), (2, 3, 5));
    Qwen35Model::from_gguf(&gguf, cfg, probe, QGateLayout::PerHeadInterleaved).unwrap()
}

#[test]
fn revert_and_readvance_is_bit_exact() {
    let model = load_tiny(1);
    let mut state = model.new_state();
    let tokens = [5u32, 2, 3, 2, 2, 3, 3, 2, 3, 2];
    let mut logits = Vec::new();
    for &t in &tokens {
        logits.push(model.advance(&mut state, t));
    }
    let full_digest = state.state_digest();

    assert!(state.revert_to(4));
    assert_eq!(state.pos, 4);
    for &t in &tokens[4..] {
        let l = model.advance(&mut state, t);
        assert_eq!(
            (l[0].to_bits(), l[1].to_bits()),
            (
                logits[state.pos - 1][0].to_bits(),
                logits[state.pos - 1][1].to_bits()
            ),
            "re-advanced logits differ at pos {}",
            state.pos
        );
    }
    assert_eq!(state.state_digest(), full_digest);
}

#[test]
fn deltanet_state_actually_carries_information() {
    // The recurrent path must make logits depend on distant context: two
    // histories with identical last tokens but different prefixes.
    let model = load_tiny(2);
    let mut a = model.new_state();
    let mut b = model.new_state();
    for &t in &[5u32, 2, 2, 3, 3] {
        model.advance(&mut a, t);
    }
    for &t in &[5u32, 3, 2, 3, 3] {
        model.advance(&mut b, t);
    }
    let la = model.advance(&mut a, 2);
    let lb = model.advance(&mut b, 2);
    assert!(
        la[0] != lb[0] || la[1] != lb[1],
        "prefix had no effect on logits"
    );
}

#[test]
fn env_model_contract_exact_revert_and_normalization() {
    let model = load_tiny(3);
    let mut m = LlmModel::new(model);
    let mut rng = seeded(9);
    use rand::Rng;

    // Base history.
    for _ in 0..6 {
        m.append_history_symbols(&[u8::from(rng.random_bool(0.5))]);
        m.learn_symbols(&[
            u8::from(rng.random_bool(0.5)),
            u8::from(rng.random_bool(0.5)),
        ]);
    }
    let root = m.root_log_probability();
    let p0 = m.predict_bit_probability(0);
    let p1 = m.predict_bit_probability(1);
    assert!(
        ((p0 + p1) - 1.0).abs() < 1e-12,
        "predictions must normalize"
    );
    assert!(root.is_finite() && root < 0.0);

    // predict == learn-ratio.
    let before = m.root_log_probability();
    m.learn_symbols(&[1]);
    let ratio = (m.root_log_probability() - before).exp();
    m.revert_learned_symbols(1);
    assert!((ratio - p1).abs() < 1e-12);
    assert_eq!(m.root_log_probability().to_bits(), root.to_bits());

    // Random excursions with LIFO unwinding restore everything bit-exactly.
    for _ in 0..30 {
        let mut ops = Vec::new();
        for _ in 0..rng.random_range(1..6) {
            let learned = rng.random_bool(0.5);
            let count = rng.random_range(1..3);
            let bits: Vec<u8> = (0..count).map(|_| u8::from(rng.random_bool(0.5))).collect();
            if learned {
                m.learn_symbols(&bits);
            } else {
                m.append_history_symbols(&bits);
            }
            ops.push((learned, count));
        }
        let _ = m.predict_bit_probability(1); // force advance mid-excursion
        for (learned, count) in ops.into_iter().rev() {
            if learned {
                m.revert_learned_symbols(count);
            } else {
                m.revert_history_symbols(count);
            }
        }
        assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
        assert_eq!(m.predict_bit_probability(0).to_bits(), p0.to_bits());
    }
    assert_eq!(
        m.deep_replays(),
        0,
        "shallow excursions must not trigger replay"
    );
}

#[test]
fn checkpoint_eviction_falls_back_to_exact_replay() {
    let model = load_tiny(4);
    let mut m = LlmModel::new(model);
    m.learn_symbols(&[1]);
    let root_after_one = m.root_log_probability();
    let p_after_one = m.predict_bit_probability(1);

    // Push far beyond CHECKPOINT_CAP, then unwind everything.
    let n = mc_aixi::llm::state::CHECKPOINT_CAP + 16;
    let deep: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    m.learn_symbols(&deep);
    m.revert_learned_symbols(deep.len());

    assert!(m.deep_replays() >= 1, "eviction must have forced a replay");
    assert_eq!(m.root_log_probability().to_bits(), root_after_one.to_bits());
    assert_eq!(
        m.predict_bit_probability(1).to_bits(),
        p_after_one.to_bits()
    );
}
