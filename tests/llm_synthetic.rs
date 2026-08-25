//! The dissected forward pass exercised on a synthetic random-weight qwen35
//! hybrid (1 DeltaNet + 1 attention block) written through our own GGUF
//! writer — proving the *state machinery* (checkpointed recurrence, KV
//! truncation, deterministic replay, EnvModel contract) independently of the
//! real checkpoint. Absolute numeric fidelity of the real 2B forward is the
//! oracle script's job.

use mc_aixi::llm::config::{Qwen35Config, TokenProbe};
use mc_aixi::llm::env_model::LlmModel;
use mc_aixi::llm::gguf::{write_gguf, GgufFile, GgufValue};
use mc_aixi::llm::model::{QGateLayout, Qwen35Model};
use mc_aixi::llm::quant::GgmlType;
use mc_aixi::models::EnvModel;
use mc_aixi::rng::seeded;
use rand::Rng;
use std::path::PathBuf;

const HIDDEN: u64 = 16;
const FFN: u64 = 32;
const HEADS: u64 = 2;
const KV_HEADS: u64 = 1;
const HEAD_DIM: u64 = 8;
const SSM_HEADS: u64 = 2;
const SSM_DK: u64 = 4;
const SSM_INNER: u64 = 8; // 2 heads × d_v 4
const CONV_K: u64 = 4;

fn f32_tensor(
    rng: &mut mc_aixi::rng::AgentRng,
    ne: Vec<u64>,
    lo: f32,
    hi: f32,
) -> (Vec<u64>, Vec<u8>) {
    let n: u64 = ne.iter().product();
    let data = (0..n)
        .flat_map(|_| rng.random_range(lo..hi).to_le_bytes())
        .collect();
    (ne, data)
}

fn build_tiny_model(seed: u64) -> PathBuf {
    let mut rng = seeded(seed);
    let dir = std::env::temp_dir().join("mc_aixi_llm_synth");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("tiny-{seed}.gguf"));

    let kvs: Vec<(String, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufValue::Str("qwen35".into()),
        ),
        ("qwen35.block_count".into(), GgufValue::U32(2)),
        ("qwen35.context_length".into(), GgufValue::U32(512)),
        (
            "qwen35.embedding_length".into(),
            GgufValue::U32(HIDDEN as u32),
        ),
        (
            "qwen35.feed_forward_length".into(),
            GgufValue::U32(FFN as u32),
        ),
        (
            "qwen35.attention.head_count".into(),
            GgufValue::U32(HEADS as u32),
        ),
        (
            "qwen35.attention.head_count_kv".into(),
            GgufValue::U32(KV_HEADS as u32),
        ),
        (
            "qwen35.attention.key_length".into(),
            GgufValue::U32(HEAD_DIM as u32),
        ),
        (
            "qwen35.attention.value_length".into(),
            GgufValue::U32(HEAD_DIM as u32),
        ),
        (
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(1e-6),
        ),
        ("qwen35.rope.freq_base".into(), GgufValue::F32(1e7)),
        ("qwen35.rope.dimension_count".into(), GgufValue::U32(4)),
        (
            "qwen35.ssm.conv_kernel".into(),
            GgufValue::U32(CONV_K as u32),
        ),
        (
            "qwen35.ssm.state_size".into(),
            GgufValue::U32(SSM_DK as u32),
        ),
        (
            "qwen35.ssm.group_count".into(),
            GgufValue::U32(SSM_HEADS as u32),
        ),
        (
            "qwen35.ssm.time_step_rank".into(),
            GgufValue::U32(SSM_HEADS as u32),
        ),
        (
            "qwen35.ssm.inner_size".into(),
            GgufValue::U32(SSM_INNER as u32),
        ),
        // llama.cpp derives layer kinds from this interval (we detect by
        // tensor presence): 2 ⇒ blk.0 recurrent, blk.1 full attention.
        ("qwen35.full_attention_interval".into(), GgufValue::U32(2)),
        (
            "qwen35.rope.dimension_sections".into(),
            GgufValue::IntArray(vec![1, 1, 0, 0]),
        ),
        // Enough tokenizer metadata for llama.cpp to load this file too (the
        // f32 exact-graph oracle comparison): a gpt2-BPE vocab of 7 tokens.
        ("tokenizer.ggml.model".into(), GgufValue::Str("gpt2".into())),
        (
            "tokenizer.ggml.pre".into(),
            GgufValue::Str("default".into()),
        ),
        ("tokenizer.ggml.padding_token_id".into(), GgufValue::U32(5)),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(6)),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::IntArray(vec![1; 7]),
        ),
        ("tokenizer.ggml.merges".into(), GgufValue::StrArray(vec![])),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::StrArray(
                ["!", "\"", "0", "1", "2", "<pad>", "<eos>"]
                    .map(String::from)
                    .to_vec(),
            ),
        ),
    ];

    let qkv_rows = 2 * SSM_HEADS * SSM_DK + SSM_INNER; // q + k + v = 24
    let mut tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = Vec::new();
    let mut push = |name: &str, spec: (Vec<u64>, Vec<u8>)| {
        tensors.push((name.into(), spec.0, GgmlType::F32, spec.1));
    };

    push(
        "token_embd.weight",
        f32_tensor(&mut rng, vec![HIDDEN, 7], -0.5, 0.5),
    );
    push(
        "output_norm.weight",
        f32_tensor(&mut rng, vec![HIDDEN], 0.8, 1.2),
    );
    for i in 0..2 {
        let t = |s: &str| format!("blk.{i}.{s}");
        push(
            &t("attn_norm.weight"),
            f32_tensor(&mut rng, vec![HIDDEN], 0.8, 1.2),
        );
        push(
            &t("post_attention_norm.weight"),
            f32_tensor(&mut rng, vec![HIDDEN], 0.8, 1.2),
        );
        push(
            &t("ffn_gate.weight"),
            f32_tensor(&mut rng, vec![HIDDEN, FFN], -0.3, 0.3),
        );
        push(
            &t("ffn_up.weight"),
            f32_tensor(&mut rng, vec![HIDDEN, FFN], -0.3, 0.3),
        );
        push(
            &t("ffn_down.weight"),
            f32_tensor(&mut rng, vec![FFN, HIDDEN], -0.3, 0.3),
        );
    }
    // blk.0: DeltaNet
    push(
        "blk.0.attn_qkv.weight",
        f32_tensor(&mut rng, vec![HIDDEN, qkv_rows], -0.3, 0.3),
    );
    push(
        "blk.0.ssm_conv1d.weight",
        f32_tensor(&mut rng, vec![CONV_K, qkv_rows], -0.5, 0.5),
    );
    push(
        "blk.0.ssm_alpha.weight",
        f32_tensor(&mut rng, vec![HIDDEN, SSM_HEADS], -0.3, 0.3),
    );
    push(
        "blk.0.ssm_beta.weight",
        f32_tensor(&mut rng, vec![HIDDEN, SSM_HEADS], -0.3, 0.3),
    );
    push(
        "blk.0.ssm_a",
        f32_tensor(&mut rng, vec![SSM_HEADS], -1.0, 0.0),
    );
    push(
        "blk.0.ssm_dt.bias",
        f32_tensor(&mut rng, vec![SSM_HEADS], -0.5, 0.5),
    );
    push(
        "blk.0.ssm_norm.weight",
        f32_tensor(&mut rng, vec![SSM_INNER / SSM_HEADS], 0.8, 1.2),
    );
    push(
        "blk.0.attn_gate.weight",
        f32_tensor(&mut rng, vec![HIDDEN, SSM_INNER], -0.3, 0.3),
    );
    push(
        "blk.0.ssm_out.weight",
        f32_tensor(&mut rng, vec![SSM_INNER, HIDDEN], -0.3, 0.3),
    );
    // blk.1: gated attention
    push(
        "blk.1.attn_q.weight",
        f32_tensor(&mut rng, vec![HIDDEN, 2 * HEADS * HEAD_DIM], -0.3, 0.3),
    );
    push(
        "blk.1.attn_k.weight",
        f32_tensor(&mut rng, vec![HIDDEN, KV_HEADS * HEAD_DIM], -0.3, 0.3),
    );
    push(
        "blk.1.attn_v.weight",
        f32_tensor(&mut rng, vec![HIDDEN, KV_HEADS * HEAD_DIM], -0.3, 0.3),
    );
    push(
        "blk.1.attn_q_norm.weight",
        f32_tensor(&mut rng, vec![HEAD_DIM], 0.8, 1.2),
    );
    push(
        "blk.1.attn_k_norm.weight",
        f32_tensor(&mut rng, vec![HEAD_DIM], 0.8, 1.2),
    );
    push(
        "blk.1.attn_output.weight",
        f32_tensor(&mut rng, vec![HEADS * HEAD_DIM, HIDDEN], -0.3, 0.3),
    );

    write_gguf(&path, &kvs, &tensors).unwrap();
    path
}

fn load_tiny(seed: u64) -> Qwen35Model {
    let path = build_tiny_model(seed);
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
    let deep: Vec<u8> = (0..40).map(|i| (i % 2) as u8).collect();
    m.learn_symbols(&deep);
    m.revert_learned_symbols(deep.len());

    assert!(m.deep_replays() >= 1, "eviction must have forced a replay");
    assert_eq!(m.root_log_probability().to_bits(), root_after_one.to_bits());
    assert_eq!(
        m.predict_bit_probability(1).to_bits(),
        p_after_one.to_bits()
    );
}
