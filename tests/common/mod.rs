//! Shared synthetic-checkpoint builder for the LLM integration tests: a
//! tiny random-weight qwen35 hybrid (1 DeltaNet + 1 attention block)
//! written through the crate's own GGUF writer. The vocabulary is
//! parameterized: the classic 7-token bit vocab, or a byte-level BPE vocab
//! carrying all 256 single-byte tokens for the byte-carve tests.

// Each integration-test binary compiles this module separately and uses
// only the pieces it needs; dead-code analysis is per-binary noise here.
#![allow(dead_code)]

use mc_aixi::llm::config::gpt2_byte_char;
use mc_aixi::llm::gguf::{write_gguf, GgufValue};
use mc_aixi::llm::quant::GgmlType;
use std::path::PathBuf;

pub const HIDDEN: u64 = 16;
pub const FFN: u64 = 32;
pub const HEADS: u64 = 2;
pub const KV_HEADS: u64 = 1;
pub const HEAD_DIM: u64 = 8;
pub const SSM_HEADS: u64 = 2;
pub const SSM_DK: u64 = 4;
pub const SSM_INNER: u64 = 8; // 2 heads × d_v 4
pub const CONV_K: u64 = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Vocab {
    /// 7 tokens: ["!", "\"", "0", "1", "2", "<pad>", "<eos>"].
    Bits,
    /// 258 tokens: the 256 GPT-2 byte chars, then "<pad>", "<eos>".
    Bytes,
}

fn f32_tensor(
    rng: &mut mc_aixi::rng::AgentRng,
    ne: Vec<u64>,
    lo: f32,
    hi: f32,
) -> (Vec<u64>, Vec<u8>) {
    use rand::Rng;
    let n: u64 = ne.iter().product();
    let data = (0..n)
        .flat_map(|_| rng.random_range(lo..hi).to_le_bytes())
        .collect();
    (ne, data)
}

pub fn build_tiny_model(seed: u64, vocab: Vocab) -> PathBuf {
    let mut rng = mc_aixi::rng::seeded(seed);
    let dir = std::env::temp_dir().join("mc_aixi_llm_synth");
    std::fs::create_dir_all(&dir).unwrap();
    let tag = match vocab {
        Vocab::Bits => "bits",
        Vocab::Bytes => "bytes",
    };
    let path = dir.join(format!("tiny-{tag}-{seed}.gguf"));

    let tokens: Vec<String> = match vocab {
        Vocab::Bits => ["!", "\"", "0", "1", "2", "<pad>", "<eos>"]
            .map(String::from)
            .to_vec(),
        Vocab::Bytes => (0..=255u8)
            .map(|b| gpt2_byte_char(b).to_string())
            .chain(["<pad>".to_string(), "<eos>".to_string()])
            .collect(),
    };
    let n_vocab = tokens.len() as u64;
    let (pad_id, eos_id) = (n_vocab as u32 - 2, n_vocab as u32 - 1);

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
        // f32 exact-graph oracle comparison).
        ("tokenizer.ggml.model".into(), GgufValue::Str("gpt2".into())),
        (
            "tokenizer.ggml.pre".into(),
            GgufValue::Str("default".into()),
        ),
        (
            "tokenizer.ggml.padding_token_id".into(),
            GgufValue::U32(pad_id),
        ),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(eos_id)),
        (
            "tokenizer.ggml.token_type".into(),
            // Text tokens are NORMAL (1); <pad>/<eos> are CONTROL (3) like
            // the real checkpoint, so first-byte bucketing must skip them.
            GgufValue::IntArray(
                (0..n_vocab)
                    .map(|i| if i >= n_vocab - 2 { 3 } else { 1 })
                    .collect(),
            ),
        ),
        ("tokenizer.ggml.merges".into(), GgufValue::StrArray(vec![])),
        ("tokenizer.ggml.tokens".into(), GgufValue::StrArray(tokens)),
    ];

    let qkv_rows = 2 * SSM_HEADS * SSM_DK + SSM_INNER; // q + k + v = 24
    let mut tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = Vec::new();
    let mut push = |name: &str, spec: (Vec<u64>, Vec<u8>)| {
        tensors.push((name.into(), spec.0, GgmlType::F32, spec.1));
    };

    push(
        "token_embd.weight",
        f32_tensor(&mut rng, vec![HIDDEN, n_vocab], -0.5, 0.5),
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
