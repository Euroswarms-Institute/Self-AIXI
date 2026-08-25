//! Writer → parser round-trip for the hand-rolled GGUF container, plus the
//! qwen35 config extraction from synthetic metadata.

use mc_aixi::llm::config::{Qwen35Config, TokenProbe};
use mc_aixi::llm::gguf::{write_gguf, GgufFile, GgufValue};
use mc_aixi::llm::quant::GgmlType;
use mc_aixi::llm::tensor::QTensor;
use mc_aixi::rng::seeded;
use rand::Rng;

fn synthetic_kvs() -> Vec<(String, GgufValue)> {
    let mut kvs: Vec<(String, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufValue::Str("qwen35".into()),
        ),
        ("qwen35.block_count".into(), GgufValue::U32(3)),
        ("qwen35.context_length".into(), GgufValue::U32(4096)),
        ("qwen35.embedding_length".into(), GgufValue::U32(16)),
        ("qwen35.feed_forward_length".into(), GgufValue::U32(32)),
        ("qwen35.attention.head_count".into(), GgufValue::U32(2)),
        ("qwen35.attention.head_count_kv".into(), GgufValue::U32(1)),
        ("qwen35.attention.key_length".into(), GgufValue::U32(8)),
        ("qwen35.attention.value_length".into(), GgufValue::U32(8)),
        (
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(1e-6),
        ),
        ("qwen35.rope.freq_base".into(), GgufValue::F32(1e7)),
        ("qwen35.rope.dimension_count".into(), GgufValue::U32(4)),
        ("qwen35.ssm.conv_kernel".into(), GgufValue::U32(4)),
        ("qwen35.ssm.state_size".into(), GgufValue::U32(4)),
        ("qwen35.ssm.group_count".into(), GgufValue::U32(2)),
        ("qwen35.ssm.inner_size".into(), GgufValue::U32(8)),
        (
            "qwen35.rope.dimension_sections".into(),
            GgufValue::IntArray(vec![1, 1, 0, 0]),
        ),
        ("tokenizer.ggml.padding_token_id".into(), GgufValue::U32(5)),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(6)),
    ];
    let tokens: Vec<String> = vec![
        "!".into(),
        "\"".into(),
        "0".into(),
        "1".into(),
        "2".into(),
        "<pad>".into(),
        "<eos>".into(),
    ];
    kvs.push(("tokenizer.ggml.tokens".into(), GgufValue::StrArray(tokens)));
    kvs
}

#[test]
fn roundtrip_metadata_and_tensors() {
    let mut rng = seeded(4);
    let dir = std::env::temp_dir().join("mc_aixi_gguf_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.gguf");

    // token_embd: 7 vocab rows × 16 (f32); one Q8_0 tensor 2 rows × 64.
    let embd: Vec<u8> = (0..7 * 16)
        .flat_map(|_| rng.random_range(-1.0f32..1.0).to_le_bytes())
        .collect();
    let mut q8: Vec<u8> = vec![0; GgmlType::Q8_0.row_bytes(64).unwrap() * 2];
    rng.fill(&mut q8[..]);
    // Random bytes are valid quant payloads except the f16 scale, which
    // could be NaN/Inf — pin each block's scale to a finite value.
    for block in q8.chunks_exact_mut(GgmlType::Q8_0.block_bytes()) {
        block[0..2].copy_from_slice(
            &half::f16::from_f32(rng.random_range(0.01f32..0.1))
                .to_bits()
                .to_le_bytes(),
        );
    }

    write_gguf(
        &path,
        &synthetic_kvs(),
        &[
            (
                "token_embd.weight".into(),
                vec![16, 7],
                GgmlType::F32,
                embd.clone(),
            ),
            (
                "blk.0.test.weight".into(),
                vec![64, 2],
                GgmlType::Q8_0,
                q8.clone(),
            ),
        ],
    )
    .unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.kv_str("general.architecture").unwrap(), "qwen35");
    assert_eq!(gguf.kv_u64("qwen35.block_count").unwrap(), 3);
    assert!((gguf.kv_f32("qwen35.rope.freq_base").unwrap() - 1e7).abs() < 1.0);
    match gguf.kvs.get("qwen35.rope.dimension_sections") {
        Some(GgufValue::IntArray(v)) => assert_eq!(v, &vec![1, 1, 0, 0]),
        other => panic!("sections not retained: {other:?}"),
    }

    let te = gguf.tensor("token_embd.weight").unwrap();
    assert_eq!((te.cols(), te.rows()), (16, 7));
    assert_eq!(gguf.tensor_data(te).unwrap(), &embd[..]);
    let qt = gguf.tensor("blk.0.test.weight").unwrap();
    assert_eq!(gguf.tensor_data(qt).unwrap(), &q8[..]);

    // Config + probe extraction.
    let cfg = Qwen35Config::from_gguf(&gguf).unwrap();
    assert_eq!((cfg.n_layers, cfg.hidden, cfg.vocab), (3, 16, 7));
    assert_eq!(cfg.ssm_head_v(), 4);
    let probe = TokenProbe::from_gguf(&gguf).unwrap();
    assert_eq!((probe.bit0, probe.bit1, probe.prime), (2, 3, 5));

    // QTensor loading straight from the container.
    let t = QTensor::from_gguf(&gguf, "blk.0.test.weight").unwrap();
    assert_eq!((t.rows, t.cols), (2, 64));
    let mut out = vec![0f32; 2];
    let x = vec![1.0f32; 64];
    t.gemv(&x, &mut out);
    let naive: f32 = t.dequant_row_f32(0).iter().sum();
    assert!((out[0] - naive).abs() < 1e-4);
}
