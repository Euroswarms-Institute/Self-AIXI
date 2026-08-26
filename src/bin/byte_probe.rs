//! Dev diagnostic: run the byte-carved model alone (no mixture, no CTW)
//! over the embedded corpus as a percept stream and report windowed
//! nats/byte and modal-guess accuracy. This is the tool that exposed the
//! naive 256-row carve's failure (renormalizing over single-byte tokens
//! costs ~10 nats per word-internal letter) and measured the in-context
//! adaptation of the token-healing marginal (3.7 → 2.3 nats/byte over the
//! first 400 bytes).
//!
//! Usage: byte_probe [N_BYTES]   (default 40; wants models/*.gguf fetched)

use mc_aixi::env::text_bytes::EMBEDDED_CORPUS;
use mc_aixi::llm::byte_model::{load_byte_carved, LlmByteModel};
use mc_aixi::llm::model::QGateLayout;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::modal_byte::byte_observation_marginal;
use std::path::PathBuf;

fn bits(b: u8) -> Vec<u8> {
    (0..8).rev().map(|k| (b >> k) & 1).collect()
}

fn main() {
    let gguf = PathBuf::from("models/Qwen3.8-2B-Q4_K_M.gguf");
    let (model, probe, head) = load_byte_carved(&gguf, QGateLayout::PerHeadInterleaved).unwrap();
    // Report a few byte-token ids for sanity.
    for c in [b'0', b'1', b'T', b'h', b'e', b' '] {
        eprintln!("byte {:?} -> token {}", c as char, probe.byte_tokens[c as usize]);
    }
    eprintln!(
        "head: {}x{} {} ({:.1} MiB), text tokens {}",
        head.rows,
        head.cols,
        head.ty.name(),
        head.byte_len() as f64 / (1024.0 * 1024.0),
        probe.first_byte.iter().flatten().count()
    );
    let mut m = LlmByteModel::new(model, probe, head, 8, 8, 1);
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let corpus = EMBEDDED_CORPUS.as_bytes();
    let mut window_nats = 0.0;
    let mut window_hits = 0u32;
    for (i, &b) in corpus.iter().take(n).enumerate() {
        // Dummy action byte (the agent's guess): append zeros.
        m.append_history_symbols(&bits(0));
        let marginal = byte_observation_marginal(&mut m);
        let modal = marginal
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.total_cmp(c.1))
            .unwrap()
            .0 as u8;
        let before = m.root_log_probability();
        m.learn_symbols(&bits(b));
        let nats = -(m.root_log_probability() - before);
        window_nats += nats;
        window_hits += u32::from(modal == b);
        m.learn_symbols(&[u8::from(modal == b)]);
        if (i + 1) % 50 == 0 {
            println!(
                "bytes {:>3}-{:>3}: {:.2} nats/byte, modal accuracy {:.2}",
                i + 1 - 49,
                i + 1,
                window_nats / 50.0,
                window_hits as f64 / 50.0
            );
            window_nats = 0.0;
            window_hits = 0;
        }
    }
}
