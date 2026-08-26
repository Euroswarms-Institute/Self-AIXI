//! `LlmByteModel` — the byte carve as an MC-AIXI environment-model component.
//!
//! Where `LlmModel` maps every stream bit to a bit token, this component
//! knows the cycle layout (action bits, observation bits, reward bits) and
//! feeds the network *observed bytes only*: the token stream is the prime
//! followed by one byte token per completed observation byte, i.e. exactly
//! the text the environment emits.
//!
//! The output side keeps the **full tied unembedding**: a trained BPE model
//! concentrates next-token mass on merged multi-character tokens (" agent",
//! "The"), so renormalizing over the 256 single-byte rows alone measures the
//! wrong slice of the distribution (measured: ~10 nats per word-internal
//! letter). Instead P(next byte = b) is the token-healing marginal — the
//! full-vocabulary softmax bucketed by each token's first byte
//! (`ByteProbe::first_byte`), control/special tokens contributing to the
//! normalizer but no bucket. The eight bit conditionals renormalize within
//! the byte, so the per-cycle observation model is a proper measure over the
//! 256 bytes. This is the one place the amputated 508.6M-parameter head
//! comes back: one fused-quantized GEMV per *observed* byte.
//!
//! Per-bit semantics against the six-method `EnvModel` contract:
//! - **action-phase bits** advance the cycle position only. They carry no
//!   token and predict 0.5 (they are appended by the agent, never learned,
//!   but the fallback keeps the semimeasure well-defined either way);
//! - **observation-phase bits** are predicted by marginalizing the cached
//!   256-way next-byte distribution over the bytes consistent with the bits
//!   already pending (MSB-first, so each marginal is one contiguous range
//!   sum); the 8th bit completes the byte and pushes its token. The chain
//!   rule telescopes: eight bit log-probs sum to the byte's log-prob;
//! - **reward-phase bits** get one order-0 KT estimator per position —
//!   honest online adaptation for scalar bookkeeping the base model has no
//!   prior about, with exact integer-count revert.
//!
//! Laziness matters twice: tokens advance through the network only when an
//! observation-phase prediction actually needs the next distribution, so (a)
//! completing a byte during an imagined rollout costs nothing if it is
//! reverted before the next byte is examined, which makes exact byte-tree
//! enumeration (`planning::modal_byte`) cache arithmetic instead of forward
//! passes, and (b) revert stays the checkpoint-stack pop it is for
//! `LlmModel`, with the same deterministic full-replay fallback.

use super::config::{ByteProbe, Qwen35Config, TokenProbe};
use super::gguf::GgufFile;
use super::model::{QGateLayout, Qwen35Model};
use super::state::LlmState;
use super::tensor::QTensor;
use crate::models::EnvModel;
use std::path::Path;

/// Open a GGUF and carve it for byte streams: the 256 single-byte token
/// embeddings on the input side, and the full quantized-resident tied
/// unembedding (`token_embd`) as the marginalization head.
pub fn load_byte_carved(
    path: &Path,
    layout: QGateLayout,
) -> Result<(Qwen35Model, ByteProbe, QTensor), String> {
    let gguf = GgufFile::open(path)?;
    let cfg = Qwen35Config::from_gguf(&gguf)?;
    let probe = TokenProbe::from_gguf(&gguf)?;
    let byte_probe = ByteProbe::from_gguf(&gguf)?;
    let model = Qwen35Model::from_gguf_carved(&gguf, cfg, probe, layout, &byte_probe.byte_tokens)?;
    let head = QTensor::from_gguf(&gguf, "token_embd.weight")?;
    Ok((model, byte_probe, head))
}

#[derive(Clone, Copy)]
struct Event {
    learned: bool,
    bit: u8,
}

pub struct LlmByteModel {
    model: Qwen35Model,
    state: LlmState,
    probe: ByteProbe,
    /// The full tied unembedding, quantized-resident.
    head: QTensor,
    /// Reused full-vocabulary logit buffer for the head GEMV.
    logits_buf: Vec<f32>,
    action_bits: usize,
    obs_bits: usize,
    reward_bits: usize,
    /// Prime + one token per completed observation byte.
    tokens: Vec<u32>,
    /// First-byte-bucketed softmax over the full vocabulary after each
    /// advanced position: dists[p][b] = P(next token starts with byte b).
    /// Plain probabilities so bit marginals are pure range sums; entries
    /// sum to slightly below 1 (control tokens keep their normalizer mass
    /// but have no bucket) and the bit chain renormalizes within the byte.
    dists: Vec<Vec<f64>>,
    /// One entry per stream symbol; phase is index arithmetic, so this plus
    /// the bit value reconstructs every side effect on revert.
    events: Vec<Event>,
    /// Observation bits pending toward the current byte, MSB-first.
    pending: Vec<u8>,
    /// KT counts per reward-bit position: [zeros, ones].
    kt: Vec<[u64; 2]>,
    /// Cumulative ln ρ per learned symbol (popped on revert — exact).
    loglik_prefix: Vec<f64>,
}

/// Softmax over the full vocabulary, bucketed by first byte (None buckets —
/// control/special tokens — count in the normalizer only).
fn bucket_softmax(logits: &[f32], first_byte: &[Option<u8>]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let mut z = 0f64;
    let mut buckets = vec![0f64; 256];
    for (&l, &fb) in logits.iter().zip(first_byte) {
        let e = ((l as f64) - max).exp();
        z += e;
        if let Some(b) = fb {
            buckets[b as usize] += e;
        }
    }
    for v in buckets.iter_mut() {
        *v /= z;
    }
    buckets
}

impl LlmByteModel {
    pub fn new(
        model: Qwen35Model,
        probe: ByteProbe,
        head: QTensor,
        action_bits: usize,
        obs_bits: usize,
        reward_bits: usize,
    ) -> Self {
        assert_eq!(obs_bits, 8, "the byte carve models 8-bit observations");
        assert_eq!(
            head.rows,
            probe.first_byte.len(),
            "head rows must cover the vocabulary"
        );
        assert_eq!(head.cols, model.cfg.hidden, "head width != hidden");
        let state = model.new_state();
        let prime = probe.prime;
        let vocab = head.rows;
        let mut m = LlmByteModel {
            model,
            state,
            probe,
            head,
            logits_buf: vec![0f32; vocab],
            action_bits,
            obs_bits,
            reward_bits,
            tokens: vec![prime],
            dists: Vec::new(),
            events: Vec::new(),
            pending: Vec::new(),
            kt: vec![[0, 0]; reward_bits],
            loglik_prefix: Vec::new(),
        };
        m.ensure_advanced();
        m
    }

    pub fn deep_replays(&self) -> u64 {
        self.state.deep_replays
    }

    /// Tokens currently conditioning the model (prime included).
    pub fn context_len(&self) -> usize {
        self.tokens.len()
    }

    fn total_bits(&self) -> usize {
        self.action_bits + self.obs_bits + self.reward_bits
    }

    /// Cycle phase of the NEXT symbol.
    fn phase(&self) -> usize {
        self.events.len() % self.total_bits()
    }

    fn ensure_advanced(&mut self) {
        assert!(
            self.tokens.len() <= self.model.cfg.ctx_len,
            "history of {} tokens exceeds the model context window {}",
            self.tokens.len(),
            self.model.cfg.ctx_len
        );
        while self.state.pos < self.tokens.len() {
            let t = self.tokens[self.state.pos];
            let hidden = self.model.advance_hidden(&mut self.state, t);
            self.head.gemv(&hidden, &mut self.logits_buf);
            self.dists
                .push(bucket_softmax(&self.logits_buf, &self.probe.first_byte));
        }
    }

    /// P(next observation bit = 1) by marginalizing the byte distribution
    /// over the contiguous range consistent with the pending prefix.
    fn obs_p_one(&mut self) -> f64 {
        self.ensure_advanced();
        let dist = &self.dists[self.state.pos - 1];
        let k = self.pending.len();
        let width = self.obs_bits - k; // bits still free, incl. this one
        let prefix = self
            .pending
            .iter()
            .fold(0usize, |v, &b| (v << 1) | b as usize);
        let lo = prefix << width;
        let mid = lo + (1 << (width - 1));
        let hi = lo + (1 << width);
        let s0: f64 = dist[lo..mid].iter().sum();
        let s1: f64 = dist[mid..hi].iter().sum();
        if s0 + s1 <= 0.0 {
            0.5 // the whole prefix underflowed; stay a semimeasure
        } else {
            s1 / (s0 + s1)
        }
    }

    fn p_one(&mut self) -> f64 {
        let ph = self.phase();
        if ph < self.action_bits {
            0.5
        } else if ph < self.action_bits + self.obs_bits {
            self.obs_p_one()
        } else {
            let [c0, c1] = self.kt[ph - self.action_bits - self.obs_bits];
            (c1 as f64 + 0.5) / ((c0 + c1) as f64 + 1.0)
        }
    }

    fn push_symbol(&mut self, bit: u8, learned: bool) {
        debug_assert!(bit <= 1);
        let ph = self.phase();
        let obs_end = self.action_bits + self.obs_bits;
        if ph >= self.action_bits && ph < obs_end {
            self.pending.push(bit);
            if ph == obs_end - 1 {
                let byte = self
                    .pending
                    .iter()
                    .fold(0usize, |v, &b| (v << 1) | b as usize);
                self.tokens.push(self.probe.byte_tokens[byte]);
                self.pending.clear();
            }
        } else if ph >= obs_end && learned {
            self.kt[ph - obs_end][bit as usize] += 1;
        }
        self.events.push(Event { learned, bit });
    }

    fn pop_symbol(&mut self, expect_learned: bool) {
        let ev = self.events.pop().expect("revert past the stream start");
        assert_eq!(
            ev.learned, expect_learned,
            "revert out of LIFO order (learned/history mismatch)"
        );
        if ev.learned {
            self.loglik_prefix.pop();
        }
        let ph = self.events.len() % self.total_bits(); // phase of the popped symbol
        let obs_end = self.action_bits + self.obs_bits;
        if ph >= self.action_bits && ph < obs_end {
            if ph == obs_end - 1 {
                // This bit had completed a byte: un-push its token and
                // restore the other seven bits (the immediately preceding
                // events — phases within one byte are consecutive symbols).
                self.tokens.pop();
                let start = self.events.len() - (self.obs_bits - 1);
                self.pending = self.events[start..].iter().map(|e| e.bit).collect();
            } else {
                self.pending.pop();
            }
        } else if ph >= obs_end && ev.learned {
            let c = &mut self.kt[ph - obs_end][ev.bit as usize];
            *c = c.checked_sub(1).expect("KT count underflow on revert");
        }
    }

    fn rollback_state(&mut self) {
        if self.state.pos <= self.tokens.len() {
            return;
        }
        let target = self.tokens.len();
        if self.state.revert_to(target) {
            self.dists.truncate(target);
        } else {
            // Checkpoint evicted: deterministic full replay (exact, slow).
            self.state = self.model.new_state();
            self.state.deep_replays += 1;
            self.dists.clear();
            let tokens = std::mem::take(&mut self.tokens);
            for &t in &tokens {
                let hidden = self.model.advance_hidden(&mut self.state, t);
                self.head.gemv(&hidden, &mut self.logits_buf);
                self.dists
                    .push(bucket_softmax(&self.logits_buf, &self.probe.first_byte));
            }
            self.tokens = tokens;
        }
    }
}

impl EnvModel for LlmByteModel {
    fn root_log_probability(&self) -> f64 {
        self.loglik_prefix.last().copied().unwrap_or(0.0)
    }

    fn predict_bit_probability(&mut self, bit: u8) -> f64 {
        let p1 = self.p_one();
        if bit == 1 {
            p1
        } else {
            1.0 - p1
        }
    }

    fn learn_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            let p = self.predict_bit_probability(b).max(1e-300);
            self.loglik_prefix
                .push(self.root_log_probability() + p.ln());
            self.push_symbol(b, true);
        }
    }

    fn append_history_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            self.push_symbol(b, false);
        }
    }

    fn revert_learned_symbols(&mut self, n: usize) {
        for _ in 0..n {
            self.pop_symbol(true);
        }
        self.rollback_state();
    }

    fn revert_history_symbols(&mut self, n: usize) {
        for _ in 0..n {
            self.pop_symbol(false);
        }
        self.rollback_state();
    }

    fn model_id(&self) -> String {
        "byte-llm".to_string()
    }
}
