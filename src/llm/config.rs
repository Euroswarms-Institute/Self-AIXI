//! Architecture configuration and the bit-token probe, both extracted from
//! GGUF metadata — the checkpoint fully describes itself; no config.json,
//! no tokenizer runtime.

use super::gguf::{GgufFile, GgufValue};

/// The `qwen35` hybrid architecture hyperparameters (verified against the
/// parsed metadata of empero-ai/Qwen3.8-2B-Distill-GGUF: 25 blocks, 18
/// Gated-DeltaNet + 7 gated-attention, hidden 2048).
#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Per-head key/value width of the full-attention blocks (256 here).
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Rotary dims per head (partial RoPE: 64 of 256).
    pub rope_dims: usize,
    pub ctx_len: usize,
    pub vocab: usize,
    /// DeltaNet: depthwise causal conv width over the fused qkv channels.
    pub ssm_conv_kernel: usize,
    /// DeltaNet: per-head key/query width (d_k = 128).
    pub ssm_state_size: usize,
    /// DeltaNet: number of key/query heads (`group_count`, 16).
    pub ssm_heads: usize,
    /// DeltaNet: number of value heads (`time_step_rank` in GGUF terms, 16;
    /// llama.cpp's num_v_heads).
    pub ssm_v_heads: usize,
    /// DeltaNet: total value width (inner = v_heads · d_v = 2048).
    pub ssm_inner: usize,
}

impl Qwen35Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.kv_str("general.architecture")?;
        if arch != "qwen35" {
            return Err(format!(
                "architecture '{arch}' is not qwen35 — this dissection targets the \
                 Qwen3.8/Qwen3.5 hybrid family"
            ));
        }
        let p = |k: &str| gguf.kv_u64(&format!("{arch}.{k}"));
        let f = |k: &str| gguf.kv_f32(&format!("{arch}.{k}"));
        let cfg = Qwen35Config {
            n_layers: p("block_count")? as usize,
            hidden: p("embedding_length")? as usize,
            ffn: p("feed_forward_length")? as usize,
            n_heads: p("attention.head_count")? as usize,
            n_kv_heads: p("attention.head_count_kv")? as usize,
            head_dim: p("attention.key_length")? as usize,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            rope_theta: f("rope.freq_base")?,
            rope_dims: p("rope.dimension_count")? as usize,
            ctx_len: p("context_length")? as usize,
            vocab: gguf
                .tensor("token_embd.weight")
                .map(|t| t.rows())
                .ok_or("token_embd.weight tensor missing")?,
            ssm_conv_kernel: p("ssm.conv_kernel")? as usize,
            ssm_state_size: p("ssm.state_size")? as usize,
            ssm_heads: p("ssm.group_count")? as usize,
            ssm_v_heads: p("ssm.time_step_rank")? as usize,
            ssm_inner: p("ssm.inner_size")? as usize,
        };
        if p("attention.value_length")? as usize != cfg.head_dim {
            return Err("attention value_length != key_length is unsupported".into());
        }
        if !cfg.ssm_inner.is_multiple_of(cfg.ssm_v_heads) {
            return Err("ssm inner size not divisible by value-head count".into());
        }
        if cfg.ssm_heads != cfg.ssm_v_heads {
            // The general family allows fewer k-heads repeated across
            // v-heads; qwen3_5 uses symmetric heads, so keep the code honest.
            return Err("k-head/v-head repeat is not implemented (k != v head count)".into());
        }
        if cfg.rope_dims > cfg.head_dim {
            return Err("rope dims exceed head dim".into());
        }
        Ok(cfg)
    }

    /// DeltaNet per-head value width d_v (128 here).
    pub fn ssm_head_v(&self) -> usize {
        self.ssm_inner / self.ssm_v_heads
    }
}

/// The whole tokenizer, reduced to four integers.
#[derive(Clone, Copy, Debug)]
pub struct TokenProbe {
    /// Token id whose string is literally "0".
    pub bit0: u32,
    /// Token id whose string is literally "1".
    pub bit1: u32,
    /// Stream-prime token fed at position 0 (this family has no BOS; the
    /// padding/endoftext id anchors the empty history).
    pub prime: u32,
    pub eos: Option<u32>,
}

impl TokenProbe {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let tokens = match gguf.kvs.get("tokenizer.ggml.tokens") {
            Some(GgufValue::StrArray(v)) => v,
            _ => return Err("tokenizer.ggml.tokens missing from metadata".into()),
        };
        let find =
            |s: &str| -> Option<u32> { tokens.iter().position(|t| t == s).map(|i| i as u32) };
        let bit0 = find("0").ok_or("tokenizer has no literal \"0\" token")?;
        let bit1 = find("1").ok_or("tokenizer has no literal \"1\" token")?;
        let eos = gguf
            .kv_u64("tokenizer.ggml.eos_token_id")
            .ok()
            .map(|v| v as u32);
        let prime = gguf
            .kv_u64("tokenizer.ggml.padding_token_id")
            .ok()
            .map(|v| v as u32)
            .or(eos)
            .ok_or("no padding/eos token to prime the stream with")?;
        Ok(TokenProbe {
            bit0,
            bit1,
            prime,
            eos,
        })
    }

    pub fn token_for_bit(&self, bit: u8) -> u32 {
        if bit == 0 {
            self.bit0
        } else {
            self.bit1
        }
    }
}
