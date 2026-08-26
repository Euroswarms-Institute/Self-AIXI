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

/// Is byte `b` one GPT-2's byte-level BPE maps to its own code point?
/// (printable ASCII and most of Latin-1; everything else gets remapped).
fn gpt2_direct_byte(b: u8) -> bool {
    (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
}

/// GPT-2's `bytes_to_unicode`: the printable-substitute character under
/// which raw byte `b` appears in byte-level BPE vocabulary strings
/// (space → 'Ġ', newline → 'Ċ', ...). Qwen's tokenizer inherits this map.
pub fn gpt2_byte_char(b: u8) -> char {
    if gpt2_direct_byte(b) {
        return b as char;
    }
    let mut n = 0u32;
    for x in 0..=255u8 {
        if gpt2_direct_byte(x) {
            continue;
        }
        if x == b {
            return char::from_u32(256 + n).unwrap();
        }
        n += 1;
    }
    unreachable!("every byte is direct or remapped")
}

/// The byte carve's tokenizer reduction: the 256 single-byte token ids (the
/// base alphabet of the byte-level BPE — always present in the vocabulary),
/// the stream prime, and the per-token first-byte map the full-vocabulary
/// marginalization buckets with.
#[derive(Clone, Debug)]
pub struct ByteProbe {
    /// `byte_tokens[b]` = token id whose string is byte b's mapped char.
    pub byte_tokens: [u32; 256],
    pub prime: u32,
    /// For every vocabulary entry: the raw byte its string starts with, or
    /// None for control/special tokens and non-text entries. P(next byte=b)
    /// = Σ P(token t) over t with first_byte[t] == Some(b).
    pub first_byte: Vec<Option<u8>>,
}

impl ByteProbe {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let tokens = match gguf.kvs.get("tokenizer.ggml.tokens") {
            Some(GgufValue::StrArray(v)) => v,
            _ => return Err("tokenizer.ggml.tokens missing from metadata".into()),
        };
        // llama token types: 1 = NORMAL, 6 = BYTE are text; CONTROL and
        // USER_DEFINED specials ("<|endoftext|>", ...) must not be bucketed
        // by their literal first char. Absent array ⇒ treat all as text.
        let token_types = match gguf.kvs.get("tokenizer.ggml.token_type") {
            Some(GgufValue::IntArray(v)) => Some(v),
            _ => None,
        };
        let is_text = |i: usize| -> bool {
            token_types.is_none_or(|tt| matches!(tt.get(i), Some(1) | Some(6) | None))
        };
        let mut char_to_byte = std::collections::HashMap::new();
        for b in 0..=255u8 {
            char_to_byte.insert(gpt2_byte_char(b), b);
        }
        // One pass over the vocabulary: the single-char byte tokens (input
        // side) and every token's first byte (output-side bucketing).
        let mut byte_tokens = [u32::MAX; 256];
        let mut first_byte = vec![None; tokens.len()];
        for (i, t) in tokens.iter().enumerate() {
            if !is_text(i) {
                continue;
            }
            let mut chars = t.chars();
            let Some(c0) = chars.next() else { continue };
            let Some(&b0) = char_to_byte.get(&c0) else {
                continue;
            };
            first_byte[i] = Some(b0);
            if chars.next().is_none() && byte_tokens[b0 as usize] == u32::MAX {
                byte_tokens[b0 as usize] = i as u32;
            }
        }
        if let Some(missing) = byte_tokens.iter().position(|&t| t == u32::MAX) {
            return Err(format!(
                "vocabulary has no single-byte token for byte {missing:#04x} — \
                 not a byte-level BPE vocab?"
            ));
        }
        let prime = gguf
            .kv_u64("tokenizer.ggml.padding_token_id")
            .ok()
            .or_else(|| gguf.kv_u64("tokenizer.ggml.eos_token_id").ok())
            .map(|v| v as u32)
            .ok_or("no padding/eos token to prime the stream with")?;
        Ok(ByteProbe {
            byte_tokens,
            prime,
            first_byte,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::gpt2_byte_char;

    #[test]
    fn gpt2_byte_map_known_vectors() {
        // Identity range.
        assert_eq!(gpt2_byte_char(b'0'), '0');
        assert_eq!(gpt2_byte_char(b'1'), '1');
        assert_eq!(gpt2_byte_char(b'!'), '!');
        assert_eq!(gpt2_byte_char(b'~'), '~');
        assert_eq!(gpt2_byte_char(0xFF), '\u{FF}');
        // Remapped controls: the classic constants.
        assert_eq!(gpt2_byte_char(b' '), '\u{120}'); // Ġ
        assert_eq!(gpt2_byte_char(b'\n'), '\u{10A}'); // Ċ
        assert_eq!(gpt2_byte_char(b'\t'), '\u{109}'); // ĉ
        assert_eq!(gpt2_byte_char(0x00), '\u{100}'); // Ā
                                                     // The map is injective over all 256 bytes.
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            assert!(seen.insert(gpt2_byte_char(b)));
        }
    }
}
