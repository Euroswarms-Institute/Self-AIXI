//! The dissected qwen3_5 hybrid forward pass — hand-rolled from the GGUF
//! tensors and the published architecture (HF `Qwen3_5ForConditionalGeneration`
//! text config; Gated DeltaNet after Yang et al. 2024), with the surgery
//! applied at load time:
//!
//! - vision tower: absent from the GGUF already; multimodal machinery ignored;
//! - MTP block(s): the last `nextn_predict_layers` blocks are never loaded;
//! - vocabulary: only the bit-token and stream-prime rows of `token_embd`
//!   are dequantized (input side), and the tied unembedding is carved to the
//!   two bit rows — the model's output IS a 2-logit distribution.
//!
//! Per block (pre-norm residual): x += mixer(rmsnorm(x)); x += ffn(rmsnorm(x)).
//! Mixer is either gated GQA attention (QK-norm → partial RoPE → causal
//! softmax over the KV arena → per-head sigmoid output gate) or Gated
//! DeltaNet (causal conv-4 + SiLU over fused qkv → per-head L2-normalized
//! q,k → gated delta-rule state update S ← γS(I − βkkᵀ) + βvkᵀ, o = Sq →
//! per-head gated RMSNorm with SiLU(z)).

use super::config::{Qwen35Config, TokenProbe};
use super::gguf::GgufFile;
use super::quant::dequant_row;
use super::rope::Rope;
use super::state::{LlmState, StateShape};
use super::tensor::QTensor;
use std::collections::HashMap;
use std::path::Path;

/// Row layout of the fused q|gate projection in attention blocks.
/// `PerHeadInterleaved` is the raw HF layout (head h = [q_h(256) | g_h(256)]);
/// `Blocked` is [all q | all gates]. The oracle script pins the truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QGateLayout {
    PerHeadInterleaved,
    Blocked,
}

struct Ffn {
    gate: QTensor,
    up: QTensor,
    down: QTensor,
}

struct LayerCommon {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    ffn: Ffn,
}

struct DeltaNetLayer {
    common: LayerCommon,
    qkv: QTensor,
    /// conv_w[channel · taps + t], t oldest→current over `taps` taps.
    conv_w: Vec<f32>,
    alpha: QTensor,
    beta: QTensor,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    gate: QTensor,
    out: QTensor,
}

struct AttentionLayer {
    common: LayerCommon,
    q: QTensor,
    k: QTensor,
    v: QTensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o: QTensor,
}

enum Layer {
    DeltaNet(Box<DeltaNetLayer>),
    Attention(Box<AttentionLayer>),
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    pub probe: TokenProbe,
    pub q_gate_layout: QGateLayout,
    /// Layers actually run (MTP blocks amputated).
    pub n_active_layers: usize,
    rope: Rope,
    embed_rows: HashMap<u32, Vec<f32>>,
    unembed: [Vec<f32>; 2],
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / ((ms as f32) + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
}

fn rmsnorm_head(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / ((ms as f32) + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v *= inv * w;
    }
}

/// L2 normalization with the ggml semantics: eps floors the *norm* itself
/// (scale = 1/max(‖x‖, eps)), matching `ggml_l2_norm`.
fn l2norm_head(x: &mut [f32], eps: f32) {
    let ss: f64 = x.iter().map(|v| (v * v) as f64).sum();
    let inv = 1.0 / (ss as f32).sqrt().max(eps);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Diagnostic tensor trace, enabled by MC_AIXI_TRACE=1: prints per-tensor
/// sums in the same shape as llama.cpp's eval-callback, for oracle debugging.
fn trace(name: &str, layer: isize, x: &[f32]) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("MC_AIXI_TRACE").is_some()) {
        let sum: f64 = x.iter().map(|&v| v as f64).sum();
        if layer >= 0 {
            eprintln!("trace {name}-{layer} sum = {sum:.6}");
        } else {
            eprintln!("trace {name} sum = {sum:.6}");
        }
    }
}

impl Qwen35Model {
    pub fn load(path: &Path, q_gate_layout: QGateLayout) -> Result<Self, String> {
        let gguf = GgufFile::open(path)?;
        let cfg = Qwen35Config::from_gguf(&gguf)?;
        let probe = TokenProbe::from_gguf(&gguf)?;
        Self::from_gguf(&gguf, cfg, probe, q_gate_layout)
    }

    pub fn from_gguf(
        gguf: &GgufFile,
        cfg: Qwen35Config,
        probe: TokenProbe,
        q_gate_layout: QGateLayout,
    ) -> Result<Self, String> {
        let nextn = gguf.kv_u64("qwen35.nextn_predict_layers").unwrap_or(0) as usize;
        let n_active = cfg
            .n_layers
            .checked_sub(nextn)
            .ok_or("more MTP layers than layers")?;

        // The carve: dequantize exactly the token_embd rows the bit alphabet
        // needs; the rest of the vocabulary is never touched.
        let embd_info = gguf
            .tensor("token_embd.weight")
            .ok_or("token_embd missing")?;
        let embd_ty = embd_info.ggml_type()?;
        let embd_data = gguf.tensor_data(embd_info)?;
        let row_bytes = embd_ty.row_bytes(embd_info.cols())?;
        if embd_info.cols() != cfg.hidden {
            return Err("token_embd width != hidden".into());
        }
        let carve_row = |id: u32| -> Result<Vec<f32>, String> {
            let r = id as usize;
            if r >= embd_info.rows() {
                return Err(format!("token id {id} outside vocabulary"));
            }
            let mut out = vec![0f32; cfg.hidden];
            dequant_row(
                embd_ty,
                &embd_data[r * row_bytes..(r + 1) * row_bytes],
                &mut out,
            );
            Ok(out)
        };
        let mut embed_rows = HashMap::new();
        for id in [probe.bit0, probe.bit1, probe.prime] {
            embed_rows.insert(id, carve_row(id)?);
        }
        // Tied unembedding: logits are dots with the same two bit rows.
        let unembed = [carve_row(probe.bit0)?, carve_row(probe.bit1)?];

        let vec1 = |name: &str| -> Result<Vec<f32>, String> {
            Ok(QTensor::from_gguf(gguf, name)?.dequant_row_f32(0))
        };
        let output_norm = vec1("output_norm.weight")?;

        let mut layers = Vec::with_capacity(n_active);
        for i in 0..n_active {
            let t = |suffix: &str| format!("blk.{i}.{suffix}");
            let common = LayerCommon {
                attn_norm: vec1(&t("attn_norm.weight"))?,
                post_norm: vec1(&t("post_attention_norm.weight"))?,
                ffn: Ffn {
                    gate: QTensor::from_gguf(gguf, &t("ffn_gate.weight"))?,
                    up: QTensor::from_gguf(gguf, &t("ffn_up.weight"))?,
                    down: QTensor::from_gguf(gguf, &t("ffn_down.weight"))?,
                },
            };
            if gguf.tensor(&t("attn_qkv.weight")).is_some() {
                let conv = QTensor::from_gguf(gguf, &t("ssm_conv1d.weight"))?;
                let taps = conv.cols;
                if taps != cfg.ssm_conv_kernel {
                    return Err(format!(
                        "blk.{i}: conv kernel {taps} != {}",
                        cfg.ssm_conv_kernel
                    ));
                }
                let mut conv_w = vec![0f32; conv.rows * taps];
                for c in 0..conv.rows {
                    let row = conv.dequant_row_f32(c);
                    conv_w[c * taps..(c + 1) * taps].copy_from_slice(&row);
                }
                let l = DeltaNetLayer {
                    qkv: QTensor::from_gguf(gguf, &t("attn_qkv.weight"))?,
                    conv_w,
                    alpha: QTensor::from_gguf(gguf, &t("ssm_alpha.weight"))?,
                    beta: QTensor::from_gguf(gguf, &t("ssm_beta.weight"))?,
                    a_log: vec1(&t("ssm_a"))?,
                    dt_bias: vec1(&t("ssm_dt.bias"))?,
                    ssm_norm: vec1(&t("ssm_norm.weight"))?,
                    gate: QTensor::from_gguf(gguf, &t("attn_gate.weight"))?,
                    out: QTensor::from_gguf(gguf, &t("ssm_out.weight"))?,
                    common,
                };
                let d_k = cfg.ssm_state_size;
                let qk = cfg.ssm_heads * d_k;
                if l.qkv.rows != 2 * qk + cfg.ssm_inner {
                    return Err(format!(
                        "blk.{i}: attn_qkv rows {} != q{qk}+k{qk}+v{}",
                        l.qkv.rows, cfg.ssm_inner
                    ));
                }
                if l.a_log.len() != cfg.ssm_heads || l.ssm_norm.len() != cfg.ssm_head_v() {
                    return Err(format!("blk.{i}: deltanet head-parameter shapes wrong"));
                }
                layers.push(Layer::DeltaNet(Box::new(l)));
            } else {
                let l = AttentionLayer {
                    q: QTensor::from_gguf(gguf, &t("attn_q.weight"))?,
                    k: QTensor::from_gguf(gguf, &t("attn_k.weight"))?,
                    v: QTensor::from_gguf(gguf, &t("attn_v.weight"))?,
                    q_norm: vec1(&t("attn_q_norm.weight"))?,
                    k_norm: vec1(&t("attn_k_norm.weight"))?,
                    o: QTensor::from_gguf(gguf, &t("attn_output.weight"))?,
                    common,
                };
                if l.q.rows != 2 * cfg.n_heads * cfg.head_dim {
                    return Err(format!(
                        "blk.{i}: attn_q rows {} != 2·heads·head_dim (gated attention)",
                        l.q.rows
                    ));
                }
                if l.k.rows != cfg.n_kv_heads * cfg.head_dim {
                    return Err(format!("blk.{i}: attn_k rows {}", l.k.rows));
                }
                layers.push(Layer::Attention(Box::new(l)));
            }
        }

        Ok(Qwen35Model {
            rope: Rope::new(cfg.rope_dims, cfg.rope_theta),
            cfg,
            probe,
            q_gate_layout,
            n_active_layers: n_active,
            embed_rows,
            unembed,
            layers,
            output_norm,
        })
    }

    pub fn state_shape(&self) -> StateShape {
        let n_attention = self
            .layers
            .iter()
            .filter(|l| matches!(l, Layer::Attention(_)))
            .count();
        StateShape {
            n_attention,
            kv_stride: self.cfg.n_kv_heads * self.cfg.head_dim,
            n_deltanet: self.layers.len() - n_attention,
            ssm_heads: self.cfg.ssm_v_heads,
            ssm_d_k: self.cfg.ssm_state_size,
            ssm_d_v: self.cfg.ssm_head_v(),
            conv_taps: self.cfg.ssm_conv_kernel - 1,
            conv_channels: 2 * self.cfg.ssm_heads * self.cfg.ssm_state_size + self.cfg.ssm_inner,
        }
    }

    pub fn new_state(&self) -> LlmState {
        LlmState::new(self.state_shape())
    }

    /// Total resident weight bytes (diagnostics).
    pub fn weight_bytes(&self) -> usize {
        let ffn = |f: &Ffn| f.gate.byte_len() + f.up.byte_len() + f.down.byte_len();
        self.layers
            .iter()
            .map(|l| match l {
                Layer::DeltaNet(l) => {
                    l.qkv.byte_len()
                        + l.alpha.byte_len()
                        + l.beta.byte_len()
                        + l.gate.byte_len()
                        + l.out.byte_len()
                        + ffn(&l.common.ffn)
                }
                Layer::Attention(l) => {
                    l.q.byte_len()
                        + l.k.byte_len()
                        + l.v.byte_len()
                        + l.o.byte_len()
                        + ffn(&l.common.ffn)
                }
            })
            .sum()
    }

    /// Advance one token through the network, mutating `state` (checkpoint
    /// pushed first, so the step is exactly revertible), and return the
    /// carved 2-logit output [logit("0"), logit("1")].
    pub fn advance(&self, state: &mut LlmState, token: u32) -> [f32; 2] {
        let eps = self.cfg.rms_eps;
        state.push_checkpoint();
        let pos = state.pos;
        let mut x = self
            .embed_rows
            .get(&token)
            .unwrap_or_else(|| panic!("token {token} was not carved into the embedding"))
            .clone();

        trace("model.input_embed", -1, &x);
        let mut att_i = 0;
        let mut dn_i = 0;
        for (li, layer) in self.layers.iter().enumerate() {
            let li = li as isize;
            match layer {
                Layer::DeltaNet(l) => {
                    let xn = rmsnorm(&x, &l.common.attn_norm, eps);
                    trace("attn_norm", li, &xn);
                    let y = self.deltanet_mix(l, &mut state.deltanet[dn_i], &xn, li);
                    trace("linear_attn_out", li, &y);
                    add_into(&mut x, &y);
                    dn_i += 1;
                    trace("attn_residual", li, &x);
                    let xn = rmsnorm(&x, &l.common.post_norm, eps);
                    let y = self.ffn(&l.common.ffn, &xn);
                    trace("ffn_out", li, &y);
                    add_into(&mut x, &y);
                }
                Layer::Attention(l) => {
                    let xn = rmsnorm(&x, &l.common.attn_norm, eps);
                    trace("attn_norm", li, &xn);
                    let y = self.attention_mix(l, &mut state.kv[att_i], &xn, pos, li);
                    add_into(&mut x, &y);
                    att_i += 1;
                    trace("attn_residual", li, &x);
                    let xn = rmsnorm(&x, &l.common.post_norm, eps);
                    let y = self.ffn(&l.common.ffn, &xn);
                    trace("ffn_out", li, &y);
                    add_into(&mut x, &y);
                }
            }
            trace("l_out", li, &x);
        }

        let xn = rmsnorm(&x, &self.output_norm, eps);
        trace("result_norm", -1, &xn);
        state.pos += 1;
        let dot = |w: &[f32]| -> f32 {
            let mut acc = 0f64;
            for (a, b) in w.iter().zip(&xn) {
                acc += (a * b) as f64;
            }
            acc as f32
        };
        [dot(&self.unembed[0]), dot(&self.unembed[1])]
    }

    fn ffn(&self, f: &Ffn, xn: &[f32]) -> Vec<f32> {
        let mut g = vec![0f32; f.gate.rows];
        f.gate.gemv(xn, &mut g);
        let mut u = vec![0f32; f.up.rows];
        f.up.gemv(xn, &mut u);
        for (gv, uv) in g.iter_mut().zip(&u) {
            *gv = silu(*gv) * uv;
        }
        let mut d = vec![0f32; f.down.rows];
        f.down.gemv(&g, &mut d);
        d
    }

    fn attention_mix(
        &self,
        l: &AttentionLayer,
        kv: &mut super::state::KvArena,
        xn: &[f32],
        pos: usize,
        li: isize,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let eps = cfg.rms_eps;

        let mut qg = vec![0f32; l.q.rows];
        l.q.gemv(xn, &mut qg);
        trace("Qcur_full", li, &qg);
        let mut k = vec![0f32; l.k.rows];
        l.k.gemv(xn, &mut k);
        let mut v = vec![0f32; l.v.rows];
        l.v.gemv(xn, &mut v);

        for j in 0..cfg.n_kv_heads {
            let kh = &mut k[j * hd..(j + 1) * hd];
            rmsnorm_head(kh, &l.k_norm, eps);
        }
        trace("Kcur_normed", li, &k);
        for j in 0..cfg.n_kv_heads {
            self.rope.apply(&mut k[j * hd..(j + 1) * hd], pos);
        }
        kv.k.extend_from_slice(&k);
        kv.v.extend_from_slice(&v);

        // Split fused q|gate, per-head QK-norm (before RoPE, as the graph).
        let mut q_all = vec![0f32; cfg.n_heads * hd];
        let mut gates = vec![0f32; cfg.n_heads * hd];
        for h in 0..cfg.n_heads {
            let (qs, gs) = match self.q_gate_layout {
                QGateLayout::PerHeadInterleaved => (h * 2 * hd, h * 2 * hd + hd),
                QGateLayout::Blocked => (h * hd, cfg.n_heads * hd + h * hd),
            };
            q_all[h * hd..(h + 1) * hd].copy_from_slice(&qg[qs..qs + hd]);
            gates[h * hd..(h + 1) * hd].copy_from_slice(&qg[gs..gs + hd]);
            rmsnorm_head(&mut q_all[h * hd..(h + 1) * hd], &l.q_norm, eps);
        }
        trace("Qcur_normed", li, &q_all);

        let n_ctx = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = cfg.n_heads / cfg.n_kv_heads;
        let mut out_flat = vec![0f32; cfg.n_heads * hd];
        let mut scores = vec![0f32; n_ctx];
        for h in 0..cfg.n_heads {
            let qh = &mut q_all[h * hd..(h + 1) * hd];
            self.rope.apply(qh, pos);
            let j = h / group;

            let mut max = f32::NEG_INFINITY;
            for (t, s) in scores.iter_mut().enumerate() {
                let key = &kv.keys(t)[j * hd..(j + 1) * hd];
                let mut acc = 0f32;
                for d in 0..hd {
                    acc += qh[d] * key[d];
                }
                *s = acc * scale;
                max = max.max(*s);
            }
            let mut z = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                z += *s;
            }
            let inv_z = 1.0 / z;
            let oh = &mut out_flat[h * hd..(h + 1) * hd];
            for (t, s) in scores.iter().enumerate() {
                let val = &kv.values(t)[j * hd..(j + 1) * hd];
                let w = s * inv_z;
                for d in 0..hd {
                    oh[d] += w * val[d];
                }
            }
        }
        trace("attn_pregate", li, &out_flat);
        for g in gates.iter_mut() {
            *g = sigmoid(*g);
        }
        trace("gate_sigmoid", li, &gates);
        for (o, g) in out_flat.iter_mut().zip(&gates) {
            *o *= g;
        }
        trace("attn_gated", li, &out_flat);

        let mut out = vec![0f32; l.o.rows];
        l.o.gemv(&out_flat, &mut out);
        out
    }

    fn deltanet_mix(
        &self,
        l: &DeltaNetLayer,
        dn: &mut super::state::DeltaNetState,
        xn: &[f32],
        li: isize,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let heads = cfg.ssm_v_heads;
        let d_k = cfg.ssm_state_size;
        let d_v = cfg.ssm_head_v();
        let taps = cfg.ssm_conv_kernel - 1;
        let channels = l.qkv.rows;
        let eps = cfg.rms_eps;

        let mut qkv_raw = vec![0f32; channels];
        l.qkv.gemv(xn, &mut qkv_raw);
        trace("linear_attn_qkv_mixed", li, &qkv_raw);

        // Depthwise causal conv over time (tail = previous `taps` raw inputs,
        // oldest first) followed by SiLU; then rotate the tail.
        let mut y = vec![0f32; channels];
        for c in 0..channels {
            let w = &l.conv_w[c * (taps + 1)..(c + 1) * (taps + 1)];
            let mut acc = w[taps] * qkv_raw[c];
            for (t, &wt) in w[..taps].iter().enumerate() {
                acc += wt * dn.conv_tail[t * channels + c];
            }
            y[c] = acc;
        }
        trace("conv_output_raw", li, &y);
        for v in y.iter_mut() {
            *v = silu(*v);
        }
        trace("conv_output_silu", li, &y);
        dn.conv_tail.copy_within(channels.., 0);
        dn.conv_tail[(taps - 1) * channels..].copy_from_slice(&qkv_raw);

        let qk = heads * d_k;
        let (q_all, rest) = y.split_at_mut(qk);
        let (k_all, v_all) = rest.split_at_mut(qk);
        for h in 0..heads {
            l2norm_head(&mut q_all[h * d_k..(h + 1) * d_k], eps);
            l2norm_head(&mut k_all[h * d_k..(h + 1) * d_k], eps);
        }
        trace("q_conv_predelta", li, q_all);
        trace("k_conv_predelta", li, k_all);
        trace("v_conv_predelta", li, v_all);

        let mut alpha = vec![0f32; heads];
        l.alpha.gemv(xn, &mut alpha);
        let mut beta = vec![0f32; heads];
        l.beta.gemv(xn, &mut beta);
        let mut z = vec![0f32; l.gate.rows];
        l.gate.gemv(xn, &mut z);
        // Gated delta rule (Yang et al. 2024): per-head decay
        // γ = exp(a·softplus(α(x)+dt_bias)) with the stored `ssm_a` already
        // equal to −exp(A_log) (the GGUF converter pre-negates it); write
        // strength β = σ(·).
        let sp: Vec<f32> = (0..heads)
            .map(|h| softplus(alpha[h] + l.dt_bias[h]))
            .collect();
        trace("a_softplus", li, &sp);
        let gs: Vec<f32> = (0..heads).map(|h| l.a_log[h] * sp[h]).collect();
        trace("gate", li, &gs);
        let bs: Vec<f32> = beta.iter().map(|&b| sigmoid(b)).collect();
        trace("beta_sigmoid", li, &bs);
        trace("z", li, &z);

        // The delta-rule query is scaled by 1/√d_k (reference:
        // build_delta_net_autoregressive in llama.cpp's delta-net base).
        let q_scale = 1.0 / (d_k as f32).sqrt();
        let mut out_flat = vec![0f32; heads * d_v];
        let mut qh = vec![0f32; d_k];
        let mut kh = vec![0f32; d_k];
        for h in 0..heads {
            qh.copy_from_slice(&q_all[h * d_k..(h + 1) * d_k]);
            kh.copy_from_slice(&k_all[h * d_k..(h + 1) * d_k]);
            for qv in qh.iter_mut() {
                *qv *= q_scale;
            }
            let vh = &v_all[h * d_v..(h + 1) * d_v];
            let gamma = gs[h].exp();
            let b = bs[h];

            let s = &mut dn.s[h * d_v * d_k..(h + 1) * d_v * d_k];
            let oh = &mut out_flat[h * d_v..(h + 1) * d_v];
            for dv in 0..d_v {
                let row = &mut s[dv * d_k..(dv + 1) * d_k];
                let mut mem = 0f32;
                for dk in 0..d_k {
                    row[dk] *= gamma;
                    mem += row[dk] * kh[dk];
                }
                let delta = (vh[dv] - mem) * b;
                let mut o = 0f32;
                for dk in 0..d_k {
                    row[dk] += delta * kh[dk];
                    o += row[dk] * qh[dk];
                }
                oh[dv] = o;
            }
            rmsnorm_head(oh, &l.ssm_norm, eps);
            let zh = &z[h * d_v..(h + 1) * d_v];
            for (o, zv) in oh.iter_mut().zip(zh) {
                *o *= silu(*zv);
            }
        }
        trace("final_output", li, &out_flat);

        let mut out = vec![0f32; l.out.rows];
        l.out.gemv(&out_flat, &mut out);
        out
    }
}
