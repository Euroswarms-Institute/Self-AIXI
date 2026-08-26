//! GGML quantization block formats and fused dequant·dot kernels.
//!
//! Weights stay in their on-disk quantized blocks in RAM; the GEMV kernels
//! dequantize inside the dot product (per 256-element super-block, partial
//! sums accumulated into f64), so the only f32 tensors ever materialized are
//! activations. Layouts follow ggml's reference `dequantize_row_*`
//! implementations; `dequant_row` is the readable reference the fused
//! kernels are tested against, and the llama.cpp oracle (scripts/) provides
//! the external ground truth end-to-end.

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, String> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            30 => GgmlType::BF16,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            other => return Err(format!("unsupported ggml tensor type {other}")),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
        }
    }

    /// Elements per quantization block.
    pub fn block_elems(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => 1,
            GgmlType::Q8_0 => 32,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => 256,
        }
    }

    /// Bytes per quantization block.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 | GgmlType::BF16 => 2,
            GgmlType::Q8_0 => 34, // f16 scale + 32×i8
            GgmlType::Q4K => 144, // 2×f16 + 12 scale bytes + 128 nibbles
            GgmlType::Q5K => 176, // Q4_K + 32 high-bit bytes
            GgmlType::Q6K => 210, // 128 ql + 64 qh + 16 scales + f16
        }
    }

    pub fn row_bytes(self, elems: usize) -> Result<usize, String> {
        if !elems.is_multiple_of(self.block_elems()) {
            return Err(format!(
                "row of {elems} elements is not a multiple of {} block size {}",
                self.name(),
                self.block_elems()
            ));
        }
        Ok(elems / self.block_elems() * self.block_bytes())
    }
}

fn f16_at(bytes: &[u8], off: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[off], bytes[off + 1]])).to_f32()
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// ggml's 6-bit K-quant scale/min unpacking (get_scale_min_k4).
fn scale_min_k4(j: usize, scales: &[u8]) -> (f32, f32) {
    let (sc, m) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    };
    (sc as f32, m as f32)
}

/// Reference dequantization of one full row into `out` (len = element count).
pub fn dequant_row(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    let bb = ty.block_bytes();
    match ty {
        GgmlType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_at(row, 2 * i);
            }
        }
        GgmlType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            }
        }
        GgmlType::Q8_0 => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let d = f16_at(block, 0);
                for (l, o) in out[32 * bi..32 * (bi + 1)].iter_mut().enumerate() {
                    *o = d * (block[2 + l] as i8) as f32;
                }
            }
        }
        GgmlType::Q4K => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let d = f16_at(block, 0);
                let dmin = f16_at(block, 2);
                let scales = &block[4..16];
                let qs = &block[16..144];
                let y = &mut out[256 * bi..256 * (bi + 1)];
                for j in 0..4 {
                    let (sc_lo, m_lo) = scale_min_k4(2 * j, scales);
                    let (sc_hi, m_hi) = scale_min_k4(2 * j + 1, scales);
                    for l in 0..32 {
                        let q = qs[32 * j + l];
                        y[64 * j + l] = d * sc_lo * (q & 0x0F) as f32 - dmin * m_lo;
                        y[64 * j + 32 + l] = d * sc_hi * (q >> 4) as f32 - dmin * m_hi;
                    }
                }
            }
        }
        GgmlType::Q5K => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let d = f16_at(block, 0);
                let dmin = f16_at(block, 2);
                let scales = &block[4..16];
                let qh = &block[16..48];
                let qs = &block[48..176];
                let y = &mut out[256 * bi..256 * (bi + 1)];
                for j in 0..4 {
                    let (sc_lo, m_lo) = scale_min_k4(2 * j, scales);
                    let (sc_hi, m_hi) = scale_min_k4(2 * j + 1, scales);
                    let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
                    for l in 0..32 {
                        let q = qs[32 * j + l];
                        let hi_lo = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        let hi_hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        y[64 * j + l] = d * sc_lo * ((q & 0x0F) as f32 + hi_lo) - dmin * m_lo;
                        y[64 * j + 32 + l] = d * sc_hi * ((q >> 4) as f32 + hi_hi) - dmin * m_hi;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = f16_at(block, 208);
                let y = &mut out[256 * bi..256 * (bi + 1)];
                for half in 0..2 {
                    let (qlo, qho, sco, yo) = (64 * half, 32 * half, 8 * half, 128 * half);
                    for l in 0..32 {
                        let is = l / 16;
                        let h = qh[qho + l];
                        let q1 = ((ql[qlo + l] & 0x0F) | ((h & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[qlo + l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[qlo + l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[qlo + l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                        y[yo + l] = d * (scales[sco + is] as i8 as f32) * q1 as f32;
                        y[yo + 32 + l] = d * (scales[sco + is + 2] as i8 as f32) * q2 as f32;
                        y[yo + 64 + l] = d * (scales[sco + is + 4] as i8 as f32) * q3 as f32;
                        y[yo + 96 + l] = d * (scales[sco + is + 6] as i8 as f32) * q4 as f32;
                    }
                }
            }
        }
    }
}

/// Fused dequant·dot of one quantized row against `x` (len = element count).
/// Dispatches to the AVX2 kernels when the CPU has them (disable with
/// MC_AIXI_NO_SIMD=1); the scalar path is the reference the SIMD path is
/// tested against. Per-block partial sums in f32, accumulated in f64.
pub fn dot_row(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled()
        && matches!(
            ty,
            GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
        )
    {
        // Safety: gated on runtime AVX2+FMA detection.
        return unsafe { avx2::dot_row(ty, row, x) };
    }
    dot_row_scalar(ty, row, x)
}

#[cfg(target_arch = "x86_64")]
fn avx2_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("MC_AIXI_NO_SIMD").is_none()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
    })
}

/// Scalar reference implementation of the fused dequant·dot.
pub fn dot_row_scalar(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    let bb = ty.block_bytes();
    let mut total = 0f64;
    match ty {
        GgmlType::F32 => {
            let mut acc = 0f64;
            for (i, xv) in x.iter().enumerate() {
                acc += (f32::from_le_bytes(row[4 * i..4 * i + 4].try_into().unwrap()) * xv) as f64;
            }
            total = acc;
        }
        GgmlType::F16 => {
            for (i, xv) in x.iter().enumerate() {
                total += (f16_at(row, 2 * i) * xv) as f64;
            }
        }
        GgmlType::BF16 => {
            for (i, xv) in x.iter().enumerate() {
                total +=
                    (bf16_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]])) * xv) as f64;
            }
        }
        GgmlType::Q8_0 => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let d = f16_at(block, 0);
                let xs = &x[32 * bi..32 * (bi + 1)];
                let mut acc = 0f32;
                for l in 0..32 {
                    acc += (block[2 + l] as i8) as f32 * xs[l];
                }
                total += (d * acc) as f64;
            }
        }
        GgmlType::Q4K => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let d = f16_at(block, 0);
                let dmin = f16_at(block, 2);
                let scales = &block[4..16];
                let qs = &block[16..144];
                let xs = &x[256 * bi..256 * (bi + 1)];
                let mut acc = 0f32;
                for j in 0..4 {
                    let (sc_lo, m_lo) = scale_min_k4(2 * j, scales);
                    let (sc_hi, m_hi) = scale_min_k4(2 * j + 1, scales);
                    let (mut s_lo, mut s_hi, mut sx_lo, mut sx_hi) = (0f32, 0f32, 0f32, 0f32);
                    for l in 0..32 {
                        let q = qs[32 * j + l];
                        let (xl, xh) = (xs[64 * j + l], xs[64 * j + 32 + l]);
                        s_lo += (q & 0x0F) as f32 * xl;
                        s_hi += (q >> 4) as f32 * xh;
                        sx_lo += xl;
                        sx_hi += xh;
                    }
                    acc += d * (sc_lo * s_lo + sc_hi * s_hi) - dmin * (m_lo * sx_lo + m_hi * sx_hi);
                }
                total += acc as f64;
            }
        }
        GgmlType::Q5K => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let d = f16_at(block, 0);
                let dmin = f16_at(block, 2);
                let scales = &block[4..16];
                let qh = &block[16..48];
                let qs = &block[48..176];
                let xs = &x[256 * bi..256 * (bi + 1)];
                let mut acc = 0f32;
                for j in 0..4 {
                    let (sc_lo, m_lo) = scale_min_k4(2 * j, scales);
                    let (sc_hi, m_hi) = scale_min_k4(2 * j + 1, scales);
                    let (u1, u2) = (1u8 << (2 * j), 2u8 << (2 * j));
                    let (mut s_lo, mut s_hi, mut sx_lo, mut sx_hi) = (0f32, 0f32, 0f32, 0f32);
                    for l in 0..32 {
                        let q = qs[32 * j + l];
                        let (xl, xh) = (xs[64 * j + l], xs[64 * j + 32 + l]);
                        let q_lo = (q & 0x0F) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        let q_hi = (q >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        s_lo += q_lo * xl;
                        s_hi += q_hi * xh;
                        sx_lo += xl;
                        sx_hi += xh;
                    }
                    acc += d * (sc_lo * s_lo + sc_hi * s_hi) - dmin * (m_lo * sx_lo + m_hi * sx_hi);
                }
                total += acc as f64;
            }
        }
        GgmlType::Q6K => {
            for (bi, block) in row.chunks_exact(bb).enumerate() {
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let d = f16_at(block, 208);
                let xs = &x[256 * bi..256 * (bi + 1)];
                let mut acc = 0f32;
                for half in 0..2 {
                    let (qlo, qho, sco, xo) = (64 * half, 32 * half, 8 * half, 128 * half);
                    let mut s = [0f32; 4];
                    for l in 0..32 {
                        let is = l / 16;
                        let h = qh[qho + l];
                        let q1 = ((ql[qlo + l] & 0x0F) | ((h & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[qlo + l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[qlo + l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[qlo + l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                        s[0] += (scales[sco + is] as i8 as f32) * q1 as f32 * xs[xo + l];
                        s[1] += (scales[sco + is + 2] as i8 as f32) * q2 as f32 * xs[xo + 32 + l];
                        s[2] += (scales[sco + is + 4] as i8 as f32) * q3 as f32 * xs[xo + 64 + l];
                        s[3] += (scales[sco + is + 6] as i8 as f32) * q4 as f32 * xs[xo + 96 + l];
                    }
                    acc += d * (s[0] + s[1] + s[2] + s[3]);
                }
                total += acc as f64;
            }
        }
    }
    total as f32
}

/// AVX2 + FMA kernels for the four quantized block formats. Same block
/// structure as the scalar path (per-block f32 partial sums folded into an
/// f64 running total), so the numerical contract the tests check is shared;
/// only the intra-block summation order differs (8 SIMD lanes reduced at the
/// end of each sub-block instead of a sequential left fold).
///
/// Unpacking strategy: quant bytes are widened 8 at a time to 32-bit lanes
/// (`_mm256_cvtepu8_epi32`), so every shift/mask below operates on a whole
/// byte value per lane and cross-byte contamination is structurally
/// impossible; nibbles and 2-bit fields are then plain `and`/`srli` on the
/// lane. Converted to f32, multiply-accumulated against `x` with FMA.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{f16_at, scale_min_k4, GgmlType};
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 f32 lanes.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }

    /// Zero-extend 8 quant bytes at `p` into 8×u32 lanes.
    ///
    /// # Safety
    /// `p..p+8` must be in bounds (callers derive `p` from `chunks_exact`
    /// blocks and per-block `x` slices whose lengths are checked).
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn load8_u32(p: *const u8) -> __m256i {
        _mm256_cvtepu8_epi32(_mm_loadl_epi64(p.cast()))
    }

    /// # Safety
    /// Requires AVX2 + FMA (the dispatcher checks at runtime); `row` must be
    /// whole blocks of `ty` and `x` at least as long as the element count.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_row(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
        match ty {
            GgmlType::Q8_0 => dot_q8_0(row, x),
            GgmlType::Q4K => dot_q4_k(row, x),
            GgmlType::Q5K => dot_q5_k(row, x),
            GgmlType::Q6K => dot_q6_k(row, x),
            _ => unreachable!("AVX2 dispatch covers only quantized block types"),
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
        let mut total = 0f64;
        for (bi, block) in row.chunks_exact(34).enumerate() {
            let d = f16_at(block, 0);
            let xs = x[32 * bi..32 * (bi + 1)].as_ptr();
            let mut acc = _mm256_setzero_ps();
            for g in 0..4 {
                let q = _mm256_cvtepi8_epi32(_mm_loadl_epi64(block.as_ptr().add(2 + 8 * g).cast()));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), _mm256_loadu_ps(xs.add(8 * g)), acc);
            }
            total += (d * hsum(acc)) as f64;
        }
        total as f32
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_q4_k(row: &[u8], x: &[f32]) -> f32 {
        let mask4 = _mm256_set1_epi32(0x0F);
        let mut total = 0f64;
        for (bi, block) in row.chunks_exact(144).enumerate() {
            let d = f16_at(block, 0);
            let dmin = f16_at(block, 2);
            let scales = &block[4..16];
            let qs = block.as_ptr().add(16);
            let xs = x[256 * bi..256 * (bi + 1)].as_ptr();
            let mut acc = 0f32;
            for j in 0..4 {
                let (sc_lo, m_lo) = scale_min_k4(2 * j, scales);
                let (sc_hi, m_hi) = scale_min_k4(2 * j + 1, scales);
                let mut s_lo = _mm256_setzero_ps();
                let mut s_hi = _mm256_setzero_ps();
                let mut sx_lo = _mm256_setzero_ps();
                let mut sx_hi = _mm256_setzero_ps();
                for g in 0..4 {
                    let q = load8_u32(qs.add(32 * j + 8 * g));
                    let lo = _mm256_cvtepi32_ps(_mm256_and_si256(q, mask4));
                    let hi = _mm256_cvtepi32_ps(_mm256_srli_epi32::<4>(q));
                    let xl = _mm256_loadu_ps(xs.add(64 * j + 8 * g));
                    let xh = _mm256_loadu_ps(xs.add(64 * j + 32 + 8 * g));
                    s_lo = _mm256_fmadd_ps(lo, xl, s_lo);
                    s_hi = _mm256_fmadd_ps(hi, xh, s_hi);
                    sx_lo = _mm256_add_ps(sx_lo, xl);
                    sx_hi = _mm256_add_ps(sx_hi, xh);
                }
                acc += d * (sc_lo * hsum(s_lo) + sc_hi * hsum(s_hi))
                    - dmin * (m_lo * hsum(sx_lo) + m_hi * hsum(sx_hi));
            }
            total += acc as f64;
        }
        total as f32
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_q5_k(row: &[u8], x: &[f32]) -> f32 {
        let mask4 = _mm256_set1_epi32(0x0F);
        let one = _mm256_set1_epi32(1);
        let mut total = 0f64;
        for (bi, block) in row.chunks_exact(176).enumerate() {
            let d = f16_at(block, 0);
            let dmin = f16_at(block, 2);
            let scales = &block[4..16];
            let qh = block.as_ptr().add(16);
            let qs = block.as_ptr().add(48);
            let xs = x[256 * bi..256 * (bi + 1)].as_ptr();
            let mut acc = 0f32;
            for j in 0..4 {
                let (sc_lo, m_lo) = scale_min_k4(2 * j, scales);
                let (sc_hi, m_hi) = scale_min_k4(2 * j + 1, scales);
                // Bit (2j) of qh[l] is the fifth bit of the lo nibble,
                // bit (2j+1) of the hi nibble; runtime shift counts go via
                // the xmm-count form of vpsrld.
                let cnt_lo = _mm_cvtsi32_si128((2 * j) as i32);
                let cnt_hi = _mm_cvtsi32_si128((2 * j + 1) as i32);
                let mut s_lo = _mm256_setzero_ps();
                let mut s_hi = _mm256_setzero_ps();
                let mut sx_lo = _mm256_setzero_ps();
                let mut sx_hi = _mm256_setzero_ps();
                for g in 0..4 {
                    let q = load8_u32(qs.add(32 * j + 8 * g));
                    let h = load8_u32(qh.add(8 * g));
                    let b_lo = _mm256_and_si256(_mm256_srl_epi32(h, cnt_lo), one);
                    let b_hi = _mm256_and_si256(_mm256_srl_epi32(h, cnt_hi), one);
                    let lo = _mm256_cvtepi32_ps(_mm256_add_epi32(
                        _mm256_and_si256(q, mask4),
                        _mm256_slli_epi32::<4>(b_lo),
                    ));
                    let hi = _mm256_cvtepi32_ps(_mm256_add_epi32(
                        _mm256_srli_epi32::<4>(q),
                        _mm256_slli_epi32::<4>(b_hi),
                    ));
                    let xl = _mm256_loadu_ps(xs.add(64 * j + 8 * g));
                    let xh = _mm256_loadu_ps(xs.add(64 * j + 32 + 8 * g));
                    s_lo = _mm256_fmadd_ps(lo, xl, s_lo);
                    s_hi = _mm256_fmadd_ps(hi, xh, s_hi);
                    sx_lo = _mm256_add_ps(sx_lo, xl);
                    sx_hi = _mm256_add_ps(sx_hi, xh);
                }
                acc += d * (sc_lo * hsum(s_lo) + sc_hi * hsum(s_hi))
                    - dmin * (m_lo * hsum(sx_lo) + m_hi * hsum(sx_hi));
            }
            total += acc as f64;
        }
        total as f32
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_q6_k(row: &[u8], x: &[f32]) -> f32 {
        let mask4 = _mm256_set1_epi32(0x0F);
        let mask2 = _mm256_set1_epi32(3);
        let bias = _mm256_set1_epi32(32);
        let mut total = 0f64;
        for (bi, block) in row.chunks_exact(210).enumerate() {
            let ql = block.as_ptr();
            let qh = block.as_ptr().add(128);
            let scales = &block[192..208];
            let d = f16_at(block, 208);
            let xs = x[256 * bi..256 * (bi + 1)].as_ptr();
            let mut acc = 0f32;
            for half in 0..2 {
                let (qlo, qho, sco, xo) = (64 * half, 32 * half, 8 * half, 128 * half);
                let mut s = [_mm256_setzero_ps(); 4];
                for g in 0..4 {
                    let is = g / 2; // 16-lane scale groups = 2 SIMD groups each
                    let l_lo = load8_u32(ql.add(qlo + 8 * g));
                    let l_hi = load8_u32(ql.add(qlo + 32 + 8 * g));
                    let h = load8_u32(qh.add(qho + 8 * g));
                    let q1 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_and_si256(l_lo, mask4),
                            _mm256_slli_epi32::<4>(_mm256_and_si256(h, mask2)),
                        ),
                        bias,
                    );
                    let q2 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_and_si256(l_hi, mask4),
                            _mm256_slli_epi32::<4>(_mm256_and_si256(
                                _mm256_srli_epi32::<2>(h),
                                mask2,
                            )),
                        ),
                        bias,
                    );
                    let q3 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_srli_epi32::<4>(l_lo),
                            _mm256_slli_epi32::<4>(_mm256_and_si256(
                                _mm256_srli_epi32::<4>(h),
                                mask2,
                            )),
                        ),
                        bias,
                    );
                    // h is a zero-extended byte per lane, so h >> 6 needs no mask.
                    let q4 = _mm256_sub_epi32(
                        _mm256_or_si256(
                            _mm256_srli_epi32::<4>(l_hi),
                            _mm256_slli_epi32::<4>(_mm256_srli_epi32::<6>(h)),
                        ),
                        bias,
                    );
                    for (k, q) in [q1, q2, q3, q4].into_iter().enumerate() {
                        let sc = _mm256_set1_ps(scales[sco + is + 2 * k] as i8 as f32);
                        let xv = _mm256_loadu_ps(xs.add(xo + 32 * k + 8 * g));
                        s[k] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q), sc), xv, s[k]);
                    }
                }
                acc += d * (hsum(s[0]) + hsum(s[1]) + hsum(s[2]) + hsum(s[3]));
            }
            total += acc as f64;
        }
        total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;
    use rand::Rng;

    fn naive_dot(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
        let mut y = vec![0f32; x.len()];
        dequant_row(ty, row, &mut y);
        let mut acc = 0f64;
        for (a, b) in y.iter().zip(x) {
            acc += (a * b) as f64;
        }
        acc as f32
    }

    #[test]
    fn q8_0_hand_golden() {
        // d = 0.5, qs = [1, -2, 3, 0, 0, ...]
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2] = 1i8 as u8;
        block[3] = (-2i8) as u8;
        block[4] = 3i8 as u8;
        let mut y = vec![0f32; 32];
        dequant_row(GgmlType::Q8_0, &block, &mut y);
        assert_eq!(&y[..4], &[0.5, -1.0, 1.5, 0.0]);
    }

    #[test]
    fn q4_k_hand_golden() {
        // d = 1, dmin = 0, sub-block scales sc0 = 1, sc1 = 2 (packed 6-bit),
        // qs[0] = 0x21 ⇒ element 0 = 1·1 = 1, element 32 = 2·2 = 4.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.0).to_bits().to_le_bytes());
        block[4] = 1; // scales[0]: sc for sub-block 0
        block[5] = 2; // scales[1]: sc for sub-block 1
        block[16] = 0x21;
        let mut y = vec![0f32; 256];
        dequant_row(GgmlType::Q4K, &block, &mut y);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[32], 4.0);
    }

    #[test]
    fn fused_dot_matches_reference_on_random_blocks() {
        let mut rng = seeded(2718);
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::F32,
        ] {
            let elems = ty.block_elems().max(256) * 3; // a few blocks per row
            let row_len = ty.row_bytes(elems).unwrap();
            for _ in 0..8 {
                let mut row = vec![0u8; row_len];
                match ty {
                    // Random bytes are valid payloads for quantized blocks,
                    // but float formats need finite values.
                    GgmlType::F32 => {
                        for c in row.chunks_exact_mut(4) {
                            c.copy_from_slice(&(rng.random_range(-2.0f32..2.0)).to_le_bytes());
                        }
                    }
                    GgmlType::F16 => {
                        for c in row.chunks_exact_mut(2) {
                            let v = f16::from_f32(rng.random_range(-2.0f32..2.0));
                            c.copy_from_slice(&v.to_bits().to_le_bytes());
                        }
                    }
                    GgmlType::BF16 => {
                        for c in row.chunks_exact_mut(2) {
                            let bits = rng.random_range(-2.0f32..2.0).to_bits();
                            c.copy_from_slice(&((bits >> 16) as u16).to_le_bytes());
                        }
                    }
                    _ => rng.fill(&mut row[..]),
                }
                // Quantized scale fields are raw f16 bit patterns; clamp NaN/Inf
                // scales by rewriting them with small finite values.
                if matches!(ty, GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K) {
                    for b in row.chunks_exact_mut(ty.block_bytes()) {
                        b[0..2].copy_from_slice(
                            &f16::from_f32(rng.random_range(-0.1..0.1))
                                .to_bits()
                                .to_le_bytes(),
                        );
                        if ty != GgmlType::Q8_0 {
                            b[2..4].copy_from_slice(
                                &f16::from_f32(rng.random_range(-0.1..0.1))
                                    .to_bits()
                                    .to_le_bytes(),
                            );
                        }
                    }
                } else if ty == GgmlType::Q6K {
                    for b in row.chunks_exact_mut(ty.block_bytes()) {
                        b[208..210].copy_from_slice(
                            &f16::from_f32(rng.random_range(-0.1..0.1))
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                }
                let x: Vec<f32> = (0..elems).map(|_| rng.random_range(-1.0f32..1.0)).collect();
                let fused = dot_row(ty, &row, &x);
                let naive = naive_dot(ty, &row, &x);
                let scale = naive.abs().max(1.0);
                assert!(
                    (fused - naive).abs() / scale < 1e-4,
                    "{}: fused {fused} vs naive {naive}",
                    ty.name()
                );
            }
        }
    }

    /// The dispatched kernel (AVX2 where the CPU has it) against the scalar
    /// reference, on many multi-block rows. Tighter than the dequant
    /// reference test since both sides share the per-block f32 / total f64
    /// accumulation structure; only intra-block summation order differs.
    #[test]
    fn simd_matches_scalar() {
        let mut rng = seeded(31415);
        for ty in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let elems = ty.block_elems() * 8;
            let row_len = ty.row_bytes(elems).unwrap();
            for _ in 0..32 {
                let mut row = vec![0u8; row_len];
                rng.fill(&mut row[..]);
                // Rewrite raw f16 scale fields with finite values (random bit
                // patterns can be NaN/Inf, which no real checkpoint contains).
                for b in row.chunks_exact_mut(ty.block_bytes()) {
                    let (d_off, dmin) = match ty {
                        GgmlType::Q6K => (208, false),
                        GgmlType::Q8_0 => (0, false),
                        _ => (0, true),
                    };
                    b[d_off..d_off + 2].copy_from_slice(
                        &f16::from_f32(rng.random_range(-0.1..0.1))
                            .to_bits()
                            .to_le_bytes(),
                    );
                    if dmin {
                        b[2..4].copy_from_slice(
                            &f16::from_f32(rng.random_range(-0.1..0.1))
                                .to_bits()
                                .to_le_bytes(),
                        );
                    }
                }
                let x: Vec<f32> = (0..elems).map(|_| rng.random_range(-1.0f32..1.0)).collect();
                let dispatched = dot_row(ty, &row, &x);
                let scalar = dot_row_scalar(ty, &row, &x);
                let scale = scalar.abs().max(1.0);
                assert!(
                    (dispatched - scalar).abs() / scale < 1e-5,
                    "{}: dispatched {dispatched} vs scalar {scalar}",
                    ty.name()
                );
            }
        }
    }

    #[test]
    fn row_bytes_arithmetic() {
        assert_eq!(GgmlType::Q4K.row_bytes(2048).unwrap(), 2048 / 256 * 144);
        assert_eq!(GgmlType::Q8_0.row_bytes(64).unwrap(), 68);
        assert!(GgmlType::Q4K.row_bytes(100).is_err());
    }
}
