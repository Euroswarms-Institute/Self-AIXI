//! Quantized weight tensors and the rayon-parallel GEMV.
//!
//! A `QTensor` owns its raw block bytes (copied out of the mmap once at
//! load); `gemv` computes out[r] = ⟨row r, x⟩ with the fused dequant·dot
//! kernels, parallelized over independent output rows — deterministic by
//! construction since no reduction crosses a thread boundary.

use super::gguf::GgufFile;
use super::quant::{dequant_row, dot_row, GgmlType};
use rayon::prelude::*;

pub struct QTensor {
    pub name: String,
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    row_bytes: usize,
    data: Vec<u8>,
}

/// Rows per rayon work item (a row dot is ~2–6 µs; batch to amortize).
const GEMV_CHUNK: usize = 16;

impl QTensor {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf
            .tensor(name)
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        let ty = info.ggml_type()?;
        let data = gguf.tensor_data(info)?.to_vec();
        Ok(QTensor {
            name: name.to_string(),
            ty,
            rows: info.rows(),
            cols: info.cols(),
            row_bytes: ty.row_bytes(info.cols())?,
            data,
        })
    }

    pub fn from_raw(
        name: &str,
        ty: GgmlType,
        rows: usize,
        cols: usize,
        data: Vec<u8>,
    ) -> Result<Self, String> {
        let row_bytes = ty.row_bytes(cols)?;
        if data.len() != row_bytes * rows {
            return Err(format!(
                "tensor {name}: {} bytes for {rows}×{cols} {}",
                data.len(),
                ty.name()
            ));
        }
        Ok(QTensor {
            name: name.to_string(),
            ty,
            rows,
            cols,
            row_bytes,
            data,
        })
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    pub fn row(&self, r: usize) -> &[u8] {
        &self.data[r * self.row_bytes..(r + 1) * self.row_bytes]
    }

    /// Dequantize one row to f32 (used for norm weights, embed-row carving,
    /// and as the kernels' test reference).
    pub fn dequant_row_f32(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.cols];
        dequant_row(self.ty, self.row(r), &mut out);
        out
    }

    /// out[r] = ⟨row r, x⟩ for all rows, in parallel.
    pub fn gemv(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols, "{}: x len", self.name);
        assert_eq!(out.len(), self.rows, "{}: out len", self.name);
        out.par_chunks_mut(GEMV_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * GEMV_CHUNK;
                for (j, o) in chunk.iter_mut().enumerate() {
                    *o = dot_row(self.ty, self.row(base + j), x);
                }
            });
    }

    /// Single dot against one row (the 2-row unembedding uses this).
    pub fn dot(&self, r: usize, x: &[f32]) -> f32 {
        dot_row(self.ty, self.row(r), x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;
    use rand::Rng;

    fn random_f32_tensor(rows: usize, cols: usize, seed: u64) -> QTensor {
        let mut rng = seeded(seed);
        let mut data = Vec::with_capacity(rows * cols * 4);
        for _ in 0..rows * cols {
            data.extend_from_slice(&rng.random_range(-1.0f32..1.0).to_le_bytes());
        }
        QTensor::from_raw("w", GgmlType::F32, rows, cols, data).unwrap()
    }

    #[test]
    fn gemv_matches_naive_and_is_deterministic() {
        let w = random_f32_tensor(67, 40, 9); // deliberately not chunk-aligned
        let mut rng = seeded(1);
        let x: Vec<f32> = (0..40).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let mut out1 = vec![0f32; 67];
        let mut out2 = vec![0f32; 67];
        w.gemv(&x, &mut out1);
        w.gemv(&x, &mut out2);
        for r in 0..67 {
            let naive: f64 = (0..40)
                .map(|c| (w.dequant_row_f32(r)[c] * x[c]) as f64)
                .sum();
            assert!((out1[r] as f64 - naive).abs() < 1e-4);
            assert_eq!(
                out1[r].to_bits(),
                out2[r].to_bits(),
                "nondeterministic gemv"
            );
        }
    }
}
