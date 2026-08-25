//! Partial rotary position embedding (rotate-half / NeoX convention).
//!
//! qwen3_5 config: `partial_rotary_factor = 0.25` (64 of 256 dims per head),
//! θ = 10⁷, `mrope_interleaved` with sections [11, 11, 10]. For a text-only
//! token stream all three MRoPE position components carry the same index, so
//! the section assignment is irrelevant and MRoPE collapses to standard
//! RoPE — an assumption the llama.cpp oracle validates end-to-end.

pub struct Rope {
    /// θ^(−2i/rope_dims) for i in 0..rope_dims/2.
    inv_freq: Vec<f32>,
}

impl Rope {
    pub fn new(rope_dims: usize, theta: f32) -> Self {
        let half = rope_dims / 2;
        let inv_freq = (0..half)
            .map(|i| theta.powf(-2.0 * i as f32 / rope_dims as f32))
            .collect();
        Rope { inv_freq }
    }

    /// Rotate the first `2·|inv_freq|` dims of one head vector in place;
    /// pair (i, i+half): x_i ← x_i·cos − x_{i+half}·sin, and symmetrically.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.inv_freq.len();
        for i in 0..half {
            let angle = pos as f32 * self.inv_freq[i];
            let (sin, cos) = angle.sin_cos();
            let a = x[i];
            let b = x[i + half];
            x[i] = a * cos - b * sin;
            x[i + half] = b * cos + a * sin;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_identity_and_rotation_preserves_norm() {
        let rope = Rope::new(4, 1e7);
        let mut x = vec![0.3f32, -0.7, 1.1, 0.2, 9.0, -9.0]; // dims 4.. untouched
        let orig = x.clone();
        rope.apply(&mut x, 0);
        assert_eq!(x, orig);
        rope.apply(&mut x, 1234);
        assert_eq!(&x[4..], &orig[4..], "non-rotary dims must pass through");
        let n0: f32 = orig[..4].iter().map(|v| v * v).sum();
        let n1: f32 = x[..4].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
        assert!(x[..4] != orig[..4]);
    }
}
