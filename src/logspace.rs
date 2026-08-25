//! Log-domain arithmetic helpers (IMPLEMENTATION_PLAN.md §3: all model
//! probabilities live in log space; JAIR §5 computes CTW weights the same way).

/// ln(1/2), the branching prior weight of the CTW recursion (JAIR eq. 33).
pub const LOG_HALF: f64 = -core::f64::consts::LN_2;

/// Numerically stable ln(exp(a) + exp(b)).
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// Numerically stable ln Σᵢ exp(xᵢ) over a slice (NEG_INFINITY for empty).
pub fn log_sum_exp_slice(xs: &[f64]) -> f64 {
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = xs.iter().map(|&x| (x - hi).exp()).sum();
    hi + sum.ln()
}

/// Normalized weights exp(xᵢ − ln Σ exp(x)) (posterior from unnormalized logs).
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    let z = log_sum_exp_slice(xs);
    xs.iter().map(|&x| (x - z).exp()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_sum() {
        let xs: [f64; 4] = [-1.5, -2.25, -0.75, -10.0];
        let naive: f64 = xs.iter().map(|&x| x.exp()).sum::<f64>().ln();
        assert!((log_sum_exp_slice(&xs) - naive).abs() < 1e-14);
        assert!((log_sum_exp(xs[0], xs[1]) - (xs[0].exp() + xs[1].exp()).ln()).abs() < 1e-14);
    }

    #[test]
    fn handles_neg_infinity() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, -3.0), -3.0);
        assert_eq!(log_sum_exp(-3.0, f64::NEG_INFINITY), -3.0);
        assert_eq!(log_sum_exp_slice(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn softmax_normalizes() {
        let w = softmax(&[-1.0, -2.0, -3.0]);
        let total: f64 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(w[0] > w[1] && w[1] > w[2]);
    }
}
