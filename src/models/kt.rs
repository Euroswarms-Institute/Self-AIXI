//! Krichevsky–Trofimov estimator (JAIR §4.2, eq. 30–31).
//!
//! The KT estimator is the Bayes-optimal predictor for an unknown Bernoulli
//! parameter under the Jeffreys prior Beta(½, ½):
//!     P(next = b | c₀ zeros, c₁ ones) = (c_b + ½) / (c₀ + c₁ + 1).
//! It is the leaf estimator of every CTW node; order-0 KT itself is exactly
//! `CtwModel` with depth 0.

/// ln P_KT(next = `bit` | counts) — the log-probability increment a KT
/// estimator assigns to seeing `bit` after `counts`.
pub fn kt_log_increment(counts: [u32; 2], bit: u8) -> f64 {
    let total = (counts[0] + counts[1]) as f64;
    ((counts[bit as usize] as f64 + 0.5) / (total + 1.0)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_sequence() {
        // P_KT(1) = 1/2, P_KT(1|1) = 3/4, P_KT(0|11) = 1/6 ⇒ P(110) = 1/16.
        let mut counts = [0u32; 2];
        let mut log_p = 0.0;
        for &b in &[1u8, 1, 0] {
            log_p += kt_log_increment(counts, b);
            counts[b as usize] += 1;
        }
        assert!((log_p - (1.0f64 / 16.0).ln()).abs() < 1e-15);
    }

    #[test]
    fn symmetric_in_bit_flip() {
        assert_eq!(kt_log_increment([3, 7], 0), kt_log_increment([7, 3], 1));
    }

    #[test]
    fn sums_to_one() {
        for c0 in 0..5u32 {
            for c1 in 0..5u32 {
                let p = kt_log_increment([c0, c1], 0).exp() + kt_log_increment([c0, c1], 1).exp();
                assert!((p - 1.0).abs() < 1e-15);
            }
        }
    }
}
