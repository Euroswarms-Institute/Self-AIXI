//! Planning — the expectimax side of MC-AIXI
//! (Family A of the research roadmap).
//!
//! `rho_uct` approximates the finite-horizon expectimax over ξ by Monte-Carlo
//! tree search (JAIR §3); `expectimax` computes it exactly by enumeration on
//! tiny domains, serving as the ground truth ρUCT is tested against.

pub mod expectimax;
pub mod modal_byte;
pub mod rho_uct;
pub mod root_parallel;

/// Discounted horizon mass Σ_{k=0}^{m-1} γᵏ — the value-normalization scale.
pub(crate) fn discounted_span(gamma: f64, m: u32) -> f64 {
    if (gamma - 1.0).abs() < 1e-12 {
        m as f64
    } else {
        (1.0 - gamma.powi(m as i32)) / (1.0 - gamma)
    }
}
