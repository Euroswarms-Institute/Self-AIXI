//! First-principles verification of the CTW recursion (JAIR §4.3):
//! `CtwModel`'s root log-probability must equal the explicitly enumerated
//! Bayesian mixture over ALL prediction suffix trees of depth ≤ D under the
//! natural prior 2^(−Γ_D(T)) — including zero-padded contexts and the FAC
//! learn/append split. Tolerance 1e-12 on strings of ≤ 12 bits, depths 0–4.

use mc_aixi::logspace::{log_sum_exp_slice, LOG_HALF};
use mc_aixi::models::ctw::CtwModel;
use mc_aixi::models::kt::kt_log_increment;
use mc_aixi::models::EnvModel;
use mc_aixi::rng::seeded;
use rand::Rng;
use std::collections::HashMap;

#[derive(Clone)]
enum Pst {
    Leaf,
    Split(Box<Pst>, Box<Pst>),
}

/// All suffix-tree shapes with remaining depth budget `budget`, with their
/// description cost Γ: a leaf at the depth bound is free (forced), any other
/// node costs one bit (leaf-or-split decision).
fn enumerate_trees(budget: usize) -> Vec<(Pst, u32)> {
    if budget == 0 {
        return vec![(Pst::Leaf, 0)];
    }
    let subs = enumerate_trees(budget - 1);
    let mut out = vec![(Pst::Leaf, 1)];
    for (t0, c0) in &subs {
        for (t1, c1) in &subs {
            out.push((
                Pst::Split(Box::new(t0.clone()), Box::new(t1.clone())),
                1 + c0 + c1,
            ));
        }
    }
    out
}

/// Sequence log-probability under a single suffix tree: every learned bit is
/// routed by its (zero-padded, most-recent-first) context to a leaf KT
/// estimator; appended bits only extend the context.
fn tree_log_prob(tree: &Pst, ops: &[(bool, u8)]) -> f64 {
    let mut counts: HashMap<Vec<u8>, [u32; 2]> = HashMap::new();
    let mut history: Vec<u8> = Vec::new();
    let mut log_p = 0.0;
    for &(learned, bit) in ops {
        if learned {
            let mut node = tree;
            let mut path = Vec::new();
            let mut dist = 1usize;
            while let Pst::Split(c0, c1) = node {
                let cb = if dist <= history.len() {
                    history[history.len() - dist]
                } else {
                    0
                };
                path.push(cb);
                node = if cb == 0 { c0 } else { c1 };
                dist += 1;
            }
            let leaf = counts.entry(path).or_insert([0, 0]);
            log_p += kt_log_increment(*leaf, bit);
            leaf[bit as usize] += 1;
        }
        history.push(bit);
    }
    log_p
}

fn brute_force_log_prob(depth: usize, ops: &[(bool, u8)]) -> f64 {
    let terms: Vec<f64> = enumerate_trees(depth)
        .iter()
        .map(|(tree, cost)| *cost as f64 * LOG_HALF + tree_log_prob(tree, ops))
        .collect();
    log_sum_exp_slice(&terms)
}

fn ctw_log_prob(depth: usize, ops: &[(bool, u8)]) -> f64 {
    let mut m = CtwModel::new(depth);
    for &(learned, bit) in ops {
        if learned {
            m.learn_symbols(&[bit]);
        } else {
            m.append_history_symbols(&[bit]);
        }
    }
    m.root_log_probability()
}

fn assert_parity(depth: usize, ops: &[(bool, u8)]) {
    let brute = brute_force_log_prob(depth, ops);
    let ctw = ctw_log_prob(depth, ops);
    assert!(
        (brute - ctw).abs() < 1e-12,
        "depth {depth}: brute-force {brute} vs CTW {ctw} on {ops:?}"
    );
}

#[test]
fn prior_is_normalized() {
    // Σ_T 2^(−Γ) = 1 for every depth bound: the empty-sequence mixture is 1.
    for depth in 0..=4 {
        let total = brute_force_log_prob(depth, &[]);
        assert!(total.abs() < 1e-12, "depth {depth}: prior mass {total}");
    }
}

#[test]
fn matches_brute_force_on_learned_streams() {
    let mut rng = seeded(2026);
    for depth in 0..=4 {
        for _ in 0..4 {
            let ops: Vec<(bool, u8)> = (0..12)
                .map(|_| (true, u8::from(rng.random_bool(0.65))))
                .collect();
            assert_parity(depth, &ops);
        }
    }
}

#[test]
fn matches_brute_force_with_fac_interleaving() {
    let mut rng = seeded(31337);
    for depth in 0..=4 {
        for _ in 0..4 {
            // Cycles of one appended action bit + two learned percept bits.
            let mut ops = Vec::new();
            for _ in 0..4 {
                ops.push((false, u8::from(rng.random_bool(0.5))));
                ops.push((true, u8::from(rng.random_bool(0.3))));
                ops.push((true, u8::from(rng.random_bool(0.8))));
            }
            assert_parity(depth, &ops);
        }
    }
}

/// FAC-CTW must equal the *product over percept bit positions* of
/// brute-force PST mixtures, where position p's factor learns only its own
/// bits (all other learned bits demoted to context-only appends).
#[test]
fn fac_ctw_matches_product_of_brute_force_factors() {
    use mc_aixi::models::fac_ctw::FacCtwModel;
    let mut rng = seeded(777);
    for depth in 0..=3 {
        for pbits in 1..=3usize {
            // Interleaved cycles: 1 action bit + `pbits` percept bits.
            let mut ops: Vec<(bool, u8)> = Vec::new();
            for _ in 0..4 {
                ops.push((false, u8::from(rng.random_bool(0.5))));
                for _ in 0..pbits {
                    ops.push((true, u8::from(rng.random_bool(0.7))));
                }
            }

            let mut fac = FacCtwModel::new(depth, pbits);
            for &(learned, bit) in &ops {
                if learned {
                    fac.learn_symbols(&[bit]);
                } else {
                    fac.append_history_symbols(&[bit]);
                }
            }

            let mut expected = 0.0;
            for p in 0..pbits {
                let mut pos = 0usize;
                let masked: Vec<(bool, u8)> = ops
                    .iter()
                    .map(|&(learned, bit)| {
                        if learned {
                            let mine = pos % pbits == p;
                            pos += 1;
                            (mine, bit)
                        } else {
                            (false, bit)
                        }
                    })
                    .collect();
                expected += brute_force_log_prob(depth, &masked);
            }
            let got = fac.root_log_probability();
            assert!(
                (got - expected).abs() < 1e-12,
                "depth {depth} pbits {pbits}: fac {got} vs brute {expected}"
            );
        }
    }
}

#[test]
fn matches_brute_force_on_degenerate_streams() {
    for depth in 0..=4 {
        assert_parity(depth, &[(true, 0); 12]);
        let alternating: Vec<(bool, u8)> = (0..12).map(|i| (true, (i % 2) as u8)).collect();
        assert_parity(depth, &alternating);
    }
}
