//! Offline smoke suite — the Rust analog of `experiments/run_smoke.py`:
//! fast PASS/FAIL invariant checks over the Family-A stack, no model files
//! or network required. Exit code 0 iff everything passes.

use mc_aixi::agent::{run_cycles, AixiAgent};
use mc_aixi::env::coin_flip::CoinFlip;
use mc_aixi::env::{DomainSpec, Environment};
use mc_aixi::models::ctw::CtwModel;
use mc_aixi::models::fac_ctw::FacCtwModel;
use mc_aixi::models::kt::kt_log_increment;
use mc_aixi::models::mixture::BayesMixture;
use mc_aixi::models::uniform::UniformModel;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::expectimax::exact_expectimax;
use mc_aixi::planning::rho_uct::{RhoUct, SearchBudget};
use mc_aixi::rng::seeded;
use rand::Rng;

fn report(ok: bool, name: &str, detail: String) -> bool {
    println!("{} {name} — {detail}", if ok { "PASS" } else { "FAIL" });
    ok
}

fn kt_closed_form() -> bool {
    let mut counts = [0u32; 2];
    let mut log_p = 0.0;
    for &b in &[1u8, 1, 0] {
        log_p += kt_log_increment(counts, b);
        counts[b as usize] += 1;
    }
    let err = (log_p - (1.0f64 / 16.0).ln()).abs();
    report(err < 1e-15, "kt-closed-form", format!("|err| = {err:.2e}"))
}

fn ctw_revert_exactness() -> bool {
    let mut rng = seeded(42);
    let mut m = CtwModel::new(8); // Phase-0 parity depth
    for _ in 0..64 {
        let b = u8::from(rng.random_bool(0.6));
        if rng.random_bool(0.25) {
            m.append_history_symbols(&[b]);
        } else {
            m.learn_symbols(&[b]);
        }
    }
    let digest = m.state_digest();
    let root = m.root_log_probability();
    for _ in 0..300 {
        let mut ops = Vec::new();
        for _ in 0..rng.random_range(1..10) {
            let learned = rng.random_bool(0.5);
            let n = rng.random_range(1..4);
            let bits: Vec<u8> = (0..n).map(|_| u8::from(rng.random_bool(0.5))).collect();
            if learned {
                m.learn_symbols(&bits);
            } else {
                m.append_history_symbols(&bits);
            }
            ops.push((learned, n));
        }
        for (learned, n) in ops.into_iter().rev() {
            if learned {
                m.revert_learned_symbols(n);
            } else {
                m.revert_history_symbols(n);
            }
        }
    }
    let ok = m.state_digest() == digest && m.root_log_probability().to_bits() == root.to_bits();
    report(
        ok,
        "ctw-revert-exact",
        format!("root ln P = {root:.6} restored bit-exactly"),
    )
}

fn mixture_hand_math() -> bool {
    let mut m = BayesMixture::uniform(vec![
        Box::new(CtwModel::new(0)) as Box<dyn EnvModel>,
        Box::new(UniformModel::default()),
    ]);
    m.learn_symbols(&[1, 1, 1]);
    let e1 = (m.root_log_probability() - (7.0f64 / 32.0).ln()).abs();
    let e2 = (m.posterior_weights()[0] - 5.0 / 7.0).abs();
    let e3 = (m.predict_bit_probability(1) - 43.0 / 56.0).abs();
    let ok = e1 < 1e-13 && e2 < 1e-13 && e3 < 1e-13;
    report(
        ok,
        "mixture-bayes-math",
        format!("errors {e1:.1e}/{e2:.1e}/{e3:.1e}"),
    )
}

fn rho_uct_vs_expectimax() -> bool {
    let spec = DomainSpec {
        num_actions: 2,
        action_bits: 1,
        observation_bits: 1,
        reward_bits: 1,
        reward_min: 0.0,
        reward_max: 1.0,
    };
    let mut model = BayesMixture::uniform(vec![
        Box::new(CtwModel::new(2)) as Box<dyn EnvModel>,
        Box::new(UniformModel::default()),
    ]);
    for i in 0..30u64 {
        let a = (i % 2) as u8;
        model.append_history_symbols(&[a]);
        model.learn_symbols(&[a, a]);
    }
    let (best_exact, v_exact) = exact_expectimax(&mut model, &spec, 1.0, 3);
    let mut search = RhoUct::new(spec, SearchBudget::new(4000, 3, 1.4, 1.0).unwrap());
    let mut rng = seeded(7);
    let before = model.root_log_probability();
    let best = search.plan(&mut model, &mut rng);
    let drift = (model.root_log_probability() - before).abs();
    let v_uct = search.root_stats()[best as usize].value;
    let ok = best == best_exact && (v_uct - v_exact).abs() < 0.1 && drift < 1e-9;
    report(
        ok,
        "rho-uct-expectimax",
        format!("argmax {best}=={best_exact}, V {v_uct:.3}≈{v_exact:.3}, revert drift {drift:.1e}"),
    )
}

fn coin_flip_learning() -> bool {
    let mut env = CoinFlip::default();
    let spec = DomainSpec::from_env(&env);
    let pbits = spec.percept_bits() as usize;
    let model = BayesMixture::uniform(vec![
        Box::new(FacCtwModel::new(2, pbits)) as Box<dyn EnvModel>,
        Box::new(FacCtwModel::new(8, pbits)),
        Box::new(UniformModel::default()),
    ]);
    let mut agent = AixiAgent::new(model, spec, SearchBudget::new(300, 2, 0.35, 0.99).unwrap());
    let mut rng = seeded(42);
    env.reset(&mut rng);
    let records = run_cycles(&mut agent, &mut env, 200, &mut rng);
    let tail = &records[100..];
    let avg: f64 = tail.iter().map(|r| r.reward).sum::<f64>() / tail.len() as f64;
    report(
        avg > 0.53,
        "coin-flip-learning",
        format!("late average reward {avg:.3} (optimum 0.6)"),
    )
}

fn main() {
    println!("mc-aixi smoke suite (offline)");
    let ok = [
        kt_closed_form(),
        ctw_revert_exactness(),
        mixture_hand_math(),
        rho_uct_vs_expectimax(),
        coin_flip_learning(),
    ]
    .iter()
    .all(|&b| b);
    println!(
        "{}",
        if ok {
            "ALL PASS"
        } else {
            "SMOKE FAILURES PRESENT"
        }
    );
    std::process::exit(i32::from(!ok));
}
