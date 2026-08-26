//! ρUCT vs exact expectimax (JAIR §3: ρUCT's value estimate converges to the
//! finite-horizon expectimax over the same ξ). We train a ξ whose optimal
//! action is nontrivial, enumerate the exact Q values, and require the search
//! to (a) recover the argmax and (b) approach the exact value.

use mc_aixi::env::DomainSpec;
use mc_aixi::models::ctw::CtwModel;
use mc_aixi::models::mixture::BayesMixture;
use mc_aixi::models::uniform::UniformModel;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::expectimax::{exact_expectimax, exact_q_values};
use mc_aixi::planning::rho_uct::{RhoUct, SearchBudget};
use mc_aixi::rng::seeded;

fn coin_spec() -> DomainSpec {
    DomainSpec {
        num_actions: 2,
        action_bits: 1,
        observation_bits: 1,
        reward_bits: 1,
        reward_min: 0.0,
        reward_max: 1.0,
    }
}

/// ξ that has learned "action 1 earns reward 1, action 0 earns reward 0"
/// through ordinary FAC updates.
fn trained_model() -> BayesMixture {
    let mut m = BayesMixture::uniform(vec![
        Box::new(CtwModel::new(2)) as Box<dyn EnvModel>,
        Box::new(CtwModel::new(4)),
        Box::new(UniformModel::default()),
    ]);
    for i in 0..40u64 {
        let a = (i % 2) as u8;
        m.append_history_symbols(&[a]);
        m.learn_symbols(&[a, a]); // observation echoes action; reward = action
    }
    m
}

#[test]
fn recovers_exact_argmax_and_value() {
    let spec = coin_spec();
    let mut model = trained_model();
    let before = model.root_log_probability();

    let (best_exact, v_exact) = exact_expectimax(&mut model, &spec, 1.0, 3);
    let qs = exact_q_values(&mut model, &spec, 1.0, 3);
    assert_eq!(best_exact, 1, "trained xi must prefer the rewarded action");
    assert!(qs[1] > qs[0]);
    assert!((model.root_log_probability() - before).abs() < 1e-12);

    let mut search = RhoUct::new(spec, SearchBudget::new(50_000, 3, 1.4, 1.0).unwrap());
    let mut rng = seeded(1);
    let best_uct = search.plan(&mut model, &mut rng);
    assert!((model.root_log_probability() - before).abs() < 1e-12);
    assert_eq!(best_uct, best_exact);

    let stats = search.root_stats();
    let uct_value = stats[best_uct as usize].value;
    assert!(
        (uct_value - v_exact).abs() < 0.05,
        "rho-UCT value {uct_value} vs exact {v_exact}"
    );
    // The optimal arm must dominate the visit distribution.
    assert!(stats[1].visits > 10 * stats[0].visits);
}

#[test]
fn horizon_one_matches_exact_q_closely() {
    let spec = coin_spec();
    let mut model = trained_model();
    let qs = exact_q_values(&mut model, &spec, 1.0, 1);

    let mut search = RhoUct::new(spec, SearchBudget::new(30_000, 1, 1.4, 1.0).unwrap());
    let mut rng = seeded(9);
    let best = search.plan(&mut model, &mut rng);
    assert_eq!(best, 1);
    for s in search.root_stats() {
        assert!(
            (s.value - qs[s.action as usize]).abs() < 0.05,
            "action {}: uct {} exact {}",
            s.action,
            s.value,
            qs[s.action as usize]
        );
    }
}
