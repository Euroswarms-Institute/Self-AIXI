//! End-to-end Family-A regression: the MC-AIXI agent on CoinFlip(0.6) with a
//! FAC-CTW mixture must learn to bet on the biased side (average reward
//! → 0.6) and the mixture posterior must abandon the uniform noise floor.

use mc_aixi::agent::{run_cycles, AixiAgent};
use mc_aixi::env::coin_flip::CoinFlip;
use mc_aixi::env::{DomainSpec, Environment};
use mc_aixi::models::fac_ctw::FacCtwModel;
use mc_aixi::models::mixture::BayesMixture;
use mc_aixi::models::uniform::UniformModel;
use mc_aixi::models::EnvModel;
use mc_aixi::planning::rho_uct::SearchBudget;
use mc_aixi::rng::seeded;

#[test]
fn learns_to_bet_on_the_biased_side() {
    let mut env = CoinFlip::default();
    let spec = DomainSpec::from_env(&env);
    let pbits = spec.percept_bits() as usize;
    let model = BayesMixture::uniform(vec![
        Box::new(FacCtwModel::new(2, pbits)) as Box<dyn EnvModel>,
        Box::new(FacCtwModel::new(8, pbits)), // Phase-0 parity depth
        Box::new(UniformModel::default()),
    ]);
    // CoinFlip's normalized Q-gap is ~0.16 at horizon 2, so the budget must
    // separate visits: 300 simulations at c = 0.35.
    let budget = SearchBudget::new(300, 2, 0.35, 0.99).unwrap();
    let mut agent = AixiAgent::new(model, spec, budget);
    let mut rng = seeded(42);
    env.reset(&mut rng);

    let records = run_cycles(&mut agent, &mut env, 400, &mut rng);
    let tail = &records[200..];
    let avg: f64 = tail.iter().map(|r| r.reward).sum::<f64>() / tail.len() as f64;
    assert!(
        avg > 0.53,
        "late average reward {avg} — agent failed to exploit the bias"
    );

    // The biased side should dominate late actions outright.
    let heads = tail.iter().filter(|r| r.action == 1).count();
    assert!(
        heads * 10 > tail.len() * 8,
        "action-1 rate {}/{}",
        heads,
        tail.len()
    );

    // Posterior: the uniform floor must lose to the FAC-CTW components.
    let weights = agent.model.posterior_weights();
    let ids = agent.model.component_ids();
    let uniform_idx = ids.iter().position(|s| s == "uniform").unwrap();
    assert!(
        weights[uniform_idx] < 0.2,
        "uniform kept weight {}",
        weights[uniform_idx]
    );

    // Root log-probability is a genuine log-likelihood: finite and negative.
    let lp = agent.model.root_log_probability();
    assert!(lp.is_finite() && lp < 0.0);
}
