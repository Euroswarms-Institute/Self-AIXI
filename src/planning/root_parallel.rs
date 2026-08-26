//! Root-parallel ρUCT (root parallelization in the sense of Chaslot,
//! Winands & van den Herik 2008, applied to the JAIR search): K workers
//! each run an independent ρUCT search against a private clone of ξ with a
//! private RNG, and the root action statistics are merged by visit count
//! and visit-weighted value. No locks, no shared tree, no communication
//! until the join.
//!
//! Determinism: worker seeds derive from the caller's seed by a fixed
//! mixing function and the merge reduces in worker order, so the chosen
//! action is a pure function of (model state, budget, seed, K) regardless
//! of thread scheduling. The clones make it sound: each worker mutates its
//! own ξ through the usual learn/revert discipline, and the committed model
//! is never touched, so there is nothing to restore afterwards.

use crate::env::DomainSpec;
use crate::models::EnvModel;
use crate::planning::rho_uct::{ActionStats, RhoUct, SearchBudget};
use crate::rng::seeded;

fn worker_seed(seed: u64, worker: usize) -> u64 {
    // splitmix64 step on (seed + worker + 1): decorrelates workers whose
    // base seeds differ by small integers.
    let mut z = seed
        .wrapping_add(worker as u64 + 1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Split a total simulation budget across `workers`, remainder to the
/// earliest workers, preserving the exact total.
fn split_budget(budget: SearchBudget, workers: usize) -> Vec<SearchBudget> {
    let total = budget.mc_simulations as usize;
    (0..workers)
        .map(|i| {
            let sims = total / workers + usize::from(i < total % workers);
            SearchBudget {
                mc_simulations: sims as u32,
                ..budget
            }
        })
        .collect()
}

/// Run `workers` independent ρUCT searches over clones of `model` (total
/// simulation count = `budget.mc_simulations`, split across workers) and
/// return the merged decision plus merged per-action root statistics.
/// Fails if the model does not support cloning (`EnvModel::try_clone_box`).
pub fn plan_root_parallel(
    model: &dyn EnvModel,
    spec: &DomainSpec,
    budget: SearchBudget,
    workers: usize,
    seed: u64,
) -> Result<(u64, Vec<ActionStats>), String> {
    assert!(workers >= 1);
    let budgets = split_budget(budget, workers);
    let mut clones = Vec::with_capacity(workers);
    for _ in 0..workers {
        clones.push(model.try_clone_box().ok_or_else(|| {
            format!(
                "{} does not support cloning; root-parallel search needs \
                 clonable models (CTW catalogs do, the LLM carves do not)",
                model.model_id()
            )
        })?);
    }

    let all_stats: Vec<Vec<ActionStats>> = std::thread::scope(|scope| {
        let handles: Vec<_> = clones
            .into_iter()
            .zip(budgets)
            .enumerate()
            .map(|(i, (mut m, b))| {
                scope.spawn(move || {
                    let mut search = RhoUct::new(*spec, b);
                    let mut rng = seeded(worker_seed(seed, i));
                    if b.mc_simulations > 0 {
                        search.plan(&mut *m, &mut rng);
                    }
                    search.root_stats()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("root-parallel worker panicked"))
            .collect()
    });

    // Merge in worker order: total visits, visit-weighted mean value.
    let n_actions = spec.num_actions as usize;
    let mut merged: Vec<ActionStats> = (0..n_actions as u64)
        .map(|action| ActionStats {
            action,
            visits: 0,
            value: 0.0,
        })
        .collect();
    for stats in &all_stats {
        for s in stats {
            let m = &mut merged[s.action as usize];
            let total = m.visits + s.visits;
            if total > 0 {
                m.value = (m.value * m.visits as f64 + s.value * s.visits as f64) / total as f64;
            }
            m.visits = total;
        }
    }
    // Same decision convention as the serial search: most visits, ties by
    // higher mean, then lower action id.
    let best = merged
        .iter()
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then(a.value.total_cmp(&b.value))
                .then(b.action.cmp(&a.action))
        })
        .map(|s| s.action)
        .unwrap();
    Ok((best, merged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctw::CtwModel;
    use crate::models::mixture::BayesMixture;
    use crate::models::uniform::UniformModel;

    fn trained_mixture() -> BayesMixture {
        let mut m = BayesMixture::uniform(vec![
            Box::new(CtwModel::new(2)) as Box<dyn EnvModel>,
            Box::new(UniformModel::default()),
        ]);
        // Deterministic pattern: percept equals the action, reward follows.
        for i in 0..30u64 {
            let a = (i % 2) as u8;
            m.append_history_symbols(&[a]);
            m.learn_symbols(&[a, a]);
        }
        m
    }

    fn spec() -> DomainSpec {
        DomainSpec {
            num_actions: 2,
            action_bits: 1,
            observation_bits: 1,
            reward_bits: 1,
            reward_min: 0.0,
            reward_max: 1.0,
        }
    }

    #[test]
    fn clones_are_independent_and_the_original_is_untouched() {
        let m = trained_mixture();
        let root = m.root_log_probability();
        let mut clone = m.try_clone_box().expect("ctw mixtures clone");
        clone.learn_symbols(&[1, 0, 1]);
        assert_ne!(clone.root_log_probability().to_bits(), root.to_bits());
        assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
    }

    #[test]
    fn parallel_matches_the_exact_action_and_is_deterministic() {
        let m = trained_mixture();
        let spec = spec();
        let budget = SearchBudget::new(2000, 3, 1.4, 1.0).unwrap();
        let root_before = m.root_log_probability();
        let (a1, s1) = plan_root_parallel(&m, &spec, budget, 4, 7).unwrap();
        let (a2, s2) = plan_root_parallel(&m, &spec, budget, 4, 7).unwrap();
        assert_eq!(a1, a2, "same seed must reproduce the decision");
        assert_eq!(
            s1.iter().map(|s| s.visits).collect::<Vec<_>>(),
            s2.iter().map(|s| s.visits).collect::<Vec<_>>()
        );
        // The trained pattern rewards repeating action 1's percept: exact
        // expectimax picks 1 (checked against the serial search elsewhere);
        // the merged parallel decision must agree.
        assert_eq!(a1, 1);
        assert_eq!(
            s1.iter().map(|s| s.visits).sum::<u64>(),
            2000,
            "split budget must preserve the total"
        );
        assert_eq!(m.root_log_probability().to_bits(), root_before.to_bits());
    }
}
