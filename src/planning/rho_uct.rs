//! ρUCT — Monte-Carlo expectimax approximation over ξ
//! (JAIR §3, Algorithms 1–3; IMPLEMENTATION_PLAN.md §4.1).
//!
//! The search tree alternates decision nodes (agent chooses, UCB1 with
//! horizon-normalized values, JAIR eq. 14) and chance nodes (percepts sampled
//! bit-by-bit from ξ so intra-percept bits condition on one another). One new
//! decision node is expanded per simulation; beyond the frontier a
//! uniform-random rollout policy estimates the return (JAIR Alg. 4).
//!
//! Every simulation extends ξ with imagined bits via `append`/`learn` and
//! unwinds them afterwards in strict per-cycle LIFO order — the
//! `xi_rollouts.py` contract; a debug assertion checks the root
//! log-probability is restored.

use crate::encoding::{decode_bits_msb, encode_bits_msb};
use crate::env::DomainSpec;
use crate::models::EnvModel;
use crate::rng::AgentRng;
use rand::Rng;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Finite per-decision search budget (§1.1: every knob validated, no
/// unbounded anything; §1 ledger: `gamma`).
#[derive(Clone, Copy, Debug)]
pub struct SearchBudget {
    pub mc_simulations: u32,
    pub horizon: u32,
    pub uct_c: f64,
    pub gamma: f64,
}

impl SearchBudget {
    pub fn new(mc_simulations: u32, horizon: u32, uct_c: f64, gamma: f64) -> Result<Self, String> {
        if mc_simulations == 0 {
            return Err("mc_simulations must be >= 1".into());
        }
        if horizon == 0 {
            return Err("horizon must be >= 1".into());
        }
        if !uct_c.is_finite() || uct_c < 0.0 {
            return Err(format!("uct_c must be finite and >= 0, got {uct_c}"));
        }
        if !(gamma > 0.0 && gamma <= 1.0) {
            return Err(format!("gamma must be in (0, 1], got {gamma}"));
        }
        Ok(SearchBudget {
            mc_simulations,
            horizon,
            uct_c,
            gamma,
        })
    }
}

#[derive(Default)]
struct DecisionNode {
    visits: u64,
    children: HashMap<u64, ChanceNode>,
}

#[derive(Default)]
struct ChanceNode {
    visits: u64,
    /// Incremental mean of sampled returns from this (history, action).
    value: f64,
    children: HashMap<u64, DecisionNode>,
}

/// Aggregate statistics of one root action after a `plan` call.
#[derive(Clone, Copy, Debug)]
pub struct ActionStats {
    pub action: u64,
    pub visits: u64,
    pub value: f64,
}

pub struct RhoUct {
    spec: DomainSpec,
    budget: SearchBudget,
    root: DecisionNode,
}

impl RhoUct {
    pub fn new(spec: DomainSpec, budget: SearchBudget) -> Self {
        RhoUct {
            spec,
            budget,
            root: DecisionNode::default(),
        }
    }

    pub fn budget(&self) -> SearchBudget {
        self.budget
    }

    /// Run `mc_simulations` simulations from the current root and return the
    /// action with the most visits (ties: higher mean, then lower action id —
    /// the `aixi/planning/mcts.py` convention).
    pub fn plan(&mut self, model: &mut dyn EnvModel, rng: &mut AgentRng) -> u64 {
        for _ in 0..self.budget.mc_simulations {
            let root_log_p = model.root_log_probability();
            let mut cycles = 0u32;
            sample_decision(
                &self.spec,
                &self.budget,
                &mut self.root,
                model,
                self.budget.horizon,
                rng,
                &mut cycles,
            );
            unwind_imagination(&self.spec, model, cycles);
            debug_assert!(
                (model.root_log_probability() - root_log_p).abs() < 1e-9,
                "xi root log-probability drifted after imagination revert"
            );
        }
        self.best_root_action()
    }

    fn best_root_action(&self) -> u64 {
        let mut best = 0u64;
        let mut best_key = (0u64, f64::NEG_INFINITY);
        for a in 0..self.spec.num_actions {
            let key = match self.root.children.get(&a) {
                Some(c) if c.visits > 0 => (c.visits, c.value),
                _ => (0, f64::NEG_INFINITY),
            };
            if key.0 > best_key.0 || (key.0 == best_key.0 && key.1 > best_key.1) {
                best_key = key;
                best = a;
            }
        }
        best
    }

    pub fn root_stats(&self) -> Vec<ActionStats> {
        (0..self.spec.num_actions)
            .map(|a| match self.root.children.get(&a) {
                Some(c) => ActionStats {
                    action: a,
                    visits: c.visits,
                    value: c.value,
                },
                None => ActionStats {
                    action: a,
                    visits: 0,
                    value: 0.0,
                },
            })
            .collect()
    }

    /// Discard the search tree (fresh root for the next real decision).
    ///
    /// This is the default between real cycles: under a *receding* horizon,
    /// advancing the root shifts every kept node one step closer to the
    /// agent, so cached chance-node means — averages of returns truncated at
    /// the OLD remaining horizon — understate the value at their NEW depth.
    /// At small horizons that bias is large enough to invert action
    /// rankings, so soundness demands a rebuild.
    pub fn reset_root(&mut self) {
        self.root = DecisionNode::default();
    }

    /// Tree reuse across real cycles (JAIR §3.4): the subtree under the real
    /// (action, percept) becomes the next root; unseen branches start fresh.
    ///
    /// Caveat (documented deviation): reused value means were computed for a
    /// shorter remaining horizon than the node now has, an inconsistency the
    /// JAIR setup tolerates only because its per-cycle simulation counts
    /// dwarf the carried visit counts. Prefer `reset_root` unless simulation
    /// budgets are large relative to the reused subtree.
    pub fn advance_root(&mut self, action: u64, percept_code: u64) {
        let next = self
            .root
            .children
            .remove(&action)
            .and_then(|mut chance| chance.children.remove(&percept_code));
        self.root = next.unwrap_or_default();
    }
}

/// Undo one simulation's imagined bits: per interaction cycle, the percept
/// bits (learned) then the action bits (appended), most recent cycle first.
fn unwind_imagination(spec: &DomainSpec, model: &mut dyn EnvModel, cycles: u32) {
    for _ in 0..cycles {
        model.revert_learned_symbols(spec.percept_bits() as usize);
        model.revert_history_symbols(spec.action_bits as usize);
    }
}

fn sample_decision(
    spec: &DomainSpec,
    budget: &SearchBudget,
    node: &mut DecisionNode,
    model: &mut dyn EnvModel,
    depth: u32,
    rng: &mut AgentRng,
    cycles: &mut u32,
) -> f64 {
    if depth == 0 {
        return 0.0;
    }
    let action = select_action(spec, budget, node, depth, rng);
    append_action(spec, model, action);
    let chance = node.children.entry(action).or_default();

    let (percept_code, reward) = sample_percept(spec, model, rng, cycles);
    let future = if depth == 1 {
        0.0
    } else {
        match chance.children.entry(percept_code) {
            Entry::Occupied(child) => sample_decision(
                spec,
                budget,
                child.into_mut(),
                model,
                depth - 1,
                rng,
                cycles,
            ),
            Entry::Vacant(slot) => {
                // Expansion: one new decision node per simulation; its value
                // is estimated by a random-policy rollout (JAIR Alg. 2/4).
                slot.insert(DecisionNode::default());
                rollout(spec, budget, model, depth - 1, rng, cycles)
            }
        }
    };
    let ret = reward + budget.gamma * future;
    chance.visits += 1;
    chance.value += (ret - chance.value) / chance.visits as f64;
    node.visits += 1;
    ret
}

/// UCB1 action selection with values normalized onto [0,1] by the reachable
/// return span over the remaining horizon (JAIR eq. 14 generalized to γ ≤ 1).
fn select_action(
    spec: &DomainSpec,
    budget: &SearchBudget,
    node: &DecisionNode,
    depth: u32,
    rng: &mut AgentRng,
) -> u64 {
    let mut unexplored = [0u64; 64];
    let mut n_unexplored = 0usize;
    for a in 0..spec.num_actions {
        let fresh = match node.children.get(&a) {
            Some(c) => c.visits == 0,
            None => true,
        };
        if fresh {
            unexplored[n_unexplored] = a;
            n_unexplored += 1;
        }
    }
    if n_unexplored > 0 {
        return unexplored[rng.random_range(0..n_unexplored)];
    }

    let span = super::discounted_span(budget.gamma, depth);
    let value_span = span * (spec.reward_max - spec.reward_min);
    let value_lo = span * spec.reward_min;
    let ln_n = (node.visits.max(1) as f64).ln();
    let mut best = 0u64;
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..spec.num_actions {
        let c = &node.children[&a];
        let exploit = if value_span > 0.0 {
            (c.value - value_lo) / value_span
        } else {
            0.5
        };
        let score = exploit + budget.uct_c * (ln_n / c.visits as f64).sqrt();
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

fn append_action(spec: &DomainSpec, model: &mut dyn EnvModel, action: u64) {
    let mut bits = Vec::with_capacity(spec.action_bits as usize);
    encode_bits_msb(action, spec.action_bits, &mut bits);
    model.append_history_symbols(&bits);
}

/// Draw a percept from ξ bit-by-bit: each bit is predicted, sampled, then
/// learned so the following bits are conditioned on it. Returns the percept's
/// code (all bits MSB-first) and its decoded reward. Increments `cycles`.
fn sample_percept(
    spec: &DomainSpec,
    model: &mut dyn EnvModel,
    rng: &mut AgentRng,
    cycles: &mut u32,
) -> (u64, f64) {
    let n = spec.percept_bits() as usize;
    let mut bits = [0u8; 64];
    for slot in bits.iter_mut().take(n) {
        let p1 = model.predict_bit_probability(1).clamp(0.0, 1.0);
        let b = u8::from(rng.random_bool(p1));
        model.learn_symbols(&[b]);
        *slot = b;
    }
    *cycles += 1;
    let code = decode_bits_msb(&bits[..n]);
    let reward_code = decode_bits_msb(&bits[spec.observation_bits as usize..n]);
    (code, spec.decode_reward(reward_code))
}

/// Random-policy playout to the horizon (JAIR Alg. 4).
fn rollout(
    spec: &DomainSpec,
    budget: &SearchBudget,
    model: &mut dyn EnvModel,
    depth: u32,
    rng: &mut AgentRng,
    cycles: &mut u32,
) -> f64 {
    let mut total = 0.0;
    let mut discount = 1.0;
    for _ in 0..depth {
        let action = rng.random_range(0..spec.num_actions);
        append_action(spec, model, action);
        let (_, reward) = sample_percept(spec, model, rng, cycles);
        total += discount * reward;
        discount *= budget.gamma;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ctw::CtwModel;
    use crate::rng::seeded;

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

    #[test]
    fn budget_validation() {
        assert!(SearchBudget::new(0, 4, 1.4, 0.99).is_err());
        assert!(SearchBudget::new(2, 0, 1.4, 0.99).is_err());
        assert!(SearchBudget::new(2, 4, -1.0, 0.99).is_err());
        assert!(SearchBudget::new(2, 4, 1.4, 0.0).is_err());
        assert!(SearchBudget::new(2, 4, 1.4, 1.1).is_err());
        assert!(SearchBudget::new(2, 4, 1.4, 1.0).is_ok());
    }

    #[test]
    fn plan_restores_model_exactly() {
        let mut model = CtwModel::new(4);
        model.learn_symbols(&[1, 0, 1, 1, 0, 1]);
        let digest = model.state_digest();
        let mut search = RhoUct::new(coin_spec(), SearchBudget::new(200, 4, 1.4, 0.99).unwrap());
        let mut rng = seeded(5);
        let _ = search.plan(&mut model, &mut rng);
        assert_eq!(model.state_digest(), digest);
    }

    #[test]
    fn plan_is_deterministic_under_seed() {
        let run = || {
            let mut model = CtwModel::new(4);
            model.learn_symbols(&[1, 1, 0, 1]);
            let mut search = RhoUct::new(coin_spec(), SearchBudget::new(100, 3, 1.4, 1.0).unwrap());
            let mut rng = seeded(21);
            let a = search.plan(&mut model, &mut rng);
            let stats: Vec<(u64, u64)> = search
                .root_stats()
                .iter()
                .map(|s| (s.action, s.visits))
                .collect();
            (a, stats)
        };
        assert_eq!(run(), run());
    }
}
