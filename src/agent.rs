//! The MC-AIXI agent loop (JAIR Alg. 5):
//! plan with ρUCT over ξ, act, then commit the real experience — action bits
//! appended, percept bits learned *permanently* (no revert for reality).

use crate::encoding::encode_bits_msb;
use crate::env::{DomainSpec, Environment, Percept};
use crate::models::EnvModel;
use crate::planning::rho_uct::{RhoUct, SearchBudget};
use crate::planning::root_parallel::plan_root_parallel;
use crate::rng::AgentRng;

pub struct AixiAgent<M: EnvModel> {
    pub model: M,
    pub search: RhoUct,
    pub spec: DomainSpec,
    /// Reuse the (action, percept) subtree across cycles instead of
    /// rebuilding. Off by default: under a receding horizon, carried value
    /// means are biased low (see `RhoUct::advance_root`).
    pub reuse_tree: bool,
    /// Some((workers, base_seed)): each decision runs `workers` independent
    /// root-parallel searches over clones of ξ (total simulations =
    /// budget.mc_simulations, split across workers). The model must support
    /// `try_clone_box`; incompatible with `reuse_tree`.
    pub root_parallel: Option<(usize, u64)>,
    decisions: u64,
}

impl<M: EnvModel> AixiAgent<M> {
    pub fn new(model: M, spec: DomainSpec, budget: SearchBudget) -> Self {
        AixiAgent {
            model,
            search: RhoUct::new(spec, budget),
            spec,
            reuse_tree: false,
            root_parallel: None,
            decisions: 0,
        }
    }

    /// Choose the next action by ρUCT search over the current ξ state.
    pub fn act(&mut self, rng: &mut AgentRng) -> u64 {
        if let Some((workers, base_seed)) = self.root_parallel {
            let seed = base_seed.wrapping_add(self.decisions);
            self.decisions += 1;
            let (action, _) =
                plan_root_parallel(&self.model, &self.spec, self.search.budget(), workers, seed)
                    .expect("root-parallel planning needs a clonable model catalog");
            return action;
        }
        self.search.plan(&mut self.model, rng)
    }

    /// Commit one real interaction cycle into ξ and re-root the search tree.
    pub fn perceive(&mut self, action: u64, percept: Percept) {
        let mut bits = Vec::with_capacity(self.spec.action_bits as usize);
        encode_bits_msb(action, self.spec.action_bits, &mut bits);
        self.model.append_history_symbols(&bits);

        bits.clear();
        percept.encode_into(self.spec.observation_bits, self.spec.reward_bits, &mut bits);
        self.model.learn_symbols(&bits);

        if self.reuse_tree {
            let percept_code = (percept.observation << self.spec.reward_bits) | percept.reward_code;
            self.search.advance_root(action, percept_code);
        } else {
            self.search.reset_root();
        }
    }
}

/// One completed agent–environment cycle (§7 metrics source).
#[derive(Clone, Copy, Debug)]
pub struct CycleRecord {
    pub action: u64,
    pub observation: u64,
    pub reward: f64,
    pub root_log_probability: f64,
}

/// Drive `cycles` interaction cycles between agent and environment.
pub fn run_cycles<M: EnvModel, E: Environment + ?Sized>(
    agent: &mut AixiAgent<M>,
    env: &mut E,
    cycles: usize,
    rng: &mut AgentRng,
) -> Vec<CycleRecord> {
    let mut records = Vec::with_capacity(cycles);
    for _ in 0..cycles {
        let action = agent.act(rng);
        let percept = env.step(action, rng);
        agent.perceive(action, percept);
        records.push(CycleRecord {
            action,
            observation: percept.observation,
            reward: env.decode_reward(percept.reward_code),
            root_log_probability: agent.model.root_log_probability(),
        });
    }
    records
}
