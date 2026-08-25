//! # mc-aixi — Monte-Carlo AIXI in Rust (Family A)
//!
//! A computable approximation of AIXI in the sense of Veness, Ng, Hutter,
//! Uther & Silver, *A Monte-Carlo AIXI Approximation*, JAIR 40 (2011):
//! ρUCT expectimax search over a Bayesian mixture environment model ξ.
//!
//! This crate is the Rust realization of Family A from
//! `IMPLEMENTATION_PLAN.md` §0/§4.1, honoring the §1 notation ledger, the
//! §1.1 runtime-computability discipline (every budget finite, every
//! `predict`/`learn`/`revert` terminating), and the revert-exactness contract
//! of `aixi/planning/xi_rollouts.py` (root log-probability must be restored
//! after every imagined rollout — here restored *bit-exactly*).
//!
//! Mixture components:
//! - FAC-CTW (action-conditional Context Tree Weighting, JAIR §4–§5) at one
//!   or more depths, and
//! - a surgically dissected base LLM (Qwen3.8-2B, hybrid Gated-DeltaNet /
//!   full-attention) reduced to a conditional probability engine over the
//!   two-token bit alphabet (`src/llm`).

pub mod agent;
pub mod encoding;
pub mod env;
pub mod llm;
pub mod logspace;
pub mod models;
pub mod planning;
pub mod rng;
