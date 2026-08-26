//! The surgically dissected base model.
//!
//! Everything under this module exists to turn a raw GGUF checkpoint of
//! Qwen3.8-2B (architecture `qwen35`: a hybrid of Gated-DeltaNet linear
//! recurrence and gated full attention) into one thing only: a conditional
//! probability engine over the two-token bit alphabet, satisfying the same
//! `EnvModel` contract as CTW — including exact revert, which for a
//! recurrent model means checkpointed state, not just cache truncation.
//!
//! Deliberately absent: tokenizer runtime, sampling, chat templates, the
//! multi-token-prediction head, and the 248 320-row vocabulary (only the
//! bit-token and stream-prime rows are ever materialized).
//!
//! No inference framework is used anywhere: `gguf` parses the container,
//! `quant` implements the block formats and fused dequant·dot kernels,
//! `tensor` provides the GEMV, and the forward pass lives in `model`.

pub mod byte_model;
pub mod config;
pub mod env_model;
pub mod gguf;
pub mod model;
pub mod quant;
pub mod rope;
pub mod state;
pub mod tensor;
