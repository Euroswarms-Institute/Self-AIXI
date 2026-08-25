//! Mutable inference state with **exact LIFO rollback** — what makes an
//! autoregressive *recurrent* hybrid satisfy the MC-AIXI revert contract.
//!
//! Two kinds of per-layer state:
//! - Full attention: append-only K/V arenas — revert is truncation, O(1),
//!   trivially exact.
//! - Gated DeltaNet: a recurrent matrix state S (d_v×d_k per head) plus the
//!   causal-conv tail. Recurrences don't truncate, so every advanced token
//!   pushes a **checkpoint** (deep copy of all DeltaNet layer states,
//!   ~19 MB for the 2B model) onto a bounded stack; revert pops and restores
//!   bit-exactly. The stack holds the most recent `CHECKPOINT_CAP` positions
//!   — far deeper than any ρUCT imagination — and if a revert ever reaches
//!   past it, the state is rebuilt by deterministic replay from position 0
//!   (exact, just slow; counted in `deep_replays` for diagnostics).

/// Positions of retained checkpoints (≈19 MB each on the 2B model, so the
/// cap bounds checkpoint memory at ~1.2 GB). Must exceed the deepest ρUCT
/// imagination in tokens — horizon × (action_bits + percept_bits) — or
/// unwinding falls back to full deterministic replay each simulation.
pub const CHECKPOINT_CAP: usize = 64;

#[derive(Clone, PartialEq)]
pub struct DeltaNetState {
    /// Per head: S[d_v][d_k], row-major, mapping key-space → value-space.
    pub s: Vec<f32>,
    /// Raw (pre-conv) fused-qkv values of the last `conv_kernel − 1` tokens,
    /// oldest first: [k−1][conv_channels].
    pub conv_tail: Vec<f32>,
}

impl DeltaNetState {
    fn zeros(heads: usize, d_k: usize, d_v: usize, conv_taps: usize, conv_channels: usize) -> Self {
        DeltaNetState {
            s: vec![0.0; heads * d_v * d_k],
            conv_tail: vec![0.0; conv_taps * conv_channels],
        }
    }
}

pub struct KvArena {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Per-position K/V width (n_kv_heads · head_dim).
    pub stride: usize,
}

impl KvArena {
    pub fn keys(&self, pos: usize) -> &[f32] {
        &self.k[pos * self.stride..(pos + 1) * self.stride]
    }
    pub fn values(&self, pos: usize) -> &[f32] {
        &self.v[pos * self.stride..(pos + 1) * self.stride]
    }
}

struct Checkpoint {
    pos: usize,
    deltanet: Vec<DeltaNetState>,
}

pub struct LlmState {
    /// Tokens advanced so far (equals every arena's logical length).
    pub pos: usize,
    pub kv: Vec<KvArena>,
    pub deltanet: Vec<DeltaNetState>,
    checkpoints: Vec<Checkpoint>,
    pub deep_replays: u64,
}

/// Geometry needed to (re)build empty state.
#[derive(Clone, Copy)]
pub struct StateShape {
    pub n_attention: usize,
    pub kv_stride: usize,
    pub n_deltanet: usize,
    pub ssm_heads: usize,
    pub ssm_d_k: usize,
    pub ssm_d_v: usize,
    pub conv_taps: usize,
    pub conv_channels: usize,
}

impl LlmState {
    pub fn new(shape: StateShape) -> Self {
        LlmState {
            pos: 0,
            kv: (0..shape.n_attention)
                .map(|_| KvArena {
                    k: Vec::new(),
                    v: Vec::new(),
                    stride: shape.kv_stride,
                })
                .collect(),
            deltanet: (0..shape.n_deltanet)
                .map(|_| {
                    DeltaNetState::zeros(
                        shape.ssm_heads,
                        shape.ssm_d_k,
                        shape.ssm_d_v,
                        shape.conv_taps,
                        shape.conv_channels,
                    )
                })
                .collect(),
            checkpoints: Vec::new(),
            deep_replays: 0,
        }
    }

    /// Snapshot the recurrent state *before* the caller mutates it for the
    /// token at the current position. Attention arenas need no snapshot.
    pub fn push_checkpoint(&mut self) {
        self.checkpoints.push(Checkpoint {
            pos: self.pos,
            deltanet: self.deltanet.clone(),
        });
        if self.checkpoints.len() > CHECKPOINT_CAP {
            self.checkpoints.remove(0);
        }
    }

    /// Roll every arena and recurrence back to exactly `target` advanced
    /// tokens. Returns false if only a deterministic full replay can get
    /// there (checkpoint evicted) — the caller owns the replay.
    #[must_use]
    pub fn revert_to(&mut self, target: usize) -> bool {
        assert!(
            target <= self.pos,
            "revert_to({target}) ahead of pos {}",
            self.pos
        );
        if target == self.pos {
            return true;
        }
        // Pop checkpoints newer than the target; the one AT the target's
        // position carries the recurrent state as it stood before that
        // token was applied.
        let mut restored = false;
        while let Some(top) = self.checkpoints.last() {
            if top.pos > target {
                self.checkpoints.pop();
            } else if top.pos == target {
                let cp = self.checkpoints.pop().unwrap();
                self.deltanet = cp.deltanet;
                restored = true;
                break;
            } else {
                break;
            }
        }
        if !restored {
            return false;
        }
        for arena in &mut self.kv {
            arena.k.truncate(target * arena.stride);
            arena.v.truncate(target * arena.stride);
        }
        self.pos = target;
        true
    }

    pub fn checkpoint_depth(&self) -> usize {
        self.checkpoints.len()
    }

    /// Bit-exact fingerprint of every mutable field (tests/diagnostics).
    pub fn state_digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.pos.hash(&mut h);
        for arena in &self.kv {
            for v in arena.k.iter().chain(&arena.v) {
                v.to_bits().hash(&mut h);
            }
        }
        for dn in &self.deltanet {
            for v in dn.s.iter().chain(&dn.conv_tail) {
                v.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }
}
