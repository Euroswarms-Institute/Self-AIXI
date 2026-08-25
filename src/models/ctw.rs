//! Action-conditional Context Tree Weighting (AC-CTW, JAIR §4–§5.3) — one
//! shared context tree over the whole interleaved bit stream, the variant
//! pyaixi implements. The *factored* FAC-CTW the JAIR experiments actually
//! use (one tree per percept bit position, immune to positional aliasing)
//! is built on top of this type in `models::fac_ctw`.
//!
//! `CtwModel` computes, in O(D) per bit, the Bayesian mixture over **all**
//! prediction suffix trees of depth ≤ D with the natural prior 2^(−Γ_D(T))
//! (JAIR eq. 33): a leaf holds a KT estimator; an internal node weights
//! "stop here" against "split on one more context bit":
//!     log P_w = log_kt                                        at depth D,
//!     log P_w = ln½·[P_kt + P_w(child₀)·P_w(child₁)]          otherwise.
//!
//! Conventions (documented because the Python reference is not runnable):
//! - The context of a bit is the previous `depth` history bits, **most recent
//!   first**, with the empty prefix zero-padded (Willems et al. 1995), so
//!   every bit has a full-depth context from t = 0.
//! - FAC split: learned (percept) bits update KT counts along their context
//!   path; appended (action) bits enter the history/context only (JAIR §5.3).
//! - Revert is *bit-exact*: undo frames record previous node values, and node
//!   creation is undone by truncating the arena — the design flaw pyaixi's
//!   adapter had to work around (`aixi/models/ctw_pyaixi.py` docstring)
//!   cannot occur because `predict` never touches the tree.

use super::kt::kt_log_increment;
use super::EnvModel;
use crate::logspace::{log_sum_exp, LOG_HALF};

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    log_kt: f64,
    log_weighted: f64,
    counts: [u32; 2],
    children: [u32; 2],
}

impl Node {
    fn empty() -> Self {
        Node {
            log_kt: 0.0,
            log_weighted: 0.0,
            counts: [0, 0],
            children: [NONE, NONE],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SymbolKind {
    Learned,
    Appended,
}

/// One learned bit's undo frame; the flat record vectors are shared across
/// frames (ranges start at `path_start`/`created_start`) to avoid per-bit
/// allocation.
struct Frame {
    bit: u8,
    arena_len_before: u32,
    path_start: u32,
    created_start: u32,
}

pub struct CtwModel {
    depth: usize,
    nodes: Vec<Node>,
    /// Interleaved action+percept bit history (context source for both kinds).
    history: Vec<u8>,
    kinds: Vec<SymbolKind>,
    frames: Vec<Frame>,
    path_records: Vec<(u32, f64, f64)>, // (node, prev_log_kt, prev_log_weighted)
    created_records: Vec<(u32, u8)>,    // (parent, child slot) set in this frame
    walk_scratch: Vec<u32>,
}

impl CtwModel {
    /// Depth 0 is legal and equals a single order-0 KT estimator.
    pub fn new(depth: usize) -> Self {
        assert!(depth <= 64, "ct depth {depth} unreasonably large");
        CtwModel {
            depth,
            nodes: vec![Node::empty()],
            history: Vec::new(),
            kinds: Vec::new(),
            frames: Vec::new(),
            path_records: Vec::new(),
            created_records: Vec::new(),
            walk_scratch: Vec::new(),
        }
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of allocated context-tree nodes (§7 "CTW nodes touched" metric).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Context bit at `distance` (1 = most recent) with zero padding.
    fn context_bit(&self, distance: usize) -> u8 {
        let len = self.history.len();
        if distance <= len {
            self.history[len - distance]
        } else {
            0
        }
    }

    fn child_log_weighted(&self, node: u32, slot: u8) -> f64 {
        let c = self.nodes[node as usize].children[slot as usize];
        if c == NONE {
            0.0 // empty subtree: probability 1 of the empty sequence
        } else {
            self.nodes[c as usize].log_weighted
        }
    }

    fn learn_one(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        let frame = Frame {
            bit,
            arena_len_before: self.nodes.len() as u32,
            path_start: self.path_records.len() as u32,
            created_start: self.created_records.len() as u32,
        };

        // Walk root → depth-D leaf along the context, creating missing nodes.
        let mut walk = std::mem::take(&mut self.walk_scratch);
        walk.clear();
        walk.push(0);
        let mut cur = 0u32;
        for level in 1..=self.depth {
            let cbit = self.context_bit(level);
            let mut child = self.nodes[cur as usize].children[cbit as usize];
            if child == NONE {
                child = self.nodes.len() as u32;
                self.nodes.push(Node::empty());
                self.nodes[cur as usize].children[cbit as usize] = child;
                self.created_records.push((cur, cbit));
            }
            walk.push(child);
            cur = child;
        }

        // Update leaf → root so each parent sees its updated path child.
        for (level, &idx) in walk.iter().enumerate().rev() {
            self.path_records.push((
                idx,
                self.nodes[idx as usize].log_kt,
                self.nodes[idx as usize].log_weighted,
            ));
            let new_log_kt = self.nodes[idx as usize].log_kt
                + kt_log_increment(self.nodes[idx as usize].counts, bit);
            let new_lw = if level == self.depth {
                new_log_kt
            } else {
                let lw0 = self.child_log_weighted(idx, 0);
                let lw1 = self.child_log_weighted(idx, 1);
                log_sum_exp(LOG_HALF + new_log_kt, LOG_HALF + lw0 + lw1)
            };
            let node = &mut self.nodes[idx as usize];
            node.log_kt = new_log_kt;
            node.counts[bit as usize] += 1;
            node.log_weighted = new_lw;
        }

        self.walk_scratch = walk;
        self.frames.push(frame);
        self.history.push(bit);
        self.kinds.push(SymbolKind::Learned);
    }

    /// Test/diagnostic fingerprint of the complete mutable state.
    pub fn state_digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for n in &self.nodes {
            n.log_kt.to_bits().hash(&mut h);
            n.log_weighted.to_bits().hash(&mut h);
            n.counts.hash(&mut h);
            n.children.hash(&mut h);
        }
        self.history.hash(&mut h);
        (
            self.frames.len(),
            self.path_records.len(),
            self.created_records.len(),
        )
            .hash(&mut h);
        h.finish()
    }
}

impl EnvModel for CtwModel {
    fn root_log_probability(&self) -> f64 {
        self.nodes[0].log_weighted
    }

    fn predict_bit_probability(&mut self, bit: u8) -> f64 {
        debug_assert!(bit <= 1);
        // Hypothetical update computed in temporaries; the tree is untouched.
        // Path nodes below the allocated frontier are virtual empty nodes.
        let mut existing: Vec<Option<u32>> = Vec::with_capacity(self.depth + 1);
        existing.push(Some(0));
        let mut cur = Some(0u32);
        for level in 1..=self.depth {
            let cbit = self.context_bit(level);
            cur = cur.and_then(|i| {
                let c = self.nodes[i as usize].children[cbit as usize];
                if c == NONE {
                    None
                } else {
                    Some(c)
                }
            });
            existing.push(cur);
        }

        let mut child_new_lw = 0.0; // filled by the leaf iteration first
        for level in (0..=self.depth).rev() {
            let (counts, log_kt) = match existing[level] {
                Some(i) => (self.nodes[i as usize].counts, self.nodes[i as usize].log_kt),
                None => ([0, 0], 0.0),
            };
            let new_log_kt = log_kt + kt_log_increment(counts, bit);
            child_new_lw = if level == self.depth {
                new_log_kt
            } else {
                // The path child's *new* weighted prob is child_new_lw; its
                // sibling keeps its current value.
                let path_slot = self.context_bit(level + 1);
                let sibling_lw = match existing[level] {
                    Some(i) => self.child_log_weighted(i, 1 - path_slot),
                    None => 0.0,
                };
                log_sum_exp(LOG_HALF + new_log_kt, LOG_HALF + child_new_lw + sibling_lw)
            };
        }
        (child_new_lw - self.nodes[0].log_weighted).exp()
    }

    fn learn_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            self.learn_one(b);
        }
    }

    fn append_history_symbols(&mut self, bits: &[u8]) {
        for &b in bits {
            debug_assert!(b <= 1);
            self.history.push(b);
            self.kinds.push(SymbolKind::Appended);
        }
    }

    fn revert_learned_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(SymbolKind::Learned),
                "revert_learned out of LIFO order"
            );
            self.history.pop();
            let frame = self.frames.pop().expect("no learned bit to revert");
            for &(idx, prev_kt, prev_lw) in
                self.path_records[frame.path_start as usize..].iter().rev()
            {
                let node = &mut self.nodes[idx as usize];
                node.log_kt = prev_kt;
                node.log_weighted = prev_lw;
                node.counts[frame.bit as usize] -= 1;
            }
            self.path_records.truncate(frame.path_start as usize);
            for &(parent, slot) in self.created_records[frame.created_start as usize..]
                .iter()
                .rev()
            {
                self.nodes[parent as usize].children[slot as usize] = NONE;
            }
            self.created_records.truncate(frame.created_start as usize);
            self.nodes.truncate(frame.arena_len_before as usize);
        }
    }

    fn revert_history_symbols(&mut self, n: usize) {
        for _ in 0..n {
            assert_eq!(
                self.kinds.pop(),
                Some(SymbolKind::Appended),
                "revert_history out of LIFO order"
            );
            self.history.pop();
        }
    }

    fn model_id(&self) -> String {
        format!("ctw-d{}", self.depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seeded;
    use rand::Rng;

    #[test]
    fn depth_zero_is_kt() {
        let mut m = CtwModel::new(0);
        m.learn_symbols(&[1, 1, 0]);
        assert!((m.root_log_probability() - (1.0f64 / 16.0).ln()).abs() < 1e-14);
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn predict_is_learn_ratio_and_normalized() {
        let mut m = CtwModel::new(3);
        m.append_history_symbols(&[1]);
        m.learn_symbols(&[0, 1, 1, 0, 1]);
        m.append_history_symbols(&[0]);
        for bit in [0u8, 1] {
            let predicted = m.predict_bit_probability(bit);
            let before = m.root_log_probability();
            m.learn_symbols(&[bit]);
            let ratio = (m.root_log_probability() - before).exp();
            m.revert_learned_symbols(1);
            assert!((predicted - ratio).abs() < 1e-14, "bit {bit}");
        }
        let total = m.predict_bit_probability(0) + m.predict_bit_probability(1);
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn revert_restores_state_bit_exactly() {
        let mut rng = seeded(1234);
        let mut m = CtwModel::new(6);
        // Build a persistent base first.
        for _ in 0..40 {
            let b = u8::from(rng.random_bool(0.7));
            if rng.random_bool(0.3) {
                m.append_history_symbols(&[b]);
            } else {
                m.learn_symbols(&[b]);
            }
        }
        let digest = m.state_digest();
        let root = m.root_log_probability();
        let nodes = m.node_count();

        // Random imagined excursions, unwound in strict LIFO order.
        for _ in 0..200 {
            let mut ops: Vec<(bool, usize)> = Vec::new();
            for _ in 0..rng.random_range(1..12) {
                let learned = rng.random_bool(0.5);
                let count = rng.random_range(1..4);
                let bits: Vec<u8> = (0..count).map(|_| u8::from(rng.random_bool(0.5))).collect();
                if learned {
                    m.learn_symbols(&bits);
                } else {
                    m.append_history_symbols(&bits);
                }
                ops.push((learned, count));
            }
            for (learned, count) in ops.into_iter().rev() {
                if learned {
                    m.revert_learned_symbols(count);
                } else {
                    m.revert_history_symbols(count);
                }
            }
            assert_eq!(m.state_digest(), digest);
            assert_eq!(m.root_log_probability().to_bits(), root.to_bits());
            assert_eq!(m.node_count(), nodes);
        }
    }

    #[test]
    fn appended_bits_do_not_learn_but_do_condition() {
        let mut m = CtwModel::new(2);
        let before = m.root_log_probability();
        let nodes = m.node_count();
        m.append_history_symbols(&[1, 0, 1, 1]);
        assert_eq!(m.root_log_probability(), before);
        assert_eq!(m.node_count(), nodes);

        // Conditioning check: teach "percept mirrors previous action" and
        // verify the prediction tracks the appended action bit.
        let mut m = CtwModel::new(1);
        for _ in 0..30 {
            m.append_history_symbols(&[1]);
            m.learn_symbols(&[1]);
            m.append_history_symbols(&[0]);
            m.learn_symbols(&[0]);
        }
        m.append_history_symbols(&[1]);
        assert!(m.predict_bit_probability(1) > 0.9);
        m.revert_history_symbols(1);
        m.append_history_symbols(&[0]);
        assert!(m.predict_bit_probability(0) > 0.9);
    }

    #[test]
    #[should_panic(expected = "LIFO")]
    fn out_of_order_revert_is_rejected() {
        let mut m = CtwModel::new(2);
        m.learn_symbols(&[1]);
        m.append_history_symbols(&[0]);
        m.revert_learned_symbols(1); // top of stack is Appended — must panic
    }
}
