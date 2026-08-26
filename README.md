# AIXI research track (CR-CA)

This repository is the **AIXI** line of work (extracted from CR-CA): **computable** approximations to universal reinforcement learning—finite environment classes, budgeted planning, and explicit revert/replay—aimed at long-horizon **AGI research**, not a claim of full Solomonoff optimality.

---

## What the theory targets

AIXI is a **Bayes-optimal reinforcement learner** under a mixture prior over environments. Informally, at history \(h_{<t}\) the agent maintains a posterior over models \(\nu\) and acts to maximize expected **discounted return** with discount \(\gamma \in (0,1)\). Writing \(\xi(\cdot \mid h_{<t})\) for the **Bayes mixture** over a countable or finite class \(\mathcal{M}\) of environment hypotheses,

$$
V_\xi^\pi(h_{<t}) = \mathbb{E}_{\xi,\pi}\Big[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \,\Big|\, h_{<t}\Big], \qquad
\pi_\xi^\ast \in \arg\max_\pi V_\xi^\pi(h_{<t}).
$$

The **AIXI policy** is \(\pi_\xi^\ast\) when \(\xi\) is a **universal** semimeasure over computable environments; that construction is **not computable** in finite time. This codebase implements **finite** \(\mathcal{M}\), **budgeted** planning, and explicit **revert/replay** contracts so that \(\xi\) updates remain ordinary Turing-bounded operations—see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and [`SUPERTASK_BOUNDARY.md`](SUPERTASK_BOUNDARY.md).

---

## The implementation: MC-AIXI in Rust (Family A)

**The implementation of this repository is the Rust crate at the repository
root.** The earlier Python prototype has been retired and fully replaced (it
remains in git history); its load-bearing contracts — the six-method
`MixtureEnvModel` interface, the FAC learn/append split, and the
revert-exactness invariant — carry over into the Rust `EnvModel` trait,
strengthened from ≤ 1e-8 drift to **bit-exact** restoration.

What the crate contains: ρUCT expectimax search over a Bayesian mixture ξ of
FAC-CTW models **and a surgically dissected Qwen3.8-2B** (GGUF container,
GGML quantization kernels, and the hybrid Gated-DeltaNet/attention forward
pass all hand-rolled with no inference framework; the recurrent state made
exactly revertible via a checkpoint stack; validated against a llama.cpp
oracle to 9e-4 on an f32 graph), plus the five JAIR §7 domains and an exact
enumerated expectimax as ground truth. Full details, measured results, and
honest limitations: [`RUST_IMPLEMENTATION.md`](RUST_IMPLEMENTATION.md).

---

## Implementation reality (what exists today)

| Area | Status | Where |
|------|--------|--------|
| **Model class \(\xi\)** | Bayes mixture with exact weight undo and dominance tests; components: FAC-CTW (per-percept-bit trees, JAIR §5.4), single-tree AC-CTW, order-0 KT, uniform floor | [`src/models/`](src/models/) |
| **CTW correctness** | Root log-probability equals a brute-force mixture over *all* depth ≤ D suffix trees (2^(−Γ) prior) to 1e-12, FAC interleavings included | [`tests/ctw_brute_force.rs`](tests/ctw_brute_force.rs) |
| **Planning (Family A)** | Full ρUCT tree (JAIR Alg. 1–4) with per-simulation bit-exact revert of \(\xi\); verified against exact enumerated expectimax | [`src/planning/`](src/planning/), [`tests/rho_uct_expectimax.rs`](tests/rho_uct_expectimax.rs) |
| **The dissected base model** | Qwen3.8-2B (qwen35 hybrid) as a 2-logit conditional probability engine: 508.6 M vocabulary params never materialized, MTP block amputated, checkpointed recurrent state for exact revert | [`src/llm/`](src/llm/) |
| **Oracle validation** | Exact-graph parity with llama.cpp (9e-4, synthetic f32 hybrid) + bounded evaluator noise on the real Q4_K_M | [`scripts/oracle_check.sh`](scripts/oracle_check.sh) |
| **Environments** | CoinFlip, Biased RPS, Cheese Maze, Tiger, Kuhn Poker (JAIR §7 bit layouts) | [`src/env/`](src/env/) |
| **Agent + CLI** | act/perceive loop, per-cycle metrics incl. mixture posterior trajectory, CSV output | [`src/agent.rs`](src/agent.rs), [`src/bin/aixi.rs`](src/bin/aixi.rs) |
| **Smoke suite** | Five offline PASS/FAIL invariants, exit-code gated | [`src/bin/smoke.rs`](src/bin/smoke.rs) |

**Smoke entrypoint (recommended):**

```bash
cargo test                          # full offline correctness spine (62 tests)
cargo run --release --bin smoke     # PASS/FAIL invariants, exit code 0 iff green
```

Source analyses and paper-to-module mapping live under [`analyses/`](analyses/) and [`modules/`](modules/); the consolidated design doc is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## Roadmap: from V0 toward broader AGI research

The implementation plan phases (abbreviated) are:

1. **Phase 0 — Correctness baseline:** done, superseded in strength — instead
   of `pyaixi` parity, the Rust CTW is verified against a from-first-principles
   brute-force suffix-tree mixture, and ρUCT against exact expectimax.
2. **Phase 1 — Unified \(\xi\) API:** done — the `EnvModel` trait
   ([`src/models/mod.rs`](src/models/mod.rs)) serves CTW, the mixture, and the
   dissected LLM interchangeably under one bit-exact revert contract.
3. **Phase 2 — Self-AIXI v0 (Family B):** open — a `MixturePolicy`/\(Q_{\zeta\xi}\)
   head built on the same `EnvModel` trait; the Rust search and mixture are the
   intended substrate.
4. **Phase 3+ — Scale & AIQI (Family C):** open — return-mixture induction with
   explicit **on-policy** training assumptions.
5. **Later:** optional empowerment / FEP regularizers, joint prediction
   interfaces, quantum **classical** simulators—each gated as research, not
   core product claims.

This roadmap is **not** a promise of timelines; it is an ordering that keeps **regression baselines** and **computable** interfaces ahead of speculative extensions.

---

## Installation

A stable Rust toolchain is the only requirement (pure-Rust dependency tree:
`memmap2`, `half`, `rand`, `rand_chacha`, `rayon`):

```bash
git clone https://github.com/Euroswarms-Institute/Self-AIXI
cd Self-AIXI
cargo test                        # correctness spine
cargo run --release --bin smoke   # PASS/FAIL invariants
```

The base model for the `llm` / `full-mix` catalogs is fetched separately
(1.31 GB, sha256-checked; not needed for CTW modes or any test):

```bash
bash scripts/fetch_model.sh
cargo run --release --bin inspect_model -- --gguf models/Qwen3.8-2B-Q4_K_M.gguf
```

---

## Developer workflow

| Task | Command |
|------|---------|
| Correctness spine | `cargo test` |
| Smoke invariants | `cargo run --release --bin smoke` |
| Run the agent | `cargo run --release --bin aixi -- --env coin_flip --cycles 400` |
| The dissection showcase | `cargo run --release --bin aixi -- --env coin_flip --model full-mix --ct-depths 4,8 --cycles 12 --mc-simulations 10 --horizon 2` |
| Forward-pass ground truth | `bash scripts/oracle_check.sh` (dev-only; needs a llama.cpp build, see script header) |
| Format / lint | `cargo fmt` / `cargo clippy --all-targets -- -D warnings` |

---

## Contributing & review

Pull requests should keep **finite-model** and **budget** assumptions explicit in new code paths. For LaTeX-heavy README or plan changes, pair with **Math & CS Wizard** for notation consistency before merge.

---

## References (primary)

1. M. Hutter, *Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability*, Springer, 2005. ([AIXI definition & Bayes mixture](https://www.hutter1.net/ai/uaibook.htm))
2. J. Veness et al., *A Monte-Carlo AIXI Approximation*, JAIR 2011 (MC-AIXI / CTW line; implementation lineage via `pyaixi`).
3. J. Veness et al., *Practical Monte Carlo AIXI with Context Tree Weighting*, and related CTW literature referenced in `pyaixi`.
4. E. Catt et al., *Self-Predictive Universal AI*, NeurIPS 2023 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/56a225639da77e8f7c0409f6d5ba996b-Paper-Conference.pdf)) — Self-AIXI / \(Q_{\zeta\xi}\) line; see [`analyses/01-neurips-2023.md`](analyses/01-neurips-2023.md).
5. M. Hutter & coauthors on **AIQI** / return-induction (see [`analyses/02-arxiv-2602-23242.md`](analyses/02-arxiv-2602-23242.md) for this repo’s working notes).

---

## Research sources (analyses)

| # | Source | Analysis file |
|---|--------|---------------|
| 1 | [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/56a225639da77e8f7c0409f6d5ba996b-Paper-Conference.pdf) | [`analyses/01-neurips-2023.md`](analyses/01-neurips-2023.md) |
| 2 | [arXiv 2602.23242](https://arxiv.org/pdf/2602.23242) | [`analyses/02-arxiv-2602-23242.md`](analyses/02-arxiv-2602-23242.md) |
| 3 | [pyaixi](https://github.com/sgkasselau/pyaixi) | [`analyses/03-pyaixi-repo.md`](analyses/03-pyaixi-repo.md) |
| 4 | [arXiv 2502.15820](https://arxiv.org/html/2502.15820v2) | [`analyses/04-arxiv-2502-15820.md`](analyses/04-arxiv-2502-15820.md) |
| 5 | [arXiv 2511.22226](https://arxiv.org/pdf/2511.22226) | [`analyses/05-arxiv-2511-22226.md`](analyses/05-arxiv-2511-22226.md) |
| 6 | [arXiv 2505.21170](https://arxiv.org/html/2505.21170v2) | [`analyses/06-arxiv-2505-21170.md`](analyses/06-arxiv-2505-21170.md) |

### Supplementary module specs

| # | Source | Module spec |
|---|--------|----------------|
| 7 | [arXiv cs/0412022](https://arxiv.org/pdf/cs/0412022) | [`modules/mod-cs-0412022.md`](modules/mod-cs-0412022.md) |
| 8 | [arXiv 1411.5679](https://arxiv.org/pdf/1411.5679) | [`modules/mod-arxiv-1411-5679.md`](modules/mod-arxiv-1411-5679.md) |
| 9 | [arXiv 2505.14698](https://arxiv.org/pdf/2505.14698) | [`modules/mod-arxiv-2505-14698.md`](modules/mod-arxiv-2505-14698.md) |
| 10 | [arXiv math/0209332](https://arxiv.org/pdf/math/0209332) | [`modules/mod-math-0209332.md`](modules/mod-math-0209332.md) |
| 11 | [HilbertMachine.pdf](https://philsci-archive.pitt.edu/2869/1/HilbertMachine.pdf) | [`modules/mod-hilbert-machine.md`](modules/mod-hilbert-machine.md) |

---

## Deliverables & repo context

- **Per paper:** structured notes in `analyses/` (problem, definitions, main theorems/algorithms, notation, what is implementable vs idealized, links to CRCA patterns if any).
- **Synthesis:** `IMPLEMENTATION_PLAN.md` is updated after analyses and supplementary modules land.

Use existing CR-CA abstractions where they help (agents, templates, prediction hooks); do not force-fit causal machinery where it does not apply.
