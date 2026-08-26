# AIXI research track

Computable approximations to universal reinforcement learning: finite environment classes, budgeted planning, explicit revert/replay. The goal is long-horizon AGI research under resource bounds we can actually pay for. Full Solomonoff optimality is out of reach and everyone involved knows it.

---

## What the theory targets

AIXI is a Bayes-optimal reinforcement learner under a mixture prior over environments. At history \(h_{<t}\) the agent maintains a posterior over models \(\nu\) and acts to maximize expected discounted return with discount \(\gamma \in (0,1)\). Writing \(\xi(\cdot \mid h_{<t})\) for the Bayes mixture over a countable or finite class \(\mathcal{M}\) of environment hypotheses,

$$
V_\xi^\pi(h_{<t}) = \mathbb{E}_{\xi,\pi}\Big[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \,\Big|\, h_{<t}\Big], \qquad
\pi_\xi^\ast \in \arg\max_\pi V_\xi^\pi(h_{<t}).
$$

The AIXI policy is \(\pi_\xi^\ast\) when \(\xi\) is a universal semimeasure over computable environments. That construction does not terminate, so this codebase implements finite \(\mathcal{M}\), budgeted planning, and explicit revert/replay contracts, keeping every \(\xi\) update an ordinary Turing-bounded operation. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and [`SUPERTASK_BOUNDARY.md`](SUPERTASK_BOUNDARY.md).

---

## The implementation

The implementation is the Rust crate at the repository root. The Python prototype has been deleted; git history has it if you ever feel nostalgic. Its load-bearing contracts survived the port: the six-method `MixtureEnvModel` interface, the FAC learn/append split, and the revert invariant, which the Rust version tightens from a 1e-8 drift tolerance to bit-exact restoration, because chasing float drift through an MCTS at 2am stops being fun quickly.

Contents: ρUCT expectimax search over a Bayes mixture \(\xi\) of FAC-CTW models plus a Qwen3.8-2B that was taken apart down to raw GGUF tensors. Own container parser, own GGML quantization kernels, own hybrid Gated-DeltaNet/attention forward pass. Zero inference frameworks in the dependency tree. The recurrent state is exactly revertible through a checkpoint stack, and the forward pass agrees with a llama.cpp oracle to 9e-4 on an f32 graph, which is about as much external validation as one can extract from this universe. Also five JAIR domains and an exact enumerated expectimax as ground truth. Details, measured numbers, and the list of things that don't work yet: [`RUST_IMPLEMENTATION.md`](RUST_IMPLEMENTATION.md).

---

## Implementation reality (what exists today)

| Area | Status | Where |
|------|--------|--------|
| Model class \(\xi\) | Bayes mixture with exact weight undo and dominance tests. Components: FAC-CTW (per-percept-bit trees, JAIR §5.4), single-tree AC-CTW, order-0 KT, uniform floor | [`src/models/`](src/models/) |
| CTW correctness | Root log-probability equals a brute-force mixture over every depth ≤ D suffix tree (2^(−Γ) prior) to 1e-12, FAC interleavings included | [`tests/ctw_brute_force.rs`](tests/ctw_brute_force.rs) |
| Planning (Family A) | Full ρUCT tree (JAIR Alg. 1-4) with per-simulation bit-exact revert of \(\xi\), checked against exact enumerated expectimax | [`src/planning/`](src/planning/), [`tests/rho_uct_expectimax.rs`](tests/rho_uct_expectimax.rs) |
| The dissected base model | Qwen3.8-2B (qwen35 hybrid) reduced to a 2-logit conditional probability engine. 508.6M vocabulary params never leave the file, MTP block amputated, recurrent state checkpointed for exact revert | [`src/llm/`](src/llm/) |
| Oracle validation | Exact-graph parity with llama.cpp (9e-4, synthetic f32 hybrid) plus bounded evaluator noise on the real Q4_K_M | [`scripts/oracle_check.sh`](scripts/oracle_check.sh) |
| Environments | CoinFlip, Biased RPS, Cheese Maze, Tiger, Kuhn Poker, with the JAIR §7 bit layouts | [`src/env/`](src/env/) |
| Agent + CLI | act/perceive loop, per-cycle metrics including the mixture posterior trajectory, CSV output | [`src/agent.rs`](src/agent.rs), [`src/bin/aixi.rs`](src/bin/aixi.rs) |
| Smoke suite | Five offline PASS/FAIL invariants behind an exit code | [`src/bin/smoke.rs`](src/bin/smoke.rs) |

Smoke entrypoint:

```bash
cargo test                          # 62 tests. They pass.
cargo run --release --bin smoke     # PASS/FAIL lines, exit 0 when green
```

Source analyses and paper-to-module mapping live under [`analyses/`](analyses/) and [`modules/`](modules/). The consolidated design doc is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## Roadmap

1. Phase 0, correctness baseline: done. The pyaixi parity target got replaced by something stricter, a from-first-principles brute-force suffix-tree mixture for CTW and exact expectimax for the search.
2. Phase 1, unified \(\xi\) API: done. The `EnvModel` trait ([`src/models/mod.rs`](src/models/mod.rs)) serves CTW, the mixture, and the dissected LLM under one bit-exact revert contract.
3. Phase 2, Self-AIXI v0 (Family B): open. A `MixturePolicy` / \(Q_{\zeta\xi}\) head on top of the same trait. The substrate is there, someone just has to write it.
4. Phase 3+, scale and AIQI (Family C): open. Return-mixture induction with explicit on-policy training assumptions.
5. Later: empowerment / FEP regularizers, joint prediction interfaces, classical simulators of quantum toys. All gated behind research review.

The ordering keeps regression baselines and computable interfaces ahead of the speculative stuff. Timelines are whatever they turn out to be.

---

## Installation

A stable Rust toolchain. That's it. The dependency tree is `memmap2`, `half`, `rand`, `rand_chacha`, `rayon`.

```bash
git clone https://github.com/Euroswarms-Institute/Self-AIXI
cd Self-AIXI
cargo test
cargo run --release --bin smoke
```

The base model for the `llm` / `full-mix` catalogs is a separate 1.31 GB download (sha256-checked). CTW modes and the entire test suite run without it.

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
| LLM demo | `cargo run --release --bin aixi -- --env coin_flip --model full-mix --ct-depths 4,8 --cycles 12 --mc-simulations 10 --horizon 2` (about 30 s per cycle on 4 CPU cores; budget accordingly) |
| Forward-pass ground truth | `bash scripts/oracle_check.sh` (dev-only, wants a llama.cpp build, see the script header) |
| Format / lint | `cargo fmt` / `cargo clippy --all-targets -- -D warnings` |

---

## Contributing & review

Pull requests should keep finite-model and budget assumptions explicit in new code paths. For LaTeX-heavy README or plan changes, pair with Math & CS Wizard for notation consistency before merge.

---

## References (primary)

1. M. Hutter, *Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability*, Springer, 2005. ([AIXI definition & Bayes mixture](https://www.hutter1.net/ai/uaibook.htm))
2. J. Veness et al., *A Monte-Carlo AIXI Approximation*, JAIR 2011 (the MC-AIXI / CTW line; implementation lineage via `pyaixi`).
3. J. Veness et al., *Practical Monte Carlo AIXI with Context Tree Weighting*, and related CTW literature referenced in `pyaixi`.
4. E. Catt et al., *Self-Predictive Universal AI*, NeurIPS 2023 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/56a225639da77e8f7c0409f6d5ba996b-Paper-Conference.pdf)), the Self-AIXI / \(Q_{\zeta\xi}\) line; see [`analyses/01-neurips-2023.md`](analyses/01-neurips-2023.md).
5. M. Hutter & coauthors on AIQI / return-induction; see [`analyses/02-arxiv-2602-23242.md`](analyses/02-arxiv-2602-23242.md) for this repo's working notes.

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

- Per paper: structured notes in `analyses/` (problem, definitions, main theorems/algorithms, notation, what is implementable and what stays idealized).
- Synthesis: `IMPLEMENTATION_PLAN.md` is updated after analyses and supplementary modules land.
