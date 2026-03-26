# AIXI research track (CR-CA)

This directory is the **AIXI** line of work inside CR-CA: **computable** approximations to universal reinforcement learning—finite environment classes, budgeted planning, and explicit revert/replay—aimed at long-horizon **AGI research**, not a claim of full Solomonoff optimality.

---

## What the theory targets

AIXI is a **Bayes-optimal reinforcement learner** under a mixture prior over environments. Informally, at history \(h_{<t}\) the agent maintains a posterior over models \(\nu\) and acts to maximize expected **discounted return** with discount \(\gamma \in (0,1)\). Writing \(\xi(\cdot \mid h_{<t})\) for the **Bayes mixture** over a countable or finite class \(\mathcal{M}\) of environment hypotheses,

$$
V_\xi^\pi(h_{<t}) = \mathbb{E}_{\xi,\pi}\Big[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \,\Big|\, h_{<t}\Big], \qquad
\pi_\xi^\ast \in \arg\max_\pi V_\xi^\pi(h_{<t}).
$$

The **AIXI policy** is \(\pi_\xi^\ast\) when \(\xi\) is a **universal** semimeasure over computable environments; that construction is **not computable** in finite time. This codebase implements **finite** \(\mathcal{M}\), **budgeted** planning, and explicit **revert/replay** contracts so that \(\xi\) updates remain ordinary Turing-bounded operations—see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and [`SUPERTASK_BOUNDARY.md`](SUPERTASK_BOUNDARY.md).

---

## V0 implementation reality (what exists today)

| Area | Status | Where |
|------|--------|--------|
| **Phase 0 parity** | Symbol-level agreement between `pyaixi` CTW rollouts and a manual CTW driver on `CoinFlip` | [`aixi/parity/`](aixi/parity/), [`experiments/PHASE0_PYAIXI_PARITY_REPORT.md`](experiments/PHASE0_PYAIXI_PARITY_REPORT.md) |
| **Mixture \(\xi\) adapter** | `PyAixiCTWBitMixture` bridging CTW + search | [`aixi/models/ctw_pyaixi.py`](aixi/models/ctw_pyaixi.py) |
| **Planning (Family A)** | Root UCT-style action selection with revert checks on \(\xi\) | [`aixi/planning/mcts.py`](aixi/planning/mcts.py), smoke in [`experiments/run_smoke.py`](experiments/run_smoke.py) |
| **Self-AIXI (Family B)** | Policy mixture + \(Q_{\zeta\xi}\)-style scaffolding | [`aixi/planning/self_aixi/`](aixi/planning/self_aixi/) |
| **AIQI (Family C)** | Return-mixture / augmentation skeleton | [`aixi/planning/aiqi/`](aixi/planning/aiqi/) |
| **Formal CI properties** | Cross-family tests (revert stack, \(\omega\) normalization, augmentation schedule, budgets) | [`formal_ci.py`](formal_ci.py), `tests/test_aixi_*.py` |
| **Optional signals** | Intrinsic / empowerment-style hooks (gated) | [`aixi/signals/`](aixi/signals/) |
| **Quantum toy** | Classical CPTP toy env for lab experiments | [`experiments/quantum_toy/`](experiments/quantum_toy/) |

**Smoke entrypoint (recommended):**

```bash
uv run --extra aixi python -m aixi.experiments.run_smoke
# or from repo root:
make aixi-phase0
```

Source analyses and paper-to-module mapping live under [`analyses/`](analyses/) and [`modules/`](modules/); the consolidated design doc is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## Roadmap: from V0 toward broader AGI research

The implementation plan phases (abbreviated) are:

1. **Phase 0 — Parity:** fixed `pyaixi` config on a toy env (done; see report above).
2. **Phase 1 — Unified \(\xi\) API:** one `MixtureEnvModel` for MCTS and Self-AIXI heads (in progress; see tests and `mixture` modules).
3. **Phase 2 — Self-AIXI v0:** finite policy class, tabular \(Q\) on small envs; compare to MC-AIXI baseline.
4. **Phase 3+ — Scale & AIQI:** function approximation; AIQI-style return mixtures with explicit **on-policy** training assumptions.
5. **Later:** optional empowerment / FEP regularizers, joint prediction interfaces, quantum **classical** simulators—each gated as research, not core product claims.

This roadmap is **not** a promise of timelines; it is an ordering that keeps **regression baselines** and **computable** interfaces ahead of speculative extensions.

---

## Installation

From the **repository root** (AIXI uses optional extras on the main package):

```bash
git clone https://github.com/IlumCI/CR-CA.git
cd CR-CA
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[aixi]"        # adds pyaixi (VCS) + AIXI smoke deps
```

Using **uv** (recommended for locked local dev):

```bash
uv sync --extra dev --extra aixi
```

Optional heavy extras (e.g. `cvxpy`) are listed under `[project.optional-dependencies]` in [`../pyproject.toml`](../pyproject.toml).

---

## Developer workflow (AIXI)

| Task | Command |
|------|---------|
| Phase 0 smoke | `make aixi-phase0` or `uv run --extra aixi python -m aixi.experiments.run_smoke --help` |
| Tests (repo root) | `pytest` |
| AIXI-related markers | `pytest -m "aixi_p1 or aixi_p2"` (see `[tool.pytest.ini_options]` in `pyproject.toml`) |
| Format / lint | `black`, `ruff`, `mypy` (from `[dev]` extra) |

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
