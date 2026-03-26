# AIXI — implementation plan (synthesis)

**Status:** consolidated after SWA-4..SWA-9; supplementary `modules/` **complete** and **folded into §1–§4** (CEO merge 2026-03-25).  
**Scope:** A **computable** agent stack that composes the SWA-3 source bundle into implementable modules, explicit approximation knobs, and a research roadmap—not a claim of full Solomonoff universality.

---

## 0. Executive synthesis

The six sources split into **three implementable families** and **three theory lenses**:

| Family | Core loop | Canonical approximator in literature | Primary refs |
|--------|-----------|----------------------------------------|--------------|
| **A** Model-based Bayes + search | Maintain \(\xi(h)\); plan with expectimax / MCTS | MC-AIXI-CTW (`pyaixi`) | `03-pyaixi-repo`, classical AIXI |
| **B** Model-based Bayes + amortized policy | Maintain \(\xi\) + policy mixture \(\zeta\); act \(\arg\max_a Q_{\zeta\xi}(h,a)\) | CTW + ensemble policies + greedy Q (Self-AIXI experiments) | `01-neurips-2023` |
| **C** Model-free return induction | Maintain \(\psi\) over **discretized \(H\)-returns** on **periodically augmented** history; ε-greedy on posterior mean \(\hat Q\) | Bounded predictor class + tabular / sequence model | `02-arxiv-2602-23242` |

**Theory lenses** (inform design and evaluation metrics; not v1 code paths unless explicitly scoped):

- **Empowerment / free energy** (`04`): optional **regularizers** on top of (A) or (B)—KL(\(\pi^\*\|\zeta\)), MI surrogates, prediction-error terms—not a standalone agent spec.
- **Embedded / joint prediction** (`05`): **interface** guidance for multi-agent and self-modeling (joint action–percept heads, opponent depth); full MUPI/RUI is non-computable.
- **Quantum AIXI** (`06`): **classical simulation lab** only (small CPTP toy envs); no universal QAIXI in scope.

**Recommendation for “bleeding-edge computable” v1:** Implement **Family A** as the **regression baseline** (fork or wrap `pyaixi` patterns), then **Family B** as the **primary research line** (Self-AIXI-style amortization sharing the same \(\xi\) backend), then prototype **Family C** on **toy** domains to validate return-augmentation plumbing **separate** from (A)/(B) to avoid mixing assumptions (AIQI needs on-policy grain-of-truth; Thm 4.10 warns off off-policy reuse).

---

## 1. Notation ledger (cross-paper)

Symbols reused across analyses—**one meaning in code**:

| Symbol | Meaning | Code name suggestion |
|--------|---------|----------------------|
| \(\mathcal{A},\mathcal{O},\mathcal{R}\) | Finite action / obs / reward sets | `Action`, `Obs`, `Reward` enums or finite vocab |
| \(\mathcal{E} = \mathcal{O}\times\mathcal{R}\), \(e_t\) | Percept | `Percept` tuple |
| \(h_{<t}\) | History before \(t\) | `History` (immutable sequence or ring buffer + hash) |
| \(\nu(e\|h,a)\) | Environment kernel | `Environment.step` or learned `p(e\|h,a)` |
| \(\xi\) | Bayesian mixture over \(\mathcal{M}\) | `MixtureEnvModel` |
| \(w(\nu\|h)\) | Posterior weights | `log_weights: dict[ModelId, float]` or structured array |
| \(Q_\xi^\*, \pi_\xi^\*\) | Optimal Q / AIXI policy w.r.t. \(\xi\) | `planning.optimal_q`, `planning.aixi_policy` |
| \(\zeta,\omega(\pi\|h)\) | Policy mixture | `MixturePolicy` |
| \(Q_{\zeta\xi}(h,a)\) | Double-mixture Q | `self_aixi.q_zeta_xi(h,a)` |
| \(\pi_S\) | Self-AIXI greedy policy | `SelfAIXIAgent.act` |
| \(\psi_n,\psi\) | Return-predictor mixtures (AIQI) | `ReturnMixturePhase` |
| \(\mathrm{aug}_n(h)\) | Periodic augmentation | `augment_history_for_phase(h, n)` |
| \(\hat Q\) | Posterior mean on return grid | `aiqi.estimated_q(h,a)` |
| \(\gamma\) | Discount | `config.gamma` |
| \(H,M,N,\tau\) | AIQI horizon, grid, period, exploration | `AIQIConfig` |

**Clash resolutions:**

- **“Model-free” (AIQI):** No explicit \(\nu\) in the **action rule**; still a **predictive model of returns**. In code, name paths `aiqi.*` vs `world_model.*` to avoid conflating with AIXI’s percept model.
- **Self-AIXI \(Q_{\zeta\xi}\)** is **not** \(Q_\xi^\*\): comments and metrics must log `q_zeta_xi` vs `q_xi_star` separately when both are estimated.

### 1.1 Runtime computability (supplementary modules)

Stable boundary language shared across `aixi/modules/mod-*.md` (Potgieter, Kim, Müller, Ord, Leon)—**no new planning symbols**, but **one contract for code and docs**:

- **Computable** means **standard Turing machines** with explicit **per-tick / per-call budgets**. v1 does **not** assume Zeno clocks, supertasks, “halting from an infinite run,” TM+oracle steps, or other hypercomputer semantics.
- **`MixtureEnvModel`** (§3) implementations expose a **finite** model catalog and **`predict` / `update` / `revert`** that **terminate** under CI (and production) timeouts—operational proxies for “our \(\xi\) is not a hypercomputer.”
- **Marketing / reviewer copy:** “Universal” priors mean **finite** \(\mathcal{M}\) (and \(\mathcal{P}\)) surrogates aligned with §8, not literal Solomonoff mixture over all computable environments confused with the hypercomputation literature.
- **Reviewer rollup:** [`SUPERTASK_BOUNDARY.md`](SUPERTASK_BOUNDARY.md) states what v1 does **not** implement (no Zeno / \(\omega\)-step schedulers in the L1–L3 core) and links the cite-only `modules/mod-*.md` cluster.

---

## 2. Environment interface (all families)

```text
class Environment(Protocol):
    action_space: FiniteSpace
    observation_space: FiniteSpace
    reward_range: tuple[float, float]  # normalized to [0,1] if matching AIQI papers

    def reset(self) -> Percept: ...
    def step(self, action: Action) -> Percept: ...
```

**Encoding:** Follow `pyaixi` bit-width pattern for discrete sanity tests; for neural \(\xi\), use fixed tokenization of `(a,e)` pairs with explicit vocab size.

**Stochasticity:** All theory assumes explicit stochastic kernels; **deterministic** envs are a degenerate subclass.

**Episodic vs continuing:** Planning horizons in (A) are **search** horizons; AIQI’s \(H\) is **return truncation**—different parameters, do not overload one knob for both.

**Computability:** Each `step` is a **bounded-time** procedure on ordinary hardware; environments are not a hook for idealized infinite-time TM semantics or accelerating-time schedules (see §1.1).

---

## 3. Model class & prior (\(\xi\))

### 3.1 Tractable \(\mathcal{M}\) (recommended default)

- **CTW / variable-order Markov** over binary or tokenized interaction stream (MC-AIXI line).
- **Interface:** `predict(e|h,a)`, `update(e)`, `revert()` for search rollback (`pyaixi` pattern).
- **Finite catalog + bounded updates:** Index \(\mathcal{M}\) with finitely many ids; every call on the hot path must stay Turing-bounded (regression: CI timeouts on mixture update loops). This is the constructive reading of \(\xi\) in §1.1—**not** a claim of updating a literal universal semimeasure.

### 3.2 Neural or structured \(\mathcal{M}\) (stretch)

- Ensemble of differentiable env models with **Bayesian** or **ensemble** weighting approximating \(w(\nu|h)\).
- **Risk:** Merging theorems do not automatically transfer; treat as **heuristic** \(\xi\).

### 3.3 Quantum (`06`)

- **Out of core product** unless building `experiments/quantum_toy/` with explicit CPTP simulators and **classical** percept readouts.

---

## 4. Planning / control

### 4.1 Family A — MC-AIXI (`pyaixi` map)

| Object | Implementation hook |
|--------|---------------------|
| Rollouts from \(\xi\) | Sample percepts from CTW; revert between MCTS iterations |
| Search | UCT + `mc_simulations`, `horizon` |
| Action | `search()` then env step |

**Horizon semantics:** Search depth and rollout counts are **finite truncations** only; action selection does not depend on limits as \(t\to\infty\) or on infinite-step-in-finite-time idealizations (§1.1).

### 4.2 Family B — Self-AIXI

| Object | Implementation hook |
|--------|---------------------|
| \(\xi\) | Same backend as §3 |
| \(\mathcal{P}\) | Finite set: tabular policies, small NN ensemble, or distilled snapshots |
| \(\omega(\pi|h)\) | Dirichlet / online Bayes over policy ids from **observed** actions |
| \(Q_\nu^\pi(h,a)\) | **Estimated** via rollouts under \(\nu\) **or** TD bootstrapping—engineering choice; theorems are asymptotic |
| Action | `argmax_a sum_pi omega(pi|h) sum_nu w(nu|h) Q_nu_pi(h,a)` with budgeted eval |

**Optional hooks from `04`:** Add `lambda_kl * KL(softmax(Q*_xi) || zeta)` or empowerment surrogate **only** after stable \(\xi\) and baseline (B) trains.

### 4.3 Family C — AIQI

| Object | Implementation hook |
|--------|---------------------|
| Augmentation | Insert discretized return symbols \(z_i\) only at \(i \equiv n \pmod N\); maintain \(N\) phase models |
| \(\psi_n\) | CTW- or transformer-style sequence model on augmented alphabet |
| \(\hat Q(h,a)\) | \(\sum_z z \cdot \psi(z|h,a)\) |
| Action | ε-greedy with rate \(\tau\) |

**Deployment constraint:** Train/eval **on-policy**; do not expect self-optimization from frozen offline logs (Thm 4.10).

---

## 5. Repository layout (proposed under `aixi/` or top-level package)

```text
aixi/
  README.md
  IMPLEMENTATION_PLAN.md
  analyses/
  modules/               # Supplementary paper → module hooks (board addendum)
  aixi/
    env/                 # Environment protocol + toy envs
    encoding/            # Bit / token codecs
    models/
      ctw/               # Mixture xi (Family A/B)
      policy_mixture/    # zeta, omega updates (Family B)
      return_mixture/    # psi phases (Family C)
    planning/
      mcts/              # Family A
      self_aixi/         # Family B action + Q_zeta_xi estimators
      aiqi/              # Family C loop
    experiments/         # configs, sweeps, baselines
    tests/
```

**External:** Vendor or submodule `pyaixi` for parity runs; wrap rather than fork if license/maintenance dictates.

---

## 6. Phased rollout

1. **Phase 0 — Parity:** Reproduce one `pyaixi` conf on a toy env; fix upstream `decodePercept` naming bug if hitting `generate_percept` path (`03` note).
2. **Phase 1 — Unified \(\xi\) API:** Single `MixtureEnvModel` used by both MCTS and Self-AIXI heads.
3. **Phase 2 — Self-AIXI v0:** Finite \(\mathcal{P}\), tabular \(Q_\nu^\pi\) estimates on small envs; log suboptimality vs MC-AIXI on identical \(\xi\).
4. **Phase 3 — Scale \(\mathcal{P}\):** Function approximation for policies + \(Q\); regression tests against Phase 2.
5. **Phase 4 — AIQI prototype:** Implement augmentation + \(\psi\) on smallest env; verify ε-greedy loop; **no** claim of grain-of-truth without explicit construction.
6. **Phase 5 — Optional signals:** Empowerment / FEP regularizers (`04`); joint action–percept models for multi-agent (`05`); quantum toy env (`06`).

---

## 7. Evaluation metrics

- **Return:** discounted sum (match \(\gamma\) across agents).
- **Compute:** wall-clock, rollouts/search steps, CTW nodes touched.
- **Correctness:** A/B on identical \(\xi\) and \(\mathcal{P}\) should match within numerical tolerance for greedy action on tiny envs.
- **Theory alignment:** Report **on-policy** vs **off-policy** data regime explicitly for AIQI.

---

## 8. Open risks (explicit)

| Risk | Mitigation |
|------|------------|
| Universal classes are **incomputable** | Finite \(\mathcal{M},\mathcal{P},\mathcal{P}_n\); document misspecification |
| Self-AIXI **sensible off-policy** condition unverified for finite approx | Prefer on-policy \(\zeta\) updates; cite gap in writeups |
| AIQI **grain-of-truth** is reflective / fixed-point | Small constructed classes only; or treat as heuristic |
| Mixing (A)+(C) in one update rule | **Do not** without new analysis—separate agents or staged training |
| MUPI/RUI/QAIXI | Research direction only; not production paths |
| Empowerment (`04`) | Intrinsic terms can increase **power-seeking**; gate behind safety review |

---

## 9. Analysis traceability

| Analysis file | Pulls into plan |
|----------------|-----------------|
| `01-neurips-2023.md` | §1, §4.2, §6 Phase 2–3, Family B |
| `02-arxiv-2602-23242.md` | §1, §4.3, §6 Phase 4, Family C |
| `03-pyaixi-repo.md` | §3.1, §4.1, §5, §6 Phase 0 |
| `04-arxiv-2502-15820.md` | §0, §4.2 optional, §8 |
| `05-arxiv-2511-22226.md` | §0, §5 joint-model hook, §8 |
| `06-arxiv-2505-21170.md` | §0, §3.3, §6 Phase 5, §8 |

### Supplementary modules → plan

| Module file | Folded into |
|-------------|-------------|
| `modules/mod-cs-0412022.md` | §1.1, §2, §8 |
| `modules/mod-arxiv-1411-5679.md` | §1.1, §8 |
| `modules/mod-arxiv-2505-14698.md` | §1.1, §8 |
| `modules/mod-math-0209332.md` | §1.1, §3.1, §8 |
| `modules/mod-hilbert-machine.md` | §1.1, §8 |

---

### Analysis checklist (complete)

- [x] `analyses/01-neurips-2023.md`
- [x] `analyses/02-arxiv-2602-23242.md`
- [x] `analyses/03-pyaixi-repo.md`
- [x] `analyses/04-arxiv-2502-15820.md`
- [x] `analyses/05-arxiv-2511-22226.md`
- [x] `analyses/06-arxiv-2505-21170.md`

### Supplementary module checklist (complete)

- [x] `modules/mod-cs-0412022.md` (+ validation notes)
- [x] `modules/mod-arxiv-1411-5679.md` (+ validation notes, Math & CS Wizard 2026-03-24)
- [x] `modules/mod-arxiv-2505-14698.md` (+ validation notes)
- [x] `modules/mod-math-0209332.md` (+ validation notes, Math & CS Wizard 2026-03-24)
- [x] `modules/mod-hilbert-machine.md` (+ validation notes, Math & CS Wizard 2026-03-24)

---

## 10. Supplementary theory & modules (board addendum, 2026-03-24)

Five additional references extend **priors, convergence guarantees, discounted settings, recent variants, and computability boundaries**. They are tracked as **first-class module hooks** under `aixi/modules/` (see `modules/README.md`). Each must state **implement vs cite-only**, map to **Families A/B/C**, and update the notation ledger if it introduces symbols reused in code.

| Source | Module file | Intended use (to be refined by Research Engineer) |
|--------|-------------|---------------------------------------------------|
| [cs/0412022](https://arxiv.org/pdf/cs/0412022) | `mod-cs-0412022.md` | **Corrected:** Potgieter surveys **Zeno machines & hypercomputation** (Church–Turing, halting, “Turing barrier” rhetoric). **Cite-only** scope/computability boundary for v1—reinforces finite \(\mathcal{M},\mathcal{P}\) and Turing-bounded runtime; pairs with §8 and `mod-hilbert-machine.md`, not §2–§3 env APIs. |
| [1411.5679](https://arxiv.org/pdf/1411.5679) | `mod-arxiv-1411-5679.md` | **Corrected:** Kim (cs.FL) on **Zeno machines and infinite-time Turing machines**—**cite-only** boundary (no universal halting via infinite-time exploitation in the paper’s setup). Pairs with Potgieter/Müller on hypercomputation; **not** a discounted-AIXI reference. |
| [2505.14698](https://arxiv.org/pdf/2505.14698) | `mod-arxiv-2505-14698.md` | **Corrected:** Müller (cs.LO) on **hypercomputing supertasks** vs Church–Turing—**cite-only** scope/computability boundary; **no** algorithmic bridge to MC-AIXI / Self-AIXI / AIQI. Pairs with §8 and `mod-cs-0412022.md` / `mod-arxiv-1411-5679.md`. |
| [math/0209332](https://arxiv.org/pdf/math/0209332) | `mod-math-0209332.md` | **Corrected:** Ord surveys **hypercomputation** (models beyond TMs) and Church–Turing misconceptions—**cite-only** boundary for §3 \(\xi\): explains why **finite** \(\mathcal{M}\) and Turing-bounded updates are required; **not** mixture-convergence theory. Pairs with §8 and `mod-cs-0412022.md` / `mod-arxiv-2505-14698.md`. Optional **tests**: bounded-time mixture updates in CI. |
| [HilbertMachine.pdf](https://philsci-archive.pitt.edu/2869/1/HilbertMachine.pdf) | `mod-hilbert-machine.md` | Computability / limits narrative; feeds §8 risks and **scope boundaries** for v1. |

**Process:** Research Engineer completes each module spec **first**. Math & CS Wizard fills **Validation notes** in the same file (or follow-up PR), checking consistency with §1 and flagging non-computable claims before implementation milestones.

**CEO merge rule:** When all five module specs are **content-complete** and validated, fold any stable definitions into §1–§4 and close the supplementary checklist above. **Done 2026-03-25** (§1.1 + §2–§4 hooks + §10 row correction for `1411.5679`).
