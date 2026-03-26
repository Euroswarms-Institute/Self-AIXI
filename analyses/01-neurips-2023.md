# Analysis: NeurIPS 2023 — *Self-Predictive Universal AI* (56a22563)

**Authors:** Elliot Catt, Jordi Grau-Moya, Marcus Hutter, Matthew Aitchison, Tim Genewein, Grégoire Delétang, Kevin Li Wenliang, Joel Veness (Google DeepMind).

**URL:** https://proceedings.neurips.cc/paper_files/paper/2023/file/56a225639da77e8f7c0409f6d5ba996b-Paper-Conference.pdf

**OpenReview:** https://openreview.net/forum?id=psXVkKO9No

## Summary

The paper defines **Self-AIXI** (denoted policy \(\pi_S\)), a **Bayes-universal** agent that selects actions with a **single greedy step** \(\arg\max_a Q_{\zeta\xi}(h,a)\) instead of AIXI’s full **planning / search** over futures. Here \(\xi\) is the usual **Bayesian mixture over environments** \(\mathcal{M}\) (as in AIXI), and \(\zeta\) is a **new Bayesian mixture over policies** \(\mathcal{P}\). The agent “**self-predicts**” in the sense that \(\zeta\) is updated on the **stream of actions the agent itself produces** after each greedy improvement step.

**Main claim (informal):** Under stated assumptions (including a technical “**sensible off-policy**” condition for \(\pi_S\)), Self-AIXI’s performance converges to **AIXI’s** in appropriate **expectation** limits, and it inherits **Legg–Hutter universal intelligence** and **self-optimization** analogues to classical AIXI results.

**Positioning:** Motivated by **AlphaZero / MuZero**-style **distillation** of search into a parametric policy; Self-AIXI is offered as a **principled** universal counterpart where “planning effort” is replaced by **learning / prediction** over one’s own policy.

## Notation (as in the paper)

| Symbol | Meaning |
|--------|---------|
| \(\mathcal{A}, \mathcal{O}, \mathcal{R}\) | Finite action, observation, reward sets |
| \(\mathcal{E} := \mathcal{O} \times \mathcal{R}\) | Percepts |
| \(\mathcal{H} := (\mathcal{A} \times \mathcal{E})^\*\) | Histories |
| \(\Delta_X\) | Probability simplex / distributions on \(X\) |
| \(h_{<t} = a_1 e_1 \cdots a_{t-1} e_{t-1}\) | History strictly before time \(t\) |
| \(\pi : \mathcal{H} \to \Delta_{\mathcal{A}}\) | Policy |
| \(\nu : \mathcal{H} \times \mathcal{A} \to \Delta_{\mathcal{E}}\) | Environment (next-percept kernel) |
| \(\mu^\pi\) | Induced law on histories under \(\pi\) in \(\mu\) |
| \(\gamma \in (0,1)\) | Discount factor |
| \(\Gamma_t\) | Normalization factor for discounted returns from \(t\) (paper uses \(\Gamma_t = \sum_{k=t}^\infty \gamma^{k-t}\); for geometric discount this is \(1/(1-\gamma)\)) |
| \(V_\nu^\pi(h_{<t})\) | Normalized discounted expected return from \(h_{<t}\) under \(\pi,\nu\) |
| \(Q_\nu^\pi(h_{<t}, a_t) := V_\nu^\pi(h_{<t} a_t)\) | State–action value |
| \(D_m(\nu_1^{\pi_1}, \nu_2^{\pi_2} \mid h_{<t})\) | Total-variation distance between laws of length-\(m\) futures |
| \(\xi\) | Bayesian mixture over \(\mathcal{M}\) with prior weights \(w(\nu)\), posterior \(w(\nu \mid h)\) |
| \(\pi_\xi^\*\) / **AIXI** | \(\arg\max_a Q_\xi^\*(h,a)\) — optimal w.r.t. mixture environment (standard definition in paper as **AI\(_\xi\)**) |
| \(\zeta\) | Bayesian mixture over \(\mathcal{P}\) with prior \(\omega(\pi)\), posterior \(\omega(\pi \mid h)\) |
| \(Q_{\zeta\xi}(h,a)\) | Double mixture: \(\sum_{\pi \in \mathcal{P}} \omega(\pi|h) \sum_{\nu \in \mathcal{M}} w(\nu|h)\, Q_\nu^\pi(h,a)\) |
| \(\pi_S\) / **Self-AIXI** | \(\arg\max_{a_t} Q_{\zeta\xi}(h_{<t}, a_t)\) |
| \(\Upsilon(\pi|h)\) | Legg–Hutter intelligence: \(\sum_{\nu} w(\nu|h)\, V_\nu^\pi(h) = V_\xi^\pi(h)\) |

## Formal objects

### Definitions

- **Value and Q-functions** (Definition 1): \(V_\nu^\pi\), \(Q_\nu^\pi\), optimal \(V_\nu^\*\), \(\pi_\nu^\*\), with Bellman-style sums over actions and percepts (Eq. (1) in paper).
- **TV distance** (Definition 2): \(D_m\) between future rollouts; used to bound value differences.
- **Bayesian mixture environment** (Definition 4): \(\xi(e_t \mid h_{<t} a_t) = \sum_\nu w(\nu|h_{<t})\, \nu(e_t \mid h_{<t} a_t)\).
- **AI\(_\xi\) / AIXI** (Definition 5): greedy policy w.r.t. \(Q_\xi^\*\) (their notation \(\pi_\xi^\*\)).
- **Legg–Hutter intelligence** (Definition 6): \(\Upsilon(\pi|h_{<t}) = V_\xi^\pi(h_{<t})\).
- **Bayesian mixture policy** (Definition 8): \(\zeta(a_t|h_{<t}) = \sum_{\pi \in \mathcal{P}} \omega(\pi|h_{<t})\, \pi(a_t|h_{<t})\).
- **Self-AIXI** (Definition 10): \(\pi_S(h_{<t}) \in \arg\max_{a_t} Q_{\zeta\xi}(h_{<t}, a_t)\).

### Lemmas (selected)

- **Lemma 3:** TV distance upper-bounds \(|V_{\nu_1}^{\pi_1} - V_{\nu_2}^{\pi_2}|\) (from Leike).
- **Lemma 9 (linearity):** \(Q_\xi^\pi\) and \(Q_\nu^\zeta\) decompose as convex combinations of \(Q_\nu^\pi\) under mixture posteriors.
- **Lemma 13:** For \(\mu \in \mathcal{M}\), \(D_\infty(\xi^\pi, \mu^\pi \mid h_{<t}) \to 0\) a.s. under \(\mu^\pi\) (merging / dominance).
- **Lemma 14:** Dual merging statement for \(\zeta\) approximating a fixed \(\pi \in \mathcal{P}\) in expectation over policies.
- **Lemma 15:** If a policy is “one-step good” (bounded \(\mathbb{E}|\max_a Q_\xi^\pi(h,a) - V_\xi^\pi(h)|\)) and a mild contraction-style condition holds, then suboptimality gap \(\mathbb{E}\Delta(h)\) is bounded by \(\beta/(1-\gamma(1+\alpha))\).
- **Lemma 17, 20:** Bridge convergence statements from \(\xi\)-expectation to \(\mu\)-expectation under additional merging assumptions.

### Theorems / corollaries (main narrative)

- **Theorem 7** (cited): Classical **AIXI self-optimization** (Hutter).
- **Theorem 16:** Under “sensible off-policy” assumption on \(\pi_S\): \(\mathbb{E}_{\pi_S}^{\xi}\big[V_\xi^\*(h) - V_\xi^{\pi_S}(h)\big] \to 0\) as \(t \to \infty\).
- **Theorem 18:** For **all** \(\mu \in \mathcal{M}\): \(\mathbb{E}_{\pi_S}^{\mu}\big[V_\xi^\*(h) - V_\xi^{\pi_S}(h)\big] \to 0\) (combine Theorem 16 + Lemma 17).
- **Corollary 19:** LH form: \(\mathbb{E}_{\pi_S}^{\mu}\big[\max_\pi \Upsilon(\pi|h) - \Upsilon(\pi_S|h)\big] \to 0\).
- **Theorem 21:** If \(D_\infty(\mu^{\pi_\xi^\*}, \xi^{\pi_\xi^\*} \mid h) \to 0\) \(\mu^{\pi_S}\)-a.s., then \(\mathbb{E}_{\pi_S}^{\mu}\big|V_\mu^{\pi_\xi^\*}(h) - V_\mu^{\pi_S}(h)\big| \to 0\) (performance in **true** \(\mu\) matches AIXI asymptotically in expectation).
- **Theorem 22:** **Self-optimization** analogue for \(\pi_S\), with an **extra assumption** \(V_\xi^\zeta(h) \geq V_\xi^{\pi_t}(h) - \epsilon_t\), \(\epsilon_t \to 0\), parallel to Theorem 7’s conditions. Authors note finding interesting \((\mathcal{M}, \mathcal{P})\) where this holds is **future work**.

### Algorithms / empirical slice

- **Appendix B** (summarized in §3.2): Compares a **Self-AIXI approximation** (CTW-style environment model + **on-policy** Q-learning style updates for the mixture policy) against an **AIXI approximation** using **MCTS** for Q-estimates — Self-AIXI approximation **wins or ties** on their small suite.

## Implementable slice

What can be **directly implemented** in code (in principle):

- **Finite / tractable \(\mathcal{M}\):** Mixture predictor with efficient Bayes updates (e.g. **context-tree weighting** as in MC-AIXI line of work).
- **Finite / structured \(\mathcal{P}\):** Explicit ensemble of policies with a tractable prior \(\omega(\pi)\); maintain posteriors online from observed actions.
- **One-step greedy action:** \(\arg\max_a Q_{\zeta\xi}(h,a)\) given current posteriors \(w(\nu|h)\), \(\omega(\pi|h)\) and estimates of \(Q_\nu^\pi(h,a)\) (e.g. tabular or function approximation — **not** part of the formal theorems).

What remains **idealized / non-computable** in the full universal story:

- Taking \(\mathcal{M}\) as **all computable environments** and \(\mathcal{P}\) as **all computable policies** with **Solomonoff-style** priors reproduces the **same incomputability** as standard AIXI: exact \(\xi\), \(\zeta\), and \(Q_{\zeta\xi}\) are not computable.
- **Theorems** rely on **merging**, **expectation** limits, and (for Theorem 16) the **“sensible off-policy”** condition — a **conjectured** plausibility assumption rather than a derived property for general universal classes.
- **Theorem 22**’s extra assumption on \(V_\xi^\zeta\) vs. reference policy sequence is **explicitly** left open for interesting classes.

**Engineering reading:** The actionable blueprint is: **(environment mixture + policy mixture) \(\Rightarrow\) define \(Q_{\zeta\xi}\) \(\Rightarrow\) greedy action \(\Rightarrow\) feed actions back into \(\zeta\)** — i.e. **amortized policy improvement** with a **Bayesian self-model**, analogous in spirit to distilling MCTS into a policy, but with a cleaner universal-Bayes story than heuristic distillation.

## Links to other SWA-3 sources

| This paper | Relation | See analysis |
|------------|----------|----------------|
| AIXI / \(\xi\), \(Q_\xi^\*\), self-optimization | Self-AIXI is defined **relative to** the same mixture-environment formalism as classic AIXI | [arXiv 2602.23242](02-arxiv-2602-23242.md) (companion theory line in SWA-3 bundle) |
| MC-AIXI, CTW, planning vs learning | Paper cites **MC-AIXI-CTW** as **planning-heavy** baseline; experiments use **CTW + MCTS (AIXI approx.)** vs **CTW + self-prediction (Self-AIXI approx.)** | [pyaixi](03-pyaixi-repo.md) — reference code landscape for AIXI approximations |
| Practical RL (MuZero, policy distillation) | Motivation for **self-prediction** as theoretical counterpart to **search distillation** | [arXiv 2502.15820](04-arxiv-2502-15820.md), [arXiv 2511.22226](05-arxiv-2511-22226.md), [arXiv 2505.21170](06-arxiv-2505-21170.md) — compare what each adds for **implementable** agents vs this **Bayes-optimal** ideal |
| CRCA / repo | No direct causal-inference hook; relevant if we **implement** a bounded \(\mathcal{M},\mathcal{P}\) agent or **interface** predictors (e.g. templates, agents) as **mixture components** | `aixi/README.md`, `aixi/IMPLEMENTATION_PLAN.md` |

## Notes

- **Naming:** The PDF filename uses **“self-predictive-universal-ai”**; the mathematical object is **Self-AIXI** (\(\pi_S\)).
- **“Self-AIXI does not optimize the future”** (Remark 11): it greedes w.r.t. **mixture-policy** Q-values \(Q_{\zeta\xi}\), not w.r.t. the **optimal** \(Q_\xi^\*\) at each step — the **optimality** is **asymptotic** (expectation / merging), not per-step Bellman optimality.
- **Safety discussion (§6):** Authors argue **interpretability** from **separate** beliefs over environment vs. **self** (policy mixture); still a **theoretical** agent — none of the safety claims remove misuse risk from **approximate** or **misspecified** \(\mathcal{M},\mathcal{P}\).
