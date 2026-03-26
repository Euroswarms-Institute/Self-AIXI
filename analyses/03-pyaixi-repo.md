# Analysis: pyaixi (reference implementation)

**Repository:** https://github.com/sgkasselau/pyaixi  
**Primary reference:** J. Veness, K. S. Ng, M. Hutter, W. Uther, D. Silver, *A Monte Carlo AIXI Approximation*, JAIR 40 (2011), [doi:10.1613/jair.3125](http://dx.doi.org/10.1613/jair.3125) — TechReport: https://arxiv.org/abs/0909.0801  
**Lineage:** Python port of the C++ MC-AIXI-CTW stack (e.g. [moridinamael/mc-aixi](https://github.com/moridinamael/mc-aixi)); README also points to Joel Veness’s software page.

## Summary

**pyaixi** is a **discrete**, **finite-resource** implementation of **MC-AIXI-CTW**: a **Bayes-adaptive** agent that (1) maintains a **Context Tree Weighting (CTW)** model over a **binary symbol stream** encoding the alternation **action → percept (reward, then observation)**, and (2) uses **Monte Carlo tree search** with **UCT-style** action selection and **model-based rollouts** (sample percepts from the CTW) to approximate **planning** toward high discounted return over a **fixed search horizon**.

This matches the **classical AIXI picture** at the algorithmic level: an **explicit environment model** (here: CTW mixture over histories) plus **look-ahead** to pick actions — **not** Self-AIXI’s policy mixture, and **not** AIQI-style **return-predictor** induction without a world model in the loop.

## Code map (theory ↔ implementation)

| Theoretical object (MC-AIXI / AIXI family) | Where it lives in pyaixi |
|-------------------------------------------|---------------------------|
| Finite **horizon** planning; Monte Carlo return estimates | `MC_AIXI_CTW_Agent.horizon`, `mc_simulations`; `MonteCarloSearchNode.sample()` in `pyaixi/search/monte_carlo_search_tree.py` |
| **Mixture environment model** \(\xi\) over next symbols (bits) given context | `CTWContextTree` / `CTWContextTreeNode` in `pyaixi/prediction/ctw_context_tree.py` — **CTW** + **Krichevsky–Trofimov (KT)** estimators at leaves, **log-space** probabilities, `predict`, `update`, `revert` |
| **Generative rollouts** from \(\xi\) (imagined percepts) | `generate_percept_and_update()`, `generate_random_symbols*` on the context tree; used inside MCTS at **chance** nodes |
| **Action selection** as **planning** / search (AIXI-style “think then act”) | `MC_AIXI_CTW_Agent.search()` — builds a fresh `MonteCarloSearchNode`, runs `sample()` `mc_simulations` times with `model_revert` between samples |
| **UCT / upper-confidence exploration** along the tree | `MonteCarloSearchNode.select_action()` — exploration constant, `unexplored_bias` for unvisited actions |
| **Playout policy** after tree expansion | `playout()`: **uniform random** actions + **model-sampled** percepts |
| **History** as symbol sequence (actions and percepts interleaved) | `context_tree.history`; strict alternation enforced via `last_update` (`action_update` / `percept_update`) in `pyaixi/agent.py` |
| **Discrete** \(\mathcal{A}\), observations, rewards | Each environment sets `valid_*` and **bit widths** via `action_bits()`, `observation_bits()`, `reward_bits()` on `pyaixi/environment.py` |
| Agent / env **interface** | Base `Agent` (`pyaixi/agent.py`), `Environment` (`pyaixi/environment.py`); concrete envs under `pyaixi/environments/` |
| **CLI** and interaction cycle | `aixi.py`: load config → instantiate env + `mc_aixi_ctw` → loop: `model_update_percept` → optional ε-random **exploration** or `search()` → `performAction` on env |

**Entry points:** run `python aixi.py -v conf/<env>.conf` (see `README.md`). Config keys align with constructor options: `agent-horizon`, `ct-depth`, `mc-simulations`, `learning-period`, exploration flags, etc.

## Extension points (where to plug in new science)

1. **New agent algorithms** — Subclass `pyaixi.agent.Agent` and implement `model_update_action`, `model_update_percept`, and `search` (README: set `agent` in conf or `-a`). Example pattern: replace CTW with another **sequence model** but keep the **MCTS shell**, or replace MCTS with **greedy one-step** \(\arg\max_a Q\) if targeting a **Self-AIXI-style** outer loop (you would still need a separate **policy mixture** not present here).

2. **New environments** — Subclass `Environment`, define discrete actions/obs/rewards and transition logic; add a `conf/*.conf` with `environment` module name (`README.md`).

3. **Search** — `MonteCarloSearchNode` encodes the **alternating** decision/chance structure; constants (`exploration_constant`, `unexplored_bias`) and the **playout** policy in `playout()` are natural tuning / replacement points (e.g. different rollout policies, priors, or partial expansion).

4. **Prediction** — `CTWContextTree` is the **entire** world model; swapping in **other Bayesian sequence predictors** (bounded variants, different context schemes) is the main path to “new \(\xi\)” without touching env code.

5. **Tooling** — `pyaixi/util.py` (bit encode/decode, enums); `six` / Python 2–3 shims for legacy compatibility.

## Gaps vs theory papers (SWA-3 bundle)

| Paper / idea | vs pyaixi |
|--------------|-----------|
| **Universal AIXI** (Hutter) | CTW depth and **finite** symbol alphabet give a **computable, misspecified** \(\xi\), not Solomonoff / full \(\mathcal{M}\). |
| **NeurIPS 2023 Self-AIXI** ([01-neurips-2023.md](01-neurips-2023.md)) | No **policy mixture** \(\zeta\) and no **\(Q_{\zeta\xi}\)** greedy step; this codebase is **planning-heavy MC-AIXI**, the paper’s **baseline** contrast. |
| **arXiv 2602.23242 AIQI** ([02-arxiv-2602-23242.md](02-arxiv-2602-23242.md)) | **Model-based** control loop (CTW predicts percepts); AIQI’s **return-predictor** \(\psi\) and **periodic augmentation** are absent. |
| **Asymptotic optimality / merging** | JAIR MC-AIXI theory is about the **approximation class**; this repo is an **engineering reference**, not a proof artifact. |
| **Exploration in `aixi.py`** | ε-random actions in the **main loop** are a **practical heuristic**, not part of the idealized **Bayes-optimal** AIXI definition. |

**Implementation hygiene note:** `generate_percept()` in `mc_aixi_ctw.py` calls `self.decodePercept(...)` while the method is named `decode_percept` — likely a **Python 3 bug** if that path is exercised (MCTS uses `generate_percept_and_update`, which is fine). Worth fixing upstream if you rely on `generate_percept`.

## Links to other SWA-3 sources

| Relation | See |
|----------|-----|
| Self-AIXI vs MC-AIXI role in the literature | [NeurIPS 2023 Self-AIXI](01-neurips-2023.md) |
| Model-free universal agent (AIQI) contrast | [arXiv 2602.23242](02-arxiv-2602-23242.md) |
| Empirical / algorithmic cousins (distributional RL, etc.) | [arXiv 2502.15820](04-arxiv-2502-15820.md), [arXiv 2511.22226](05-arxiv-2511.22226.md), [arXiv 2505.21170](06-arxiv-2505-21170.md) |
| Repo integration / planning | `aixi/README.md`, `aixi/IMPLEMENTATION_PLAN.md` |

## Notes

- **Performance:** README recommends **PyPy**; Python is for clarity and prototyping, not parity with C++ speed.
- **License:** CC BY-SA 3.0 (with author note on possible LGPL/GPL dual licensing where legal).
