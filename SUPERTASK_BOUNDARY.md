# Computability boundary

What this codebase assumes about time, steps, and supertask idealizations, spelled out once so the question stops coming up in review. Everything on the execution path is standard Turing computability with explicit per-call budgets. The idealized limits live in the referenced papers and stay there.

## Layer labels (L1-L3)

Documentation labels for where a computability claim lands. There is no separate runtime API behind them.

| Layer | Code anchor | Meaning |
|-------|-------------|---------|
| **L1** | [`src/env/`](src/env/) | `Environment::step` and related I/O: each step is a bounded-time procedure on ordinary hardware. |
| **L2** | [`src/models/`](src/models/), [`src/llm/`](src/llm/) | The `EnvModel` catalog and finite \(\mathcal{M}\): finite component list, terminating `predict` / `learn` / `revert` on every path, bit-exact revert as a tested contract. |
| **L3** | [`src/planning/`](src/planning/) | ρUCT and exact expectimax: finite search depth, simulation counts, and horizons, every knob validated by `SearchBudget`. Nothing depends on limits as \(t \to \infty\) or on infinitely many steps completing inside a tick. |

Theory lenses, quantum toys, and the supplementary `modules/` hooks sit outside this execution core unless explicitly scoped in.

## What the execution core will not implement

- Zeno clocks or accelerating-time schedules that pack infinitely many machine steps into a finite physical or logical tick.
- Completed supertasks: no scheduler API that assumes an \(\omega\)-sequence (or longer) of updates has finished inside a single bounded user-visible call.
- Halting-from-an-infinite-run or other TM-plus-oracle semantics on the hot path.
- Hypercomputer output as a dependency of any default configuration.

"Computable" in code and tests means standard Turing computability with explicit per-call and per-tick budgets. The hypercomputation literature's idealized machines are cited for scope; none of them run here.

## Cite-only module cluster

These papers inform scope and reviewer Q&A. They add zero planning symbols and zero runtime oracle APIs:

- [`modules/mod-cs-0412022.md`](modules/mod-cs-0412022.md), Potgieter (Zeno machines, hypercomputation survey)
- [`modules/mod-arxiv-1411-5679.md`](modules/mod-arxiv-1411-5679.md), Kim (Zeno / infinite-time TM framing)
- [`modules/mod-arxiv-2505-14698.md`](modules/mod-arxiv-2505-14698.md), Müller (supertasks vs Church-Turing)
- [`modules/mod-math-0209332.md`](modules/mod-math-0209332.md), Ord (hypercomputation survey)
- [`modules/mod-hilbert-machine.md`](modules/mod-hilbert-machine.md), Leon (philosophical supertask "Hilbert machine")
