# v1 computability boundary (reviewer-facing)

This page rolls up what the CR-CA **AIXI-style** stack **does** and **does not** assume about time, steps, and “supertask” idealizations. It complements the normative contract in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) **§1.1** (*Runtime computability*) and the risk table in **§8** (*Open risks*).

## Layer labels (L1–L3)

For documentation only—**not** a separate runtime API:

| Layer | Plan anchor | Meaning |
|-------|-------------|---------|
| **L1** | [§2 Environment interface](IMPLEMENTATION_PLAN.md#2-environment-interface-all-families) | `Environment.step` and related I/O: each step is a **bounded-time** procedure on ordinary hardware. |
| **L2** | [§3 Model class & prior](IMPLEMENTATION_PLAN.md#3-model-class--prior-xi) | `MixtureEnvModel` and finite \(\mathcal{M}\): **finite** catalog, **terminating** `predict` / `update` / `revert` under CI and production timeouts. |
| **L3** | [§4 Planning / control](IMPLEMENTATION_PLAN.md#4-planning--control) | MCTS, Self-AIXI, AIQI-style control: **finite** search depth, rollout counts, and truncations—no dependence on limits as \(t \to \infty\) or on infinite-step-in-finite-time idealizations (see §1.1 in the plan). |

Optional theory lenses, quantum toys, and supplementary `modules/` hooks sit **outside** this L1–L3 execution core unless explicitly scoped.

## What v1 **will not** implement (L1–L3)

- **Zeno clocks** or **accelerating-time schedules** that pack infinitely many machine steps into a finite physical or logical tick.
- **Completed supertasks**: no scheduler API that assumes an **\(\omega\)-sequence** (or longer) of updates has “finished” inside a single bounded user-visible step or oracle call.
- **Halting-from-an-infinite-run** or other **TM+oracle** semantics on the hot path.
- **Hypercomputer output** or “digital hypercomputing” machinery as part of Families **A**, **B**, or **C** defaults.

“Computable” in code and tests means **standard Turing computability** with explicit **per-call / per-tick budgets**, as in plan §1.1—not the hypercomputation literature’s idealized machines.

## Cite-only module cluster

These papers inform **scope and reviewer Q&A** only; they do **not** add planning symbols or runtime oracle APIs:

- [`modules/mod-cs-0412022.md`](modules/mod-cs-0412022.md) — Potgieter (Zeno machines, hypercomputation survey)
- [`modules/mod-arxiv-1411-5679.md`](modules/mod-arxiv-1411-5679.md) — Kim (Zeno / infinite-time TM framing)
- [`modules/mod-arxiv-2505-14698.md`](modules/mod-arxiv-2505-14698.md) — Müller (supertasks vs Church–Turing)
- [`modules/mod-math-0209332.md`](modules/mod-math-0209332.md) — Ord (hypercomputation survey)
- [`modules/mod-hilbert-machine.md`](modules/mod-hilbert-machine.md) — Leon (philosophical supertask “Hilbert machine”; cite-only)

## Related

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §1.1, §2 (computability paragraph), §4.1 (horizon semantics), §8  
- Parent planning context: supertask track under project CR-CA (board plan §2–§5 on issue SWA-21).
