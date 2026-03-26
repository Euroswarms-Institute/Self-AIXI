# Phase 0 — pyaixi parity on toy env (run report)

**Scope:** One fixed `pyaixi` configuration on the **`CoinFlip`** environment, matching [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §6 Phase 0.

## Configuration (frozen for regression)

| Item | Value |
|------|--------|
| Environment | `pyaixi.environments.coin_flip.CoinFlip` |
| Options | `{"coin-flip-p": 0.6}` (parity loop); `{"coin-flip-p": 0.55}` (MCTS smoke in `run_smoke.py`) |
| Agent | `MC_AIXI_CTW_Agent` with `coin_flip_options_toy()` from `aixi.aixi.parity.coin_flip` |
| CTW depth | `8` (`ct-depth` in options) |
| Other agent options | `agent-horizon=4`, `mc-simulations=2`, `learning-period=0` |

## Metric

**Symbol-level ξ parity:** After each percept update, `agent.context_tree.root.log_probability` equals a **manually driven** `CTWContextTree` fed the same encoded symbol history (tolerance `1e-9` absolute). Implementation: `run_coin_flip_symbol_parity` in `aixi/aixi/parity/coin_flip.py`.

## How to reproduce

From repository root (requires `crca[aixi]`):

```bash
uv run --extra aixi python -m aixi.experiments.run_smoke --seed 42 --parity-cycles 20 --mcts-horizon 6
```

Or:

```bash
make aixi-phase0
```

Default CLI flags: `--seed 42`, `--parity-cycles 20`, `--mcts-horizon 6` (parity + short MCTS episode using `decode_percept` on imagined symbols — not the legacy `generate_percept` path).

## Last recorded smoke (local)

| Check | Seed | Result |
|-------|------|--------|
| Parity (21 rows: initial percept + 20 cycles) | 42 | PASS |
| MCTS short episode (6 env steps) | 42 | PASS (`return_sum=2.0` for that stochastic run) |

**CI:** `pytest tests/test_aixi_phase01.py` (marked skips if `pyaixi` unavailable).

## `decodePercept` / `generate_percept` note

Per [`analyses/03-pyaixi-repo.md`](../analyses/03-pyaixi-repo.md), upstream `generate_percept()` may call `decodePercept` while the method is `decode_percept`. **This Phase 0 harness does not exercise that path** — MCTS rollouts use `generate_percept_and_update` and the local smoke uses explicit `decode_percept` in `run_smoke.py`. Track upstream if you switch to the broken call path.
