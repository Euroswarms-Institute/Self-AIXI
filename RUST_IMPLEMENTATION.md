# MC-AIXI in Rust — Family A with a surgically dissected base model

This crate is the Rust realization of **Family A** from `IMPLEMENTATION_PLAN.md`:
a computable approximation of AIXI in the sense of Veness, Ng, Hutter, Uther &
Silver, *A Monte-Carlo AIXI Approximation*, JAIR 40 (2011) — ρUCT expectimax
search over a Bayesian mixture environment model ξ — with one modern addition:
the mixture's catalog contains, next to FAC-CTW, a frozen large language model
(`empero-ai/Qwen3.8-2B`, a Gated-DeltaNet/attention hybrid) reduced by surgery
to a conditional probability engine over the two-token bit alphabet.

No inference framework is used anywhere: GGUF container parsing, GGML
quantization kernels, the hybrid transformer forward pass, and the search all
live in this crate (`llama.cpp` appears only as a **development-time test
oracle**, never as a dependency).

## Theory → code map

| Theory | Approximation | Code |
|---|---|---|
| Solomonoff prior M | finite Bayes mixture ξ = Σ_ν w⁰_ν ν; dominance ⇒ extra log-loss vs any component ≤ ln(1/w⁰_ν) | `models::mixture::BayesMixture` |
| environment class | FAC-CTW: per-percept-bit context trees, each a mixture over ALL depth ≤ D suffix trees, prior 2^(−Γ_D) (JAIR §5.4) | `models::fac_ctw::FacCtwModel` over `models::ctw::CtwModel` |
| modern universal prior | frozen Qwen3.8-2B conditioned on the interleaved action/percept bit-token stream, renormalized onto {"0","1"} | `llm::env_model::LlmModel` |
| expectimax (JAIR eq. 10) | ρUCT: UCT over alternating decision/chance nodes, horizon-normalized UCB, percepts sampled bit-by-bit from ξ (JAIR Alg. 1–4) | `planning::rho_uct` |
| — ground truth — | exact enumeration of the expectimax on tiny domains | `planning::expectimax` |
| online learning | real percepts learned permanently; imagined ones learned-then-reverted **bit-exactly** | the `EnvModel` contract (`models`) |

The `EnvModel` trait mirrors the repo's Python `MixtureEnvModel` Protocol
(`aixi/models/mixture.py`): `root_log_probability`, `predict_bit_probability`,
`learn_symbols` (percept path), `append_history_symbols` (action path — the
FAC split: conditioned on, never predicted), and LIFO
`revert_learned/history_symbols`. The revert-exactness contract of
`aixi/planning/xi_rollouts.py` (≤ 1e-8 drift) is strengthened to **bit-exact**
throughout: undo frames record previous values rather than replaying deltas.

## The surgery

`Qwen3.8-2B-Q4_K_M.gguf` (the user's LM Studio model, fetched from its
Hugging Face source by `scripts/fetch_model.sh`) is a 1.94 G-parameter qwen35
hybrid: 18 Gated-DeltaNet blocks + 6 gated-attention blocks + 1
multi-token-prediction block, vocabulary 248 320. What the dissection does:

- **discards** the tokenizer runtime (BPE, merges, chat template), the
  sampling stack, the MTP block (`blk.24` + `nextn.*`), and 508.6 M
  parameters of vocabulary — 26 % of the model is never materialized. The
  tokenizer is reduced to four integers: `"0"` → 15, `"1"` → 16, stream prime
  `<|endoftext|>` → 248044, eos → 248046 (`llm::config::TokenProbe`);
- **keeps** the 24 hybrid blocks as a bare conditional-probability engine:
  the output head is literally two dot products (the tied-embedding rows for
  "0"/"1"), so P(bit | history) is a softmax over 2 logits — the LLM *is* a
  sequential binary predictor, exactly what MC-AIXI's ξ needs;
- weights stay **quantized in RAM** (Q4_K/Q6_K blocks, ~1.0 GiB resident);
  the GEMV kernels fuse dequantization into the dot product
  (`llm::quant::dot_row`), so the only f32 tensors are activations.

Per block (`llm::model`): pre-norm residual, then either
gated GQA attention (per-head-interleaved fused q|gate, QK-RMSNorm before
partial RoPE — 64 of 256 dims, θ = 10⁷, MRoPE sections collapse for text-only
streams — causal softmax over the KV arena, sigmoid output gate) or Gated
DeltaNet (depthwise causal conv-4 + SiLU over the fused qkv, per-head
L2-normalized q/k with ggml's norm-floor eps, query scaled 1/√d_k, the gated
delta rule **S ← γ S (I − β k kᵀ) + β v kᵀ**, o = S q, per-head RMSNorm gated
by SiLU(z)), followed by a SwiGLU FFN.

### Exact revert on a recurrent model

Attention state is append-only (K/V arenas): revert = truncate, O(1), exact.
The DeltaNet recurrence cannot truncate, so every advanced token first pushes
a checkpoint of all recurrent state (~19 MB on this model) onto a bounded
LIFO stack (`llm::state`, cap 64 ≈ 1.2 GiB); reverts pop and restore
bit-exactly. Past the cap — deeper than any ρUCT imagination — the state is
rebuilt by deterministic replay, so exactness is never traded away. Tokens are
advanced lazily and per-position 2-logit outputs are logged, making
predict-after-revert O(1).

### Validation against llama.cpp (dev-only oracle)

`scripts/oracle_check.sh` compares this crate's forward pass with llama.cpp
on identical **raw token-id streams** (no tokenizer on either side), in two
stages:

1. **Exact graph** — a synthetic all-F32 qwen35 hybrid written by our own
   GGUF writer and loaded by both engines: max logit difference observed
   9e-4 over 16 positions through both layer types and live recurrent state.
   The graph is the same computation.
2. **Real checkpoint (Q4_K_M)** — the engines evaluate the same quantized
   weights differently (ggml quantizes activations to Q8_1 for integer
   matmuls; we compute exact dequant·f32 dots), so small per-layer
   differences accumulate across 24 blocks: ≤ ~1.2 logits in the first three
   positions (before the conv-4 window fills), ≤ ~0.3 after. Layer-by-layer
   traces (`MC_AIXI_TRACE=1`, format-compatible with llama.cpp's
   eval-callback) agree to ~1e-2 everywhere except quantized GEMV outputs,
   which is where evaluator rounding lives.

## Correctness spine (all offline, `cargo test`)

- **CTW theorem from first principles**: `CtwModel`'s root log-probability
  equals an explicitly enumerated Bayes mixture over *all* depth ≤ D suffix
  trees under the 2^(−Γ) prior (depths 0–4, tol 1e-12), including FAC
  interleavings and zero-padded contexts; FAC-CTW equals the product of
  per-position brute-force factors (`tests/ctw_brute_force.rs`).
- **ρUCT vs exact expectimax**: on a trained ξ the search recovers the
  enumerated argmax and value (`tests/rho_uct_expectimax.rs`).
- **Bit-exact revert everywhere**: randomized learn/append/revert
  interleavings restore CTW node arenas, mixture weights, and the LLM's
  recurrent state to identical digests; out-of-LIFO-order reverts panic.
- **Bayes math by hand**: mixture posteriors and predictives against exact
  fractions; dominance ln ξ ≥ ln w⁰_ν + ln ν asserted against standalone
  component runs.
- **End-to-end learning**: CoinFlip(0.6) agent regression
  (`tests/agent_coin_flip.rs`); five PASS/FAIL invariants in
  `cargo run --release --bin smoke` (the `experiments/run_smoke.py` analog).

## Two findings the implementation surfaced

- **AC-CTW vs FAC-CTW is not a nicety.** A single shared context tree over
  the interleaved stream (pyaixi's construction, our `CtwModel`) suffers
  *positional aliasing*: leaves mix statistics of deterministic reward bits
  with stochastic observation bits, so CoinFlip's 2-bit percept needs depth 8+
  and thousands of cycles. The factored construction (JAIR §5.4, our
  `FacCtwModel` — one tree per percept bit position, built compositionally so
  all exactness proofs carry over) separates them and learns the same domain
  in ~100 cycles at depth 2. Measured: on 1 000 cycles, single-tree depth-8
  ≈ −707 nats (≈ the source entropy) but depth-2 ≈ −1132; factored depth-2
  nails both bits immediately.
- **Tree reuse under a receding horizon is value-inconsistent.** Advancing
  the root promotes every kept chance node one step closer to the agent, so
  its cached mean — an average of returns truncated at the *old* remaining
  horizon — understates its value at the new depth; at small horizons this
  inverts action rankings (observed as systematic anti-selection on
  CoinFlip). Agents therefore rebuild the tree each decision by default;
  `RhoUct::advance_root` remains available with the caveat documented.

## Running it

```
cargo test                                   # the full offline correctness spine
cargo run --release --bin smoke              # PASS/FAIL invariants, exit code

# classical MC-AIXI-CTW (fast path, µs–ms per decision):
cargo run --release --bin aixi -- --env coin_flip   --cycles 400
cargo run --release --bin aixi -- --env biased_rps  --cycles 2000
cargo run --release --bin aixi -- --env cheese_maze --cycles 12000 --horizon 6
cargo run --release --bin aixi -- --env tiger --ct-depths 8,16,24 --cycles 3000
cargo run --release --bin aixi -- --env kuhn_poker  --cycles 3000

# the dissection:
bash scripts/fetch_model.sh                  # 1.31 GB Q4_K_M, sha256-checked
cargo run --release --bin inspect_model -- --gguf models/Qwen3.8-2B-Q4_K_M.gguf
cargo run --release --bin aixi -- --env coin_flip --model full-mix \
    --ct-depths 4,8 --cycles 12 --mc-simulations 10 --horizon 2

# dev-only ground truth (needs a llama.cpp build; see script header):
bash scripts/oracle_check.sh
```

Measured on this 4-vCPU AVX-512 container (seed 42):

| run | result |
|---|---|
| CoinFlip(0.6), ctw-mix, 400 cycles | average reward **+0.605** (optimum 0.6), uniform's posterior → 0, 9 ms/cycle |
| Biased RPS, ctw-mix, 2 000 cycles | average **+0.196**, late windows ≈ +0.22–0.24 (random = 0); posterior selects depth 8 — the bias spans one full cycle of context, exactly what d4 cannot see |
| Cheese Maze, ctw-mix, 3 000 cycles, m=4 | window average −1.63 → −1.00 (wall-bumps eliminated), posterior → d16; JAIR-scale budgets (tens of thousands of cycles) apply to this domain |
| Tiger, ctw-mix(8,16,24), 3 000 cycles, m=3 | window average → −1.000 exactly: the agent learns to *never open on insufficient evidence* (no −100s after the first phase) and parks at the safe listening policy — the classic small-budget plateau; breaking out needs larger m and simulation counts |
| Kuhn Poker, ctw-mix, 4 000 cycles, m=3 | window average −0.182 → −0.079 and still climbing toward the +0.056 second-player Nash value; posterior selects d16, 31 ms/cycle |
| Qwen3.8-2B forward | ~0.4 s/token on 4 CPU cores (quantized-resident, scalar kernels; 805 MiB resident after the carve, 0.8 s load) |
| **CoinFlip, full-mix** (fac-d4 + fac-d8 + uniform + llm), 12 cycles, 10 sims, m=2 | average **+0.667**; ~31 s/cycle. The posterior trajectory is the point: the LLM starts at its ¼ prior, is measured against the bit stream, and is demoted to 0.02 while FAC-CTW rises to 0.78 — Bayes pricing a neural prior against compression priors online, with dominance bounding the cost of having tried it at ln 4 nats |

The per-cycle posterior trajectory (also written by `--csv`) is the point of
the exercise: **Bayes arbitrates between compression-era and neural priors
online**, with dominance guaranteeing the mixture never pays more than ln K
nats against the best component in hindsight.

## Deviations and limitations (documented, deliberate)

- Percept wire order is observation-bits-then-reward-bits, MSB-first;
  contexts are zero-padded (Willems' convention) so every bit has a
  full-depth context. The non-runnable Python reference imposed no binary
  convention; ours is pinned by the brute-force tests.
- ρUCT rebuilds its tree each decision (see finding above); JAIR's tree
  reuse is opt-in.
- The DeltaNet implementation requires symmetric k/v head counts (true for
  the whole released qwen3_5 dense family); the k-head repeat path of the
  general Gated DeltaNet is intentionally unimplemented and errors loudly.
- The LLM component conditions on at most the model's context window
  (262 144 tokens here — not a practical constraint for these domains);
  weights are frozen, so all *online* adaptation lives in the mixture
  weights, which is the theoretically honest place for it.
- `SearchBudget` validates every knob; forbidden API names
  (`run_unbounded`, `hypercompute`, …) do not exist in this crate
  (`supertask_surrogate.py` discipline).

## Layout

```
src/logspace.rs  src/encoding.rs  src/rng.rs      numeric + codec substrate
src/env/                                          JAIR §7 domains (5)
src/models/      kt, ctw (AC), fac_ctw (FAC), uniform, mixture — the ξ side
src/planning/    rho_uct + exact expectimax
src/agent.rs     act/perceive loop (§4.1)
src/llm/         gguf, quant, tensor, config, rope, state, model, env_model
src/bin/         aixi (CLI), smoke, inspect_model, oracle_probe (dev)
tests/           brute-force PST mixture, expectimax parity, contract suites
scripts/         fetch_model.sh, oracle_check.sh, oracle_probe.cpp (dev)
```
