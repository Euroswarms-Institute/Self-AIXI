#!/usr/bin/env bash
# Dev-only ground-truth check of the dissected qwen35 forward pass against
# llama.cpp (a TEST ORACLE only — never a dependency of the crate), in two
# stages:
#
#  1. EXACT GRAPH: a synthetic all-F32 hybrid checkpoint (written by our own
#     GGUF writer, loadable by both engines). No quantization anywhere, so
#     any real graph difference would show; tolerance is float-accumulation
#     noise (1e-2 on logits).
#  2. REAL CHECKPOINT (Q4_K_M): the two engines evaluate the same quantized
#     weights differently — ggml quantizes activations to Q8_1 for integer
#     matmuls, we compute exact dequant·f32 dots — so small per-layer
#     differences accumulate across 24 blocks. Tolerances bound that
#     evaluator noise (~1 logit early, before the conv window fills), not
#     graph correctness, which stage 1 pins.
#
# Prereqs: llama.cpp with qwen35 support built at $LLAMA_CPP_DIR:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
#   cmake --build build --target llama -j4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$ROOT/models/Qwen3.8-2B-Q4_K_M.gguf"
LCPP="${LLAMA_CPP_DIR:-/home/user/ggml-org/llama.cpp}"
OUT="$ROOT/oracle"
SYNTH="${TMPDIR:-/tmp}/mc_aixi_llm_synth/tiny-1.gguf"
mkdir -p "$OUT"

[ -f "$LCPP/build/bin/libllama.so" ] || { echo "libllama.so missing — build llama.cpp first"; exit 1; }

echo "== building probes =="
g++ -O2 -std=c++17 "$ROOT/scripts/oracle_probe.cpp" \
    -I"$LCPP/include" -I"$LCPP/ggml/include" \
    -L"$LCPP/build/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LCPP/build/bin" \
    -o "$OUT/oracle_probe_cpp"
cargo build --release --bin oracle_probe --manifest-path "$ROOT/Cargo.toml" 2>/dev/null

compare() { # ref got gap_tol abs_tol label
    python3 - "$1" "$2" "$3" "$4" "$5" <<'EOF'
import sys
ref = [tuple(map(float, l.split()[1:])) for l in open(sys.argv[1])]
got = [tuple(map(float, l.split()[1:])) for l in open(sys.argv[2])]
gap_tol, abs_tol, label = float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
assert len(ref) == len(got) and ref, f"length mismatch {len(ref)} vs {len(got)}"
worst_gap = max(abs((r1-r0)-(g1-g0)) for (r0,r1),(g0,g1) in zip(ref,got))
worst_abs = max(max(abs(r0-g0),abs(r1-g1)) for (r0,r1),(g0,g1) in zip(ref,got))
ok = worst_gap < gap_tol and worst_abs < abs_tol
print(f"{'PASS' if ok else 'FAIL'} {label}: max|Δ(l1−l0)|={worst_gap:.4f} (tol {gap_tol}) max|Δlogit|={worst_abs:.4f} (tol {abs_tol})")
sys.exit(0 if ok else 1)
EOF
}

status=0

echo "== stage 1: exact graph on synthetic F32 hybrid =="
if [ ! -f "$SYNTH" ]; then
    cargo test --release --manifest-path "$ROOT/Cargo.toml" \
        --test llm_synthetic revert_and_readvance_is_bit_exact >/dev/null 2>&1
fi
SEQ="5,2,3,2,2,3,3,2,3,2,2,2,3,3,2,3"
"$OUT/oracle_probe_cpp" "$SYNTH" "$SEQ" 2 3 >"$OUT/synth_llama.txt" 2>"$OUT/synth_llama.log"
"$ROOT/target/release/oracle_probe" --gguf "$SYNTH" --tokens "$SEQ" >"$OUT/synth_ours.txt"
compare "$OUT/synth_llama.txt" "$OUT/synth_ours.txt" 0.01 0.01 "exact-graph(F32)" || status=1

echo "== stage 2: real checkpoint (quantization-evaluator noise bound) =="
if [ -f "$MODEL" ]; then
    STREAMS=(
        "248044,15,16,15,15,16,16,15,16,15,16,16,16,15,15,16"
        "248044,16,16,16,16,16,16,16,16,15,15,15,15,15,15,15"
        "248044,15,16,15,16,15,16,15,16,15,16,15,16,15,16,15"
    )
    for i in "${!STREAMS[@]}"; do
        s="${STREAMS[$i]}"
        "$OUT/oracle_probe_cpp" "$MODEL" "$s" >"$OUT/llama_$i.txt" 2>"$OUT/llama_$i.log"
        "$ROOT/target/release/oracle_probe" --gguf "$MODEL" --tokens "$s" >"$OUT/ours_$i.txt"
        compare "$OUT/llama_$i.txt" "$OUT/ours_$i.txt" 1.5 2.0 "real-Q4_K_M[$i]" || status=1
    done
else
    echo "SKIP real-checkpoint stage (run scripts/fetch_model.sh first)"
fi

[ $status -eq 0 ] && echo "ORACLE PARITY OK" || echo "ORACLE PARITY FAILED"
exit $status
