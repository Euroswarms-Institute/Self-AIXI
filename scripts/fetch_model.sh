#!/usr/bin/env bash
# Fetch the base model for the MC-AIXI dissection: the user's own
# empero-ai/Qwen3.8-2B-Distill-GGUF (Apache-2.0), Q4_K_M quantization
# (the model card's recommended file). ~1.31 GB into models/ (gitignored).
#
#   bash scripts/fetch_model.sh          # Q4_K_M (default)
#   bash scripts/fetch_model.sh Q8_0     # any other published quant
set -euo pipefail

QUANT="${1:-Q4_K_M}"
REPO="empero-ai/Qwen3.8-2B-Distill-GGUF"
FILE="Qwen3.8-2B-${QUANT}.gguf"
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DIR"

if [ -f "$DIR/$FILE" ]; then
    echo "already present: $DIR/$FILE"
else
    echo "downloading $REPO/$FILE ..."
    curl -L --fail --retry 3 -C - \
        -o "$DIR/$FILE.part" \
        "https://huggingface.co/$REPO/resolve/main/$FILE"
    mv "$DIR/$FILE.part" "$DIR/$FILE"
fi

echo "verifying sha256 against the repo's SHA256SUMS ..."
curl -sL --fail "https://huggingface.co/$REPO/resolve/main/SHA256SUMS" -o "$DIR/SHA256SUMS"
EXPECT="$(grep "$FILE" "$DIR/SHA256SUMS" | awk '{print $1}')"
if [ -z "$EXPECT" ]; then
    echo "WARN: $FILE not listed in SHA256SUMS; skipping verification"
else
    ACTUAL="$(sha256sum "$DIR/$FILE" | awk '{print $1}')"
    if [ "$EXPECT" != "$ACTUAL" ]; then
        echo "FAIL: sha256 mismatch: expected $EXPECT got $ACTUAL"
        exit 1
    fi
    echo "sha256 OK"
fi
echo "ready: $DIR/$FILE"
