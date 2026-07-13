#!/usr/bin/env bash
# Download the GIFT-Eval train/test datasets (~1.6 GB) and set GIFT_EVAL.
# GiftEvalPretrain (~975 GB) is NOT downloaded here; only needed for Exp 7.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=${1:-"$ROOT/data/gifteval"}
mkdir -p "$DEST"

pip install -q "huggingface_hub[cli]"
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir "$DEST"

# Absolute path so later shells / tmux sessions resolve it reliably.
ABS="$(cd "$DEST" && pwd)"
echo "GIFT_EVAL=$ABS" > "$ROOT/.env"
export GIFT_EVAL="$ABS"
echo "Wrote GIFT_EVAL=$ABS to $ROOT/.env"
echo "In this shell: export GIFT_EVAL=$ABS"
echo "Also install the GIFT-Eval package for the data loader (if not already):"
echo "  pip install git+https://github.com/SalesforceAIResearch/gift-eval.git"
