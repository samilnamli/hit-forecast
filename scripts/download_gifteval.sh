#!/usr/bin/env bash
# Download the GIFT-Eval train/test datasets (~1.6 GB) and set GIFT_EVAL.
# GiftEvalPretrain (~975 GB) is NOT downloaded here; only needed for Exp 7.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST=${1:-"$ROOT/data/gifteval"}
mkdir -p "$DEST"

pip install -q "huggingface_hub>=0.24"

# Newer huggingface_hub dropped `huggingface-cli`; use `hf download`.
if command -v hf >/dev/null 2>&1; then
  hf download Salesforce/GiftEval --repo-type dataset --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir "$DEST"
else
  python - <<'PY' "$DEST"
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Salesforce/GiftEval",
    repo_type="dataset",
    local_dir=sys.argv[1],
)
print("Downloaded to", sys.argv[1])
PY
fi

# Sanity-check: refuse to write a "success" .env for an empty download.
if [[ -z "$(ls -A "$DEST" 2>/dev/null || true)" ]]; then
  echo "ERROR: $DEST is still empty after download. Check HF auth / network and retry."
  exit 1
fi

ABS="$(cd "$DEST" && pwd)"
echo "GIFT_EVAL=$ABS" > "$ROOT/.env"
export GIFT_EVAL="$ABS"
echo "Wrote GIFT_EVAL=$ABS to $ROOT/.env"
echo "Contents (first 20 entries):"
ls "$ABS" | head -20
echo
echo "In this shell:  source $ROOT/.env"
echo "If gift_eval is not installed yet:"
echo "  pip install git+https://github.com/SalesforceAIResearch/gift-eval.git"
