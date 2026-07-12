#!/usr/bin/env bash
# Download the GIFT-Eval train/test datasets (~1.6 GB) and set GIFT_EVAL.
# GiftEvalPretrain (~975 GB) is NOT downloaded here; only needed for Exp 7.
set -euo pipefail

DEST=${1:-$PWD/data/gifteval}
mkdir -p "$DEST"

pip install -q "huggingface_hub[cli]"
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir "$DEST"

echo "GIFT_EVAL=$DEST" > .env
echo "Wrote GIFT_EVAL=$DEST to .env"
echo "Also install the GIFT-Eval package for the data loader:"
echo "  pip install git+https://github.com/SalesforceAIResearch/gift-eval.git"
