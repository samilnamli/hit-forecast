#!/usr/bin/env bash
# Full main-results pipeline on a single L40S (48 GB). Assumes setup_env.sh and
# download_gifteval.sh have been run and the experts extra is installed.
set -euo pipefail

DEVICE=${DEVICE:-cuda}

echo "== [0/5] Synthetic sanity (no downloads) =="
hitf-synthetic --device "$DEVICE" --out results/synthetic_regime_switch

echo "== [1/5] Cache clean-pool features on GIFT-Eval =="
hitf-cache --config configs/experiments/gifteval_main_clean.yaml --device "$DEVICE"

echo "== [2/5] Phase-0 diagnostics (go/no-go) =="
hitf-diagnose feature_cache/gifteval_main_clean/*::train* \
  --out results/gifteval_main_clean/diagnostics.json || true

echo "== [3/5] Pilot (fast validation of the GPU path) =="
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_pilot.yaml --device "$DEVICE"

echo "== [4/5] Full main table (clean pool) =="
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_main_clean.yaml --device "$DEVICE"

echo "== [5/5] Contamination contrast (old leaking pool) =="
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_contaminated.yaml --device "$DEVICE"

echo "Done. See results/*/metrics.csv and results/*/diagnostics.json"
