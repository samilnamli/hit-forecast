#!/usr/bin/env bash
# Full main-results pipeline on a single GPU host (L40S / A100 / H100).
# Assumes setup_env.sh and download_gifteval.sh have been run and the experts
# extra is installed.
#
# Usage:
#   bash scripts/run_l40s.sh              # foreground
#   bash scripts/run_l40s.sh --tmux       # detachable tmux session (recommended)
#   DEVICE=cuda bash scripts/run_l40s.sh --tmux
set -euo pipefail

DEVICE=${DEVICE:-cuda}
SESSION=${TMUX_SESSION:-hitf}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/results/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_l40s_$(date +%Y%m%d_%H%M%S).log"

USE_TMUX=0
for arg in "$@"; do
  case "$arg" in
    --tmux|-t) USE_TMUX=1 ;;
    --help|-h)
      echo "Usage: $0 [--tmux]"
      echo "  --tmux   run inside a detached tmux session named '${SESSION}'"
      exit 0
      ;;
  esac
done

run_pipeline() {
  set -euo pipefail
  cd "$ROOT"
  # Prefer the project venv if present.
  if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
  fi

  echo "== [0/5] Synthetic sanity (no downloads) =="
  hitf-synthetic --device "$DEVICE" --out results/synthetic_regime_switch

  echo "== [1/5] Cache clean-pool features on GIFT-Eval =="
  hitf-cache --config configs/experiments/gifteval_main_clean.yaml --device "$DEVICE"

  echo "== [2/5] Phase-0 diagnostics (go/no-go) =="
  # Expand shards; do not fail the whole run if the gate warns.
  mapfile -t TRAIN_CACHES < <(find feature_cache/gifteval_main_clean -maxdepth 1 -type d -name '*::train*' 2>/dev/null || true)
  if ((${#TRAIN_CACHES[@]})); then
    hitf-diagnose "${TRAIN_CACHES[@]}" \
      --out results/gifteval_main_clean/diagnostics.json || true
  else
    echo "No train caches found yet under feature_cache/gifteval_main_clean; skipping diagnose."
  fi

  echo "== [3/5] Pilot (fast validation of the GPU path) =="
  python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_pilot.yaml --device "$DEVICE"

  echo "== [4/5] Full main table (clean pool) =="
  python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_main_clean.yaml --device "$DEVICE"

  echo "== [5/5] Contamination contrast (old leaking pool) =="
  python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_contaminated.yaml --device "$DEVICE"

  echo "Done. See results/*/metrics.csv and results/*/diagnostics.json"
}

if [[ "$USE_TMUX" -eq 1 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Installing (apt) ..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
    else
      echo "ERROR: tmux is not installed and apt-get is unavailable. Install tmux and re-run."
      exit 1
    fi
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "  attach:  tmux attach -t $SESSION"
    echo "  kill:    tmux kill-session -t $SESSION"
    exit 1
  fi
  # Export DEVICE into the session environment, then run the pipeline with logging.
  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && DEVICE='$DEVICE' bash -lc '$(declare -f run_pipeline); run_pipeline' 2>&1 | tee '$LOG'; echo; echo \"[hitf] finished. log: $LOG\"; exec bash"
  echo "Started detached tmux session: $SESSION"
  echo "  attach:   tmux attach -t $SESSION"
  echo "  detach:   Ctrl-b then d"
  echo "  log file: $LOG"
  exit 0
fi

run_pipeline 2>&1 | tee "$LOG"
