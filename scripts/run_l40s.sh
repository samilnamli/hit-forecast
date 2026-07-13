#!/usr/bin/env bash
# Full main-results pipeline on a single GPU host (L40S / A100 / H100).
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
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/run_l40s_${STAMP}.log"
INNER="${LOG_DIR}/run_l40s_${STAMP}.inner.sh"

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

write_inner() {
  cat > "$INNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
if [[ -z "\${GIFT_EVAL:-}" && -d "$ROOT/data/gifteval" ]]; then
  export GIFT_EVAL="$ROOT/data/gifteval"
fi
if [[ -z "\${GIFT_EVAL:-}" ]]; then
  echo "ERROR: GIFT_EVAL is unset. Run: bash scripts/download_gifteval.sh"
  exit 1
fi
# Fail fast if a config path is missing (common naming mismatch).
if [[ ! -d "\${GIFT_EVAL}/electricity/H" ]]; then
  echo "ERROR: \${GIFT_EVAL}/electricity/H not found."
  echo "Listing \${GIFT_EVAL}/electricity :"
  ls -la "\${GIFT_EVAL}/electricity" || true
  exit 1
fi
export DEVICE="$DEVICE"
echo "Using GIFT_EVAL=\$GIFT_EVAL DEVICE=\$DEVICE"
echo "Log: $LOG"

echo "== [0/5] Synthetic sanity (no downloads) =="
hitf-synthetic --device "\$DEVICE" --out results/synthetic_regime_switch

echo "== [1/5] Cache clean-pool features on GIFT-Eval =="
hitf-cache --config configs/experiments/gifteval_main_clean.yaml --device "\$DEVICE"

echo "== [2/5] Phase-0 diagnostics (go/no-go) =="
mapfile -t TRAIN_CACHES < <(find feature_cache/gifteval_main_clean -maxdepth 1 -type d -name '*::train*' 2>/dev/null || true)
if ((\${#TRAIN_CACHES[@]})); then
  hitf-diagnose "\${TRAIN_CACHES[@]}" --out results/gifteval_main_clean/diagnostics.json || true
else
  echo "No train caches found yet; skipping diagnose."
fi

echo "== [3/5] Pilot (fast validation of the GPU path) =="
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_pilot.yaml --device "\$DEVICE"

echo "== [4/5] Full main table (clean pool) =="
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_main_clean.yaml --device "\$DEVICE"

echo "== [5/5] Contamination contrast (old leaking pool) =="
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_contaminated.yaml --device "\$DEVICE"

echo "Done. See results/*/metrics.csv and results/*/diagnostics.json"
EOF
  chmod +x "$INNER"
}

write_inner

if [[ "$USE_TMUX" -eq 1 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Installing (apt) ..."
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "  status:  tmux ls"
    echo "  attach:  tmux attach -t $SESSION"
    echo "  log:     ls -lt $LOG_DIR | head"
    echo "  kill:    tmux kill-session -t $SESSION"
    exit 1
  fi
  # Keep the pane open even if the pipeline fails, so attach always shows something.
  tmux new-session -d -s "$SESSION" \
    "bash '$INNER' 2>&1 | tee '$LOG'; ec=\$?; echo; echo \"[hitf] exit=\$ec  log=$LOG\"; exec bash"
  sleep 1
  echo "Started detached tmux session: $SESSION"
  echo "  list:     tmux ls"
  echo "  attach:   tmux attach -t $SESSION"
  echo "  detach:   Ctrl-b then d   (do NOT Ctrl-c unless you want to stop the job)"
  echo "  follow:   tail -f $LOG"
  echo "  log file: $LOG"
  # Show the first lines so you can confirm it started without attaching.
  sleep 2
  echo "----- log head -----"
  head -n 30 "$LOG" 2>/dev/null || echo "(log not ready yet)"
  exit 0
fi

bash "$INNER" 2>&1 | tee "$LOG"
