#!/usr/bin/env bash
# Paper-ready GiftEval package after Chronos-2 / TimesFM adapter fixes.
#
# Usage (GPU host):
#   git pull && source .venv/bin/activate && source .env && pip install -e . -q
#   bash scripts/run_paper.sh              # foreground
#   bash scripts/run_paper.sh --tmux       # detached
#   bash scripts/run_paper.sh --export-only  # LaTeX from existing results/
#
# Wipes clean-pool feature caches (Chronos/TimesFM were wrong), reuses contaminated
# caches, then: main_clean → Phase-0 both → 2 ablations → seeds 0/1/2 → export.
set -euo pipefail

DEVICE=${DEVICE:-cuda}
SESSION=${TMUX_SESSION:-hitf-paper}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/run_paper_${STAMP}.log"
INNER="${LOG_DIR}/run_paper_${STAMP}.inner.sh"

USE_TMUX=0
EXPORT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --tmux|-t) USE_TMUX=1 ;;
    --export-only) EXPORT_ONLY=1 ;;
    --help|-h)
      echo "Usage: $0 [--tmux] [--export-only]"
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
export DEVICE="$DEVICE"
echo "Using GIFT_EVAL=\$GIFT_EVAL DEVICE=\$DEVICE"
echo "Log: $LOG"

if [[ "$EXPORT_ONLY" -eq 1 ]]; then
  echo "== Export paper tables only =="
  hitf-export-tables --out results/paper
  echo "Done. See results/paper/"
  exit 0
fi

echo "== [0] Wipe caches: new diverse pool + expert-specific features require re-cache =="
rm -rf feature_cache/gifteval_main_clean feature_cache/gifteval_pilot feature_cache/gifteval_contaminated
# Both pools re-cache so clean vs contaminated use the identical feature pipeline
# (encoder/rich-stat patches + forecast-conditioned tokens).

echo "== [1] Synthetic sanity =="
hitf-synthetic --device "\$DEVICE" --out results/synthetic_regime_switch

echo "== [2] Clean main table (cache + train + eval, seed=0) =="
python -m hit_forecast.cli.run_all \\
  --config configs/experiments/gifteval_main_clean.yaml \\
  --device "\$DEVICE" \\
  --out results/gifteval_main_clean/seed_0 \\
  seed=0
# Convenience copies for exporters that look at results/gifteval_main_clean/
mkdir -p results/gifteval_main_clean
cp -f results/gifteval_main_clean/seed_0/metrics.csv results/gifteval_main_clean/metrics.csv
cp -f results/gifteval_main_clean/seed_0/diagnostics.json results/gifteval_main_clean/diagnostics.json
cp -f results/gifteval_main_clean/seed_0/aggregates.json results/gifteval_main_clean/aggregates.json 2>/dev/null || true
cp -f results/gifteval_main_clean/seed_0/history.json results/gifteval_main_clean/history.json 2>/dev/null || true

echo "== [3] Contaminated contrast (re-cache with identical feature pipeline) =="
python -m hit_forecast.cli.run_all \\
  --config configs/experiments/gifteval_contaminated.yaml \\
  --device "\$DEVICE" \\
  --out results/gifteval_contaminated

echo "== [4] Phase-0 diagnose clean + contaminated =="
mapfile -t CLEAN_TRAIN < <(find feature_cache/gifteval_main_clean -maxdepth 1 -type d \\( -name '*_train' -o -name '*::train*' \\) 2>/dev/null || true)
if ((\${#CLEAN_TRAIN[@]})); then
  hitf-diagnose "\${CLEAN_TRAIN[@]}" --out results/gifteval_main_clean/diagnostics.json
fi
mapfile -t CONT_TRAIN < <(find feature_cache/gifteval_contaminated -maxdepth 1 -type d \\( -name '*_train' -o -name '*::train*' \\) 2>/dev/null || true)
if ((\${#CONT_TRAIN[@]})); then
  mkdir -p results/gifteval_contaminated
  hitf-diagnose "\${CONT_TRAIN[@]}" --out results/gifteval_contaminated/diagnostics.json
else
  echo "No contaminated train caches; skipping contaminated diagnose."
fi

echo "== [5] Ablation: MASE-only loss =="
python -m hit_forecast.cli.run_all \\
  --config configs/experiments/gifteval_main_clean.yaml \\
  --device "\$DEVICE" \\
  --out results/ablation_loss_mase \\
  train.loss.lambda_hard=0.0 train.loss.lambda_soft=0.0 seed=0

echo "== [6] Ablation: no cross-attention =="
python -m hit_forecast.cli.run_all \\
  --config configs/experiments/gifteval_main_clean.yaml \\
  --device "\$DEVICE" \\
  --out results/ablation_no_ca \\
  router.cross_attention=false seed=0

echo "== [7] Extra seeds 1 and 2 (reuse clean feature cache) =="
for S in 1 2; do
  python -m hit_forecast.cli.run_all \\
    --config configs/experiments/gifteval_main_clean.yaml \\
    --device "\$DEVICE" \\
    --out "results/gifteval_main_clean/seed_\${S}" \\
    "seed=\${S}"
done

echo "== [8] Export LaTeX =="
hitf-export-tables \\
  --clean results/gifteval_main_clean \\
  --contaminated results/gifteval_contaminated \\
  --ablation-loss results/ablation_loss_mase \\
  --ablation-arch results/ablation_no_ca \\
  --out results/paper

echo "Done. Paper artifacts in results/paper/"
ls -la results/paper/
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
    echo "tmux session '$SESSION' already exists. Kill it or attach: tmux attach -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "bash '$INNER' 2>&1 | tee '$LOG'; ec=\$?; echo; echo \"[hitf-paper] exit=\$ec  log=$LOG\"; exec bash"
  sleep 1
  echo "Started detached tmux session: $SESSION"
  echo "  attach:   tmux attach -t $SESSION"
  echo "  follow:   tail -f $LOG"
  exit 0
fi

bash "$INNER" 2>&1 | tee "$LOG"
