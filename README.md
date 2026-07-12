# HiT-Forecast

**Hierarchical Transformer Routing over frozen time-series foundation models**, with a
**contamination-aware GIFT-Eval protocol**.

HiT-Forecast routes each forecasting *window* to the best expert in a pool of frozen
time-series foundation models (TSFMs), operating on **patch-level encoder
representations** rather than pooled summaries. A two-stage transformer router (per-expert
temporal encoder + cross-expert fusion, with an optional cross-attention bridge) is trained
with a **MASE-aware composite objective** (weighted-MASE + hard CE + temperature soft CE).

This repository reworks the original draft's experiments to fix a concrete failure mode
reported during earlier attempts: **the previous expert pool was contaminated** (3 of 4
foundation models are tagged `testdata_leakage = Yes` on GIFT-Eval), which collapses the
per-window MASE margins the router needs and inflates absolute metrics. We therefore:

1. move training + evaluation onto **[GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval)** (official, leakage-controlled splits);
2. swap in a **clean, diverse expert pool** (all `testdata_leakage = No`);
3. add **Phase-0 diagnostics** that gate training on whether a learnable routing signal actually exists;
4. add a **contamination contrast** experiment that measures the effect directly.

`GiftEvalPretrain` (~975 GB) is only needed for the optional "own-the-experts" path (Exp 7);
it is **not** required for the main results.

---

## Why the previous attempt failed (and how this fixes it)

| Failure | Mechanism | Fix in this repo |
|---|---|---|
| "Model can't learn enough signal" | Contaminated / redundant experts → per-window MASE gaps collapse → weak gradients | Clean, diverse pool + Phase-0 margin gate (`hitf-diagnose`) |
| Features don't discriminate | Encoders already "know" the eval corpora | GIFT-Eval leakage-controlled splits; contrast experiment |
| Inflated absolute MASE | Test overlaps pretraining | Report on GIFT-Eval held-out horizons |

Official GIFT-Eval tags for the original vs the new pool:

| Original pool | leakage | | New clean pool | leakage |
|---|---|---|---|---|
| Chronos-Small | **Yes** | | Chronos-2 | No |
| Moirai-Base | No | | Moirai-Base | No |
| TimesFM 1.0 | **Yes** | | TimesFM-2.5 | No |
| Lag-Llama | **Yes** | | TiRex | No |

---

## Architecture

```
window x_n
  └─ for each frozen expert E_k:  H^(k) ∈ R^{T_k × D_k}   (patch-level hidden states; NOT pooled)
       └─ per-model Linear(D_k→d) + sinusoidal PE + model-identity embed e_k + [CLS]_k
            └─ Stage-1 shared transformer → per-expert summary  c̃_k
                 └─ (optional) cross-attention bridge: c̃_k attends to other experts' patches
                      └─ Stage-2 fusion over [c*, c̃_1..c̃_K] → MLP → softmax → w_n
                           └─ route: argmax w_n (hard) / Σ w_n·A_k (soft) / regression combiner
```

Implemented in `src/hit_forecast/models/router.py`. Losses in `models/losses.py`.

## Design: cache → train → evaluate

Feature extraction (the only GPU-heavy stage) is decoupled from router training:

1. **`hitf-cache`** runs each frozen expert once per window and stores forecasts, patch
   features `H^(k)`, and per-window MASE (draft Algorithm 2). Written per config under
   `feature_cache/`.
2. **`hitf-train`** / **`run_all`** train the router purely on cached tensors — fast, fits an
   L40S trivially, no TSFM in the loop.
3. **`hitf-eval`** applies the router to cached test features and reports MASE / sMAPE / MSE
   against every baseline + the per-window oracle, aggregated GIFT-Eval-style.

## Install

```bash
bash scripts/setup_env.sh                 # venv + core + gifteval + dev
source .venv/bin/activate
pip install -e '.[experts]'               # foundation-model adapters (GPU host)
bash scripts/download_gifteval.sh          # ~1.6 GB → data/gifteval, writes .env
pip install git+https://github.com/SalesforceAIResearch/gift-eval.git
```

## Run the main results (single L40S)

```bash
bash scripts/run_l40s.sh
```

Or step by step:

```bash
# 0) sanity (no downloads): synthetic regime-switch, draft Table I
hitf-synthetic --device cuda --out results/synthetic_regime_switch

# 1) cache clean-pool features on GIFT-Eval
hitf-cache --config configs/experiments/gifteval_main_clean.yaml --device cuda

# 2) Phase-0 go/no-go diagnostics
hitf-diagnose feature_cache/gifteval_main_clean/*::train* \
  --out results/gifteval_main_clean/diagnostics.json

# 3) full main table + ablationable runner
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_main_clean.yaml --device cuda

# 4) contamination contrast
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_contaminated.yaml --device cuda
```

Outputs per experiment (`results/<name>/`): `metrics.csv`, `metrics.json`,
`diagnostics.json`, `aggregates.json` (by domain/freq/term), `history.json`, and router
checkpoints `hit_router.pt` / `pooled_router.pt`.

## Experiments

| # | Config / command | Purpose |
|---|---|---|
| 0 | `hitf-synthetic` | Isolates patch-level vs pooled routing (draft Table I). Runs anywhere. |
| 1 | `gifteval_pilot.yaml` | Fast GPU-path validation on 3 domains. |
| 2 | `gifteval_main_clean.yaml` | **Main table**: HiT vs experts / ensembles / pooled-MLP / oracle. |
| 3 | `gifteval_contaminated.yaml` | Same router, old leaking pool → quantifies contamination. |
| 4–6 | override `router.*` / `train.loss.*` | Architecture & loss ablations, operating modes, pool scaling. |
| 7 | GiftEvalPretrain (optional) | Re-pretrain/own the experts for a strict zero-shot claim. |

Ablation example (draft Tables III–IV), no re-caching required:

```bash
python -m hit_forecast.cli.run_all --config configs/experiments/gifteval_main_clean.yaml \
  train.loss.lambda_hard=0.0 train.loss.lambda_soft=0.0 router.cross_attention=false
```

## Expert adapters

`src/hit_forecast/experts/` provides a small `ExpertAdapter` interface returning, per window,
a point forecast and a patch-level hidden-state sequence. Concrete adapters: `chronos`
(Chronos / Chronos-Bolt / Chronos-2), `moirai`, `timesfm`, `tirex`, plus dependency-free
`dummy_*` experts used by the synthetic experiment and CI.

Patch features come from the frozen encoder where the library exposes it
(`feature_source: auto|encoder`); otherwise a deterministic per-patch statistic fallback
(`stat`) keeps the pipeline runnable. The patch-vs-pooled scientific comparison holds under
either source, since the pooled-MLP baseline pools the *same* features.

> On the L40S, verify each expert's `encoder` feature path with a small
> `hitf-cache ... --splits test` smoke run before the full table if you want encoder-native
> patch states rather than the stat fallback for a given library version.

## Repository layout

```
src/hit_forecast/
  data/       metrics (MASE/sMAPE/MSE), windowing, synthetic, GIFT-Eval loader
  experts/    adapter interface + clean pool (chronos/moirai/timesfm/tirex) + dummy
  features/   forecast/patch/MASE caching
  models/     hierarchical router, pooled-MLP baseline, composite loss, torch dataset
  train/      trainer, Phase-0 diagnostics
  eval/       baselines, metrics, GIFT-Eval-style aggregation
  cli/        cache / diagnose / train / eval / run_all / synthetic
configs/experiments/   synthetic, pilot, main_clean, contaminated
scripts/               setup_env, download_gifteval, run_l40s
tests/                 metrics, router shapes, end-to-end synthetic pipeline
```

## Citation

If you use GIFT-Eval, cite Aksu et al., 2024 (arXiv:2410.10393).
