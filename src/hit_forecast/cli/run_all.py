"""End-to-end experiment driver: cache -> diagnose -> train -> evaluate -> report.

Reads a single experiment YAML (see ``configs/experiments/``). Produces the main
results table (HiT-Forecast vs experts, ensembles, pooled-MLP, oracle) plus
Phase-0 diagnostics and GIFT-Eval-style aggregations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..models import HierarchicalRouter, PooledMLPRouter, RouterConfig, combine_caches
from ..train import TrainConfig, signal_diagnostics, train_router
from ..eval import aggregate_by, evaluate_all
from ..eval.baselines import all_baseline_forecasts, model_routing
from ..utils import get_logger, load_config, merge_overrides, seed_everything
from ._common import build_experts, cache_entries, split_train_val

_log = get_logger("run_all")


def run_experiment(cfg: dict, device: str, out_dir: Path, overwrite: bool = False) -> dict:
    seed_everything(cfg.get("seed", 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cfg.get("cache_root", out_dir / "feature_cache"))
    batch_size = cfg.get("cache_batch_size", 64)

    experts = build_experts(cfg, device)
    expert_names = [e.name for e in experts]
    _log.info("Expert pool: %s", expert_names)

    train_caches = cache_entries(cfg["datasets"]["train"], experts, cache_root, device,
                                 batch_size, overwrite)
    test_caches = cache_entries(cfg["datasets"]["test"], experts, cache_root, device,
                                batch_size, overwrite)

    train_all = combine_caches(train_caches)
    diag = signal_diagnostics(train_all.mase, expert_names)
    _log.info("Phase-0 diagnostics: gate_pass=%s rel_margin_median=%.4f entropy=%.3f",
              diag["gate_pass"], diag["rel_margin_median"], diag["win_rate_entropy_norm"])
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    tr_idx, val_idx = split_train_val(train_all.N, cfg.get("val_frac", 0.1),
                                      cfg.get("seed", 0))
    train_data = _subset(train_all, tr_idx)
    val_data = _subset(train_all, val_idx)
    train_mase_mean = train_data.mase.mean(axis=0)

    rcfg = RouterConfig(**cfg.get("router", {}))
    tcfg = TrainConfig(**cfg.get("train", {}))

    _log.info("Training HiT-Forecast router ...")
    hit = HierarchicalRouter(train_all.hidden_dims, rcfg)
    _log.info("HiT trainable params: %.2fM", hit.num_trainable_params() / 1e6)
    hit_out = train_router(hit, train_data, val_data, tcfg, device)
    torch.save(hit.state_dict(), out_dir / "hit_router.pt")

    _log.info("Training pooled-MLP baseline ...")
    pooled = PooledMLPRouter(train_all.hidden_dims, d=rcfg.d, dropout=rcfg.dropout)
    pooled_out = train_router(pooled, train_data, val_data, tcfg, device)
    torch.save(pooled.state_dict(), out_dir / "pooled_router.pt")

    models = {"hit_forecast": hit, "pooled_mlp": pooled}

    results = {}
    aggregates = {}
    for cache in test_caches:
        data = combine_caches([cache])
        res = evaluate_all(data, models=models, train_mase_mean=train_mase_mean,
                           device=device, seed=cfg.get("seed", 0))
        results[cache.meta["name"]] = res
        # aggregate HiT (hard) by domain/freq/term
        w, jn = model_routing(hit, data, device)
        hit_fc = np.take_along_axis(data.forecasts, jn[:, None, None], axis=1)[:, 0, :]
        aggregates[cache.meta["name"]] = {
            dim: aggregate_by(hit_fc, data, dim=dim, metric="MASE")
            for dim in ("domain", "freq", "term")
        }

    _write_results(out_dir, results, aggregates, diag, hit_out, pooled_out, expert_names)
    _log.info("Done. Results in %s", out_dir)
    return {"results": results, "diagnostics": diag}


def _subset(data, idx):
    from ..models.dataset import CombinedData

    idx = np.asarray(idx)
    return CombinedData(
        feats=[f[idx] for f in data.feats],
        masks=[m[idx] for m in data.masks],
        mase=data.mase[idx],
        expert_names=data.expert_names,
        hidden_dims=data.hidden_dims,
        patch_counts=data.patch_counts,
        forecasts=data.forecasts[idx] if data.forecasts.size else data.forecasts,
        targets=data.targets[idx] if data.targets is not None else None,
        contexts=data.contexts[idx] if data.contexts is not None else None,
        m=data.m[idx] if data.m is not None else None,
        window_meta=[data.window_meta[i] for i in idx] if data.window_meta else [],
    )


def _write_results(out_dir, results, aggregates, diag, hit_out, pooled_out, expert_names):
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2))
    (out_dir / "history.json").write_text(json.dumps(
        {"hit": hit_out["history"], "pooled": pooled_out["history"],
         "hit_best_val_expected_mase": hit_out["best_val_expected_mase"],
         "pooled_best_val_expected_mase": pooled_out["best_val_expected_mase"]}, indent=2))

    rows = []
    for ds_name, res in results.items():
        for method, metrics in res.items():
            row = {"dataset": ds_name, "method": method}
            row.update({k: round(v, 5) for k, v in metrics.items()})
            rows.append(row)
    if rows:
        keys = ["dataset", "method"] + sorted({k for r in rows for k in r
                                               if k not in ("dataset", "method")})
        with open(out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HiT-Forecast end-to-end experiment runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)
    out = Path(args.out or f"results/{cfg.get('name', 'experiment')}")
    run_experiment(cfg, args.device, out, args.overwrite)


if __name__ == "__main__":
    main()
