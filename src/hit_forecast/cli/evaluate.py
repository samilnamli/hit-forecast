"""Evaluate trained routers + baselines on cached test features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models import HierarchicalRouter, PooledMLPRouter, RouterConfig, combine_caches
from ..eval import aggregate_by, evaluate_all
from ..eval.baselines import model_routing
from ..utils import get_logger, load_config, merge_overrides
from ._common import load_caches_by_split

_log = get_logger("evaluate")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate routers + baselines")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)
    cache_root = Path(cfg.get("cache_root", f"feature_cache/{cfg.get('name', 'exp')}"))
    results_dir = Path(args.results_dir or f"results/{cfg.get('name', 'experiment')}")

    train_all = combine_caches(load_caches_by_split(cache_root, "train"))
    train_mase_mean = train_all.mase.mean(axis=0)
    rcfg = RouterConfig(**cfg.get("router", {}))

    models = {}
    for arch, cls in (("hit", HierarchicalRouter), ("pooled", PooledMLPRouter)):
        ckpt = results_dir / f"{arch}_router.pt"
        if not ckpt.exists():
            continue
        if arch == "hit":
            m = HierarchicalRouter(train_all.hidden_dims, rcfg)
        else:
            m = PooledMLPRouter(train_all.hidden_dims, d=rcfg.d, dropout=rcfg.dropout)
        m.load_state_dict(torch.load(ckpt, map_location=args.device))
        models[{"hit": "hit_forecast", "pooled": "pooled_mlp"}[arch]] = m

    results, aggregates = {}, {}
    for cache in load_caches_by_split(cache_root, "test"):
        data = combine_caches([cache])
        results[cache.meta["name"]] = evaluate_all(
            data, models=models, train_mase_mean=train_mase_mean, device=args.device
        )
        if "hit_forecast" in models:
            _, jn = model_routing(models["hit_forecast"], data, args.device)
            hit_fc = np.take_along_axis(data.forecasts, jn[:, None, None], 1)[:, 0, :]
            aggregates[cache.meta["name"]] = {
                dim: aggregate_by(hit_fc, data, dim=dim) for dim in ("domain", "freq", "term")
            }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (results_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
