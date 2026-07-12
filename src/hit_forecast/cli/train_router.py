"""Train a router (HiT-Forecast or pooled-MLP) from cached features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..models import HierarchicalRouter, PooledMLPRouter, RouterConfig, combine_caches
from ..train import TrainConfig, train_router
from ..utils import get_logger, load_config, merge_overrides, seed_everything
from ._common import load_caches_by_split, split_train_val
from .run_all import _subset

_log = get_logger("train_router")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a router from cached features")
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", choices=["hit", "pooled"], default="hit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)
    seed_everything(cfg.get("seed", 0))
    cache_root = Path(cfg.get("cache_root", f"feature_cache/{cfg.get('name', 'exp')}"))
    out_dir = Path(args.out or f"results/{cfg.get('name', 'experiment')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = combine_caches(load_caches_by_split(cache_root, "train"))
    tr_idx, val_idx = split_train_val(train_all.N, cfg.get("val_frac", 0.1), cfg.get("seed", 0))
    train_data, val_data = _subset(train_all, tr_idx), _subset(train_all, val_idx)

    rcfg = RouterConfig(**cfg.get("router", {}))
    tcfg = TrainConfig(**cfg.get("train", {}))
    if args.arch == "hit":
        model = HierarchicalRouter(train_all.hidden_dims, rcfg)
    else:
        model = PooledMLPRouter(train_all.hidden_dims, d=rcfg.d, dropout=rcfg.dropout)
    _log.info("%s params: %.2fM", args.arch, model.num_trainable_params() / 1e6)

    out = train_router(model, train_data, val_data, tcfg, args.device)
    ckpt = out_dir / f"{args.arch}_router.pt"
    torch.save(model.state_dict(), ckpt)
    (out_dir / f"{args.arch}_history.json").write_text(json.dumps(out["history"], indent=2))
    _log.info("Saved %s (best val E[MASE]=%.4f)", ckpt, out["best_val_expected_mase"])


if __name__ == "__main__":
    main()
