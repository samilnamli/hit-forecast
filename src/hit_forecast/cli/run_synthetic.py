"""Synthetic regime-switch experiment (draft Table I).

Self-contained: uses four dependency-free dummy experts whose inductive biases
match the three regimes (+ a hard seasonal-naive distractor). Runs anywhere,
no downloads, and produces genuine MASE / selection-accuracy numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .run_all import run_experiment
from ..utils import get_logger

_log = get_logger("run_synthetic")

SYNTHETIC_CONFIG = {
    "name": "synthetic_regime_switch",
    "seed": 0,
    "cache_batch_size": 512,
    "experts": [
        {"kind": "dummy_trend"},
        {"kind": "dummy_spiky"},
        {"kind": "dummy_season"},
        {"kind": "dummy_snaive"},
    ],
    "datasets": {
        "train": [{"source": "synthetic", "name": "synthetic_regime_switch", "split": "train",
                   "n_windows": 8000, "L": 192, "H": 96, "m": 24, "seed": 1}],
        "test": [{"source": "synthetic", "name": "synthetic_regime_switch", "split": "test",
                  "n_windows": 2000, "L": 192, "H": 96, "m": 24, "seed": 2}],
    },
    "router": {"d": 128, "nhead": 4, "stage1_layers": 2, "stage2_layers": 2,
               "ffn": 256, "dropout": 0.1, "share_stage1": True, "cross_attention": True},
    "train": {"lr": 3e-4, "weight_decay": 1e-2, "warmup_steps": 100, "max_epochs": 40,
              "batch_size": 256, "patience": 8, "amp": False,
              "loss": {"lambda_mase": 1.0, "lambda_hard": 0.3, "lambda_soft": 0.5,
                       "tau": 1.5, "label_smoothing": 0.05}},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Synthetic regime-switch experiment")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/synthetic_regime_switch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = {**SYNTHETIC_CONFIG}
    cfg["cache_root"] = str(Path(args.out) / "feature_cache")
    if args.epochs:
        cfg["train"] = {**cfg["train"], "max_epochs": args.epochs}
    if args.n_train:
        cfg["datasets"]["train"][0]["n_windows"] = args.n_train

    run_experiment(cfg, args.device, Path(args.out), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
