"""Cache expert forecasts + patch features + MASE for an experiment config.

This is the GPU-heavy stage. Run it once per host; training/eval reuse the cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..utils import get_logger, load_config, merge_overrides
from ._common import build_experts, cache_entries

_log = get_logger("cache_features")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cache FM features/forecasts/MASE")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)
    cache_root = Path(cfg.get("cache_root", f"feature_cache/{cfg.get('name', 'exp')}"))
    batch_size = cfg.get("cache_batch_size", 64)
    experts = build_experts(cfg, args.device)

    for split in args.splits:
        entries = cfg["datasets"].get(split, [])
        if entries:
            cache_entries(entries, experts, cache_root, args.device, batch_size,
                          args.overwrite)
    _log.info("Caching complete under %s", cache_root)


if __name__ == "__main__":
    main()
