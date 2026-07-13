"""Shared CLI helpers: build window sets, expert pools, and caches from config."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.synthetic import make_regime_switch_dataset
from ..data.windows import WindowSet
from ..experts import build_pool
from ..features.cache import FeatureCache, build_cache
from ..utils.logging import get_logger

_log = get_logger(__name__)


def build_windowset(entry: dict) -> WindowSet:
    source = entry.get("source", "gifteval")
    if source == "synthetic":
        windows, _ = make_regime_switch_dataset(
            n_windows=entry.get("n_windows", 10_000),
            L=entry.get("L", 192),
            H=entry.get("H", 96),
            m=entry.get("m", 24),
            seed=entry.get("seed", 0),
        )
        name = entry.get("name", "synthetic_regime_switch")
        return WindowSet(windows=windows, name=name)
    if source == "gifteval":
        from ..data.gifteval import load_gifteval_windows

        return load_gifteval_windows(
            name=entry["name"],
            term=entry.get("term", "short"),
            split=entry.get("split", "test"),
            context_length=entry.get("context_length"),
            to_univariate=entry.get("to_univariate", False),
            max_windows=entry.get("max_windows"),
            max_series=entry.get("max_series"),
        )
    raise ValueError(f"Unknown data source: {source!r}")


def build_experts(cfg: dict, device: str):
    return build_pool(cfg["experts"], device=device)


def cache_entries(entries: list[dict], experts, cache_root: str | Path, device: str,
                  batch_size: int = 64, overwrite: bool = False) -> list[FeatureCache]:
    caches = []
    for entry in entries:
        try:
            ws = build_windowset(entry)
        except RuntimeError as e:
            _log.warning("Skipping GiftEval entry %s: %s", entry.get("name"), e)
            continue
        if len(ws) == 0:
            _log.warning("Skipping empty WindowSet for %s", entry.get("name"))
            continue
        tag = entry.get("split", "test")
        ws.name = f"{ws.name}::{tag}"
        _log.info("Caching %s (%d windows)", ws.name, len(ws))
        caches.append(build_cache(ws, experts, cache_root, batch_size=batch_size,
                                  overwrite=overwrite))
    if not caches:
        raise RuntimeError(
            "No cache shards were produced. Check GiftEval dataset names / lengths."
        )
    return caches


def load_caches_by_split(cache_root: str | Path, split: str) -> list[FeatureCache]:
    """Load all cache shards under ``cache_root`` whose name ends with ``::split``."""
    from ..features.cache import load_cache

    cache_root = Path(cache_root)
    caches = []
    for meta_path in sorted(cache_root.glob("*/meta.json")):
        import json

        name = json.loads(meta_path.read_text()).get("name", "")
        if name.endswith(f"::{split}"):
            caches.append(load_cache(meta_path.parent))
    if not caches:
        raise FileNotFoundError(
            f"No '::{split}' caches under {cache_root}. Run hitf-cache first."
        )
    return caches


def split_train_val(N: int, val_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(val_frac * N))
    return perm[n_val:], perm[:n_val]
