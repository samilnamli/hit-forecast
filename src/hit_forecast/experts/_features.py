"""Shared patch-feature helpers for foundation-model adapters.

The router only requires a per-window patch sequence ``H^{(k)} in R^{T_k x D_k}``.
Two sources are supported:

- ``encoder``: the frozen model's own patch/token hidden states (preferred; the
  draft's design). Adapters implement this per model where the API exposes it.
- ``stat``: a deterministic fallback that summarises each patch with 6 statistics
  and projects to ``D`` via a seeded random matrix. This keeps the whole
  pipeline runnable even when deep hidden-state extraction is unavailable for a
  given library version.

Adapters default to ``auto`` = try ``encoder`` then fall back to ``stat`` with a
warning, so a GPU run never hard-fails on feature extraction.
"""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

_log = get_logger(__name__)
_STATS = 6


def stat_patches(context: np.ndarray, patch_len: int, proj: np.ndarray) -> np.ndarray:
    ctx = np.asarray(context, dtype=np.float64).ravel()
    L = ctx.shape[0]
    T = max(1, int(np.ceil(L / patch_len)))
    feats = np.zeros((T, _STATS), dtype=np.float64)
    for i in range(T):
        seg = ctx[i * patch_len : (i + 1) * patch_len]
        if seg.size == 0:
            continue
        x = np.arange(seg.size, dtype=np.float64)
        slope = np.polyfit(x, seg, 1)[0] if seg.size > 1 else 0.0
        feats[i] = [seg.mean(), seg.std(), slope, seg.min(), seg.max(), seg[-1]]
    mu = feats.mean(0, keepdims=True)
    sd = feats.std(0, keepdims=True) + 1e-6
    feats = (feats - mu) / sd
    return (feats @ proj).astype(np.float32)


def make_stat_proj(name: str, hidden: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name)) % (2**32) if seed == 0 else seed)
    return rng.standard_normal((_STATS, hidden)) / np.sqrt(_STATS)


def warn_fallback(model_name: str, err: Exception) -> None:
    _log.warning(
        "Encoder feature extraction failed for %s (%s); using deterministic "
        "stat-patch fallback. This is fine for a first run but for the paper's "
        "patch-level claim verify the encoder path on the GPU host.",
        model_name,
        type(err).__name__,
    )
