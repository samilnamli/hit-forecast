"""Shared patch-feature helpers for foundation-model adapters.

The router consumes, per window and per expert, a patch sequence
``H^{(k)} in R^{T_k x D_k}``. Three sources are supported:

- ``encoder``: the frozen model's own per-patch/token hidden states, captured via
  a forward hook on the model backbone during the forecast pass
  (:class:`HookedEncoderExtractor`). This is the draft's design and the only
  source that is genuinely *expert-specific* at the representation level.
- ``stat``: a deterministic fallback summarising each patch with a rich set of
  statistics (mean/std/slope/curvature/min/max/first/last/range/MAD/acf1) and
  projecting to ``D`` with a seeded matrix. Keeps the pipeline runnable when
  deep hidden-state extraction is unavailable for a library version.
- forecast-conditioned tokens (:func:`forecast_summary_tokens`) are appended by
  the cache for *every* expert, injecting an expert-specific signal (the shape of
  that expert's own forecast) even when patch features are the shared stat set.

Adapters default to ``auto`` = try ``encoder`` then fall back to ``stat`` with a
warning, so a GPU run never hard-fails on feature extraction.
"""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

_log = get_logger(__name__)

# Rich per-patch statistics (was 6). More discriminative for routing.
_STATS = 11
# Forecast-summary statistics per half-horizon token.
_FC_STATS = 8


def stat_patches(context: np.ndarray, patch_len: int, proj: np.ndarray) -> np.ndarray:
    """Rich per-patch summary features projected to ``D`` (proj is (``_STATS``, D))."""
    ctx = np.asarray(context, dtype=np.float64).ravel()
    L = ctx.shape[0]
    T = max(1, int(np.ceil(L / patch_len)))
    feats = np.zeros((T, _STATS), dtype=np.float64)
    for i in range(T):
        seg = ctx[i * patch_len : (i + 1) * patch_len]
        if seg.size == 0:
            continue
        x = np.arange(seg.size, dtype=np.float64)
        if seg.size > 2:
            curv, slope, _ = np.polyfit(x, seg, 2)
        elif seg.size == 2:
            slope, curv = float(seg[1] - seg[0]), 0.0
        else:
            slope, curv = 0.0, 0.0
        mad = float(np.mean(np.abs(seg - seg.mean())))
        if seg.size > 1 and seg.std() > 1e-8:
            acf1 = float(np.corrcoef(seg[:-1], seg[1:])[0, 1])
            if not np.isfinite(acf1):
                acf1 = 0.0
        else:
            acf1 = 0.0
        feats[i] = [
            seg.mean(), seg.std(), slope, curv, seg.min(), seg.max(),
            seg[0], seg[-1], seg.max() - seg.min(), mad, acf1,
        ]
    mu = feats.mean(0, keepdims=True)
    sd = feats.std(0, keepdims=True) + 1e-6
    feats = (feats - mu) / sd
    return (feats @ proj).astype(np.float32)


def make_stat_proj(name: str, hidden: int, seed: int = 0, n_stats: int = _STATS) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name)) % (2**32) if seed == 0 else seed)
    return rng.standard_normal((n_stats, hidden)) / np.sqrt(n_stats)


def make_forecast_proj(name: str, hidden: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name + ":fc")) % (2**32))
    return (rng.standard_normal((_FC_STATS, hidden)) / np.sqrt(_FC_STATS)).astype(np.float32)


def _fc_summary(seg: np.ndarray, level: float, last: float) -> np.ndarray:
    if seg.size == 0:
        return np.zeros(_FC_STATS, dtype=np.float64)
    x = np.arange(seg.size, dtype=np.float64)
    slope = float(np.polyfit(x, seg, 1)[0]) if seg.size > 1 else 0.0
    return np.array([
        seg.mean() / level,
        seg.std() / level,
        slope / level,
        (seg[-1] - seg[0]) / level,
        (seg[0] - last) / level,
        seg.min() / level,
        seg.max() / level,
        (seg.mean() - last) / level,
    ], dtype=np.float64)


def forecast_summary_tokens(fc: np.ndarray, context: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Two expert-specific tokens (first/second half of the horizon) describing the
    shape of *this* expert's forecast relative to the context. Shape (2, D)."""
    fc = np.asarray(fc, dtype=np.float64).ravel()
    ctx = np.asarray(context, dtype=np.float64).ravel()
    level = float(np.mean(np.abs(ctx))) + 1e-8
    last = float(ctx[-1]) if ctx.size else 0.0
    if fc.size == 0:
        rows = np.zeros((2, _FC_STATS), dtype=np.float64)
    else:
        half = max(1, fc.size // 2)
        rows = np.stack([_fc_summary(fc[:half], level, last),
                         _fc_summary(fc[half:], level, last)])
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    return (rows @ proj).astype(np.float32)


class HookedEncoderExtractor:
    """Capture per-patch hidden states from a frozen model via a forward hook.

    Model-agnostic: resolves the deepest submodule whose qualified name matches
    one of ``name_candidates`` (e.g. an encoder/backbone/transformer block) and
    records its output during the ordinary forecast forward pass, so no second
    model call is needed. Returns ``None`` on any shape surprise so callers can
    fall back to stat features without crashing.
    """

    def __init__(self, name_candidates: tuple[str, ...] = ("encoder", "backbone", "transformer")):
        self.name_candidates = tuple(c.lower() for c in name_candidates)
        self._handle = None
        self._captured = None

    def attach(self, model) -> "HookedEncoderExtractor":
        if self._handle is not None:
            return self
        target = self._resolve(model)
        if target is None:
            raise RuntimeError("No encoder-like submodule found for hook.")

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h = getattr(h, "last_hidden_state", h)
            if hasattr(h, "detach"):
                self._captured = h.detach()

        self._handle = target.register_forward_hook(hook)
        return self

    def _resolve(self, model):
        matches = []
        for n, m in model.named_modules():
            ln = n.lower()
            if any(c in ln for c in self.name_candidates):
                matches.append((n, m))
        return matches[-1][1] if matches else None

    def pop(self, batch_size: int):
        """Return captured (B, T, D) tensor for this pass, or None if unusable."""
        h = self._captured
        self._captured = None
        if h is None:
            return None
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if h.dim() != 3 or h.shape[0] != batch_size:
            return None
        return h

    def detach_hook(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def pool_tokens(h_row: np.ndarray, max_patches: int = 64) -> np.ndarray:
    """Average-pool a (T, D) hidden sequence down to <= max_patches rows."""
    h = np.asarray(h_row, dtype=np.float32)
    if h.ndim == 1:
        h = h[None, :]
    T = h.shape[0]
    if T <= max_patches:
        return h
    bins = np.array_split(np.arange(T), max_patches)
    return np.stack([h[b].mean(axis=0) for b in bins]).astype(np.float32)


def warn_fallback(model_name: str, err: Exception) -> None:
    _log.warning(
        "Encoder feature extraction failed for %s (%s); using deterministic "
        "rich stat-patch fallback. Forecast-conditioned tokens still provide an "
        "expert-specific signal. Verify the encoder hook on the GPU host for the "
        "paper's patch-level claim.",
        model_name,
        type(err).__name__,
    )
