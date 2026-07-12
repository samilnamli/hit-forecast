"""Lightweight, dependency-free experts.

These are *not* foundation models. They exist so that the whole pipeline
(caching -> router training -> evaluation -> aggregation) runs end-to-end on any
machine with no downloads, and so the synthetic regime-switch experiment
produces genuine numbers. Each dummy expert is a classical forecaster with a
distinct inductive bias that matches one synthetic regime, mirroring the roles
the draft assigns to TimesFM / Chronos / Moirai / Lag-Llama.

Patch features are deterministic per-patch summary statistics of the context,
projected to a fixed hidden dimension by a seeded random matrix. They preserve
intra-window temporal structure (trend/spikiness/seasonality position), which is
exactly what the router is meant to exploit.
"""

from __future__ import annotations

import numpy as np

from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert

_PATCH_STATS = 6  # mean, std, slope, min, max, last


def _patch_features(context: np.ndarray, patch_len: int, proj: np.ndarray) -> np.ndarray:
    L = context.shape[0]
    T = int(np.ceil(L / patch_len))
    feats = np.zeros((T, _PATCH_STATS), dtype=np.float64)
    for i in range(T):
        seg = context[i * patch_len : (i + 1) * patch_len]
        if seg.size == 0:
            continue
        x = np.arange(seg.size, dtype=np.float64)
        slope = np.polyfit(x, seg, 1)[0] if seg.size > 1 else 0.0
        feats[i] = [seg.mean(), seg.std(), slope, seg.min(), seg.max(), seg[-1]]
    # standardise then project to hidden dim D
    mu = feats.mean(0, keepdims=True)
    sd = feats.std(0, keepdims=True) + 1e-6
    feats = (feats - mu) / sd
    return (feats @ proj).astype(np.float32)  # (T, D)


class _DummyBase(ExpertAdapter):
    def __init__(self, name: str, patch_len: int, hidden: int, device: str = "cpu",
                 seed: int = 0):
        super().__init__(name=name, device=device)
        self._patch_len = patch_len
        self._hidden = hidden
        rng = np.random.default_rng(abs(hash(name)) % (2**32) if seed == 0 else seed)
        self._proj = rng.standard_normal((_PATCH_STATS, hidden)) / np.sqrt(_PATCH_STATS)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def _predict(self, context: np.ndarray, horizon: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def batch_forecast_and_features(self, contexts, horizon):
        contexts = np.asarray(contexts, dtype=np.float64)
        outs = []
        for ctx in contexts:
            yhat = self._predict(ctx, horizon)
            patches = _patch_features(ctx, self._patch_len, self._proj)
            outs.append(ExpertOutput(forecast=np.asarray(yhat, dtype=np.float64),
                                     patches=patches))
        return outs


@register_expert("dummy_trend")
class TrendExpert(_DummyBase):
    """Linear-trend extrapolator. Best on the ``trended`` regime."""

    def __init__(self, name="dummy_trend", patch_len=16, hidden=64, device="cpu"):
        super().__init__(name, patch_len, hidden, device)

    def _predict(self, context, horizon):
        x = np.arange(context.size, dtype=np.float64)
        b, a = np.polyfit(x, context, 1)
        fx = np.arange(context.size, context.size + horizon, dtype=np.float64)
        return a + b * fx


@register_expert("dummy_spiky")
class SpikyExpert(_DummyBase):
    """Robust local-level (median of recent window). Best on the ``spiky`` regime."""

    def __init__(self, name="dummy_spiky", patch_len=8, hidden=48, device="cpu"):
        super().__init__(name, patch_len, hidden, device)

    def _predict(self, context, horizon):
        level = np.median(context[-16:]) if context.size >= 16 else np.median(context)
        return np.full(horizon, level, dtype=np.float64)


@register_expert("dummy_season")
class SeasonExpert(_DummyBase):
    """Long-period sinusoid fit. Best on the ``long_season`` regime."""

    def __init__(self, name="dummy_season", patch_len=24, hidden=80, device="cpu"):
        super().__init__(name, patch_len, hidden, device)

    def _predict(self, context, horizon):
        n = context.size
        t = np.arange(n, dtype=np.float64)
        best = None
        for period in np.linspace(max(8, n // 4), n * 1.5, 12):
            X = np.stack([np.sin(2 * np.pi * t / period),
                          np.cos(2 * np.pi * t / period),
                          np.ones_like(t)], 1)
            coef, res, *_ = np.linalg.lstsq(X, context, rcond=None)
            resid = float(np.sum((X @ coef - context) ** 2))
            if best is None or resid < best[0]:
                best = (resid, period, coef)
        _, period, coef = best
        ft = np.arange(n, n + horizon, dtype=np.float64)
        Xf = np.stack([np.sin(2 * np.pi * ft / period),
                       np.cos(2 * np.pi * ft / period),
                       np.ones_like(ft)], 1)
        return Xf @ coef


@register_expert("dummy_snaive")
class SeasonalNaiveExpert(_DummyBase):
    """Seasonal-naive with period m. A hard-to-beat generalist (Lag-Llama analogue)."""

    def __init__(self, name="dummy_snaive", patch_len=12, hidden=56, device="cpu", m=24):
        super().__init__(name, patch_len, hidden, device)
        self.m = m

    def _predict(self, context, horizon):
        m = min(self.m, context.size)
        reps = int(np.ceil(horizon / m))
        tail = context[-m:]
        return np.tile(tail, reps)[:horizon]
