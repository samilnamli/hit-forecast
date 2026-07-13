"""Decorrelated classical forecasters for the routing pool (real, not toy).

The clean foundation-model pool (Chronos-2 / Moirai / TimesFM / TiRex) is highly
*correlated*: on most GIFT-Eval windows one strong TSFM wins and the per-window
oracle gap over the best single expert is small (~12%). That caps how much any
router can gain.

These classical experts have genuinely different inductive biases and failure
modes, so they win a *different* subset of windows (strongly seasonal series,
clean linear trends, M-competition-style economic series). Adding them widens the
oracle ceiling and forces the router to make non-trivial, learnable decisions.
They are numpy-only (no GPU, no downloads), so re-caching stays cheap.

Patch features use the shared rich-stat helper; the cache additionally appends
expert-specific forecast-conditioned tokens (see ``features/cache.py``), so even
these simple experts contribute a distinct routing signal.
"""

from __future__ import annotations

import numpy as np

from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F


def estimate_period(y: np.ndarray, min_p: int = 2, max_p: int | None = None) -> int:
    """Dominant seasonal period via autocorrelation; 1 if none is significant."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 8:
        return 1
    max_p = max_p or n // 2
    max_p = min(max_p, n - 1)
    y = y - y.mean()
    denom = float(np.dot(y, y)) + 1e-12
    best_p, best_r = 1, 0.0
    for p in range(min_p, max_p + 1):
        r = float(np.dot(y[:-p], y[p:])) / denom
        if r > best_r:
            best_r, best_p = r, p
    # require a meaningful peak to call it seasonal
    return best_p if best_r > 0.2 else 1


def _seasonal_indices(y: np.ndarray, m: int) -> np.ndarray:
    """Multiplicative-ish seasonal factors of period m (additive, mean-removed)."""
    n = y.size
    idx = np.zeros(m, dtype=np.float64)
    cnt = np.zeros(m, dtype=np.float64)
    detr = y - _moving_average(y, m)
    for t in range(n):
        s = t % m
        if np.isfinite(detr[t]):
            idx[s] += detr[t]
            cnt[s] += 1
    cnt[cnt == 0] = 1
    idx = idx / cnt
    return idx - idx.mean()


def _moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y.copy()
    kernel = np.ones(w) / w
    ma = np.convolve(y, kernel, mode="same")
    return ma


def _ses_level(y: np.ndarray, alpha: float = 0.5) -> float:
    level = float(y[0])
    for v in y[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return level


class _ClassicalBase(ExpertAdapter):
    def __init__(self, name: str, device: str = "cpu", patch_len: int = 16,
                 hidden: int = 64):
        super().__init__(name=name, device=device)
        self._patch_len = patch_len
        self._hidden = hidden
        self._proj = F.make_stat_proj(name, hidden)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def _predict(self, ctx: np.ndarray, horizon: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def batch_forecast_and_features(self, contexts, horizon):
        contexts = np.asarray(contexts, dtype=np.float64)
        outs = []
        for ctx in contexts:
            try:
                yhat = np.asarray(self._predict(ctx, horizon), dtype=np.float64)
            except Exception:  # noqa: BLE001 - classical numerics must never crash caching
                yhat = np.full(horizon, float(ctx[-1]) if ctx.size else 0.0)
            if yhat.shape[0] < horizon:
                yhat = np.pad(yhat, (0, horizon - yhat.shape[0]), mode="edge")
            yhat = np.nan_to_num(yhat[:horizon], nan=float(ctx[-1] if ctx.size else 0.0))
            outs.append(ExpertOutput(
                forecast=yhat,
                patches=F.stat_patches(ctx, self._patch_len, self._proj),
            ))
        return outs


@register_expert("theta")
class ThetaExpert(_ClassicalBase):
    """Theta method (Assimakopoulos & Nikolopoulos): SES level + linear trend,
    with optional additive seasonal component. A perennial M-competition winner,
    strongly decorrelated from deep foundation models."""

    def __init__(self, name: str = "theta", device: str = "cpu",
                 patch_len: int = 16, hidden: int = 64, alpha: float = 0.5):
        super().__init__(name, device, patch_len, hidden)
        self.alpha = alpha

    def _predict(self, ctx, horizon):
        y = np.asarray(ctx, dtype=np.float64).ravel()
        n = y.size
        if n < 3:
            return np.full(horizon, float(y[-1]) if n else 0.0)
        m = estimate_period(y)
        seas = None
        work = y
        if m > 1 and n >= 2 * m:
            seas_idx = _seasonal_indices(y, m)
            work = y - seas_idx[np.arange(n) % m]
            fut_phase = (np.arange(n, n + horizon)) % m
            seas = seas_idx[fut_phase]
        x = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(x, work, 1)
        fx = np.arange(n, n + horizon, dtype=np.float64)
        trend = intercept + slope * fx
        level = _ses_level(work, self.alpha)
        # blend long-run trend line (theta=0) and SES level (theta=2)
        fc = 0.5 * trend + 0.5 * level
        if seas is not None:
            fc = fc + seas
        return fc


@register_expert("snaive")
class SeasonalNaiveExpert(_ClassicalBase):
    """Seasonal-naive with an autocorrelation-estimated period. Hard to beat on
    strongly periodic series (electricity/solar/traffic); the MASE denominator
    baseline, so it wins whenever deep models over-smooth seasonality."""

    def __init__(self, name: str = "snaive", device: str = "cpu",
                 patch_len: int = 12, hidden: int = 56):
        super().__init__(name, device, patch_len, hidden)

    def _predict(self, ctx, horizon):
        y = np.asarray(ctx, dtype=np.float64).ravel()
        n = y.size
        if n == 0:
            return np.zeros(horizon)
        m = estimate_period(y)
        if m <= 1:
            return np.full(horizon, float(y[-1]))
        m = min(m, n)
        tail = y[-m:]
        reps = int(np.ceil(horizon / m))
        return np.tile(tail, reps)[:horizon]


@register_expert("drift")
class DampedDriftExpert(_ClassicalBase):
    """Damped-trend drift from a robust local level. Complements seasonal experts
    on smooth trending series without over-committing to the trend (damping)."""

    def __init__(self, name: str = "drift", device: str = "cpu",
                 patch_len: int = 16, hidden: int = 48, phi: float = 0.9):
        super().__init__(name, device, patch_len, hidden)
        self.phi = phi

    def _predict(self, ctx, horizon):
        y = np.asarray(ctx, dtype=np.float64).ravel()
        n = y.size
        if n < 2:
            return np.full(horizon, float(y[-1]) if n else 0.0)
        level = float(np.median(y[-min(8, n):]))
        drift = (float(y[-1]) - float(y[0])) / (n - 1)
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        damp = np.cumsum(self.phi ** steps)
        return level + drift * damp
