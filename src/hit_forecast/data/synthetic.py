"""Controlled synthetic regime-switch benchmark (draft §IV-A / §IV-D1).

Each window of length ``L`` is split into three contiguous segments by two
switch points. The segments are drawn from three regimes:

- ``trended``      : strong linear trend + mild noise  (a TimesFM-style expert wins)
- ``spiky``        : high-frequency noise + sparse spikes (a Chronos-style expert wins)
- ``long_season``  : long smooth seasonality           (a Moirai-style expert wins)

The horizon target continues the *last* regime of the context, so the locally
best expert is determined by the temporal structure at the end of the window,
while the regime-switch position inside the window is the signal a pooled router
cannot see.
"""

from __future__ import annotations

import numpy as np

from .windows import Window

REGIMES = ("trended", "spiky", "long_season")


def _gen_segment(regime: str, n: int, rng: np.random.Generator, phase: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    if regime == "trended":
        slope = rng.uniform(0.05, 0.15) * rng.choice([-1.0, 1.0])
        return slope * t + rng.normal(0, 0.15, n)
    if regime == "spiky":
        base = rng.normal(0, 0.6, n)
        n_spikes = max(1, int(0.08 * n))
        idx = rng.integers(0, n, size=n_spikes)
        base[idx] += rng.normal(0, 3.0, n_spikes)
        return base
    if regime == "long_season":
        period = rng.uniform(n * 0.8, n * 1.5)
        return 2.0 * np.sin(2 * np.pi * (t + phase) / period) + rng.normal(0, 0.1, n)
    raise ValueError(regime)


def _gen_full_series(L: int, H: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Return the full (L+H) series and the regime that governs the horizon."""
    rho1 = rng.integers(L // 4, L // 2)
    rho2 = rng.integers(L // 2, 3 * L // 4)
    order = list(REGIMES)
    rng.shuffle(order)
    seg_bounds = [(0, rho1), (rho1, rho2), (rho2, L + H)]
    phase = rng.uniform(0, 100)
    parts = []
    level = 0.0
    last_regime = order[-1]
    for regime, (a, b) in zip(order, seg_bounds):
        seg = _gen_segment(regime, b - a, rng, phase)
        seg = seg - seg[0] + level  # keep continuity
        level = seg[-1]
        parts.append(seg)
    series = np.concatenate(parts)
    return series, last_regime


def make_regime_switch_dataset(
    n_windows: int = 10_000,
    L: int = 192,
    H: int = 96,
    m: int = 24,
    seed: int = 0,
) -> tuple[list[Window], np.ndarray]:
    """Build the synthetic regime-switch windows.

    Returns ``(windows, oracle_regime_ids)`` where ``oracle_regime_ids[i]`` is
    the index into :data:`REGIMES` of the regime governing window ``i``'s
    horizon. Dummy experts (see ``hit_forecast.experts.dummy``) map one-to-one
    onto these regimes, so the per-window oracle expert is recoverable.
    """
    rng = np.random.default_rng(seed)
    windows: list[Window] = []
    regime_ids = np.empty(n_windows, dtype=np.int64)
    for i in range(n_windows):
        series, last_regime = _gen_full_series(L, H, rng)
        ctx, tgt = series[:L], series[L : L + H]
        regime_ids[i] = REGIMES.index(last_regime)
        windows.append(
            Window(
                context=ctx,
                target=tgt,
                m=m,
                meta={"dataset": "synthetic_regime_switch", "regime": last_regime},
            )
        )
    return windows, regime_ids
