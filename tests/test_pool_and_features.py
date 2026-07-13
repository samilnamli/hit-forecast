"""Tests for pool-diversity (#1) and expert-specific features (#2)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hit_forecast.experts import build_pool
from hit_forecast.experts._features import (
    HookedEncoderExtractor,
    forecast_summary_tokens,
    make_forecast_proj,
    make_stat_proj,
    pool_tokens,
    stat_patches,
    _STATS,
    _FC_STATS,
)
from hit_forecast.experts.classical import (
    ThetaExpert,
    SeasonalNaiveExpert,
    DampedDriftExpert,
    estimate_period,
)


def _seasonal_series(n=200, period=24):
    t = np.arange(n)
    return 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / period)


def test_estimate_period_detects_seasonality():
    y = _seasonal_series(period=24)
    p = estimate_period(y)
    # multiple of the true period is acceptable; must be seasonal (>1)
    assert p > 1


def test_classical_experts_shapes_and_finite():
    ctx = _seasonal_series()
    H = 24
    for cls in (ThetaExpert, SeasonalNaiveExpert, DampedDriftExpert):
        exp = cls()
        outs = exp.batch_forecast_and_features(ctx[None, :], H)
        assert len(outs) == 1
        fc = outs[0].forecast
        assert fc.shape == (H,)
        assert np.all(np.isfinite(fc))
        assert outs[0].patches.shape[1] == exp.hidden_dim


def test_snaive_repeats_last_season():
    period = 12
    y = np.tile(np.arange(period, dtype=float), 10)  # perfectly periodic
    exp = SeasonalNaiveExpert()
    fc = exp.batch_forecast_and_features(y[None, :], period)[0].forecast
    # next period should match the seasonal pattern closely
    assert np.allclose(fc[:period], np.arange(period), atol=1e-6)


def test_build_pool_with_classical_and_diverse_forecasts():
    specs = [
        {"kind": "theta", "name": "theta"},
        {"kind": "snaive", "name": "snaive"},
        {"kind": "drift", "name": "drift"},
    ]
    pool = build_pool(specs, device="cpu")
    assert [e.name for e in pool] == ["theta", "snaive", "drift"]
    ctx = _seasonal_series()
    fcs = [e.batch_forecast_and_features(ctx[None, :], 24)[0].forecast for e in pool]
    # experts must be decorrelated enough to disagree somewhere
    assert not np.allclose(fcs[0], fcs[1])
    assert not np.allclose(fcs[1], fcs[2])


def test_rich_stat_patches_shape():
    ctx = _seasonal_series()
    proj = make_stat_proj("x", 32)
    assert proj.shape == (_STATS, 32)
    p = stat_patches(ctx, 16, proj)
    assert p.shape[1] == 32
    assert np.all(np.isfinite(p))


def test_forecast_tokens_are_expert_specific():
    ctx = _seasonal_series()
    fc = np.linspace(ctx[-1], ctx[-1] + 5, 24)
    D = 32
    pa = make_forecast_proj("expertA", D)
    pb = make_forecast_proj("expertB", D)
    ta = forecast_summary_tokens(fc, ctx, pa)
    tb = forecast_summary_tokens(fc, ctx, pb)
    assert ta.shape == (2, D)
    assert pa.shape == (_FC_STATS, D)
    # same forecast, different expert projection => different tokens (expert-specific)
    assert not np.allclose(ta, tb)


def test_forecast_tokens_differ_for_different_forecasts():
    ctx = _seasonal_series()
    proj = make_forecast_proj("e", 16)
    flat = np.full(24, ctx[-1])
    trend = np.linspace(ctx[-1], ctx[-1] + 20, 24)
    assert not np.allclose(
        forecast_summary_tokens(flat, ctx, proj),
        forecast_summary_tokens(trend, ctx, proj),
    )


class _ToyEncoder(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.encoder = nn.Linear(4, d)

    def forward(self, x):
        return self.encoder(x)


def test_hooked_extractor_captures_hidden_states():
    model = _ToyEncoder(d=8)
    ext = HookedEncoderExtractor(("encoder",)).attach(model)
    B, T = 3, 5
    _ = model(torch.randn(B, T, 4))
    cap = ext.pop(B)
    assert cap is not None
    assert cap.shape == (B, T, 8)
    ext.detach_hook()


def test_hooked_extractor_rejects_batch_mismatch():
    model = _ToyEncoder(d=8)
    ext = HookedEncoderExtractor(("encoder",)).attach(model)
    _ = model(torch.randn(2, 5, 4))
    assert ext.pop(3) is None  # requested B=3 but ran B=2


def test_pool_tokens_downsamples():
    h = np.random.randn(200, 16).astype(np.float32)
    pooled = pool_tokens(h, max_patches=64)
    assert pooled.shape == (64, 16)
    small = np.random.randn(10, 16).astype(np.float32)
    assert pool_tokens(small, max_patches=64).shape == (10, 16)
