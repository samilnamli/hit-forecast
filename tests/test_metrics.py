import numpy as np

from hit_forecast.data.metrics import mase, smape, mse, seasonal_naive_scale


def test_perfect_forecast_zero_error():
    ctx = np.sin(np.linspace(0, 10, 96))
    tgt = np.array([1.0, 2.0, 3.0])
    assert mase(tgt, tgt, ctx, m=24) == 0.0
    assert smape(tgt, tgt) == 0.0
    assert mse(tgt, tgt) == 0.0


def test_mase_scale_positive():
    ctx = np.cumsum(np.random.default_rng(0).normal(size=100))
    assert seasonal_naive_scale(ctx, m=12) > 0


def test_mase_matches_manual():
    ctx = np.arange(48, dtype=float)
    tgt = np.array([48.0, 49.0, 50.0])
    pred = np.array([47.0, 48.0, 49.0])
    scale = seasonal_naive_scale(ctx, m=1)  # constant diff of 1 -> scale 1
    expected = np.mean(np.abs(tgt - pred)) / scale
    assert abs(mase(tgt, pred, ctx, m=1) - expected) < 1e-9
