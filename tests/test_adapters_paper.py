"""Unit tests for Chronos-2 / TimesFM helpers and MASE hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from hit_forecast.data.metrics import (
    MASE_CLIP,
    MASE_SENTINEL,
    mase,
    sanitize_mase,
    sanitize_mase_array,
    seasonal_naive_scale,
)
from hit_forecast.experts.chronos import extract_chronos_point_forecast
from hit_forecast.experts.timesfm import patch_aligned_context


class _FakeChronos2Pipe:
    """Mimics Chronos-2: inputs= API, list of (n_variates, H) means."""

    def predict_quantiles(self, **kwargs):
        if "inputs" not in kwargs:
            raise TypeError("unexpected keyword argument 'context'")
        inputs = kwargs["inputs"]
        H = int(kwargs["prediction_length"])
        means = []
        for i, _ in enumerate(inputs):
            # Distinct median-like level per series
            means.append(np.full((1, H), 10.0 + i, dtype=np.float32))
        quantiles = [np.stack([m, m, m], axis=-1) for m in means]  # (1,H,3)
        return quantiles, means

    def predict(self, contexts, prediction_length):
        # Wrong path if used naively: (1, Q=9, H) — median(axis=0) would pick q0.1
        H = int(prediction_length)
        return [np.arange(9 * H, dtype=np.float32).reshape(1, 9, H)]


class _FakeChronos1Pipe:
    def predict_quantiles(self, **kwargs):
        if "context" not in kwargs and "inputs" not in kwargs:
            raise TypeError("need context or inputs")
        if "inputs" in kwargs and "context" not in kwargs:
            # Simulate Chronos-1 rejecting inputs=
            raise TypeError("unexpected keyword argument 'inputs'")
        contexts = kwargs.get("context") or kwargs["inputs"]
        H = int(kwargs["prediction_length"])
        B = len(contexts)
        # (B, H, Q=1)
        q = np.ones((B, H, 1), dtype=np.float32) * 3.0
        return q, None


def test_chronos2_extracts_mean_not_quantile01():
    pipe = _FakeChronos2Pipe()
    ctx = [np.zeros(16), np.zeros(16)]
    fc = extract_chronos_point_forecast(pipe, ctx, horizon=4)
    assert fc.shape == (2, 4)
    np.testing.assert_allclose(fc[0], 10.0)
    np.testing.assert_allclose(fc[1], 11.0)


def test_chronos1_batched_quantiles():
    pipe = _FakeChronos1Pipe()
    ctx = [np.zeros(16), np.zeros(16)]
    fc = extract_chronos_point_forecast(pipe, ctx, horizon=5)
    assert fc.shape == (2, 5)
    np.testing.assert_allclose(fc, 3.0)


def test_patch_aligned_context():
    assert patch_aligned_context(96) == 96
    assert patch_aligned_context(97) == 128
    assert patch_aligned_context(1) == 32
    assert patch_aligned_context(3000, cap=2048) == 2048


def test_mase_relative_floor_not_eps_only():
    # Near-constant context: absolute 1e-8 floor would explode MASE.
    ctx = np.ones(48) * 100.0
    ctx[-1] = 100.01
    scale = seasonal_naive_scale(ctx, m=1)
    assert scale >= 1e-3 * 100.0 * 0.99  # relative floor


def test_mase_clips_extreme_finite():
    ctx = np.arange(48, dtype=float)
    tgt = np.array([1000.0, 1000.0, 1000.0])
    pred = np.zeros(3)
    val = mase(tgt, pred, ctx, m=1)
    assert val <= MASE_CLIP
    assert val < MASE_SENTINEL


def test_mase_sentinel_on_nan_pred():
    ctx = np.arange(48, dtype=float)
    tgt = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, np.nan, 3.0])
    assert mase(tgt, pred, ctx, m=1) == MASE_SENTINEL


def test_sanitize_mase_array():
    arr = np.array([[1.0, np.nan], [5e4, MASE_SENTINEL]], dtype=np.float64)
    out = sanitize_mase_array(arr)
    assert out[0, 0] == 1.0
    assert out[0, 1] == MASE_SENTINEL
    assert out[1, 0] == MASE_CLIP
    assert out[1, 1] == MASE_SENTINEL
    assert sanitize_mase(float("inf")) == MASE_SENTINEL


def test_export_tables_smoke(tmp_path):
    from hit_forecast.cli.export_tables import booktabs_main, load_metrics_csv, booktabs_phase0

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "dataset,method,MASE,sMAPE,MSE\n"
        "electricity/H/H/short::test,hit_forecast:hard,1.2,10,0.5\n"
        "electricity/H/H/short::test,oracle,1.0,8,0.4\n"
        "solar/H/H/short::test,hit_forecast:hard,2.0,12,0.6\n"
        "solar/H/H/short::test,oracle,1.5,9,0.5\n"
        "bitbrains/H/short::test,hit_forecast:hard,134.0,50,1\n"
        "bitbrains/H/short::test,oracle,100.0,40,1\n"
    )
    data = load_metrics_csv(csv_path)
    tex = booktabs_main(
        data,
        ["hit_forecast:hard", "oracle"],
        caption="test",
        label="tab:test",
    )
    assert "electricity" in tex
    assert "bitbrains" not in tex  # excluded from rows... wait, exclude only affects means
    # Actually booktabs_main excludes bitbrains from dataset rows too via _excluded
    assert r"\bottomrule" in tex
    phase = booktabs_phase0(
        {"rel_margin_median": 0.5, "gate_pass": True, "win_rate": {"a": 1.0},
         "n_windows": 10, "win_rate_entropy_norm": 0.1, "abs_margin_median": 0.2,
         "oracle_gain_vs_best_single": 0.1},
        {"rel_margin_median": 0.08, "gate_pass": True, "win_rate": {"b": 1.0},
         "n_windows": 10, "win_rate_entropy_norm": 0.9, "abs_margin_median": 0.05,
         "oracle_gain_vs_best_single": 0.1},
        caption="p0",
        label="tab:p0",
    )
    assert "0.5000" in phase
    assert "0.0800" in phase
