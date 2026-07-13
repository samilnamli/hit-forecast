"""Per-window forecasting error metrics.

MASE follows Hyndman & Koehler (2006) as used in the draft (eq. 13): the mean
absolute error of the forecast, scaled by the in-sample mean absolute error of a
seasonal-naive forecaster of period ``m`` computed on the context window.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def seasonal_naive_scale(context: np.ndarray, m: int) -> float:
    """In-sample MAE of a period-``m`` seasonal-naive forecast over the context.

    ``context`` is the length-L input window ``(y_{t-L+1}, ..., y_t)``. When
    ``L <= m`` or the series is (near) constant, falls back to the m=1 scaling
    and finally to a small epsilon to avoid division by zero.
    """
    context = np.asarray(context, dtype=np.float64).ravel()
    L = context.shape[0]
    if m < 1:
        m = 1
    if L <= m:
        m = 1
    diffs = np.abs(context[m:] - context[:-m])
    scale = float(diffs.mean()) if diffs.size else 0.0
    if not np.isfinite(scale) or scale < _EPS:
        # non-seasonal fallback
        d1 = np.abs(np.diff(context))
        scale = float(d1.mean()) if d1.size else 0.0
    return max(scale, _EPS)


def mase(y_true: np.ndarray, y_pred: np.ndarray, context: np.ndarray, m: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    # Broken expert outputs / missing targets → large finite penalty (not NaN).
    if (
        y_true.size == 0
        or y_pred.size == 0
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(y_pred))
        or not np.all(np.isfinite(context))
    ):
        return 1e6
    scale = seasonal_naive_scale(context, m)
    val = float(np.mean(np.abs(y_true - y_pred)) / scale)
    if not np.isfinite(val):
        return 1e6
    return val


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        return 200.0
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < _EPS, _EPS, denom)
    val = float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))
    return val if np.isfinite(val) else 200.0


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        return 1e6
    val = float(np.mean((y_true - y_pred) ** 2))
    return val if np.isfinite(val) else 1e6


def batch_mase(
    y_true: np.ndarray, y_pred: np.ndarray, context: np.ndarray, m: int
) -> np.ndarray:
    """Vectorised MASE over a batch. Shapes: y_true/y_pred (B, H), context (B, L)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    out = np.empty(y_true.shape[0], dtype=np.float64)
    for i in range(y_true.shape[0]):
        out[i] = mase(y_true[i], y_pred[i], context[i], m)
    return out
