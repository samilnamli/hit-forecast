"""Per-window forecasting error metrics.

MASE follows Hyndman & Koehler (2006) as used in the draft (eq. 13): the mean
absolute error of the forecast, scaled by the in-sample mean absolute error of a
seasonal-naive forecaster of period ``m`` computed on the context window.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8
# Failed / non-finite forecasts — kept above the finite clip so drop filters work.
MASE_SENTINEL = 1e6
# Cap extreme but finite MASE so soft labels / L_mase are not dominated by outliers.
MASE_CLIP = 1e3


def seasonal_naive_scale(context: np.ndarray, m: int) -> float:
    """In-sample MAE of a period-``m`` seasonal-naive forecast over the context.

    ``context`` is the length-L input window ``(y_{t-L+1}, ..., y_t)``. When
    ``L <= m`` or the series is (near) constant, falls back to the m=1 scaling
    and finally to a relative epsilon to avoid division by zero.
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
    # Relative floor: absolute 1e-8 turns tiny noise into MASE ~1e8 on near-constant series.
    level = float(np.mean(np.abs(context))) if context.size else 0.0
    floor = max(_EPS, 1e-3 * max(level, _EPS))
    return max(scale, floor)


def sanitize_mase(val: float) -> float:
    """Map non-finite → sentinel; clip extreme finite values for routing stability."""
    if not np.isfinite(val):
        return MASE_SENTINEL
    if val >= MASE_SENTINEL - 1:
        return MASE_SENTINEL
    return float(min(max(val, 0.0), MASE_CLIP))


def sanitize_mase_array(mase: np.ndarray) -> np.ndarray:
    out = np.asarray(mase, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    out[bad] = MASE_SENTINEL
    finite = ~bad & (out < MASE_SENTINEL - 1)
    out[finite] = np.clip(out[finite], 0.0, MASE_CLIP)
    return out.astype(np.float32)


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
        return MASE_SENTINEL
    scale = seasonal_naive_scale(context, m)
    val = float(np.mean(np.abs(y_true - y_pred)) / scale)
    return sanitize_mase(val)


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
        return MASE_SENTINEL
    val = float(np.mean((y_true - y_pred) ** 2))
    return val if np.isfinite(val) else MASE_SENTINEL


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
