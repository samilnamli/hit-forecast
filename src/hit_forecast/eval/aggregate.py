"""GIFT-Eval-style aggregation of per-window results by metadata dimensions."""

from __future__ import annotations

import numpy as np

from ..data.metrics import mase as _mase, smape as _smape, mse as _mse
from ..models.dataset import CombinedData


def _per_window_metric(forecast_NH: np.ndarray, data: CombinedData, metric: str) -> np.ndarray:
    N = forecast_NH.shape[0]
    fn = {"MASE": lambda i: _mase(data.targets[i], forecast_NH[i], data.contexts[i], int(data.m[i])),
          "sMAPE": lambda i: _smape(data.targets[i], forecast_NH[i]),
          "MSE": lambda i: _mse(data.targets[i], forecast_NH[i])}[metric]
    return np.array([fn(i) for i in range(N)])


def aggregate_by(
    forecast_NH: np.ndarray,
    data: CombinedData,
    dim: str = "domain",
    metric: str = "MASE",
) -> dict[str, float]:
    """Mean ``metric`` grouped by ``data.window_meta[i][dim]``."""
    vals = _per_window_metric(forecast_NH, data, metric)
    groups: dict[str, list[float]] = {}
    for i, wm in enumerate(data.window_meta):
        key = str(wm.get(dim, "all"))
        groups.setdefault(key, []).append(float(vals[i]))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}
