"""Baselines, routed-model inference and metric computation on cached data.

All strategies operate on the cached forecasts ``(N, K, H)`` and per-window MASE
``(N, K)`` so evaluation is FM-free. Metrics recomputed uniformly from the
resulting per-window forecast vs the cached targets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.metrics import mase as _mase, smape as _smape, mse as _mse
from ..models.dataset import CombinedData, RouterDataset, collate_router


def compute_metrics(forecast_NH: np.ndarray, data: CombinedData) -> dict:
    """Mean MASE / sMAPE / MSE for a per-window forecast ``(N, H)``."""
    if data.targets is None or data.contexts is None:
        raise ValueError("Cache lacks aligned targets/contexts (mixed H or L).")
    N = forecast_NH.shape[0]
    ma = np.empty(N)
    sm = np.empty(N)
    ms = np.empty(N)
    for i in range(N):
        t = data.targets[i]
        f = forecast_NH[i]
        ma[i] = _mase(t, f, data.contexts[i], int(data.m[i]))
        sm[i] = _smape(t, f)
        ms[i] = _mse(t, f)
    return {"MASE": float(ma.mean()), "sMAPE": float(sm.mean()), "MSE": float(ms.mean())}


def _gather(forecasts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.take_along_axis(forecasts, idx[:, None, None], axis=1)[:, 0, :]


def model_routing(model, data: CombinedData, device: str = "cpu",
                  batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return routing weights ``w (N, K)`` and argmax expert index ``(N,)``."""
    ds = RouterDataset(data)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_router)
    model = model.to(device).eval()
    ws = np.zeros((data.N, data.K), dtype=np.float64)
    with torch.no_grad():
        for feats, masks, _, idx in dl:
            feats = [f.to(device) for f in feats]
            masks = [m.to(device) for m in masks]
            w = torch.softmax(model(feats, masks), dim=-1).cpu().numpy()
            ws[idx.numpy()] = w
    return ws, ws.argmax(axis=1)


def all_baseline_forecasts(
    data: CombinedData,
    train_mase_mean: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-window forecasts ``(N, H)`` for every non-learned strategy + oracle."""
    rng = np.random.default_rng(seed)
    N, K = data.N, data.K
    fc = data.forecasts
    out: dict[str, np.ndarray] = {}

    for k, name in enumerate(data.expert_names):
        out[f"single:{name}"] = fc[:, k, :]

    out["ensemble:mean"] = fc.mean(axis=1)

    if train_mase_mean is None:
        train_mase_mean = data.mase.mean(axis=0)
    inv = 1.0 / np.clip(train_mase_mean, 1e-8, None)
    w_ens = inv / inv.sum()
    out["ensemble:weighted"] = np.einsum("nkh,k->nh", fc, w_ens)

    # BMA-style weighting: softmax over negative mean MASE
    bma = np.exp(-(train_mase_mean - train_mase_mean.min()))
    bma = bma / bma.sum()
    out["ensemble:bma"] = np.einsum("nkh,k->nh", fc, bma)

    rand_idx = rng.integers(0, K, size=N)
    out["routing:random"] = _gather(fc, rand_idx)

    wr_idx = rng.choice(K, size=N, p=w_ens)
    out["routing:weighted_random"] = _gather(fc, wr_idx)

    out["oracle"] = _gather(fc, data.mase.argmin(axis=1))
    return out


def evaluate_all(
    data: CombinedData,
    models: dict[str, object] | None = None,
    train_mase_mean: np.ndarray | None = None,
    device: str = "cpu",
    seed: int = 0,
    soft_mode: bool = True,
) -> dict[str, dict]:
    """Evaluate baselines + provided routers. ``models`` maps label -> router module."""
    results: dict[str, dict] = {}
    forecasts = all_baseline_forecasts(data, train_mase_mean, seed)
    for name, f in forecasts.items():
        results[name] = compute_metrics(f, data)

    if models:
        for label, model in models.items():
            w, jn = model_routing(model, data, device)
            hard_fc = _gather(data.forecasts, jn)
            results[f"{label}:hard"] = compute_metrics(hard_fc, data)
            results[f"{label}:hard"]["selection_acc"] = _selection_acc(jn, data)
            if soft_mode:
                soft_fc = np.einsum("nkh,nk->nh", data.forecasts, w)
                results[f"{label}:soft"] = compute_metrics(soft_fc, data)
    return results


def _selection_acc(jn: np.ndarray, data: CombinedData) -> float:
    oracle = data.mase.argmin(axis=1)
    return float((jn == oracle).mean())
