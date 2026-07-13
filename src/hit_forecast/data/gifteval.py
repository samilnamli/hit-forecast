"""GIFT-Eval loader that turns the official splits into HiT-Forecast windows.

This is a thin wrapper over ``gift_eval.data.Dataset`` (the SalesforceAIResearch
package). We deliberately reuse GIFT-Eval's own train/val/test construction so
that router training never sees the test horizons and results are comparable to
the leaderboard.

Router training windows are sliced from the *training* series (stride = H).
Evaluation windows come from GIFT-Eval's ``test_data`` (their held-out horizons).

Requires the ``gifteval`` extra and the ``GIFT_EVAL`` env var pointing at the
downloaded arrow datasets. Importing this module without gluonts installed will
raise only when a loader function is actually called.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from .windows import Window, WindowSet, channels_from_2d, sliding_windows

# GIFT-Eval seasonal period per (reconverted) frequency, for MASE scaling.
_SEASONALITY = {
    "A": 1,
    "Q": 4,
    "M": 12,
    "W": 1,
    "D": 7,
    "H": 24,
    "T": 60 * 24,  # minute-of-day is large; capped below by context length
    "S": 60,
    "U": 1,
}

# Domains per dataset stem (from GIFT-Eval dataset_properties.json).
DOMAINS = {
    "m4": "Econ/Fin",
    "electricity": "Energy",
    "ett1": "Energy",
    "ett2": "Energy",
    "solar": "Energy",
    "hospital": "Healthcare",
    "covid_deaths": "Healthcare",
    "us_births": "Healthcare",
    "saugeen": "Nature",
    "temperature_rain": "Nature",
    "kdd_cup_2018": "Nature",
    "jena_weather": "Nature",
    "car_parts": "Sales",
    "restaurant": "Sales",
    "hierarchical_sales": "Sales",
    "loop_seattle": "Transport",
    "sz_taxi": "Transport",
    "m_dense": "Transport",
    "bitbrains": "Web/CloudOps",
    "bizitobs": "Web/CloudOps",
}


def _domain_for(name: str) -> str:
    stem = name.split("/")[0]
    for key, dom in DOMAINS.items():
        if stem.startswith(key):
            return dom
    return "Unknown"


def _season_for(freq: str, context_len: int) -> int:
    base = _SEASONALITY.get(freq.upper(), 1)
    # Never let the seasonal period exceed the context (metrics.py also guards this).
    return max(1, min(base, max(1, context_len // 2)))


def _require_gift_eval():
    try:
        from gift_eval.data import Dataset, Term  # type: ignore

        return Dataset, Term
    except Exception as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "gift_eval is not importable. Install the GIFT-Eval package "
            "(`pip install -e .` inside SalesforceAIResearch/gift-eval) and set "
            "the GIFT_EVAL env var to the downloaded dataset directory."
        ) from e


def load_gifteval_windows(
    name: str,
    term: str = "short",
    split: str = "test",
    context_length: int | None = None,
    to_univariate: bool = True,
    max_windows: int | None = None,
    max_series: int | None = None,
) -> WindowSet:
    """Load one GIFT-Eval config as a :class:`WindowSet`.

    Parameters
    ----------
    name: GIFT-Eval dataset name, e.g. ``"electricity/H"`` or ``"m4_hourly"``.
    term: ``short`` | ``medium`` | ``long``.
    split: ``train`` (slice training series into windows) or ``test``
        (use GIFT-Eval held-out windows).
    context_length: override the input length ``L``. Defaults to ``2 * H``.
    """
    Dataset, Term = _require_gift_eval()
    ds = Dataset(name=name, term=Term(term), to_univariate=to_univariate)
    H = int(ds.prediction_length)
    L = int(context_length) if context_length else 2 * H
    freq = ds.freq
    m = _season_for(freq, L)
    meta_base = {
        "dataset": name,
        "config": f"{name}/{freq}/{term}",
        "freq": freq,
        "term": term,
        "domain": _domain_for(name),
    }

    windows: list[Window] = []
    if split == "train":
        series_iter = _iter_series(ds.training_dataset)
        per_series = None if max_windows is None else max(1, max_windows // 50)
        for si, arr in enumerate(series_iter):
            if max_series is not None and si >= max_series:
                break
            for ch in channels_from_2d(arr):
                windows.extend(
                    sliding_windows(ch, L=L, H=H, m=m, stride=H, meta=meta_base,
                                    max_windows=per_series)
                )
            if max_windows is not None and len(windows) >= max_windows:
                windows = windows[:max_windows]
                break
    elif split == "test":
        windows = _windows_from_test_data(ds.test_data, L=L, H=H, m=m, meta=meta_base,
                                          max_windows=max_windows)
    else:
        raise ValueError(f"Unknown split: {split!r}")

    return WindowSet(windows=windows, name=meta_base["config"])


def _iter_series(gluonts_dataset) -> Iterable[np.ndarray]:
    for entry in gluonts_dataset:
        yield np.asarray(entry["target"])


def _windows_from_test_data(test_data, L: int, H: int, m: int, meta: dict,
                            max_windows: int | None) -> list[Window]:
    """Build windows from a gluonts ``TestData`` (input/label pairs)."""
    windows: list[Window] = []
    for inp, label in test_data:
        ctx_full = np.asarray(inp["target"])
        tgt_full = np.asarray(label["target"])
        ctx_channels = channels_from_2d(ctx_full)
        tgt_channels = channels_from_2d(tgt_full)
        for ctx, tgt in zip(ctx_channels, tgt_channels):
            if ctx.shape[-1] < L or tgt.shape[-1] < H:
                continue
            ctx = ctx[-L:]
            tgt = tgt[:H]
            if not (np.all(np.isfinite(ctx)) and np.all(np.isfinite(tgt))):
                continue
            windows.append(Window(context=ctx.astype(np.float64),
                                  target=tgt.astype(np.float64), m=m, meta=dict(meta)))
            if max_windows is not None and len(windows) >= max_windows:
                return windows
    return windows


def gifteval_available() -> bool:
    try:
        _require_gift_eval()
        return bool(os.getenv("GIFT_EVAL"))
    except Exception:
        return False
