"""GIFT-Eval loader that turns the official splits into HiT-Forecast windows.

This is a thin wrapper over ``gift_eval.data.Dataset`` (the SalesforceAIResearch
package). We deliberately reuse GIFT-Eval's own train/val/test construction so
that router training never sees the test horizons and results are comparable to
the leaderboard.

Router training windows are sliced from the *training* series (stride = H).
Evaluation windows come from GIFT-Eval's ``test_data`` (their held-out horizons).

Important: we always load with ``to_univariate=False``. Gift-Eval's own
``MultivariateToUnivariate`` walks a 1-D target as a Python list and yields
*scalar* targets, which then crash GluonTS ``split`` with
``IndexError: tuple index out of range``. Multivariate series are flattened to
channels by our own :func:`channels_from_2d` instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from ..utils.logging import get_logger
from .windows import Window, WindowSet, channels_from_2d, sliding_windows

_log = get_logger(__name__)

# GIFT-Eval seasonal period per (reconverted) frequency, for MASE scaling.
_SEASONALITY = {
    "A": 1,
    "Q": 4,
    "M": 12,
    "W": 1,
    "D": 7,
    "H": 24,
    "T": 60 * 24,
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
    "LOOP_SEATTLE": "Transport",
    "loop_seattle": "Transport",
    "SZ_TAXI": "Transport",
    "sz_taxi": "Transport",
    "M_DENSE": "Transport",
    "m_dense": "Transport",
    "bitbrains": "Web/CloudOps",
    "bizitobs": "Web/CloudOps",
}


def _domain_for(name: str) -> str:
    stem = name.split("/")[0]
    for key, dom in DOMAINS.items():
        if stem.startswith(key) or stem.upper().startswith(key.upper()):
            return dom
    return "Unknown"


def _season_for(freq: str, context_len: int) -> int:
    base = _SEASONALITY.get(freq.upper(), 1)
    return max(1, min(base, max(1, context_len // 2)))


def _ensure_gift_eval_env() -> str:
    """Resolve and export ``GIFT_EVAL``; raise a clear error if missing."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    path = os.getenv("GIFT_EVAL")
    if not path:
        for candidate in (
            Path.cwd() / "data" / "gifteval",
            Path(__file__).resolve().parents[3] / "data" / "gifteval",
        ):
            if candidate.is_dir() and any(candidate.iterdir()):
                path = str(candidate.resolve())
                os.environ["GIFT_EVAL"] = path
                break
    if not path:
        raise EnvironmentError(
            "GIFT_EVAL is not set and no local data/gifteval/ was found.\n"
            "Download the benchmark first:\n"
            "  bash scripts/download_gifteval.sh\n"
            "Then either `source .env` or:\n"
            "  export GIFT_EVAL=$PWD/data/gifteval"
        )
    if not Path(path).is_dir():
        raise EnvironmentError(
            f"GIFT_EVAL={path!r} does not exist or is not a directory. "
            "Re-run: bash scripts/download_gifteval.sh"
        )
    return path


def _require_gift_eval():
    _ensure_gift_eval_env()
    try:
        from gift_eval.data import Dataset, Term  # type: ignore

        return Dataset, Term
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "gift_eval is not importable. Install with:\n"
            "  pip install git+https://github.com/SalesforceAIResearch/gift-eval.git\n"
            "and set GIFT_EVAL to the downloaded dataset directory "
            "(bash scripts/download_gifteval.sh)."
        ) from e


def load_gifteval_windows(
    name: str,
    term: str = "short",
    split: str = "test",
    context_length: int | None = None,
    to_univariate: bool | None = None,
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
    to_univariate: ignored (kept for config compatibility). Always ``False``
        toward Gift-Eval; we channel-split ourselves.
    """
    if to_univariate:
        _log.warning(
            "to_univariate=True is unsafe with gift_eval on univariate series; "
            "forcing False and channel-splitting locally for %s",
            name,
        )
    Dataset, Term = _require_gift_eval()
    ds = Dataset(name=name, term=Term(term), to_univariate=False)
    H = int(ds.prediction_length)
    freq = ds.freq
    # Prefer L=2H; for short GiftEval series (e.g. car_parts monthly) fall back.
    L_candidates = []
    if context_length is not None:
        L_candidates.append(int(context_length))
    L_candidates.extend([2 * H, H, max(H // 2, 8), max(4, H // 4)])
    # unique, preserve order
    seen = set()
    L_candidates = [L for L in L_candidates if not (L in seen or seen.add(L))]

    meta_base = {
        "dataset": name,
        "config": f"{name}/{freq}/{term}",
        "freq": freq,
        "term": term,
        "domain": _domain_for(name),
    }

    windows: list[Window] = []
    used_L = L_candidates[0]
    last_err = None
    for L in L_candidates:
        m = _season_for(freq, L)
        try:
            if split == "train":
                windows = _windows_from_train(
                    ds.training_dataset,
                    L=L,
                    H=H,
                    m=m,
                    meta=meta_base,
                    max_windows=max_windows,
                    max_series=max_series,
                )
            elif split == "test":
                windows = _windows_from_test_data(
                    ds.test_data, L=L, H=H, m=m, meta=meta_base, max_windows=max_windows
                )
            else:
                raise ValueError(f"Unknown split: {split!r}")
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("GiftEval %s split=%s L=%d failed (%s); trying shorter L",
                         name, split, L, e)
            windows = []
        if windows:
            used_L = L
            break

    if not windows:
        msg = (
            f"No windows produced for GiftEval config {name!r} split={split!r} "
            f"(tried L in {L_candidates}, H={H}). "
            f"Check that {os.getenv('GIFT_EVAL')}/{name} exists and series are long enough."
        )
        if last_err is not None:
            raise RuntimeError(msg) from last_err
        raise RuntimeError(msg)

    if used_L != L_candidates[0]:
        _log.info("GiftEval %s split=%s: using shortened context L=%d (H=%d)",
                  name, split, used_L, H)
    return WindowSet(windows=windows, name=meta_base["config"])


def _windows_from_train(
    training_dataset,
    L: int,
    H: int,
    m: int,
    meta: dict,
    max_windows: int | None,
    max_series: int | None,
) -> list[Window]:
    windows: list[Window] = []
    per_series = None if max_windows is None else max(1, max_windows // 50)
    for si, arr in enumerate(_iter_series(training_dataset)):
        if max_series is not None and si >= max_series:
            break
        if arr is None or np.asarray(arr).size == 0:
            continue
        for ch in channels_from_2d(arr):
            ch = np.asarray(ch).ravel()
            if ch.size < L + H:
                continue
            windows.extend(
                sliding_windows(
                    ch,
                    L=L,
                    H=H,
                    m=m,
                    stride=H,
                    meta=meta,
                    max_windows=per_series,
                )
            )
        if max_windows is not None and len(windows) >= max_windows:
            return windows[:max_windows]
    return windows


def _as_target_array(target) -> np.ndarray | None:
    """Coerce a gluonts target to a 1-D or 2-D float array; skip broken entries."""
    arr = np.asarray(target)
    if arr.ndim == 0:
        return None
    if arr.dtype == object:
        # list-of-arrays / ragged: try stacking if possible
        try:
            arr = np.stack([np.asarray(x, dtype=np.float64) for x in arr])
        except Exception:
            return None
    if arr.size == 0:
        return None
    return arr.astype(np.float64, copy=False)


def _iter_series(gluonts_dataset) -> Iterable[np.ndarray]:
    for entry in gluonts_dataset:
        try:
            arr = _as_target_array(entry["target"])
        except Exception as e:  # noqa: BLE001
            _log.warning("Skipping broken training entry: %s", e)
            continue
        if arr is None:
            continue
        yield arr


def _windows_from_test_data(
    test_data, L: int, H: int, m: int, meta: dict, max_windows: int | None
) -> list[Window]:
    """Build windows from a gluonts ``TestData`` (input/label pairs)."""
    windows: list[Window] = []
    for inp, label in test_data:
        try:
            ctx_full = _as_target_array(inp["target"])
            tgt_full = _as_target_array(label["target"])
        except Exception:
            continue
        if ctx_full is None or tgt_full is None:
            continue
        ctx_channels = channels_from_2d(ctx_full)
        tgt_channels = channels_from_2d(tgt_full)
        for ctx, tgt in zip(ctx_channels, tgt_channels):
            ctx = np.asarray(ctx).ravel()
            tgt = np.asarray(tgt).ravel()
            if ctx.shape[0] < L or tgt.shape[0] < H:
                continue
            ctx = ctx[-L:]
            tgt = tgt[:H]
            if not (np.all(np.isfinite(ctx)) and np.all(np.isfinite(tgt))):
                continue
            windows.append(
                Window(
                    context=ctx.astype(np.float64),
                    target=tgt.astype(np.float64),
                    m=m,
                    meta=dict(meta),
                )
            )
            if max_windows is not None and len(windows) >= max_windows:
                return windows
    return windows


def gifteval_available() -> bool:
    try:
        _require_gift_eval()
        return bool(os.getenv("GIFT_EVAL"))
    except Exception:
        return False
