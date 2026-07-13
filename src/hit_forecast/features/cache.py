"""Cache expert forecasts, patch features and per-window MASE labels.

This is the expensive, GPU-bound stage (draft Algorithm 2, lines 2-7). Once
cached, router training and evaluation are FM-free and fast. One cache shard is
written per :class:`~hit_forecast.data.windows.WindowSet`.

Layout::

    <cache_root>/<sanitized_name>/
        meta.json                 # names, dims, patch counts, N, L, H
        arrays.npz                # contexts, targets, m, mase (N,K), forecasts (N,K,H)
        feats_<expert>.npy        # (N, T_k, D_k) padded patch features
        lens_<expert>.npy         # (N,) valid patch counts (for masking)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.metrics import mase as _mase
from ..data.windows import WindowSet
from ..utils.logging import get_logger

_log = get_logger(__name__)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_cache(
    windowset: WindowSet,
    experts: list,
    out_dir: str | Path,
    batch_size: int = 64,
    overwrite: bool = False,
) -> "FeatureCache":
    out_dir = Path(out_dir) / _sanitize(windowset.name)
    if out_dir.exists() and not overwrite and (out_dir / "meta.json").exists():
        _log.info("Cache exists, loading: %s", out_dir)
        return load_cache(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(windowset)
    if N == 0:
        raise ValueError(f"WindowSet {windowset.name!r} is empty.")
    L, H = windowset.L, windowset.H
    K = len(experts)
    expert_names = [e.name for e in experts]

    contexts = np.stack([w.context for w in windowset]).astype(np.float32)
    targets = np.stack([w.target for w in windowset]).astype(np.float32)
    ms = np.array([w.m for w in windowset], dtype=np.int64)

    forecasts = np.zeros((N, K, H), dtype=np.float32)
    mase = np.zeros((N, K), dtype=np.float32)
    # collect variable patch sequences per expert
    feats: list[list[np.ndarray]] = [[] for _ in range(K)]

    for k, expert in enumerate(experts):
        try:
            # Warm-up / compile on a tiny batch so API mismatches fail before the loop.
            _ = expert.batch_forecast_and_features(contexts[:1].astype(np.float64), H)
            _log.info("Expert ready: %s (D=%d)", expert.name, expert.hidden_dim)
        except Exception as e:
            raise RuntimeError(
                f"Expert {expert.name!r} failed to initialize/forecast "
                f"(horizon={H}). Check the adapter / package version. Root cause: {e}"
            ) from e

    for start in tqdm(range(0, N, batch_size), desc=f"cache:{windowset.name}"):
        end = min(start + batch_size, N)
        ctx_batch = contexts[start:end].astype(np.float64)
        for k, expert in enumerate(experts):
            outs = expert.batch_forecast_and_features(ctx_batch, H)
            for j, o in enumerate(outs):
                idx = start + j
                fc = np.asarray(o.forecast, dtype=np.float32).ravel()[:H]
                if fc.shape[0] < H:
                    fc = np.pad(fc, (0, H - fc.shape[0]), mode="edge")
                # Experts occasionally emit NaN/Inf on messy GiftEval series.
                if not np.all(np.isfinite(fc)):
                    finite = fc[np.isfinite(fc)]
                    fill = float(finite[-1]) if finite.size else float(
                        contexts[idx][-1] if np.isfinite(contexts[idx][-1]) else 0.0
                    )
                    fc = np.nan_to_num(fc, nan=fill, posinf=fill, neginf=fill)
                patches = np.asarray(o.patches, dtype=np.float32)
                patches = np.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
                forecasts[idx, k] = fc
                mase[idx, k] = _mase(targets[idx], fc, contexts[idx], int(ms[idx]))
                feats[k].append(patches)

    # Final sanitisation: no NaNs allowed into router training.
    mase = np.nan_to_num(mase, nan=1e6, posinf=1e6, neginf=1e6).astype(np.float32)
    forecasts = np.nan_to_num(forecasts, nan=0.0, posinf=0.0, neginf=0.0)
    n_bad = int(np.any(~np.isfinite(mase), axis=1).sum())  # should be 0 now
    if n_bad:
        _log.warning("%d windows still had non-finite MASE after sanitisation", n_bad)

    meta = {
        "name": windowset.name,
        "N": N,
        "L": L,
        "H": H,
        "K": K,
        "expert_names": expert_names,
        "hidden_dims": [],
        "patch_counts": [],
    }
    for k, name in enumerate(expert_names):
        arr, lens = _pad_stack(feats[k])
        np.save(out_dir / f"feats_{_sanitize(name)}.npy", arr)
        np.save(out_dir / f"lens_{_sanitize(name)}.npy", lens)
        meta["hidden_dims"].append(int(arr.shape[2]))
        meta["patch_counts"].append(int(arr.shape[1]))

    np.savez_compressed(
        out_dir / "arrays.npz",
        contexts=contexts,
        targets=targets,
        m=ms,
        mase=mase,
        forecasts=forecasts,
    )
    # store lightweight per-window meta for aggregation
    (out_dir / "window_meta.json").write_text(
        json.dumps([w.meta for w in windowset])
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    _log.info("Wrote cache: %s (N=%d, K=%d)", out_dir, N, K)
    return load_cache(out_dir)


def _pad_stack(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    D = seqs[0].shape[1]
    Tmax = max(s.shape[0] for s in seqs)
    N = len(seqs)
    out = np.zeros((N, Tmax, D), dtype=np.float32)
    lens = np.zeros(N, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        lens[i] = s.shape[0]
    return out, lens


@dataclass
class FeatureCache:
    """In-memory view of a cache shard used by the router and evaluators."""

    dir: Path
    meta: dict
    contexts: np.ndarray
    targets: np.ndarray
    m: np.ndarray
    mase: np.ndarray  # (N, K)
    forecasts: np.ndarray  # (N, K, H)
    feats: list[np.ndarray]  # per expert (N, T_k, D_k)
    lens: list[np.ndarray]  # per expert (N,)
    window_meta: list[dict]

    @property
    def N(self) -> int:
        return int(self.meta["N"])

    @property
    def K(self) -> int:
        return int(self.meta["K"])

    @property
    def expert_names(self) -> list[str]:
        return list(self.meta["expert_names"])

    @property
    def hidden_dims(self) -> list[int]:
        return list(self.meta["hidden_dims"])

    @property
    def oracle_idx(self) -> np.ndarray:
        return self.mase.argmin(axis=1)


def load_cache(cache_dir: str | Path) -> FeatureCache:
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    arrays = np.load(cache_dir / "arrays.npz")
    feats, lens = [], []
    for name in meta["expert_names"]:
        s = _sanitize(name)
        f = np.load(cache_dir / f"feats_{s}.npy")
        feats.append(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0))
        lens.append(np.load(cache_dir / f"lens_{s}.npy"))
    wm_path = cache_dir / "window_meta.json"
    window_meta = json.loads(wm_path.read_text()) if wm_path.exists() else []
    mase = np.nan_to_num(arrays["mase"], nan=1e6, posinf=1e6, neginf=1e6).astype(np.float32)
    forecasts = np.nan_to_num(arrays["forecasts"], nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureCache(
        dir=cache_dir,
        meta=meta,
        contexts=arrays["contexts"],
        targets=arrays["targets"],
        m=arrays["m"],
        mase=mase,
        forecasts=forecasts,
        feats=feats,
        lens=lens,
        window_meta=window_meta,
    )
