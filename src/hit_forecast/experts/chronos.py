"""Chronos / Chronos-Bolt / Chronos-2 expert adapter.

Forecast path uses the official ``chronos`` package (stable). Patch features are
taken from the frozen encoder's last hidden state when available, else the
deterministic stat fallback.

GIFT-Eval leakage note: classic Chronos-T5 and Chronos-Bolt are tagged
``testdata_leakage = Yes``. Prefer ``amazon/chronos-2`` (No) or
``autogluon/chronos-2-synth`` (strict zero-shot) for the clean pool.
"""

from __future__ import annotations

import numpy as np

from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F


def _to_numpy_2d(x, horizon: int) -> np.ndarray:
    """Coerce Chronos outputs to (B, H) float64."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3:
        # (B, H, Q) or (B, Q, H) — take first quantile / median slot already chosen upstream
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :]
        else:
            arr = arr[:, :, 0]
    return arr[:, :horizon]


def extract_chronos_point_forecast(pipe, contexts_t, horizon: int) -> np.ndarray:
    """Median / mean point forecast compatible with Chronos-1, Bolt, and Chronos-2.

    Chronos-2 uses ``inputs=`` and returns list tensors; Chronos-1/Bolt use
    ``context=`` and return batched arrays. A naive ``predict`` + ``median(axis=0)``
    on Chronos-2's ``(1, Q, H)`` samples incorrectly selects quantile 0.1.
    """
    # --- Chronos-2 style ---
    try:
        q, mean = pipe.predict_quantiles(
            inputs=contexts_t, prediction_length=horizon, quantile_levels=[0.5]
        )
        if isinstance(mean, (list, tuple)):
            # list of (n_variates, H) — univariate → first variate
            rows = []
            for m in mean:
                a = np.asarray(
                    m.detach().float().cpu().numpy() if hasattr(m, "detach") else m,
                    dtype=np.float64,
                )
                rows.append(a[0] if a.ndim == 2 else a.ravel())
            return np.stack(rows, axis=0)[:, :horizon]
        if isinstance(q, (list, tuple)):
            rows = []
            for qi in q:
                a = np.asarray(
                    qi.detach().float().cpu().numpy() if hasattr(qi, "detach") else qi,
                    dtype=np.float64,
                )
                # (n_variates, H, Q) with Q=1 → [0, :, 0]
                if a.ndim == 3:
                    rows.append(a[0, :, 0])
                else:
                    rows.append(a.ravel())
            return np.stack(rows, axis=0)[:, :horizon]
        return _to_numpy_2d(q, horizon)
    except TypeError:
        pass
    except Exception:
        pass

    # --- Chronos-1 / Bolt style ---
    try:
        q, _ = pipe.predict_quantiles(
            context=contexts_t, prediction_length=horizon, quantile_levels=[0.5]
        )
        return _to_numpy_2d(q, horizon)
    except Exception:
        pass

    # Last resort: sample paths and take median over the sample axis only.
    samples = pipe.predict(contexts_t, prediction_length=horizon)
    if isinstance(samples, (list, tuple)):
        rows = []
        for s in samples:
            a = np.asarray(
                s.detach().float().cpu().numpy() if hasattr(s, "detach") else s,
                dtype=np.float64,
            )
            # Chronos-2: (n_variates, Q, H) → median over Q
            if a.ndim == 3:
                rows.append(np.median(a[0], axis=0))
            elif a.ndim == 2:
                # (Q, H) or (n_samples, H)
                rows.append(np.median(a, axis=0))
            else:
                rows.append(a.ravel())
        return np.stack(rows, axis=0)[:, :horizon]
    a = np.asarray(
        samples.detach().float().cpu().numpy() if hasattr(samples, "detach") else samples,
        dtype=np.float64,
    )
    if a.ndim == 3:
        # (B, n_samples, H)
        return np.median(a, axis=1)[:, :horizon]
    return _to_numpy_2d(a, horizon)


@register_expert("chronos")
class ChronosExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "chronos-2",
        model_id: str = "amazon/chronos-2",
        device: str = "cpu",
        dtype: str = "float32",
        feature_source: str = "auto",  # auto | encoder | stat
        stat_patch_len: int = 16,
        stat_hidden: int = 64,
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.dtype = dtype
        self.feature_source = feature_source
        self._stat_patch_len = stat_patch_len
        self._stat_hidden = stat_hidden
        self._proj = F.make_stat_proj(name, stat_hidden)
        self._pipe = None
        self._hidden = stat_hidden
        self._encoder_ok = feature_source in ("auto", "encoder")
        self._extractor = None
        self._max_patches = 64

    def _lazy(self):
        if self._pipe is not None:
            return
        import torch
        from chronos import BaseChronosPipeline

        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[self.dtype]
        self._pipe = BaseChronosPipeline.from_pretrained(
            self.model_id, device_map=self.device, torch_dtype=dt
        )
        if self._encoder_ok:
            inner = getattr(self._pipe, "model", None)
            try:
                self._extractor = F.HookedEncoderExtractor(
                    ("encoder", "backbone", "transformer")
                ).attach(inner)
            except Exception as e:  # noqa: BLE001
                if self.feature_source == "encoder":
                    raise
                F.warn_fallback(self.name, e)
                self._encoder_ok = False

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def batch_forecast_and_features(self, contexts, horizon):
        self._lazy()
        import torch

        contexts = np.asarray(contexts, dtype=np.float64)
        ctx_t = [torch.tensor(c, dtype=torch.float32) for c in contexts]

        hidden = None
        if self._encoder_ok and self._extractor is not None:
            try:
                fc = extract_chronos_point_forecast(self._pipe, ctx_t, int(horizon))
                cap = self._extractor.pop(len(contexts))
                if cap is not None:
                    hidden = cap.float().cpu().numpy()
                    self._hidden = int(hidden.shape[-1])
                else:
                    raise RuntimeError("hook capture shape mismatch")
            except Exception as e:  # noqa: BLE001
                if self.feature_source == "encoder":
                    raise
                F.warn_fallback(self.name, e)
                self._encoder_ok = False
                hidden = None
                fc = extract_chronos_point_forecast(self._pipe, ctx_t, int(horizon))
        else:
            fc = extract_chronos_point_forecast(self._pipe, ctx_t, int(horizon))

        outs = []
        for i, ctx in enumerate(contexts):
            if hidden is not None:
                patches = F.pool_tokens(hidden[i], self._max_patches)
            else:
                patches = F.stat_patches(ctx, self._stat_patch_len, self._proj)
            outs.append(ExpertOutput(forecast=fc[i][:horizon], patches=patches))
        return outs
