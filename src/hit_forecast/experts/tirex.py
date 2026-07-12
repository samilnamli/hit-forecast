"""TiRex expert adapter (NXAI).

TiRex is a strong GIFT-Eval zero-shot competitor with a different architecture
family (xLSTM-based), tagged ``testdata_leakage = No``. Good for inductive-bias
diversity in the clean pool.

Forecast via the ``tirex`` package. Patch features default to the stat fallback.
"""

from __future__ import annotations

import numpy as np

from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F


@register_expert("tirex")
class TiRexExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "tirex",
        model_id: str = "NX-AI/TiRex",
        device: str = "cpu",
        feature_source: str = "stat",
        stat_patch_len: int = 24,
        stat_hidden: int = 112,
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.feature_source = feature_source
        self._stat_patch_len = stat_patch_len
        self._stat_hidden = stat_hidden
        self._proj = F.make_stat_proj(name, stat_hidden)
        self._model = None
        self._hidden = stat_hidden

    def _lazy(self):
        if self._model is not None:
            return
        from tirex import load_model

        self._model = load_model(self.model_id, device=self.device)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def batch_forecast_and_features(self, contexts, horizon):
        self._lazy()
        import torch

        contexts = np.asarray(contexts, dtype=np.float64)
        ctx_t = torch.tensor(contexts, dtype=torch.float32)
        try:
            quantiles, mean = self._model.forecast(
                context=ctx_t, prediction_length=horizon
            )
            fc = np.asarray(mean)
        except Exception:
            out = self._model.forecast(ctx_t, prediction_length=horizon)
            fc = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
            if fc.ndim == 3:  # (B, Q, H) -> median
                fc = fc[:, fc.shape[1] // 2, :]

        return [
            ExpertOutput(forecast=fc[i][:horizon],
                         patches=F.stat_patches(ctx, self._stat_patch_len, self._proj))
            for i, ctx in enumerate(contexts)
        ]
