"""TimesFM expert adapter (Google).

Uses the ``timesfm`` package. Prefer ``timesfm-2.5`` which is tagged
``testdata_leakage = No`` / zero-shot on GIFT-Eval; TimesFM 1.0/2.0 leak.

Forecast path is the documented ``forecast`` API. Patch features default to the
stat fallback (decoder-only per-position hidden states can be wired via a
forward hook on the GPU host; see ``_features``).
"""

from __future__ import annotations

import numpy as np

from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F


@register_expert("timesfm")
class TimesFMExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "timesfm-2.5",
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "cpu",
        context_length: int = 512,
        feature_source: str = "stat",
        stat_patch_len: int = 32,
        stat_hidden: int = 128,
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.context_length = context_length
        self.feature_source = feature_source
        self._stat_patch_len = stat_patch_len
        self._stat_hidden = stat_hidden
        self._proj = F.make_stat_proj(name, stat_hidden)
        self._model = None
        self._hidden = stat_hidden

    def _lazy(self):
        if self._model is not None:
            return
        import timesfm

        # API differs across timesfm versions; try the 2.x class then 1.x hparams.
        try:
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
            self._model.compile()
        except Exception:
            hp = timesfm.TimesFmHparams(backend="gpu" if "cuda" in self.device else "cpu",
                                        context_len=self.context_length)
            ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_id)
            self._model = timesfm.TimesFm(hparams=hp, checkpoint=ckpt)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def batch_forecast_and_features(self, contexts, horizon):
        self._lazy()
        contexts = np.asarray(contexts, dtype=np.float64)
        inputs = [c.astype(np.float32) for c in contexts]
        try:
            point, _ = self._model.forecast(horizon=horizon, inputs=inputs)
            fc = np.asarray(point)
        except TypeError:
            point, _ = self._model.forecast(inputs, freq=[0] * len(inputs))
            fc = np.asarray(point)[:, :horizon]

        return [
            ExpertOutput(forecast=fc[i][:horizon],
                         patches=F.stat_patches(ctx, self._stat_patch_len, self._proj))
            for i, ctx in enumerate(contexts)
        ]

    def _features(self, ctx):  # placeholder for verified hidden-state hook
        return F.stat_patches(ctx, self._stat_patch_len, self._proj)
