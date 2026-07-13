"""TimesFM expert adapter (Google).

Uses the ``timesfm`` package. Prefer ``timesfm-2.5`` which is tagged
``testdata_leakage = No`` / zero-shot on GIFT-Eval; TimesFM 1.0/2.0 leak.

TimesFM 2.5 API (current)::

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
    model.compile(timesfm.ForecastConfig(max_context=..., max_horizon=..., ...))
    point, quantiles = model.forecast(horizon=H, inputs=[...])

Patch features default to the deterministic stat fallback (decoder-only
per-position hidden states can be wired later via a forward hook).
"""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger
from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F

_log = get_logger(__name__)


@register_expert("timesfm")
class TimesFMExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "timesfm-2.5",
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "cpu",
        context_length: int = 512,
        max_horizon: int = 256,
        per_core_batch_size: int = 64,
        feature_source: str = "stat",
        stat_patch_len: int = 32,
        stat_hidden: int = 128,
        torch_compile: bool = False,
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.per_core_batch_size = per_core_batch_size
        self.feature_source = feature_source
        self.torch_compile = torch_compile
        self._stat_patch_len = stat_patch_len
        self._stat_hidden = stat_hidden
        self._proj = F.make_stat_proj(name, stat_hidden)
        self._model = None
        self._hidden = stat_hidden
        self._api = None  # "2p5" | "legacy"

    def _wants_2p5(self) -> bool:
        mid = (self.model_id or "").lower()
        name = (self.name or "").lower()
        return "2.5" in mid or "2p5" in mid or "2.5" in name

    def _lazy(self, horizon: int | None = None):
        if self._model is not None:
            # Re-compile if a longer horizon arrives than we planned for.
            if (
                self._api == "2p5"
                and horizon is not None
                and horizon > self.max_horizon
            ):
                self.max_horizon = int(horizon)
                self._compile_2p5()
            return
        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")
        if horizon is not None:
            self.max_horizon = max(self.max_horizon, int(horizon))

        # --- TimesFM 2.5 path (current PyPI timesfm) ---
        if self._wants_2p5() and hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            try:
                self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    self.model_id, torch_compile=self.torch_compile
                )
            except TypeError:
                self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    self.model_id
                )
            self._api = "2p5"
            self._compile_2p5()
            _log.info("Loaded TimesFM 2.5 (%s) on %s", self.model_id, self.device)
            return

        # --- Legacy TimesFM 1.x / 2.0 (needs timesfm<=1.3 or v1 extra) ---
        TimesFm = getattr(timesfm, "TimesFm", None)
        TimesFmHparams = getattr(timesfm, "TimesFmHparams", None)
        TimesFmCheckpoint = getattr(timesfm, "TimesFmCheckpoint", None)
        if TimesFm is None:
            try:
                from timesfm import timesfm as _v1  # type: ignore

                TimesFm = getattr(_v1, "TimesFm", None)
                TimesFmHparams = getattr(_v1, "TimesFmHparams", None)
                TimesFmCheckpoint = getattr(_v1, "TimesFmCheckpoint", None)
            except Exception:
                pass
        if TimesFm is not None and TimesFmHparams is not None and TimesFmCheckpoint is not None:
            backend = "gpu" if "cuda" in str(self.device) else "cpu"
            hp = TimesFmHparams(
                backend=backend,
                context_len=self.context_length,
                horizon_len=max(int(self.max_horizon), 128),
            )
            ckpt = TimesFmCheckpoint(huggingface_repo_id=self.model_id)
            self._model = TimesFm(hparams=hp, checkpoint=ckpt)
            self._api = "legacy"
            _log.info("Loaded legacy TimesFM (%s)", self.model_id)
            return

        raise ImportError(
            f"Cannot load TimesFM checkpoint '{self.model_id}'. "
            "TimesFM 2.5 needs a current `timesfm[torch]` with TimesFM_2p5_200M_torch. "
            "TimesFM 1.0/2.0 need the archived v1 API (`pip install 'timesfm==1.3.0'` "
            "in a separate env, or swap the contaminated-pool expert). "
            f"Detected wants_2p5={self._wants_2p5()}."
        )

    def _compile_2p5(self):
        import timesfm

        cfg = timesfm.ForecastConfig(
            max_context=int(self.context_length),
            max_horizon=int(self.max_horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,  # GiftEval has signed / mixed-sign series
            fix_quantile_crossing=True,
            per_core_batch_size=int(self.per_core_batch_size),
        )
        self._model.compile(cfg)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def batch_forecast_and_features(self, contexts, horizon):
        self._lazy(horizon=horizon)
        contexts = np.asarray(contexts, dtype=np.float64)
        inputs = [np.asarray(c, dtype=np.float32) for c in contexts]

        if self._api == "2p5":
            point, _ = self._model.forecast(horizon=int(horizon), inputs=inputs)
            fc = np.asarray(point)
        else:
            try:
                point, _ = self._model.forecast(horizon=int(horizon), inputs=inputs)
                fc = np.asarray(point)
            except TypeError:
                point, _ = self._model.forecast(inputs, freq=[0] * len(inputs))
                fc = np.asarray(point)[:, :horizon]

        if fc.ndim == 1:
            fc = fc[None, :]
        return [
            ExpertOutput(
                forecast=np.asarray(fc[i][:horizon], dtype=np.float64),
                patches=F.stat_patches(ctx, self._stat_patch_len, self._proj),
            )
            for i, ctx in enumerate(contexts)
        ]
