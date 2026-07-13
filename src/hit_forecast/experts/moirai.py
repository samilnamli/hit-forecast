"""Moirai / Moirai-2 expert adapter (Salesforce uni2ts).

Moirai is a masked-encoder transformer and exposes patch-level encoder states
naturally. Tagged ``testdata_leakage = No`` on GIFT-Eval -> good clean-pool
member.
"""

from __future__ import annotations

import numpy as np

from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F


@register_expert("moirai")
class MoiraiExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "moirai-base",
        model_id: str = "Salesforce/moirai-1.1-R-base",
        device: str = "cpu",
        patch_size: int | str = "auto",
        num_samples: int = 100,
        feature_source: str = "auto",
        stat_patch_len: int = 32,
        stat_hidden: int = 96,
        context_length: int | None = None,
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.feature_source = feature_source
        self._stat_patch_len = stat_patch_len
        self._stat_hidden = stat_hidden
        self._proj = F.make_stat_proj(name, stat_hidden)
        self._module = None
        self._hidden = stat_hidden
        self._ctx_len = context_length
        self._encoder_ok = feature_source in ("auto", "encoder")
        self._extractor = None
        self._max_patches = 64

    def _lazy(self):
        if self._module is not None:
            return
        from uni2ts.model.moirai import MoiraiModule

        self._module = MoiraiModule.from_pretrained(self.model_id).to(self.device).eval()
        d = getattr(getattr(self._module, "hparams", None), "d_model", None) or getattr(
            getattr(self._module, "module", self._module), "d_model", None
        )
        if d and self.feature_source != "stat":
            self._hidden = int(d)
        if self._encoder_ok:
            try:
                self._extractor = F.HookedEncoderExtractor(
                    ("encoder", "layers", "transformer")
                ).attach(self._module)
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
        from uni2ts.model.moirai import MoiraiForecast

        contexts = np.asarray(contexts, dtype=np.float64)
        L = contexts.shape[1]
        ctx_len = self._ctx_len or L
        ps = 16 if self.patch_size == "auto" else int(self.patch_size)
        forecaster = MoiraiForecast(
            module=self._module,
            prediction_length=horizon,
            context_length=ctx_len,
            patch_size=ps,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device)

        outs = []
        hidden = None
        with torch.no_grad():
            past = torch.tensor(contexts, dtype=torch.float32, device=self.device)
            past = past.unsqueeze(-1)  # (B, L, 1)
            observed = torch.ones_like(past, dtype=torch.bool)
            is_pad = torch.zeros(past.shape[:2], dtype=torch.bool, device=self.device)
            try:
                samples = forecaster(
                    past_target=past,
                    past_observed_target=observed,
                    past_is_pad=is_pad,
                )  # (B, num_samples, H)
                fc = samples.median(dim=1).values.float().cpu().numpy()
                if self._encoder_ok and self._extractor is not None:
                    cap = self._extractor.pop(len(contexts))
                    if cap is not None:
                        hidden = cap.float().cpu().numpy()
                        self._hidden = int(hidden.shape[-1])
            except Exception:
                fc = np.stack([self._fallback_forecast(c, horizon) for c in contexts])
                hidden = None

        for i, ctx in enumerate(contexts):
            if hidden is not None:
                patches = F.pool_tokens(hidden[i], self._max_patches)
            else:
                patches = F.stat_patches(ctx, self._stat_patch_len, self._proj)
            outs.append(ExpertOutput(forecast=fc[i][:horizon], patches=patches))
        return outs

    @staticmethod
    def _fallback_forecast(ctx, horizon):
        m = min(24, ctx.size)
        reps = int(np.ceil(horizon / m))
        return np.tile(ctx[-m:], reps)[:horizon]
