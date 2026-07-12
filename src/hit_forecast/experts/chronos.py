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


@register_expert("chronos")
class ChronosExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "chronos-2",
        model_id: str = "amazon/chronos-2",
        device: str = "cpu",
        dtype: str = "bfloat16",
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
        inner = getattr(self._pipe, "model", None)
        cfg = getattr(getattr(inner, "config", None), "d_model", None)
        if cfg and self.feature_source != "stat":
            self._hidden = int(cfg)

    @property
    def hidden_dim(self) -> int:
        return self._hidden

    def batch_forecast_and_features(self, contexts, horizon):
        self._lazy()
        import torch

        contexts = np.asarray(contexts, dtype=np.float64)
        ctx_t = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        # Point forecast = median quantile.
        try:
            q, _ = self._pipe.predict_quantiles(
                context=ctx_t, prediction_length=horizon, quantile_levels=[0.5]
            )
            fc = q[:, :, 0].float().cpu().numpy()
        except Exception:
            samples = self._pipe.predict(ctx_t, prediction_length=horizon)
            fc = np.stack([np.median(np.asarray(s), axis=0) for s in samples])

        outs = []
        for i, ctx in enumerate(contexts):
            patches = self._features(ctx)
            outs.append(ExpertOutput(forecast=fc[i][:horizon], patches=patches))
        return outs

    def _features(self, ctx: np.ndarray) -> np.ndarray:
        if self._encoder_ok:
            try:
                return self._encoder_features(ctx)
            except Exception as e:  # noqa: BLE001
                if self.feature_source == "encoder":
                    raise
                F.warn_fallback(self.name, e)
                self._encoder_ok = False
        return F.stat_patches(ctx, self._stat_patch_len, self._proj)

    def _encoder_features(self, ctx: np.ndarray) -> np.ndarray:
        """Last encoder hidden state of the tokenised context (T, D)."""
        import torch

        model = self._pipe.model
        tok = getattr(self._pipe, "tokenizer", None)
        with torch.no_grad():
            ids = torch.tensor(ctx, dtype=torch.float32, device=self.device)[None, :]
            if tok is not None and hasattr(tok, "context_input_transform"):
                token_ids, attn, _ = tok.context_input_transform(ids)
                enc = model.model.encoder(input_ids=token_ids.to(self.device),
                                          attention_mask=attn.to(self.device))
            else:  # patch/embedding models expose encoder differently
                enc = model.encoder(inputs_embeds=ids.unsqueeze(-1))
            h = enc.last_hidden_state[0]
        self._hidden = int(h.shape[-1])
        return h.float().cpu().numpy().astype(np.float32)
