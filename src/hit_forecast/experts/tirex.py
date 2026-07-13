"""TiRex expert adapter (NXAI).

Install the correct package (NOT the unrelated PyPI project ``tirex``)::

    pip uninstall -y tirex
    pip install "tirex-ts"

TiRex is a strong GIFT-Eval zero-shot competitor (xLSTM-based), tagged
``testdata_leakage = No``. Patch features default to the stat fallback.
"""

from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger
from .base import ExpertAdapter, ExpertOutput
from .registry import register_expert
from . import _features as F

_log = get_logger(__name__)


def _import_load_model():
    """Import ``load_model`` from the real TiRex package, with a clear error."""
    try:
        from tirex import load_model  # type: ignore

        return load_model
    except ImportError as e:
        # Diagnose the common "wrong tirex package" failure.
        try:
            import tirex as _pkg  # type: ignore

            path = getattr(_pkg, "__file__", "?")
            names = [n for n in dir(_pkg) if not n.startswith("_")]
            raise ImportError(
                "Cannot import load_model from tirex. You likely installed the "
                "WRONG PyPI package `tirex` (dim-reduction, unrelated). "
                "Fix with:\n"
                "  pip uninstall -y tirex\n"
                "  pip install 'tirex-ts'\n"
                f"Currently imported tirex from: {path}\n"
                f"Exports: {names[:20]}\n"
                f"Original error: {e}"
            ) from e
        except ImportError:
            raise ImportError(
                "TiRex is not installed. Install with:\n"
                "  pip install 'tirex-ts'\n"
                f"Original error: {e}"
            ) from e


@register_expert("tirex")
class TiRexExpert(ExpertAdapter):
    def __init__(
        self,
        name: str = "tirex",
        model_id: str = "NX-AI/TiRex",
        device: str = "cpu",
        backend: str = "torch",  # "torch" | "cuda"
        feature_source: str = "stat",
        stat_patch_len: int = 24,
        stat_hidden: int = 112,
    ):
        super().__init__(name=name, device=device)
        self.model_id = model_id
        self.backend = backend
        self.feature_source = feature_source
        self._stat_patch_len = stat_patch_len
        self._stat_hidden = stat_hidden
        self._proj = F.make_stat_proj(name, stat_hidden)
        self._model = None
        self._hidden = stat_hidden

    def _lazy(self):
        if self._model is not None:
            return
        load_model = _import_load_model()
        try:
            self._model = load_model(
                self.model_id, device=self.device, backend=self.backend
            )
        except TypeError:
            # Older tirex-ts signatures.
            self._model = load_model(self.model_id)
        _log.info("Loaded TiRex (%s) on %s backend=%s", self.model_id, self.device, self.backend)

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
                context=ctx_t, prediction_length=int(horizon)
            )
            fc = np.asarray(mean.detach().cpu() if hasattr(mean, "detach") else mean)
        except Exception:
            out = self._model.forecast(ctx_t, prediction_length=int(horizon))
            fc = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
            if hasattr(fc, "detach"):
                fc = fc.detach().cpu().numpy()
            if fc.ndim == 3:  # (B, Q, H) -> median
                fc = fc[:, fc.shape[1] // 2, :]

        if fc.ndim == 1:
            fc = fc[None, :]
        return [
            ExpertOutput(
                forecast=np.asarray(fc[i][:horizon], dtype=np.float64),
                patches=F.stat_patches(ctx, self._stat_patch_len, self._proj),
            )
            for i, ctx in enumerate(contexts)
        ]
