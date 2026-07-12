"""Expert-adapter interface.

Each expert wraps a (frozen) time-series foundation model and exposes two
things per input window, matching the draft:

1. a point forecast ``yhat`` of length ``H`` (used to compute per-window MASE
   supervision and to produce the final forecast once routed), and
2. a *patch-level* hidden-state sequence ``H^{(k)} in R^{T_k x D_k}`` from the
   frozen encoder -- NOT pooled -- which the router consumes.

Adapters should implement the batched method for throughput; the single-window
method has a default batched fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertOutput:
    forecast: np.ndarray  # (H,)
    patches: np.ndarray  # (T_k, D_k) float32 patch-level encoder hidden states


class ExpertAdapter(ABC):
    """Base class for a frozen-foundation-model expert."""

    #: short unique name, used as the routing label and cache key
    name: str = "expert"

    def __init__(self, name: str | None = None, device: str = "cpu"):
        if name:
            self.name = name
        self.device = device

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Encoder hidden dimension ``D_k``."""

    @abstractmethod
    def batch_forecast_and_features(
        self, contexts: np.ndarray, horizon: int
    ) -> list[ExpertOutput]:
        """Run the expert on a batch of contexts ``(B, L)`` and return B outputs."""

    def forecast_and_features(self, context: np.ndarray, horizon: int) -> ExpertOutput:
        return self.batch_forecast_and_features(context[None, :], horizon)[0]

    # Adapters that need GPU setup can override; default is a no-op.
    def to(self, device: str) -> "ExpertAdapter":
        self.device = device
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r}, D={self.hidden_dim})"
