"""HiT-Forecast: Hierarchical Transformer Routing over frozen time-series foundation models.

Public API is intentionally small; use the subpackages directly:

- ``hit_forecast.data``      windowing, MASE, synthetic + GIFT-Eval loaders
- ``hit_forecast.experts``   expert-adapter interface and the clean expert pool
- ``hit_forecast.features``  patch-feature / forecast / MASE caching
- ``hit_forecast.models``    router, pooled-MLP baseline, losses
- ``hit_forecast.train``     trainer + Phase-0 diagnostics
- ``hit_forecast.eval``      metrics, baselines, GIFT-Eval-style aggregation
"""

__version__ = "0.2.0"
