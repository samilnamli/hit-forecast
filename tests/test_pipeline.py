import numpy as np

from hit_forecast.data.synthetic import make_regime_switch_dataset, REGIMES
from hit_forecast.data.windows import WindowSet
from hit_forecast.experts import build_pool
from hit_forecast.features.cache import build_cache
from hit_forecast.train.diagnostics import signal_diagnostics


def test_synthetic_cache_and_diagnostics(tmp_path):
    windows, regime_ids = make_regime_switch_dataset(n_windows=200, L=96, H=48, m=24, seed=0)
    ws = WindowSet(windows=windows, name="synthetic::test")
    experts = build_pool(
        [{"kind": "dummy_trend"}, {"kind": "dummy_spiky"},
         {"kind": "dummy_season"}, {"kind": "dummy_snaive"}]
    )
    cache = build_cache(ws, experts, tmp_path, batch_size=64, overwrite=True)
    assert cache.mase.shape == (200, 4)
    assert cache.forecasts.shape[0] == 200

    diag = signal_diagnostics(cache.mase, cache.expert_names)
    # There must be a non-trivial routing signal on the synthetic data.
    assert diag["n_experts"] == 4
    assert diag["rel_margin_median"] > 0.0
    assert 0.0 <= diag["win_rate_entropy_norm"] <= 1.0
    assert len(REGIMES) == 3
