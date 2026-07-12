"""Phase-0 go/no-go diagnostics.

Quantifies whether the cached expert pool provides a *learnable* routing signal
on a corpus, directly targeting the colleague's failure mode (contaminated /
redundant experts => collapsed MASE margins => weak supervision).
"""

from __future__ import annotations

import numpy as np


def signal_diagnostics(mase: np.ndarray, expert_names: list[str]) -> dict:
    """Compute margin and win-rate statistics from a cached MASE matrix (N, K)."""
    mase = np.asarray(mase, dtype=np.float64)
    N, K = mase.shape
    order = np.sort(mase, axis=1)
    best = order[:, 0]
    second = order[:, 1] if K > 1 else order[:, 0]

    # relative margin between best and 2nd-best expert per window
    denom = np.where(best < 1e-8, 1e-8, best)
    rel_margin = (second - best) / denom
    abs_margin = second - best

    win = mase.argmin(axis=1)
    win_counts = np.bincount(win, minlength=K).astype(np.float64)
    win_rate = win_counts / N
    p = np.clip(win_rate, 1e-12, 1.0)
    entropy = float(-(p * np.log(p)).sum())
    norm_entropy = entropy / np.log(K) if K > 1 else 0.0

    oracle_mase = float(best.mean())
    best_single = float(mase.mean(axis=0).min())
    oracle_gap = (best_single - oracle_mase) / max(best_single, 1e-8)

    # go/no-go heuristic: enough margin AND experts not collapsed onto one
    median_rel = float(np.median(rel_margin))
    max_win = float(win_rate.max())
    passes = (median_rel > 0.02) and (max_win < 0.85) and (norm_entropy > 0.4)

    return {
        "n_windows": int(N),
        "n_experts": int(K),
        "expert_names": list(expert_names),
        "win_rate": {n: float(r) for n, r in zip(expert_names, win_rate)},
        "win_rate_entropy_norm": norm_entropy,
        "rel_margin_median": median_rel,
        "rel_margin_mean": float(rel_margin.mean()),
        "abs_margin_median": float(np.median(abs_margin)),
        "oracle_mase": oracle_mase,
        "best_single_mase": best_single,
        "oracle_gain_vs_best_single": float(oracle_gap),
        "gate_pass": bool(passes),
    }
