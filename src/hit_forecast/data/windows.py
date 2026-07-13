"""Windowing utilities.

A *window* is the unit of routing supervision (draft §II): an input context of
length ``L`` and an ``H``-step target. Multivariate series are handled
channel-independently, matching the draft's protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class Window:
    context: np.ndarray  # (L,)
    target: np.ndarray  # (H,)
    m: int  # seasonal period for MASE
    meta: dict = field(default_factory=dict)  # dataset/freq/term/domain/channel


@dataclass
class WindowSet:
    windows: list[Window]
    name: str = "unnamed"

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self) -> Iterator[Window]:
        return iter(self.windows)

    def __getitem__(self, i: int) -> Window:
        return self.windows[i]

    @property
    def L(self) -> int:
        return int(self.windows[0].context.shape[0]) if self.windows else 0

    @property
    def H(self) -> int:
        return int(self.windows[0].target.shape[0]) if self.windows else 0


def sliding_windows(
    series: np.ndarray,
    L: int,
    H: int,
    m: int,
    stride: int | None = None,
    meta: dict | None = None,
    max_windows: int | None = None,
) -> list[Window]:
    """Slice a 1-D series into (context, target) windows.

    Stride defaults to ``H`` to prevent target leakage across windows (draft §IV-A).
    """
    series = np.asarray(series, dtype=np.float64).ravel()
    stride = stride or H
    meta = meta or {}
    out: list[Window] = []
    last_start = len(series) - (L + H)
    for start in range(0, last_start + 1, stride):
        ctx = series[start : start + L]
        tgt = series[start + L : start + L + H]
        if ctx.shape[0] != L or tgt.shape[0] != H:
            continue
        if not (np.all(np.isfinite(ctx)) and np.all(np.isfinite(tgt))):
            continue
        out.append(Window(context=ctx.copy(), target=tgt.copy(), m=m, meta=dict(meta)))
        if max_windows is not None and len(out) >= max_windows:
            break
    return out


def channels_from_2d(target_2d: np.ndarray) -> list[np.ndarray]:
    """Split a (C, T) or (T, C) multivariate array into a list of 1-D channels."""
    arr = np.asarray(target_2d)
    if arr.ndim == 1:
        return [arr]
    # gluonts stores multivariate targets as (C, T)
    if arr.shape[0] <= arr.shape[1]:
        return [arr[c] for c in range(arr.shape[0])]
    return [arr[:, c] for c in range(arr.shape[1])]
