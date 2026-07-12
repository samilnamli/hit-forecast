"""Torch dataset over cached patch features + MASE labels.

Supports combining multiple cache shards (different GIFT-Eval configs) into one
training set. Per-expert hidden dim ``D_k`` must be constant across shards
(guaranteed by a fixed pool); patch counts ``T_k`` may differ and are padded to
a global maximum with a validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.cache import FeatureCache


@dataclass
class CombinedData:
    feats: list[np.ndarray]  # per expert (N, Tmax_k, D_k) float32
    masks: list[np.ndarray]  # per expert (N, Tmax_k) bool
    mase: np.ndarray  # (N, K) float32
    expert_names: list[str]
    hidden_dims: list[int]
    patch_counts: list[int]  # global Tmax per expert
    # provenance for evaluation
    forecasts: np.ndarray  # (N, K, H) -- only valid if single H across shards
    targets: np.ndarray | None
    contexts: np.ndarray | None
    m: np.ndarray | None
    window_meta: list[dict]

    @property
    def N(self) -> int:
        return self.mase.shape[0]

    @property
    def K(self) -> int:
        return self.mase.shape[1]


def combine_caches(caches: list[FeatureCache]) -> CombinedData:
    if not caches:
        raise ValueError("No caches provided.")
    names = caches[0].expert_names
    dims = caches[0].hidden_dims
    for c in caches:
        if c.expert_names != names:
            raise ValueError(
                f"Expert-name mismatch across caches: {c.expert_names} vs {names}"
            )
        if c.hidden_dims != dims:
            raise ValueError("Hidden-dim mismatch across caches for same pool.")
    K = len(names)
    Tmax = [max(c.feats[k].shape[1] for c in caches) for k in range(K)]

    feats_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []
    for k in range(K):
        parts, mparts = [], []
        for c in caches:
            f = c.feats[k]
            T = f.shape[1]
            if T < Tmax[k]:
                pad = np.zeros((f.shape[0], Tmax[k] - T, f.shape[2]), dtype=f.dtype)
                f = np.concatenate([f, pad], axis=1)
            parts.append(f)
            mask = np.arange(Tmax[k])[None, :] < c.lens[k][:, None]
            mparts.append(mask)
        feats_all.append(np.concatenate(parts, axis=0))
        masks_all.append(np.concatenate(mparts, axis=0))

    mase = np.concatenate([c.mase for c in caches], axis=0).astype(np.float32)
    window_meta: list[dict] = []
    for c in caches:
        wm = c.window_meta or [{} for _ in range(c.N)]
        window_meta.extend(wm)

    # forecasts/targets/contexts only concatenated if H (and L) match across shards
    same_H = len({c.forecasts.shape[2] for c in caches}) == 1
    same_L = len({c.contexts.shape[1] for c in caches}) == 1
    forecasts = np.concatenate([c.forecasts for c in caches], 0) if same_H else np.empty(0)
    targets = np.concatenate([c.targets for c in caches], 0) if same_H else None
    contexts = np.concatenate([c.contexts for c in caches], 0) if same_L else None
    m = np.concatenate([c.m for c in caches], 0)

    return CombinedData(
        feats=feats_all,
        masks=masks_all,
        mase=mase,
        expert_names=names,
        hidden_dims=dims,
        patch_counts=Tmax,
        forecasts=forecasts,
        targets=targets,
        contexts=contexts,
        m=m,
        window_meta=window_meta,
    )


class RouterDataset(Dataset):
    def __init__(self, data: CombinedData, indices: np.ndarray | None = None):
        self.data = data
        self.indices = np.arange(data.N) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        feats = [self.data.feats[k][idx] for k in range(self.data.K)]
        masks = [self.data.masks[k][idx] for k in range(self.data.K)]
        mase = self.data.mase[idx]
        return feats, masks, mase, idx


def collate_router(batch):
    K = len(batch[0][0])
    feats = [torch.from_numpy(np.stack([b[0][k] for b in batch])).float() for k in range(K)]
    masks = [torch.from_numpy(np.stack([b[1][k] for b in batch])).bool() for k in range(K)]
    mase = torch.from_numpy(np.stack([b[2] for b in batch])).float()
    idx = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return feats, masks, mase, idx
