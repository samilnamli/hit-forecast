"""Pooled-feature MLP router baseline (draft §IV-B, "Pooled-feature routing").

Mean-pools each expert's patch sequence into a single vector, projects each to a
common width, concatenates the K vectors, and feeds a two-layer MLP. Uses the
same composite loss and a comparable parameter budget, isolating the value of
patch-level temporal modelling.
"""

from __future__ import annotations

import torch
from torch import nn


class PooledMLPRouter(nn.Module):
    def __init__(self, hidden_dims: list[int], d: int = 256, dropout: float = 0.1):
        super().__init__()
        self.K = len(hidden_dims)
        self.proj = nn.ModuleList([nn.Linear(D, d) for D in hidden_dims])
        self.mlp = nn.Sequential(
            nn.Linear(self.K * d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, self.K),
        )

    def forward(self, feats: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for k in range(self.K):
            x = feats[k]
            mask = masks[k].unsqueeze(-1).float()
            summed = (x * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1.0)
            mean = summed / count
            pooled.append(self.proj[k](mean))
        z = torch.cat(pooled, dim=-1)
        return self.mlp(z)

    @torch.no_grad()
    def route(self, feats, masks):
        logits = self.forward(feats, masks)
        w = torch.softmax(logits, dim=-1)
        return w.argmax(dim=-1), w

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
