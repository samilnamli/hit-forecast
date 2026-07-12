"""MASE-aware composite routing objective (draft §III-D, eqs. 14-18)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CompositeRoutingLoss(nn.Module):
    """L = lambda_mase * L_mase + lambda_hard * L_hard + lambda_soft * L_soft.

    - L_mase  : expected MASE under the routing distribution (eq. 14)
    - L_hard  : label-smoothed CE against the per-window oracle expert (eq. 15)
    - L_soft  : CE against a temperature-softmax of negative MASE (eqs. 16-17)
    """

    def __init__(
        self,
        lambda_mase: float = 1.0,
        lambda_hard: float = 0.3,
        lambda_soft: float = 0.5,
        tau: float = 1.5,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.lambda_mase = lambda_mase
        self.lambda_hard = lambda_hard
        self.lambda_soft = lambda_soft
        self.tau = tau
        self.eps = label_smoothing

    def forward(self, logits: torch.Tensor, mase: torch.Tensor) -> dict[str, torch.Tensor]:
        # logits, mase: (B, K)
        w = F.softmax(logits, dim=-1)
        log_w = F.log_softmax(logits, dim=-1)
        K = mase.shape[1]

        l_mase = (w * mase).sum(dim=-1).mean()

        oracle = mase.argmin(dim=-1)
        q_hard = torch.full_like(mase, self.eps / K)
        q_hard.scatter_(1, oracle.unsqueeze(1), 1.0 - self.eps + self.eps / K)
        l_hard = -(q_hard * log_w).sum(dim=-1).mean()

        q_soft = F.softmax(-mase / self.tau, dim=-1)
        l_soft = -(q_soft * log_w).sum(dim=-1).mean()

        total = (
            self.lambda_mase * l_mase
            + self.lambda_hard * l_hard
            + self.lambda_soft * l_soft
        )
        return {"total": total, "mase": l_mase, "hard": l_hard, "soft": l_soft}
