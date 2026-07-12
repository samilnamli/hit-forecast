"""HiT-Forecast hierarchical transformer router (draft §III-C).

Pipeline per window:

1. per-model linear projection ``W_p^{(k)}: R^{D_k} -> R^d`` (eq. 4)
2. + sinusoidal positional encoding + learnable model-identity embedding ``e_k``
3. prepend a learnable per-expert summary token ``c_k`` (eq. 5)
4. Stage-1 shared transformer encoder -> per-expert summary ``\\tilde c_k`` (eqs. 6-7)
5. optional cross-attention bridge: each summary attends to other experts'
   patch tokens (eqs. 8-9)
6. Stage-2 fusion transformer over ``[c*, \\tilde c_1..\\tilde c_K]`` (eq. 10)
7. MLP on the fused global token -> routing logits ``s_n`` (eq. 11) -> softmax
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RouterConfig:
    d: int = 256
    nhead: int = 4
    stage1_layers: int = 2
    stage2_layers: int = 2
    ffn: int = 512
    dropout: float = 0.1
    share_stage1: bool = True
    cross_attention: bool = True
    max_patches: int = 1024


def sinusoidal_pe(T: int, d: int, device) -> torch.Tensor:
    pe = torch.zeros(T, d, device=device)
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe  # (T, d)


def _encoder(cfg: RouterConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.d,
        nhead=cfg.nhead,
        dim_feedforward=cfg.ffn,
        dropout=cfg.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class HierarchicalRouter(nn.Module):
    def __init__(self, hidden_dims: list[int], config: RouterConfig | None = None):
        super().__init__()
        self.cfg = config or RouterConfig()
        d = self.cfg.d
        self.K = len(hidden_dims)
        self.hidden_dims = hidden_dims

        self.proj = nn.ModuleList([nn.Linear(D, d) for D in hidden_dims])
        self.model_embed = nn.Parameter(torch.randn(self.K, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(self.K, d) * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, d) * 0.02)

        if self.cfg.share_stage1:
            self.stage1 = _encoder(self.cfg, self.cfg.stage1_layers)
        else:
            self.stage1 = nn.ModuleList(
                [_encoder(self.cfg, self.cfg.stage1_layers) for _ in range(self.K)]
            )

        if self.cfg.cross_attention:
            self.bridge = nn.MultiheadAttention(d, self.cfg.nhead, dropout=self.cfg.dropout,
                                                batch_first=True)
            self.bridge_ln = nn.LayerNorm(d)

        self.stage2 = _encoder(self.cfg, self.cfg.stage2_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.K)
        )

    def _stage1_for(self, k: int):
        return self.stage1 if self.cfg.share_stage1 else self.stage1[k]

    def forward(self, feats: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        """feats[k]: (B, T_k, D_k); masks[k]: (B, T_k) True=valid. Returns logits (B, K)."""
        B = feats[0].shape[0]
        d = self.cfg.d
        device = feats[0].device

        summaries = []  # \tilde c_k, each (B, d)
        patch_tokens = []  # Y^{(k)}[:,1:], (B, T_k, d)
        patch_valid = []  # (B, T_k) True=valid

        for k in range(self.K):
            x = feats[k][:, : self.cfg.max_patches]
            valid = masks[k][:, : self.cfg.max_patches]
            T = x.shape[1]
            h = self.proj[k](x)  # (B, T, d)
            h = h + sinusoidal_pe(T, d, device).unsqueeze(0)
            h = h + self.model_embed[k].view(1, 1, d)
            cls = self.cls[k].view(1, 1, d).expand(B, 1, d)
            seq = torch.cat([cls, h], dim=1)  # (B, T+1, d)
            cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
            seq_valid = torch.cat([cls_valid, valid], dim=1)
            key_padding = ~seq_valid  # True = ignore
            y = self._stage1_for(k)(seq, src_key_padding_mask=key_padding)
            summaries.append(y[:, 0])
            patch_tokens.append(y[:, 1:])
            patch_valid.append(valid)

        if self.cfg.cross_attention:
            summaries = self._cross_bridge(summaries, patch_tokens, patch_valid)

        gtok = self.global_token.view(1, 1, d).expand(B, 1, d)
        z = torch.cat([gtok] + [s.unsqueeze(1) for s in summaries], dim=1)  # (B, K+1, d)
        z = self.stage2(z)
        logits = self.classifier(z[:, 0])  # (B, K)
        return logits

    def _cross_bridge(self, summaries, patch_tokens, patch_valid):
        d = self.cfg.d
        updated = []
        for k in range(self.K):
            others = [j for j in range(self.K) if j != k]
            keys = torch.cat([patch_tokens[j] for j in others], dim=1)  # (B, sumT, d)
            kvalid = torch.cat([patch_valid[j] for j in others], dim=1)  # (B, sumT)
            q = summaries[k].unsqueeze(1)  # (B, 1, d)
            attn, _ = self.bridge(q, keys, keys, key_padding_mask=~kvalid,
                                  need_weights=False)
            c = self.bridge_ln(summaries[k] + attn.squeeze(1))
            updated.append(c)
        return updated

    @torch.no_grad()
    def route(self, feats, masks) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (argmax expert index, softmax weights)."""
        logits = self.forward(feats, masks)
        w = torch.softmax(logits, dim=-1)
        return w.argmax(dim=-1), w

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
