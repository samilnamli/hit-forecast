"""Router training loop (draft §IV-C defaults)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.dataset import CombinedData, RouterDataset, collate_router
from ..models.losses import CompositeRoutingLoss
from ..utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    max_epochs: int = 100
    batch_size: int = 256
    grad_clip: float = 1.0
    patience: int = 20
    amp: bool = True
    num_workers: int = 0
    loss: dict = field(default_factory=dict)
    seed: int = 0


def _lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _expected_mase(model, data: CombinedData, device: str, batch_size: int) -> float:
    ds = RouterDataset(data)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_router)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for feats, masks, mase, _ in dl:
            feats = [f.to(device) for f in feats]
            masks = [m.to(device) for m in masks]
            mase = mase.to(device)
            w = torch.softmax(model(feats, masks), dim=-1)
            tot += float((w * mase).sum(-1).sum().item())
            n += mase.shape[0]
    return tot / max(1, n)


def train_router(
    model,
    train_data: CombinedData,
    val_data: CombinedData,
    cfg: TrainConfig,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(cfg.seed)
    model = model.to(device)
    criterion = CompositeRoutingLoss(**cfg.loss)

    ds = RouterDataset(train_data)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_router,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.max_epochs * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, cfg.warmup_steps, total_steps)
    )
    use_amp = cfg.amp and "cuda" in str(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        ep_loss = 0.0
        for feats, masks, mase, _ in dl:
            feats = [f.to(device) for f in feats]
            masks = [m.to(device) for m in masks]
            mase = mase.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(feats, masks)
                losses = criterion(logits, mase)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_loss += float(loss.item()) * mase.shape[0]
        ep_loss /= len(ds)

        val_mase = _expected_mase(model, val_data, device, cfg.batch_size)
        history.append({"epoch": epoch, "train_loss": ep_loss, "val_expected_mase": val_mase})
        _log.info("epoch %d train_loss=%.4f val_E[MASE]=%.4f", epoch, ep_loss, val_mase)

        if val_mase < best_val - 1e-5:
            best_val = val_mase
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                _log.info("Early stopping at epoch %d (best val E[MASE]=%.4f)", epoch, best_val)
                break

    model.load_state_dict(best_state)
    return {"model": model, "best_val_expected_mase": best_val, "history": history}
