import torch

from hit_forecast.models import HierarchicalRouter, PooledMLPRouter, RouterConfig
from hit_forecast.models.losses import CompositeRoutingLoss


def _fake_batch(B=5, dims=(8, 12), Ts=(6, 9)):
    feats, masks = [], []
    for D, T in zip(dims, Ts):
        feats.append(torch.randn(B, T, D))
        m = torch.ones(B, T, dtype=torch.bool)
        m[:, -1] = False  # one padded patch
        masks.append(m)
    return feats, masks


def test_hierarchical_router_shapes():
    dims = [8, 12, 16]
    Ts = [6, 9, 4]
    router = HierarchicalRouter(dims, RouterConfig(d=32, nhead=4, stage1_layers=1,
                                                   stage2_layers=1, ffn=64))
    feats, masks = _fake_batch(B=5, dims=dims, Ts=Ts)
    logits = router(feats, masks)
    assert logits.shape == (5, 3)
    jn, w = router.route(feats, masks)
    assert jn.shape == (5,)
    assert torch.allclose(w.sum(-1), torch.ones(5), atol=1e-5)


def test_pooled_router_shapes():
    dims = [8, 12]
    router = PooledMLPRouter(dims, d=16)
    feats, masks = _fake_batch(B=4, dims=dims, Ts=(6, 9))
    assert router(feats, masks).shape == (4, 2)


def test_composite_loss_decreases_on_easy_signal():
    torch.manual_seed(0)
    dims = [4, 4]
    router = HierarchicalRouter(dims, RouterConfig(d=16, nhead=2, stage1_layers=1,
                                                   stage2_layers=1, ffn=32,
                                                   cross_attention=False))
    crit = CompositeRoutingLoss()
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)
    B = 32
    feats = [torch.randn(B, 5, 4), torch.randn(B, 5, 4)]
    masks = [torch.ones(B, 5, dtype=torch.bool), torch.ones(B, 5, dtype=torch.bool)]
    # expert 0 always best
    mase = torch.stack([torch.full((B,), 0.2), torch.full((B,), 1.0)], dim=1)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = crit(router(feats, masks), mase)["total"]
        loss.backward()
        opt.step()
        first = first or float(loss.item())
    assert float(loss.item()) <= first
