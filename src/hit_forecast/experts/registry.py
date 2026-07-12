from __future__ import annotations

from typing import Callable

from .base import ExpertAdapter

_REGISTRY: dict[str, Callable[..., ExpertAdapter]] = {}


def register_expert(kind: str):
    def deco(fn: Callable[..., ExpertAdapter]):
        _REGISTRY[kind] = fn
        return fn

    return deco


def available_experts() -> list[str]:
    return sorted(_REGISTRY)


def build_expert(spec: dict, device: str = "cpu") -> ExpertAdapter:
    """Build one expert from a spec dict: ``{kind, name?, ...kwargs}``."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind not in _REGISTRY:
        raise KeyError(
            f"Unknown expert kind {kind!r}. Registered: {available_experts()}"
        )
    return _REGISTRY[kind](device=device, **spec)


def build_pool(specs: list[dict], device: str = "cpu") -> list[ExpertAdapter]:
    experts = [build_expert(s, device=device) for s in specs]
    names = [e.name for e in experts]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate expert names in pool: {names}")
    return experts
