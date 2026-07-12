from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional ``defaults: <file>`` key."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("defaults", None)
    if base is not None:
        base_path = (path.parent / base).resolve()
        merged = load_config(base_path)
        cfg = _deep_merge(merged, cfg)
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.sub=value`` CLI overrides with basic literal parsing."""
    out = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Malformed override (expected key=value): {ov!r}")
        key, raw = ov.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_literal(raw)
    return out


def _parse_literal(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw
