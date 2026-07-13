from .base import ExpertAdapter, ExpertOutput
from .registry import build_expert, build_pool, register_expert, available_experts

# Importing the concrete adapters registers them via decorators. Heavy FM
# adapters guard their own imports so this stays cheap when deps are absent.
from . import dummy  # noqa: F401
from . import classical as _classical  # noqa: F401
from . import chronos as _chronos  # noqa: F401
from . import moirai as _moirai  # noqa: F401
from . import timesfm as _timesfm  # noqa: F401
from . import tirex as _tirex  # noqa: F401

__all__ = [
    "ExpertAdapter",
    "ExpertOutput",
    "build_expert",
    "build_pool",
    "register_expert",
    "available_experts",
]
