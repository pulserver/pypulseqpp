"""Complete sequence scripts, reached as ``pypulseqpp.sequences.<name>``."""

from __future__ import annotations

import importlib

__all__ = [
    "gre2D_sequence",
]


def __getattr__(name: str):
    """Import one sequence on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sequences this package ships."""
    return sorted(__all__)
