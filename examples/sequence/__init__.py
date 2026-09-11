"""Complete sequence scripts, reached as ``pypulseqpp.sequences.<name>``."""

from __future__ import annotations

import importlib

__all__ = [
    "bssfp2D_sequence",
    "bssfp3D_sequence",
    "epi2D_sequence",
    "epi3D_sequence",
    "fse2D_sequence",
    "fse3D_sequence",
    "gre2D_sequence",
    "gre3D_sequence",
    "gre_multiecho2D_sequence",
    "gre_multiecho3D_sequence",
    "gre_radial2D_sequence",
    "gre_spiral2D_sequence",
    "gre_stack_of_spirals3D_sequence",
    "gre_stack_of_stars3D_sequence",
    "mprage3D_sequence",
    "mprage_stack_of_spirals3D_sequence",
    "se2D_sequence",
    "se3D_sequence",
    "se_propeller2D_sequence",
    "zte3D_sequence",
]


def __getattr__(name: str):
    """Import one sequence on first use."""
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
