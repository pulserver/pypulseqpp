"""The sequence zoo: one complete sequence per module.

Every module here is installed into :mod:`pypulseqpp.examples` and reached
from there -- ``pypulseqpp.examples.gre2D_sequence``, whatever directory it
was written in. Each is a worked example of the whole authoring stack: the
:mod:`pypulseqpp.sequences` modules it composes, the encoding plan it builds
from :mod:`pypulseqpp`, and the loop that writes the blocks::

    from pypulseqpp import examples

    seq = examples.gre2D_sequence(n_x=128, n_y=128, n_slices=5)
    seq.write("gre2D.seq")
"""

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
