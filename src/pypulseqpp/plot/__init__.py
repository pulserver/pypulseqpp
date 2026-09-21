"""Sequence figures, from the SeqEyes view and the publication diagram to the profiles.

Each figure is a function of the object it draws. :func:`plot` and
:func:`paper_plot` call the :class:`~pypulseqpp.Sequence` methods of the same
name, which are retained for scripts written against upstream PyPulseq.

:data:`SAMPLING`, :data:`MAGNITUDE` and :data:`SIGNED` are the colormaps these
figures are drawn with. Every tone in them is held between two luminances, so
that a figure on a transparent canvas reads against white paper and against a
dark page alike; a figure drawn beside one of these takes the same colormap so
that the two agree.
"""

from __future__ import annotations

from ._kspace import plot_kspace
from ._rf import plot_rf
from ._style import MAGNITUDE, SAMPLING, SIGNED

__all__ = [
    "MAGNITUDE",
    "SAMPLING",
    "SIGNED",
    "paper_plot",
    "plot",
    "plot_kspace",
    "plot_rf",
]


def plot(seq, *args, **kwargs):
    """Open ``seq`` in the SeqEyes viewer; see :meth:`pypulseqpp.Sequence.plot`."""
    return seq.plot(*args, **kwargs)


def paper_plot(seq, *args, **kwargs):
    """Draw a publication diagram of ``seq``; see :meth:`pypulseqpp.Sequence.paper_plot`."""
    return seq.paper_plot(*args, **kwargs)
