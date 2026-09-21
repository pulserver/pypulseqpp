"""Sequence figures, from the SeqEyes view and the publication diagram to the profiles.

Each figure is a function of the object it draws. :func:`plot` and
:func:`paper_plot` call the :class:`~pypulseqpp.Sequence` methods of the same
name, which are retained for scripts written against upstream PyPulseq.
"""

from __future__ import annotations

from ._kspace import plot_kspace
from ._rf import plot_rf

__all__ = ["paper_plot", "plot", "plot_kspace", "plot_rf"]


def plot(seq, *args, **kwargs):
    """Open ``seq`` in the SeqEyes viewer; see :meth:`pypulseqpp.Sequence.plot`."""
    return seq.plot(*args, **kwargs)


def paper_plot(seq, *args, **kwargs):
    """Draw a publication diagram of ``seq``; see :meth:`pypulseqpp.Sequence.paper_plot`."""
    return seq.paper_plot(*args, **kwargs)
