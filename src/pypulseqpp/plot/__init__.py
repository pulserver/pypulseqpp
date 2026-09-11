"""Figures of a sequence: the SeqEyes view, the publication diagram, k-space and RF profiles.

Every figure is a function taking what it draws. :func:`plot` and
:func:`paper_plot` call the :class:`~pypulseqpp.Sequence` methods of the same
name, which stay for scripts written against upstream PyPulseq.
"""

from __future__ import annotations

from ._kspace import plot_kspace
from ._rf import plot_rf

__all__ = ["paper_plot", "plot", "plot_kspace", "plot_rf"]


def plot(seq, *args, **kwargs):
    """Draw ``seq`` in SeqEyes; see :meth:`pypulseqpp.Sequence.plot`."""
    return seq.plot(*args, **kwargs)


def paper_plot(seq, *args, **kwargs):
    """Draw ``seq``'s publication diagram; see :meth:`pypulseqpp.Sequence.paper_plot`."""
    return seq.paper_plot(*args, **kwargs)
