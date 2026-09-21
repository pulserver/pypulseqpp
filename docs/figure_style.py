"""Matplotlib settings shared by the three pipelines that draw documentation figures.

The explanation figures, the ``plot_directive`` figures of the docstrings and
the gallery's own figures are produced by different machinery and are styled
from here so that they agree.

Every figure is drawn on a transparent canvas in the ink of
:mod:`pypulseqpp.plot._style`, whose tones clear 3.5:1 against white and
against the dark theme's background alike. ``_static/pypulseqpp.css`` removes
the card the theme would otherwise paint behind each image.
"""

from __future__ import annotations

# The package's own figures are drawn in these, so importing them keeps a
# gallery figure and an analysis figure in the same ink.
from pypulseqpp.plot._style import FAINT, INK, MUTED

#: Applied by `_pyplot` in `explanation_figures`, by `plot_rcparams` for the
#: docstring figures, and by `gallery_house_style` for the gallery.
FIGURE_RCPARAMS = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "savefig.transparent": True,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": FAINT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "grid.color": FAINT,
    "legend.labelcolor": INK,
}


def gallery_house_style(_gallery_conf, _fname) -> None:
    """Restore the house style after sphinx-gallery has reset matplotlib.

    sphinx-gallery calls ``rcdefaults()`` before each script, so settings made
    in ``conf.py`` do not reach the gallery. Registering this among
    ``reset_modules`` applies them again once the reset has run.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(FIGURE_RCPARAMS)
