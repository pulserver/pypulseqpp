"""Matplotlib settings shared by the three pipelines that draw documentation figures.

The explanation figures, the ``plot_directive`` figures of the docstrings and
the gallery's own figures are produced by different machinery and are styled
from here so that they agree.

Every figure is drawn on a transparent canvas in the ink of
:mod:`pypulseqpp.plot._style`, whose tones clear 3:1 against white and against
the dark theme's background alike; ``tests/test_plot_style.py`` pins that.
``_static/pypulseqpp.css`` removes the card the theme would otherwise paint
behind each image.

Text is set at 12 pt and figures are saved at 110 dpi, so a figure as wide as
the documentation column, ``COLUMN_WIDTH`` inches, is shown at its own size
and its text at the size the page's own text is. ``column_scraper`` narrows a
gallery figure that is wider than the column before it is saved, which keeps
the public plotting helpers at their own default canvas sizes.
"""

from __future__ import annotations

# The package's own figures are drawn in these, so importing them keeps a
# gallery figure and an analysis figure in the same ink.
from cycler import cycler

from pypulseqpp.plot._style import FAINT, INK, MUTED, SERIES

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
    # The default cycle is matplotlib's, whose amber and orange do not clear
    # 3:1 against white paper. A figure that names no colour draws from the
    # house palette instead, which is held between two luminances.
    "axes.prop_cycle": cycler(color=list(SERIES)),
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.titlesize": 13,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "image.interpolation": "nearest",
}

#: The width of the documentation column, in inches at ``savefig.dpi``.
COLUMN_WIDTH = 7.8


def gallery_house_style(_gallery_conf, _fname) -> None:
    """Restore the house style after sphinx-gallery has reset matplotlib.

    sphinx-gallery calls ``rcdefaults()`` before each script, so settings made
    in ``conf.py`` do not reach the gallery. Registering this among
    ``reset_modules`` applies them again once the reset has run.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(FIGURE_RCPARAMS)


def column_scraper(block, block_vars, gallery_conf, **kwargs):
    """Save the gallery's figures no wider than the documentation column.

    A figure wider than ``COLUMN_WIDTH`` is scaled to it, height in proportion,
    before matplotlib's own scraper saves it. Text keeps its point size, so it
    is not shrunk with the image when the page shows it. The bounding box is
    taken tight around what is drawn, which keeps labels a layout computed at
    the larger size would place outside the canvas.
    """
    import matplotlib.pyplot as plt
    from sphinx_gallery.scrapers import matplotlib_scraper

    for number in plt.get_fignums():
        figure = plt.figure(number)
        width, height = figure.get_size_inches()
        if width > COLUMN_WIDTH:
            figure.set_size_inches(COLUMN_WIDTH, height * COLUMN_WIDTH / width)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.05)
    return matplotlib_scraper(block, block_vars, gallery_conf, **kwargs)
