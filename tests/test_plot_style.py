"""The documentation palette is read on a white page and on a dark one alike.

A figure this package draws is transparent, so the same image is shown on the
documentation's light background and on its dark one. Every colour it is drawn
in therefore has to be legible against both, which bounds its luminance from
below and from above at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from pypulseqpp.plot import _style

#: The two backgrounds a figure is shown on. The dark one is the lighter of
#: the two the theme stack defines, pydata's ``#14181e`` over
#: sphinx-book-theme's ``#121212``, because a mid tone has less contrast
#: against it and it is therefore the binding case.
LIGHT = "#ffffff"
DARK = "#14181e"

#: WCAG's contrast floor for a graphical object.
FLOOR = 3.0

_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


def _luminance(colour) -> float:
    channels = np.asarray(to_rgb(colour), dtype=float)
    linear = np.where(
        channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4
    )
    return float(_WEIGHTS @ linear)


def contrast(colour, background) -> float:
    """The WCAG contrast ratio between two opaque colours."""
    one, other = _luminance(colour), _luminance(background)
    return (max(one, other) + 0.05) / (min(one, other) + 0.05)


@pytest.mark.parametrize(
    "name", ["INK", "MUTED", *(f"SERIES[{i}]" for i in range(len(_style.SERIES)))]
)
def test_every_drawing_tone_clears_the_contrast_floor_on_both_backgrounds(name):
    colour = (
        _style.SERIES[int(name[7:-1])]
        if name.startswith("SERIES")
        else getattr(_style, name)
    )
    assert contrast(colour, LIGHT) >= FLOOR, f"{name} {colour} on {LIGHT}"
    assert contrast(colour, DARK) >= FLOOR, f"{name} {colour} on {DARK}"


def test_the_sampling_colormap_clears_the_contrast_floor_over_its_whole_range():
    for position in np.linspace(0.0, 1.0, 33):
        colour = _style.SAMPLING(float(position))[:3]
        assert contrast(colour, LIGHT) >= FLOOR, f"SAMPLING({position:.2f})"
        assert contrast(colour, DARK) >= FLOOR, f"SAMPLING({position:.2f})"


@pytest.mark.parametrize("name", ["MAGNITUDE", "SIGNED"])
def test_a_heatmap_ramp_ends_on_a_tone_that_reads_on_both_backgrounds(name):
    """The ends carry the largest values, so neither may sink into a background.

    The light middle of a ramp is exempt: a small value is drawn as a faint
    tint of the page on purpose, and the reader reads it against the colour bar
    rather than against the page.
    """
    colormap = getattr(_style, name)
    for position in (0.0, 1.0):
        colour = colormap(position)
        if colour[3] < 0.5:  # a transparent end is the page itself
            continue
        assert contrast(colour[:3], LIGHT) >= FLOOR, f"{name}({position})"
        assert contrast(colour[:3], DARK) >= FLOOR, f"{name}({position})"


def test_the_faint_tone_is_the_muted_tone_at_a_low_alpha():
    assert _style.FAINT.startswith(_style.MUTED)
    assert int(_style.FAINT[len(_style.MUTED) :], 16) < 0x80


def _figure_sources():
    """Every module that draws a documentation figure."""
    root = Path(__file__).parents[1]
    return [
        *sorted((root / "gallery").rglob("*.py")),
        root / "docs" / "explanation_figures.py",
        *sorted((root / "src" / "pypulseqpp" / "plot").glob("*.py")),
    ]


#: Colour arguments that name matplotlib's palette rather than the house one.
_FOREIGN = re.compile(
    r"""tab:\w+                      # the default categorical cycle
      | get_cmap\(                   # a colormap fetched by name
      | cmap\s*=\s*["']              # a colormap named as a string
      | color\s*=\s*["'](?:k|black|w|white)["']
      | \.plot\([^)]*["']k[o.+x^sv*d-]{0,2}["']   # a black format string
    """,
    re.VERBOSE,
)

#: Greys light or dark enough to sink into one of the two backgrounds.
_GREY = re.compile(r"(?:color|colors|facecolor|edgecolor)\s*=\s*[\"'](0\.\d+)[\"']")


def test_no_documentation_figure_names_a_colour_outside_the_house_palette():
    """A figure names a house tone, a cycle entry or a house colormap, or nothing.

    The default cycle carries an amber and an orange that do not clear 3:1
    against white paper, and every colormap matplotlib ships runs from
    near-black to near-white, so one of its ends disappears into one of the two
    backgrounds a transparent figure is shown on.
    """
    offences = []
    for source in _figure_sources():
        for number, line in enumerate(source.read_text().splitlines(), start=1):
            if _FOREIGN.search(line):
                offences.append(f"{source.name}:{number}: {line.strip()}")
            for grey in _GREY.findall(line):
                if not 0.35 <= float(grey) <= 0.6:
                    offences.append(f"{source.name}:{number}: {line.strip()}")
    assert not offences, "\n".join(offences)
