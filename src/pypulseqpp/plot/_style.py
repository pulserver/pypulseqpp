"""Colours and axis styling shared by the analysis figures."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

#: Axis furniture. Every colour on this page is held between two luminances,
#: so that it clears 3:1 against white paper and against the dark
#: documentation theme alike; a figure drawn in them needs no canvas of its
#: own and is legible on either. `tests/test_plot_style.py` pins the bound.
INK = "#717c8b"
MUTED = "#7d8996"
FAINT = "#7d899659"

#: Categorical hues, assigned in order and never cycled.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#169869",
    "#b47900",
    "#c5678b",
    "#008300",
    "#7d6fd4",
    "#e34948",
)

#: An acquisition order, first to last. The hue order is the one a reader
#: knows from ``turbo``; the luminance rises gently across it instead of
#: running from near-black to near-white, because both of those ends
#: disappear into one background or the other.
SAMPLING = LinearSegmentedColormap.from_list(
    "pypulseqpp-sampling",
    [
        (0.000, "#4c62de"),
        (0.125, "#3373cb"),
        (0.250, "#128595"),
        (0.375, "#118f66"),
        (0.500, "#3d9436"),
        (0.625, "#6a941c"),
        (0.750, "#988d22"),
        (0.875, "#c48225"),
        (1.000, "#f56918"),
    ],
)

#: A magnitude from nothing, with zero drawn as paper. Zero is transparent
#: rather than white, so it is the paper of whichever background the figure is
#: shown on. The transparent end carries the hue it fades into, so a small
#: value is a faint tint of the ramp rather than a wash of grey. The far end
#: turns towards indigo rather than darkening, which would take it into the
#: dark theme's own background.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "pypulseqpp-magnitude",
    [
        (0.0, "#cfe1f700"),
        (0.2, "#cfe1f7"),
        (0.4, "#86b6ef"),
        (0.6, "#4a90e2"),
        (0.8, "#2a78d6"),
        (1.0, "#6657e9"),
    ],
)

#: A signed quantity about zero, with zero drawn as paper. Each side fades into
#: its own hue, so the transparent centre is approached from peach below and
#: from blue above, and each far end stops at a tone that reads against either
#: background.
SIGNED = LinearSegmentedColormap.from_list(
    "pypulseqpp-signed",
    [
        (0.0, "#bb511c"),
        (1 / 6, "#eb6834"),
        (2 / 6, "#f7c9b3"),
        (0.5, "#f7c9b300"),
        (0.5, "#cfe1f700"),
        (4 / 6, "#cfe1f7"),
        (5 / 6, "#2a78d6"),
        (1.0, "#2670ce"),
    ],
)


def axis_style(axis, title: str = "") -> None:
    """Apply the shared axis style: two faint spines, muted ticks, optional left-aligned title."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(FAINT)
    axis.set_facecolor("none")
    axis.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.grid(False)
    if title:
        axis.set_title(title, loc="left", fontsize=9, color=INK)


def image_style(axis, title: str = "") -> None:
    """As :func:`axis_style`, but keeping all four sides of a heatmap's frame."""
    for spine in axis.spines.values():
        spine.set_color(FAINT)
    axis.grid(False)
    axis.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    if title:
        axis.set_title(title, loc="left", fontsize=9, color=INK)


def figure_title(figure, text: str | None) -> None:
    if text:
        figure.suptitle(text, x=0.01, ha="left", fontsize=10, color=INK)
