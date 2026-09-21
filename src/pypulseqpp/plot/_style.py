"""Colours and axis styling shared by the analysis figures."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

#: Axis furniture, in tones that clear 3.5:1 against white paper and against a
#: dark documentation theme alike. A figure drawn in them needs no canvas, so
#: the figures this package draws are readable on either.
INK = "#6b7684"
MUTED = "#7b8794"
FAINT = "#7b879459"

#: Categorical hues, assigned in order and never cycled.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#7d6fd4",
    "#e34948",
)

#: An acquisition order, first to last.
SAMPLING = "turbo"

#: A magnitude from nothing, with zero drawn as paper. Zero is transparent
#: rather than white, so it is the paper of whichever background the figure is
#: shown on. The transparent end carries the hue it fades into, so a small
#: value is a faint tint of the ramp rather than a wash of grey.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "pypulseqpp-magnitude",
    [
        (0.0, "#cfe1f700"),
        (0.2, "#cfe1f7"),
        (0.4, "#86b6ef"),
        (0.6, "#2a78d6"),
        (0.8, "#184f95"),
        (1.0, "#0d366b"),
    ],
)

#: A signed quantity about zero, with zero drawn as paper. Each side fades into
#: its own hue, so the transparent centre is approached from peach below and
#: from blue above.
SIGNED = LinearSegmentedColormap.from_list(
    "pypulseqpp-signed",
    [
        (0.0, "#8a3a12"),
        (1 / 6, "#eb6834"),
        (2 / 6, "#f7c9b3"),
        (0.5, "#f7c9b300"),
        (0.5, "#cfe1f700"),
        (4 / 6, "#cfe1f7"),
        (5 / 6, "#2a78d6"),
        (1.0, "#123f78"),
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
