"""Colours and axis treatment shared by the analysis figures."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#b9b8b2"

#: Categorical hues, assigned in order and never cycled.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

#: An acquisition order, first to last.
SAMPLING = "turbo"

#: A magnitude from nothing, with zero drawn as paper.
MAGNITUDE = LinearSegmentedColormap.from_list(
    "pypulseqpp-magnitude",
    ["#ffffff", "#cfe1f7", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"],
)

#: A signed quantity about zero, with zero drawn as paper.
SIGNED = LinearSegmentedColormap.from_list(
    "pypulseqpp-signed",
    ["#8a3a12", "#eb6834", "#f7c9b3", "#ffffff", "#cfe1f7", "#2a78d6", "#123f78"],
)


def axis_style(axis, title: str = "") -> None:
    """Two faint spines and muted ticks, with an optional left-aligned title."""
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
    """Like :func:`axis_style`, keeping all four sides of a heatmap's frame."""
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
