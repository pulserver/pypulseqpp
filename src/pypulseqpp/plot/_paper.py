"""Publication-style diagram of one repetition, drawn with mrsd."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np

from .. import _ext as _cxx
from .._waveforms import waveforms_and_times
from . import _seqeyes as _plot

#: Diagram rows, top to bottom.
_ROWS = ("RF", "Gz", "Gy", "Gx", "ADC")

#: Fraction of a row's half-height the largest value on it reaches.
_FILL = 0.9

_RF_PARTS = {"abs": np.abs, "real": np.real, "imag": np.imag}


def select_trs(seq, tr=None, max_underlays=16):
    """Return which repetitions a diagram draws.

    Returns
    -------
    size : int
        Blocks per repetition; the whole sequence when it does not repeat.
    start : int
        1-based block where the first repetition starts.
    main : int or None
        1-based repetition drawn solid: ``tr`` if given, otherwise the one in
        which any physical axis reaches its largest magnitude.
    underlays : list of int
        1-based repetitions drawn underneath: every k-th, with k chosen so
        there are at most ``max_underlays`` of them, together with the
        repetitions in which each axis reaches its most negative and most
        positive value.
    """
    size, start = seq._detect_tr()
    repeats = (seq.num_blocks - start + 1) // size if size else 0
    if repeats == 0:
        return 0, 0, None, []

    played = np.asarray(_cxx.block_extremes(seq._native))
    per_tr = played[start - 1 : start - 1 + repeats * size].reshape(repeats, size, 3, 2)
    low = per_tr[..., 0].min(axis=1)
    high = per_tr[..., 1].max(axis=1)

    if tr is None:
        main = int(np.argmax(np.maximum(-low, high).max(axis=1)))
    elif 1 <= int(tr) <= repeats:
        main = int(tr) - 1
    else:
        raise ValueError(
            f"tr {tr} is not within the {repeats} repetitions of the sequence"
        )

    chosen = set()
    if max_underlays > 0:
        chosen.update(range(0, repeats, math.ceil(repeats / max_underlays)))
        chosen.update(int(index) for index in np.argmin(low, axis=0))
        chosen.update(int(index) for index in np.argmax(high, axis=0))
    chosen.discard(main)
    return size, start, main + 1, sorted(index + 1 for index in chosen)


def _played(seq, first, last, rf_plot):
    """Return one range's rows, ADC windows and duration, timed from its start."""
    found = waveforms_and_times(
        seq, append_RF=True, block_range=(first, last), compat=False
    )
    waves = found.waveforms
    rows = {"Gx": waves.gx, "Gy": waves.gy, "Gz": waves.gz, "RF": np.zeros((2, 0))}
    if waves.rf is not None and waves.rf.shape[1]:
        rows["RF"] = np.vstack((waves.rf[0].real, _RF_PARTS[rf_plot](waves.rf[1])))

    windows = []
    edges = np.concatenate(([0], np.cumsum(found.adc.num_samples)))
    for begin, end in itertools.pairwise(edges):
        samples = found.adc.t[begin:end]
        if samples.size == 0:
            continue
        dwell = samples[1] - samples[0] if samples.size > 1 else 0.0
        windows.append((samples[0] - dwell / 2, samples[-1] - samples[0] + dwell))

    durations = np.asarray(seq._native.block_durations())
    return rows, windows, float(durations[first - 1 : last].sum())


def _pieces(wave, scale, lift=0.0):
    """Split a (2, n) channel into polylines where it rests at zero, scaled onto a row."""
    if wave.shape[1] < 2 or scale <= 0.0:
        return []
    resting = (wave[1][:-1] == 0.0) & (wave[1][1:] == 0.0)
    points = np.column_stack((wave[0], wave[1] / scale * _FILL + lift))
    return [
        piece
        for piece in np.split(points, np.flatnonzero(resting) + 1)
        if len(piece) > 1
    ]


def _largest(drawn, names):
    values = [
        np.abs(rows[name][1]).max()
        for rows, _, _ in drawn
        for name in names
        if rows[name].shape[1]
    ]
    return max(values, default=0.0)


def paper_plot(
    seq,
    time_range=(0, np.inf),
    line_width=1.2,
    axes_color="0.9",
    rf_color="black",
    gx_color="black",
    gy_color="black",
    gz_color="black",
    rf_plot="abs",
    *,
    tr=None,
    max_underlays=16,
    underlay_color="0.8",
    ax=None,
):
    """Draw the diagram; see :meth:`pypulseqpp.Sequence.paper_plot`."""
    if rf_plot not in _RF_PARTS:
        raise ValueError(f"rf_plot is 'abs', 'real' or 'imag', not {rf_plot!r}")

    import matplotlib.pyplot as plt
    import mrsd
    from matplotlib.collections import LineCollection
    from matplotlib.path import Path
    from mrsd.event import Event

    class Trace(Event):
        """A played waveform as an mrsd event: its own path, on its own row."""

        def __init__(self, pieces, **kwargs):
            self._pieces = pieces
            begin = min(piece[0, 0] for piece in pieces)
            end = max(piece[-1, 0] for piece in pieces)
            super().__init__(end - begin, 1.0, begin=begin, **kwargs)

        def get_path(self):
            return Path.make_compound_path(*(Path(piece) for piece in self._pieces))

    main = None
    underlays = []
    if tuple(time_range) != (0, np.inf):
        ranges = [_plot.blocks_for(seq, time_range=time_range)]
    else:
        size, start, main, underlays = select_trs(seq, tr, max_underlays)
        if main is None:
            ranges = [_plot.blocks_for(seq)]
        else:
            ranges = [
                (start + (index - 1) * size, start + index * size - 1)
                for index in (main, *underlays)
            ]
    drawn = [_played(seq, first, last, rf_plot) for first, last in ranges]

    scales = {"RF": _largest(drawn, ("RF",))}
    scales.update(
        dict.fromkeys(("Gx", "Gy", "Gz"), _largest(drawn, ("Gx", "Gy", "Gz")))
    )
    colors = {"RF": rf_color, "Gx": gx_color, "Gy": gy_color, "Gz": gz_color}

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    diagram = mrsd.Diagram(ax, _ROWS)
    for baseline in ax.lines:
        baseline.set_color(axes_color)

    rows, windows, duration = drawn[0]
    for name in ("RF", "Gz", "Gy", "Gx"):
        pieces = _pieces(rows[name], scales[name])
        if pieces:
            diagram.add(
                name, Trace(pieces, edgecolor=colors[name], linewidth=line_width)
            )
        lines = [
            piece
            for other, _, _ in drawn[1:]
            for piece in _pieces(other[name], scales[name], diagram.y(name))
        ]
        if lines:
            ax.add_collection(
                LineCollection(
                    lines,
                    colors=underlay_color,
                    linewidths=0.6 * line_width,
                    zorder=0.5,
                )
            )
    for begin, length in windows:
        diagram.adc(
            "ADC", length, begin=begin, edgecolor=rf_color, linewidth=line_width
        )
    if main is not None:
        diagram.interval(0.0, duration, -1.6, "TR")

    ax.autoscale_view()
    ax.set_xlim(-0.02 * duration, 1.02 * duration)
    return SimpleNamespace(diagram=diagram, tr=main, underlays=underlays)
