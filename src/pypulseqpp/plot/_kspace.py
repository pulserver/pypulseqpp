"""Where the ADC samples land in k-space, and the path between them."""

from __future__ import annotations

import numpy as np

from . import _style
from ._seqeyes import blocks_for

#: What each letter of a ``plane`` draws: the row of the coordinates, and the
#: axis label. ``f`` is the transmit frequency of the slice a sample came from.
_AXES = {
    "x": (0, "$k_x$ [1/m]"),
    "y": (1, "$k_y$ [1/m]"),
    "z": (2, "$k_z$ [1/m]"),
    "f": (3, r"$\Delta f_z$ [Hz]"),
}

_PANELS = {
    None: (None,),
    "echo": ("echo",),
    "shot": ("shot",),
    "order": ("shot", "echo"),
}


def _marker_size(points: np.ndarray) -> float:
    """Return a scatter marker area suited to how crowded ``points`` is."""
    span = max(float(np.ptp(points)), 1e-12)
    grid = np.round(points / (span / 512.0)).astype(np.int64)
    distinct = max(len(np.unique(grid, axis=1)), 1)
    return float(np.clip(2.0e3 / distinct, 1.5, 24.0))


def _frame(axis, points: np.ndarray) -> None:
    """Hold ``axis`` to what ``points`` spans, with a small margin."""
    setters = [axis.set_xlim, axis.set_ylim]
    if hasattr(axis, "set_zlim"):
        setters.append(axis.set_zlim)
    for values, setter in zip(points, setters, strict=False):
        low, high = float(np.min(values)), float(np.max(values))
        margin = 0.05 * max(high - low, 1e-12)
        setter(low - margin, high + margin)


def _readout_trains(seq, first: int, last: int):
    """Return samples per readout, the pulses opening trains, and which each follows.

    ``train`` is 1-based into the opening pulses: a readout follows the last
    excitation played at or before its block.
    """
    found = seq.waveforms_and_times(block_range=(first, last), compat=False)
    counts = np.asarray(found.adc.num_samples, dtype=int)
    if not counts.size:
        raise ValueError("plot_kspace(): the range holds no ADC samples")
    opened = found.rf.of("excitation", "undefined")
    train = np.searchsorted(
        np.asarray(opened.block), np.asarray(found.adc.block), side="right"
    )
    return counts, opened, train


def sampling_order(seq, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
    """Which shot acquired each ADC sample, and which echo of that shot.

    The echo is the ``ECO`` label where the sequence sets one, and otherwise
    the readout's rank after the excitation that opened its train. A shot
    begins wherever the echo returns to zero. One value per sample, aligned
    with ``k_traj_adc``.
    """
    counts, _, train = _readout_trains(seq, first, last)
    echo = seq.evaluate_labels(evolution="adc", block_range=(first, last)).get("ECO")
    if echo is None or np.size(echo) != train.size:
        # ``train`` does not decrease, so the first index holding each value is
        # where that train started.
        echo = np.arange(train.size) - np.searchsorted(train, train, side="left")
    echo = np.asarray(echo, dtype=int).reshape(-1)
    shot = np.maximum(np.cumsum(echo == 0) - 1, 0)
    return np.repeat(shot, counts), np.repeat(echo, counts)


def _slice_offsets(seq, first: int, last: int) -> np.ndarray:
    """Return the frequency each sample's slice was excited at, in Hz."""
    counts, opened, train = _readout_trains(seq, first, last)
    offsets = np.asarray(opened.freq_offset, dtype=float)
    if not offsets.size:
        return np.zeros(int(counts.sum()))
    return np.repeat(offsets[np.clip(train - 1, 0, offsets.size - 1)], counts)


def plot_kspace(
    seq,
    *,
    time_range=None,
    block_range=None,
    tr_range=None,
    plane: str | None = None,
    show_trajectory: bool = True,
    color_by: str | None = None,
    plot_now: bool = True,
):
    """Draw the k-space the ADC samples visit.

    Parameters
    ----------
    seq : Sequence
        The sequence to draw.
    time_range, block_range, tr_range : sequence, optional
        The part of the sequence to draw, as for :meth:`Sequence.plot`; at
        most one. The whole sequence by default.
    plane : {"xy", "xz", "yz", ...}, optional
        Two of ``x``, ``y``, ``z`` and ``f`` to project onto, where ``f`` is
        the transmit frequency of each sample's slice. By default a trajectory
        confined to a plane is drawn in it, and any other in 3D.
    show_trajectory : bool, default True
        Draw the path between samples too. The axes are held to the samples
        either way, so a prewinder or spoiler is clipped rather than setting
        the scale.
    color_by : {"echo", "shot", "order"}, optional
        Colour samples by the echo of its shot that acquired them, by the shot,
        or both side by side. A panel whose index never varies is dropped.
    plot_now : bool, default True
        Show the figure before returning.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Coordinates are physical-axis k-space in 1/m, block rotations applied.
    """
    from matplotlib import pyplot as plt

    panels = _PANELS.get(color_by, ())
    if not panels:
        raise ValueError(
            f"plot_kspace(): color_by must be echo, shot, order or None, not {color_by!r}"
        )
    first, last = blocks_for(seq, time_range, block_range, tr_range)
    found = seq._kspace(block_range=(first, last), samples_only=not show_trajectory)
    adc = np.asarray(found["k_traj_adc"], dtype=float)
    if adc.size == 0:
        raise ValueError("plot_kspace(): the range holds no ADC samples")

    indices = {}
    if color_by is not None:
        shot, echo = sampling_order(seq, first, last)
        indices = {"shot": shot, "echo": echo}
        panels = tuple(name for name in panels if np.ptp(indices[name])) or (panels[0],)

    spread = max(float(np.ptp(adc)), 1e-12)
    used = [axis for axis in range(3) if np.ptp(adc[axis]) > 1e-9 * spread]
    if plane is None and len(used) <= 2:
        used += [axis for axis in range(3) if axis not in used]
        plane = "".join("xyz"[axis] for axis in used[:2])
    if plane is not None and (len(plane) != 2 or any(c not in _AXES for c in plane)):
        raise ValueError(
            f"plot_kspace(): plane must be two of {', '.join(_AXES)}, not {plane!r}"
        )

    coords = np.vstack([adc, np.zeros(adc.shape[1])])
    if plane is not None and "f" in plane:
        coords[3] = _slice_offsets(seq, first, last)
    drawn_rows = [_AXES[c][0] for c in plane] if plane else [0, 1, 2]
    path = (
        np.asarray(found["k_traj"], dtype=float)
        if show_trajectory and 3 not in drawn_rows
        else None
    )
    size = 1.5 if color_by is None else _marker_size(coords[drawn_rows])

    figure = plt.figure(figsize=(5.5 * len(panels), 5.0))
    for column, name in enumerate(panels, start=1):
        values = indices.get(name)
        shared = {
            "s": size,
            "c": "C0" if values is None else values,
            "cmap": None if values is None else _style.SAMPLING,
        }
        if plane is None:
            axis = figure.add_subplot(1, len(panels), column, projection="3d")
            if path is not None:
                axis.plot(path[0], path[1], path[2], lw=0.4, color="0.7")
            drawn = axis.scatter(adc[0], adc[1], adc[2], **shared)
            axis.set_xlabel(_AXES["x"][1])
            axis.set_ylabel(_AXES["y"][1])
            axis.set_zlabel(_AXES["z"][1])
        else:
            one, two = drawn_rows
            axis = figure.add_subplot(1, len(panels), column)
            if path is not None:
                axis.plot(path[one], path[two], lw=0.4, color="0.7")
            drawn = axis.scatter(coords[one], coords[two], **shared)
            axis.set_xlabel(_AXES[plane[0]][1])
            axis.set_ylabel(_AXES[plane[1]][1])
            if 3 not in drawn_rows:
                axis.set_aspect("equal", adjustable="box")
        _frame(axis, coords[drawn_rows])
        if values is not None:
            figure.colorbar(drawn, ax=axis, label=f"{name} index", shrink=0.85)

    figure.tight_layout()
    if plot_now:
        plt.show()
    return figure
