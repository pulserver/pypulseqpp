"""The sequence as what the gradients and the digitiser actually do.

A block table says which events a sequence plays and when each block starts.
Everything that looks at what a sequence *does* -- where it goes in k-space,
how fast the gradients slew, when a sample is taken, what a plot draws --
needs the other view: one waveform per axis over the whole scan, on a time
base shared by all of them.

Building it is a pass over every block, and a block contributes a handful of
points rather than one, so it is compiled. What comes back is what the
toolboxes return, in the same shapes.

A waveform is given here as its corners. A gradient is played by
interpolating between the samples it is given, so the points where its slope
changes describe it completely: a trapezoid is four points however long its
flat top, and a shape stored on the raster is restored to the corners the
interpreter will draw between.
"""

from __future__ import annotations

from warnings import warn

import numpy as np

from . import _ext as _cxx

__all__ = [
    "adc_times",
    "get_gradients",
    "rf_times",
    "waveforms",
    "waveforms_and_times",
]

#: A nanosecond, the grid a Pulseq file is written on.
_EPS = 1e-9


def _expand(seq, append_rf=False, time_range=None, block_range=None):
    """Run the compiled pass, and say what it had to complain about."""
    if block_range is not None and time_range is not None:
        raise ValueError("Specify either blockRange or time_range, not both")

    first, last, elapsed = 1, 0, 0.0
    if block_range is not None:
        if len(block_range) != 2:
            raise ValueError("parameter 'blockRange' must contain exactly two numbers")
        first = max(int(block_range[0]), 1)
        last = 0 if not np.isfinite(block_range[1]) else int(block_range[1])
    elif time_range is not None:
        first, last, elapsed = _blocks_within(seq, time_range)

    system = seq.system
    expanded = _cxx.waveforms_and_times(
        seq._native,
        append_rf=append_rf,
        first_block=first,
        last_block=last,
        b0=_of(system, "B0", 1.5),
        gamma=_of(system, "gamma", 42576000.0),
    )

    if expanded["rotated_blocks"]:
        blocks = expanded["rotated_blocks"]
        raise NotImplementedError(
            f"waveforms_and_times() cannot yet expand a rotated block; "
            f"{len(blocks)} of them, first at block {blocks[0]}. A rotation "
            f"remaps a block's gradients onto other axes, which is a different "
            f"waveform on each rather than this one moved."
        )

    for complaint in expanded["warnings"]:
        warn(complaint, stacklevel=3)

    return expanded, elapsed


def _of(system, name, fallback):
    value = getattr(system, name, None) if system is not None else None
    return fallback if value is None else float(value)


def _blocks_within(seq, time_range):
    """Return the blocks a time range touches, and when the first starts."""
    if len(time_range) != 2:
        raise ValueError("Time range must be list of two elements")
    if time_range[0] > time_range[1]:
        raise ValueError("End time of time_range must be after begin time")

    durations = np.asarray(seq._native.block_durations())
    ends = np.cumsum(durations)
    first = int(np.searchsorted(ends, time_range[0]))
    last = int(np.searchsorted(ends - durations, time_range[1], side="right"))
    elapsed = float(ends[first] - durations[first]) if first < len(durations) else 0.0
    return first + 1, last, elapsed


def _shifted(times, elapsed):
    """Return times measured from the start of the scan, not of the range."""
    return times if elapsed == 0.0 else times + elapsed


def waveforms_and_times(seq, append_RF: bool = False, time_range=None, blockRange=None):
    """Return the gradient waveforms, the RF moments and the ADC sampling.

    Parameters
    ----------
    seq : Sequence
        The sequence to expand.
    append_RF : bool, default False
        Also return the RF envelope, as a fourth channel.
    time_range : list of float, optional
        Two times in seconds; only the blocks they touch are expanded.
    blockRange : sequence of int, optional
        Two 1-based block indices. Not with ``time_range``.

    Returns
    -------
    wave_data : list of np.ndarray
        Per gradient axis, a 2-by-n array: the times over the amplitudes.
        With ``append_RF``, a fourth holding the complex RF envelope.
    tfp_excitation : np.ndarray
        3-by-n: when each excitation acts, and at what frequency and phase.
    tfp_refocusing : np.ndarray
        The same for the refocusings.
    t_adc : np.ndarray
        When every ADC sample is taken.
    fp_adc : np.ndarray
        2-by-n: the frequency and phase of each sample.
    pm_adc : np.ndarray
        The phase modulation of each sample.
    """
    expanded, elapsed = _expand(seq, append_RF, time_range, blockRange)
    waves = [np.array(channel, copy=True) for channel in expanded["wave_data"]]
    if elapsed:
        for channel in waves:
            if channel.size:
                channel[0] += elapsed
    return (
        waves,
        _shift_row(expanded["tfp_excitation"], elapsed),
        _shift_row(expanded["tfp_refocusing"], elapsed),
        _shifted(expanded["t_adc"], elapsed),
        expanded["fp_adc"],
        expanded["pm_adc"],
    )


def _shift_row(moments, elapsed):
    """Move the time row of a 3-by-n moment array."""
    if not elapsed or moments.size == 0:
        return moments
    moved = np.array(moments, copy=True)
    moved[0] += elapsed
    return moved


def waveforms(seq, append_RF: bool = False, time_range=None, blockRange=None):
    """Return the gradient waveforms alone. See :func:`waveforms_and_times`."""
    return waveforms_and_times(seq, append_RF, time_range, blockRange)[0]


def adc_times(seq, time_range=None):
    """Return when every ADC sample is taken, and each window's offsets.

    Returns
    -------
    t_adc : np.ndarray
        When every sample is taken, in seconds from the start of the scan.
    fp_adc : np.ndarray
        n-by-2, one row per ADC window rather than per sample: the frequency
        and phase offsets it was asked for, as the sequence records them.
    """
    expanded, elapsed = _expand(seq, time_range=time_range)
    return _shifted(expanded["t_adc"], elapsed), expanded["window_fp"]


def rf_times(seq, time_range=None):
    """Return when the pulses act, and at what frequency and phase.

    Returns
    -------
    t_excitation : np.ndarray
        When each excitation acts -- its centre, not its start.
    fp_excitation : np.ndarray
        2-by-n: the frequency, and the phase accumulated by then.
    t_refocusing : np.ndarray
    fp_refocusing : np.ndarray
        The same for the refocusings.
    """
    expanded, elapsed = _expand(seq, time_range=time_range)
    excitation = _shift_row(expanded["tfp_excitation"], elapsed)
    refocusing = _shift_row(expanded["tfp_refocusing"], elapsed)
    return (
        np.array(excitation[0], copy=True),
        np.array(excitation[1:3], copy=True),
        np.array(refocusing[0], copy=True),
        np.array(refocusing[1:3], copy=True),
    )


def _durations_within(seq, time_range=None, block_range=None) -> float:
    """How long the blocks a range touches last, in total."""
    durations = np.asarray(seq._native.block_durations())
    if block_range is not None:
        first = max(int(block_range[0]), 1)
        last = (
            len(durations) if not np.isfinite(block_range[1]) else int(block_range[1])
        )
        return float(durations[first - 1 : last].sum())
    if time_range is not None:
        first, last, _ = _blocks_within(seq, time_range)
        last = len(durations) if last == 0 else last
        return float(durations[first - 1 : last].sum())
    return float(durations.sum())


def get_gradients(
    seq,
    trajectory_delay=0,
    gradient_offset=0,
    time_range=None,
    blockRange=None,
):
    """Return each gradient axis as a piecewise polynomial.

    A waveform given as its corners is a first-order spline, so handing it
    back as one lets a caller read the gradient at any moment, and integrate
    it, without interpolating by hand.

    Parameters
    ----------
    seq : Sequence
        The sequence to expand.
    trajectory_delay : float or sequence of float, default 0
        How late each axis plays what it was asked to, in seconds.
    gradient_offset : float or sequence of float, default 0
        A background gradient per axis, in Hz/m.
    time_range, blockRange
        As for :func:`waveforms_and_times`.

    Returns
    -------
    list
        One `scipy.interpolate.PPoly` per axis, None where an axis plays
        nothing and has no offset.
    """
    from scipy.interpolate import PPoly

    if np.any(np.abs(trajectory_delay) > 100e-6):
        warn(
            f"Trajectory delay of {np.asarray(trajectory_delay) * 1e6} us is "
            f"suspiciously high",
            stacklevel=2,
        )

    channels = waveforms(seq, time_range=time_range, blockRange=blockRange)
    axes = len(channels)
    total = _durations_within(seq, time_range, blockRange)

    delays = _per_axis(trajectory_delay, axes)
    offsets = _per_axis(gradient_offset, axes)

    tiny = 1e-12
    splines = [None] * axes
    for axis, channel in enumerate(channels):
        if channel.shape[1] == 0:
            if abs(offsets[axis]) <= _EPS:
                continue
            wave = np.array([[0.0, total], [0.0, 0.0]], dtype=float)
        else:
            wave = np.array(channel, dtype=float, copy=True)

        if abs(delays[axis]) > _EPS:
            wave[0] -= delays[axis]
        if not np.all(np.isfinite(wave)):
            warn("Not all elements of the generated waveform are finite.", stacklevel=2)

        # A spline is only defined between its knots, so an axis that starts
        # late or stops early is held at zero either side of what it plays.
        if wave[0, 0] > tiny:
            wave = np.hstack((np.array([[-tiny, wave[0, 0] - tiny], [0.0, 0.0]]), wave))
        if wave[0, -1] < total - 1e-9:
            wave = np.hstack(
                (wave, np.array([[wave[0, -1] + tiny, total + tiny], [0.0, 0.0]]))
            )

        if abs(offsets[axis]) > _EPS:
            wave[1] += offsets[axis]

        steps = np.diff(wave[0])
        keep = np.concatenate(([True], steps > 1e-9))
        if not np.all(keep):
            if np.any(steps <= 0.0):
                warn(
                    "Warning: not all elements of the generated time vector are "
                    "unique and sorted in accending order!",
                    stacklevel=2,
                )
            wave = wave[:, keep]

        wave[1][wave[1] == -0.0] = 0.0
        splines[axis] = PPoly(
            np.vstack((np.diff(wave[1]) / np.diff(wave[0]), wave[1][:-1])),
            wave[0],
            extrapolate=True,
        )

    return splines


def _per_axis(value, axes):
    """One value per axis, whether one was given or several."""
    if isinstance(value, (int, float)):
        return [float(value)] * axes
    given = list(value)
    if len(given) != axes:
        raise ValueError(f"expected {axes} values, one per gradient axis")
    return [float(each) for each in given]
