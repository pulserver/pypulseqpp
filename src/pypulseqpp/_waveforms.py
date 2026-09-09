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
from ._results import AdcTimes, RfTimes, Waveforms, WaveformsAndTimes, use_of

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
        raise ValueError("Specify either block_range or time_range, not both")

    first, last, elapsed = 1, 0, 0.0
    if block_range is not None:
        if len(block_range) != 2:
            raise ValueError("parameter 'block_range' must contain exactly two numbers")
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


def _named(expanded, elapsed, append_rf):
    """Return the expansion as `WaveformsAndTimes`, timed where it plays."""
    channels = [np.array(channel, copy=True) for channel in expanded["wave_data"]]
    if elapsed:
        for channel in channels:
            if channel.size:
                channel[0] += elapsed
    while len(channels) < 3:
        channels.append(np.zeros((2, 0)))

    pulses = _shift_row(expanded["tfp_pulses"], elapsed)
    rf = RfTimes(
        t=np.array(pulses[0], copy=True),
        freq_offset=np.array(pulses[1], copy=True),
        phase_offset=np.array(pulses[2], copy=True),
        use=tuple(use_of(code) for code in expanded["pulse_uses"]),
        block=np.array(expanded["pulse_blocks"], copy=True),
    )

    windows = expanded["window_fp"]
    fp_adc = expanded["fp_adc"]
    adc = AdcTimes(
        t=_shifted(expanded["t_adc"], elapsed),
        freq_offset=np.array(windows[:, 0], copy=True),
        phase_offset=np.array(windows[:, 1], copy=True),
        phase_modulation=np.array(expanded["pm_adc"], copy=True),
        sample_phase=np.array(fp_adc[1], copy=True),
        sample_frequency=np.array(fp_adc[0], copy=True),
        block=np.array(expanded["window_blocks"], copy=True),
        num_samples=np.array(expanded["window_samples"], copy=True),
    )

    return WaveformsAndTimes(
        waveforms=Waveforms(
            gx=channels[0],
            gy=channels[1],
            gz=channels[2],
            rf=channels[3] if append_rf and len(channels) > 3 else None,
        ),
        rf=rf,
        adc=adc,
    )


def waveforms_and_times(
    seq,
    append_RF: bool = False,
    time_range=None,
    block_range=None,
    *,
    compat: bool = True,
):
    """Return the gradient waveforms, the RF moments and the ADC sampling.

    Parameters
    ----------
    seq : Sequence
        The sequence to expand.
    append_RF : bool, default False
        Also return the RF envelope, as a fourth channel.
    time_range : list of float, optional
        Two times in seconds; only the blocks they touch are expanded.
    block_range : sequence of int, optional
        Two 1-based block indices. Not with ``time_range``.
    compat : bool, default True
        Upstream's five values. False returns a
        :class:`pypulseqpp._results.WaveformsAndTimes`, which carries what
        those five cannot.

    Returns
    -------
    tuple or WaveformsAndTimes
        With ``compat``: ``(wave_data, tfp_excitation, tfp_refocusing, t_adc,
        fp_adc)``, which is what upstream returns and what a script written
        against it unpacks.

    Notes
    -----
    Three things the five-tuple cannot say, and ``compat=False`` is where
    they come out:

    - *Every* RF use. Pulseq has seven and the tuple carries two: an
      inversion, a saturation or a preparation pulse is not in it at all.
    - The per-sample ADC phase and phase modulation -- the phase a sample is
      actually acquired with, which is what a simulation wants. The reference
      toolbox returns the modulation as a sixth value; upstream returns
      neither.
    - Which block each pulse and each ADC window is in.
    """
    expanded, elapsed = _expand(seq, append_RF, time_range, block_range)
    found = _named(expanded, elapsed, append_RF)
    if not compat:
        return found
    return (
        found.waveforms.channels,
        found.rf.of("excitation", "undefined").tfp,
        found.rf.of("refocusing").tfp,
        found.adc.t,
        found.adc.fp,
    )


def _shift_row(moments, elapsed):
    """Move the time row of a 3-by-n moment array."""
    if not elapsed or moments.size == 0:
        return moments
    moved = np.array(moments, copy=True)
    moved[0] += elapsed
    return moved


def waveforms(seq, append_RF: bool = False, time_range=None, block_range=None):
    """Return the gradient waveforms alone. See :func:`waveforms_and_times`."""
    return waveforms_and_times(seq, append_RF, time_range, block_range)[0]


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


def rf_times(seq, time_range=None, *, compat: bool = True):
    """Return when the pulses act, and at what frequency and phase.

    Parameters
    ----------
    seq : Sequence
        The sequence to expand.
    time_range : list of float, optional
        Two times in seconds; only the blocks they touch are expanded.
    compat : bool, default True
        Upstream's four values, which describe two of Pulseq's seven RF uses
        and drop the rest. False returns a
        :class:`pypulseqpp._results.RfTimes` covering all of them.

    Returns
    -------
    tuple or RfTimes
        With ``compat``: ``(t_excitation, fp_excitation, t_refocusing,
        fp_refocusing)``. A pulse whose row records no use is counted as an
        excitation, which is what upstream does with one.
    """
    expanded, elapsed = _expand(seq, time_range=time_range)
    pulses = _named(expanded, elapsed, False).rf
    if not compat:
        return pulses

    excitation = pulses.of("excitation", "undefined").tfp
    refocusing = pulses.of("refocusing").tfp
    return (
        np.array(excitation[0], copy=True),
        np.array(excitation[1:3], copy=True),
        np.array(refocusing[0], copy=True),
        np.array(refocusing[1:3], copy=True),
    )


def get_gradients(
    seq,
    trajectory_delay=0,
    gradient_offset=0,
    time_range=None,
    block_range=None,
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
    time_range, block_range
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

    expanded, elapsed = _expand(seq, False, time_range, block_range)
    channels = [np.array(channel, copy=True) for channel in expanded["wave_data"]]
    if elapsed:
        for channel in channels:
            if channel.size:
                channel[0] += elapsed
    axes = len(channels)
    # The same running sum the waveform times were measured against: adding
    # the durations again here gives a different last bit, and whether an axis
    # stops before the end then depends on which machine is asking.
    total = elapsed + expanded["duration"]

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
