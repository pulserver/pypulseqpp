"""Where the sequence goes in k-space, and where it is sampled.

A gradient moves the spins' phase, and the phase they have accumulated is
where the sequence has got to in k-space -- so the trajectory is the integral
of the gradient waveforms, which `waveforms_and_times` has already reduced to
their corners. Between two corners a gradient is a straight line, so its
integral is a parabola, and the whole trajectory is exact rather than sampled.

Two things reset it. An excitation starts the phase over, so k returns to the
origin; a refocusing turns the accumulated phase around, so k reflects through
the origin. The trajectory is therefore built as a run of periods separated by
the pulses, each shifted to start where the pulse leaves it.
"""

from __future__ import annotations

from warnings import warn

import numpy as np

from . import _ext as _cxx
from ._waveforms import _of

__all__ = ["calculate_kspace"]

#: How close two times have to be to be one time, in seconds. Coarse next to
#: the nanosecond a file is written on, because a moment reached two ways --
#: as a corner and as an ADC sample -- has to come out as one.
_ACCURACY = 1e-10

#: A nanosecond.
_EPS = 1e-9


def calculate_kspace(
    seq,
    trajectory_delay=0.0,
    gradient_offset=0.0,
    block_range=None,
):
    """Return the trajectory, and where along it the samples are taken.

    Parameters
    ----------
    seq : Sequence
        The sequence to follow.
    trajectory_delay : float or sequence of float, default 0
        How late each axis plays what it was asked to, in seconds.
    gradient_offset : float or sequence of float, default 0
        A background gradient per axis, in Hz/m.
    block_range : sequence of int, optional
        Two 1-based block indices; only those blocks are followed.

    Returns
    -------
    k_traj_adc : np.ndarray
        3-by-n: where each ADC sample sits in k-space, in 1/m.
    k_traj : np.ndarray
        3-by-m: the whole trajectory, at every time it changes direction.
    t_excitation : np.ndarray
        When each excitation acts.
    t_refocusing : np.ndarray
        When each refocusing acts.
    t_adc : np.ndarray
        When each sample is taken.

    Notes
    -----
    This is upstream's return, in upstream's order. Everything the
    calculation produces -- the trajectory's own time base, the slice
    positions, the gradients as splines -- comes back from :func:`detail`.
    """
    found = detail(seq, trajectory_delay, gradient_offset, block_range)
    return (
        found["k_traj_adc"],
        found["k_traj"],
        found["t_excitation"],
        found["t_refocusing"],
        found["t_adc"],
    )


def detail(seq, trajectory_delay=0.0, gradient_offset=0.0, block_range=None) -> dict:
    """Return everything following the trajectory produces.

    Returns
    -------
    dict
        ``k_traj_adc``, ``t_adc``, ``k_traj``, ``t_ktraj``, ``t_excitation``,
        ``t_refocusing``, ``slicepos``, ``t_slicepos``, ``gw_pp`` and
        ``pm_adc`` -- what the reference toolbox reports, by name rather than
        by position.

    Notes
    -----
    The integration is a compiled pass. Both the gradient corners and the
    moments the trajectory is asked about are in order, so it walks them
    together rather than searching one for the other, and the moments it has
    to know are gathered and sorted once instead of an array being built per
    ramp.
    """
    if np.any(np.abs(trajectory_delay) > 100e-6):
        warn(
            f"trajectory delay of ({np.asarray(trajectory_delay) * 1e6}) us is "
            f"suspiciously high",
            stacklevel=2,
        )

    system = seq.system
    first, last = _blocks_asked_for(block_range)
    found = _cxx.calculate_kspace(
        seq._native,
        delay=_per_axis(trajectory_delay),
        offset=_per_axis(gradient_offset),
        first_block=first,
        last_block=last,
        b0=_of(system, "B0", 1.5),
        gamma=_of(system, "gamma", 42576000.0),
    )
    for complaint in found["warnings"]:
        warn(complaint, stacklevel=2)

    excitations = found["t_excitation"]
    return {
        "k_traj_adc": found["k_traj_adc"],
        "t_adc": found["t_adc"],
        "k_traj": found["k_traj"],
        "t_ktraj": found["t_ktraj"],
        "t_excitation": excitations,
        "t_refocusing": found["t_refocusing"],
        "slicepos": found["slicepos"] if excitations.size else np.array([]),
        "t_slicepos": excitations if excitations.size else np.array([]),
        "gw_pp": [_spline(channel) for channel in found["gradients"]],
        "pm_adc": found["pm_adc"],
    }


def _blocks_asked_for(block_range):
    """Return the first and last block to follow; 0 for the last means the end."""
    if block_range is None:
        return 1, 0
    if len(block_range) != 2:
        raise ValueError("parameter 'block_range' must contain exactly two numbers")
    first = max(int(block_range[0]), 1)
    last = 0 if not np.isfinite(block_range[1]) else int(block_range[1])
    if last and last < first:
        raise ValueError("block_range end must not be smaller than block_range start")
    return first, last


def _per_axis(value):
    """One value per gradient axis, whether one was given or three."""
    if np.isscalar(value):
        return [float(value)] * 3
    given = [float(each) for each in value]
    if len(given) != 3:
        raise ValueError("expected one value per gradient axis")
    return given


def _spline(channel):
    """One axis's gradient as the first-order spline it is.

    A waveform given as its corners is a straight line between them, so its
    coefficients are the slope and the value it starts each piece at.
    """
    from scipy.interpolate import PPoly

    times = channel["t"]
    values = channel["v"]
    if times.size < 2:
        return None
    return PPoly(
        np.vstack((np.diff(values) / np.diff(times), values[:-1])),
        times,
        extrapolate=True,
    )
