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

from itertools import pairwise
from warnings import warn

import numpy as np

from ._waveforms import _expand, get_gradients

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
    """
    if np.any(np.abs(trajectory_delay) > 100e-6):
        warn(
            f"trajectory delay of ({np.asarray(trajectory_delay) * 1e6}) us is "
            f"suspiciously high",
            stacklevel=2,
        )

    expanded, _ = _expand(seq, False, None, block_range)
    total = float(expanded["duration"])
    tfp_excitation = expanded["tfp_excitation"]
    tfp_refocusing = expanded["tfp_refocusing"]
    t_adc = expanded["t_adc"]

    gw_pp = get_gradients(
        seq,
        trajectory_delay=trajectory_delay,
        gradient_offset=gradient_offset,
        block_range=block_range,
    )
    axes = len(gw_pp)

    slicepos, t_slicepos = _slice_positions(gw_pp, tfp_excitation)

    # The phase a gradient has accumulated: a straight line integrates to a
    # parabola, so the trajectory is exact between the corners rather than
    # sampled at them.
    moments = [None if spline is None else spline.antiderivative() for spline in gw_pp]

    t_excitation = tfp_excitation[0] if tfp_excitation.size else np.zeros(0)
    t_refocusing = tfp_refocusing[0] if tfp_refocusing.size else np.zeros(0)

    t_ktraj = _times_to_follow(seq, moments, t_excitation, t_refocusing, t_adc, total)
    where = _index_of(t_ktraj)

    k_traj = np.zeros((3, len(t_ktraj)))
    for axis in range(min(axes, 3)):
        if moments[axis] is None:
            continue
        within = np.flatnonzero(
            (t_ktraj >= _rounded(moments[axis].x[0]))
            & (t_ktraj <= _rounded(moments[axis].x[-1]))
        )
        if within.size == 0:
            continue
        k_traj[axis, within] = moments[axis](t_ktraj[within])
        # After the last gradient the phase stays where it was left.
        if t_ktraj[within[-1]] < t_ktraj[-1]:
            k_traj[axis, within[-1] + 1 :] = k_traj[axis, within[-1]]

    _apply_pulses(
        k_traj,
        t_ktraj,
        where(t_excitation),
        where(t_refocusing),
        t_excitation,
    )

    return {
        "k_traj_adc": k_traj[:, where(t_adc)],
        "t_adc": t_adc,
        "k_traj": k_traj,
        "t_ktraj": t_ktraj,
        "t_excitation": t_excitation,
        "t_refocusing": t_refocusing,
        "slicepos": slicepos,
        "t_slicepos": t_slicepos,
        "gw_pp": gw_pp,
        "pm_adc": expanded["pm_adc"],
    }


def _rounded(when):
    """Return a time on the grid two times must agree on to be one time."""
    return _ACCURACY * np.round(when / _ACCURACY)


def _slice_positions(gw_pp, tfp_excitation):
    """Where each excitation put its slice, from the gradient it was played on.

    A slice-selective pulse excites where its frequency offset matches what
    the gradient makes the Larmor frequency there, so dividing one by the
    other is the position.
    """
    if tfp_excitation.size == 0:
        return np.array([]), np.array([])

    positions = np.zeros((len(gw_pp), tfp_excitation.shape[1]))
    for axis, spline in enumerate(gw_pp):
        if spline is None:
            positions[axis, :] = np.nan
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            positions[axis, :] = tfp_excitation[1] / spline(tfp_excitation[0])
    # An axis playing nothing at that moment says nothing about where it was.
    positions[~np.isfinite(positions)] = 0.0
    return positions, tfp_excitation[0]


def _times_to_follow(seq, moments, t_excitation, t_refocusing, t_adc, total):
    """Every moment the trajectory has to be known at.

    The corners it changes direction at, the moments the pulses act -- and
    just before them, since a pulse makes it discontinuous -- every ADC
    sample, and the ends.
    """
    raster = seq.system.grad_raster_time if seq.system is not None else 10e-6
    rf_raster = seq.system.rf_raster_time if seq.system is not None else 1e-6

    wanted = [np.zeros(1), np.asarray([total])]
    for moment in moments:
        if moment is None:
            continue
        wanted.append(moment.x)
        # Where the gradient is ramping the trajectory curves, so the raster
        # is sampled through it rather than only at its ends.
        ramping = np.flatnonzero(np.abs(moment.c[0, :]) > _EPS)
        for piece in ramping:
            wanted.append(
                np.arange(
                    np.floor(moment.x[piece] / raster),
                    np.ceil(moment.x[piece + 1] / raster) + 1,
                )
                * raster
            )

    wanted.extend(
        (
            np.asarray(t_excitation) - 2 * rf_raster,
            np.asarray(t_excitation) - rf_raster,
            np.asarray(t_excitation),
            np.asarray(t_refocusing) - rf_raster,
            np.asarray(t_refocusing),
            np.asarray(t_adc),
        )
    )
    return _ACCURACY * np.unique(np.round(np.concatenate(wanted) / _ACCURACY))


def _index_of(t_ktraj):
    """Return a lookup from a time to where it sits in the trajectory."""
    keys = np.round(t_ktraj / _ACCURACY).astype(np.int64)
    known = {key: index for index, key in enumerate(keys)}

    def find(times):
        times = np.asarray(times)
        if times.size == 0:
            return np.zeros(0, dtype=int)
        return np.array(
            [known[key] for key in np.round(times / _ACCURACY).astype(np.int64)],
            dtype=int,
        )

    return find


def _apply_pulses(k_traj, t_ktraj, i_excitation, i_refocusing, t_excitation):
    """Start the trajectory over at each excitation and turn it at each refocusing.

    Between two pulses the trajectory is the integral as it stands, shifted so
    it begins where the last pulse left it: an excitation puts it back at the
    origin, and a refocusing reflects it through the origin, which is what
    makes a spin echo come back.
    """
    boundaries = np.unique(
        np.concatenate(
            (np.zeros(1, dtype=int), i_excitation, i_refocusing, [len(t_ktraj) - 1])
        )
    )
    shift = -k_traj[:, 0]
    next_excitation = 0
    next_refocusing = 0
    ends = 0

    for start, ends in pairwise(boundaries):
        if (
            next_excitation < len(i_excitation)
            and i_excitation[next_excitation] == start
        ):
            if abs(t_ktraj[start] - t_excitation[next_excitation]) > _ACCURACY:
                warn(
                    f"abs(t_ktraj[i_period]-t_excitation[ii_next_excitation])<"
                    f"{_ACCURACY} failed for ii_next_excitation={next_excitation} "
                    f"error={t_ktraj[start] - t_excitation[next_excitation]}",
                    stacklevel=3,
                )
            shift = -k_traj[:, start]
            # The sample before an excitation belongs to the period that ended.
            if start > 0:
                k_traj[:, start - 1] = np.nan
            next_excitation += 1
        elif (
            next_refocusing < len(i_refocusing)
            and i_refocusing[next_refocusing] == start
        ):
            shift = -2 * k_traj[:, start] - shift
            next_refocusing += 1

        k_traj[:, start:ends] += shift[:, None]

    k_traj[:, ends] += shift
