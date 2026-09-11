"""Relaxation-free Bloch simulation of RF pulses."""

from __future__ import annotations

__all__ = ["sim_bloch", "sim_rf"]

import warnings as _warnings

import numpy as _np
import pypulseq as _pp

from ._calc_rf_bandwidth import calc_rf_bandwidth as _calc_rf_bandwidth
from ._ext import sim as _kernels

#: ``(bandwidth threshold in Hz, raster in s)``: ``sim_rf``'s default ``dt`` is
#: the first raster whose threshold the bandwidth exceeds, so wider pulses get
#: finer steps and a small rotation per step.
_RASTERS = ((2e4, 1e-6), (1e4, 2e-6), (4e3, 5e-6), (0.0, 10e-6))


def _about_z(angle):
    """Right-hand rotations about z by ``angle``, one ``(3, 3)`` per entry."""
    cosine, sine = _np.cos(angle), _np.sin(angle)
    turns = _np.zeros((_np.size(angle), 3, 3))
    turns[:, 0, 0] = turns[:, 1, 1] = cosine
    turns[:, 0, 1], turns[:, 1, 0] = -sine, sine
    turns[:, 2, 2] = 1.0
    return turns


def sim_bloch(b1_hz, bz_hz, dt: float, *, initial=None) -> _np.ndarray:
    """Simulate the Bloch equation without relaxation, in hard-pulse steps.

    Each step rotates the magnetisation right-handedly about
    ``(Re b1, Im b1, bz)`` by ``2 pi dt`` times that vector's length.

    Parameters
    ----------
    b1_hz : array_like
        Complex transverse field per step, in Hz: ``(T,)`` shared by every
        position, or ``(P, T)`` per position (e.g. pTx channels summed
        through their B1 maps).
    bz_hz : array_like
        Longitudinal field, in Hz: ``(P, T)``, or ``(P, 1)`` held throughout.
        Its rows define the positions.
    dt : float
        Step, in s.
    initial : array_like, optional
        Starting magnetisation, ``(3,)`` or ``(P, 3)``; ``+z`` by default.

    Returns
    -------
    numpy.ndarray
        Final magnetisation, ``(P, 3)``.

    Raises
    ------
    ValueError
        If the fields' shapes disagree on positions or steps.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp

    A 250 Hz hard pulse held for 1 ms is a 90 degree flip about ``+x``:

    >>> on_resonance = np.zeros((1, 1))
    >>> pp.sim_bloch(np.full(1000, 250.0 + 0j), on_resonance, 1e-6).round(3) + 0.0
    array([[ 0., -1.,  0.]])

    Twice the amplitude inverts:

    >>> pp.sim_bloch(np.full(1000, 500.0 + 0j), on_resonance, 1e-6).round(3) + 0.0
    array([[ 0.,  0., -1.]])

    One row of ``bz_hz`` per off-resonance:

    >>> offsets = np.array([[0.0], [500.0], [-500.0]])
    >>> pp.sim_bloch(np.full(1000, 250.0 + 0j), offsets, 1e-6).round(3) + 0.0
    array([[ 0.   , -1.   ,  0.   ],
           [ 0.773,  0.162,  0.614],
           [-0.773,  0.162,  0.614]])
    """
    b1_hz = _np.asarray(b1_hz, dtype=complex)
    bz_hz = _np.atleast_2d(_np.asarray(bz_hz, dtype=float))
    turns = _kernels.rotations(b1_hz, bz_hz, float(dt))
    start = (
        _np.array([0.0, 0.0, 1.0])
        if initial is None
        else _np.asarray(initial, dtype=float)
    )
    start = _np.broadcast_to(start, (turns.shape[0], 3))
    return _np.einsum("pij,pj->pi", turns, start)


def sim_rf(
    rf,
    rephase_factor: float | None = None,
    prephase_factor: float = 0.0,
    *,
    df: float = 1.0,
    bandwidth_multiplier: float = 4.0,
    dt: float | None = None,
):
    """Simulate an RF pulse versus off-resonance without relaxation.

    Follows MATLAB Pulseq's ``simRf``: a hard-pulse approximation with no
    selection gradient, so frequency maps to position only for a constant
    selection gradient.

    Parameters
    ----------
    rf : SimpleNamespace or RfEvent
        Its ``freq_ppm`` and ``phase_ppm`` are converted with the default
        system's gamma and B0, with a warning.
    rephase_factor : float, optional
        Free precession after the pulse, as a signed fraction of its
        duration. Defaults to zero when ``rf.use == "refocusing"`` and to
        ``-(shape_dur - center) / shape_dur``, the slice-select rephaser,
        otherwise.
    prephase_factor : float, optional
        The same, before the pulse.
    df : float, optional
        In Hz: the axis holds ``round(bandwidth / df)`` points, so their
        spacing is about ``bandwidth_multiplier * df``.
    bandwidth_multiplier : float, optional
        Width of the frequency axis, in bandwidths. The bandwidth is
        :func:`calc_rf_bandwidth` at half maximum plus ``|freq_offset|``;
        the axis is centred on ``freq_offset``.
    dt : float, optional
        Simulation raster, in s. Chosen from the bandwidth when omitted.

    Returns
    -------
    mz_z : numpy.ndarray
        ``Mz`` after the pulse, starting from ``+z``.
    mz_xy : numpy.ndarray
        ``Mx + i My`` after the pulse, starting from ``+z``.
    f : numpy.ndarray
        The frequency axis, in Hz.
    ref_eff : numpy.ndarray
        Complex refocusing efficiency: its magnitude is the refocused fraction
        and its phase twice the azimuth of the refocusing axis.
    mx_xy, my_xy : numpy.ndarray
        ``Mx + i My`` starting from ``+x`` and from ``+y``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp

    A hard 90 takes ``+z`` into the transverse plane, on resonance:

    >>> rf = pp.make_block_pulse(np.pi / 2, duration=0.5e-3)
    >>> mz_z, mz_xy, f = pp.sim_rf(rf)[:3]
    >>> on_resonance = int(np.argmin(abs(f)))
    >>> bool(abs(mz_z[on_resonance]) < 0.15), bool(abs(mz_xy[on_resonance]) > 0.85)
    (True, True)

    A hard 180 inverts it:

    >>> mz_z, _, f = pp.sim_rf(pp.make_block_pulse(np.pi, duration=0.7e-3))[:3]
    >>> bool(mz_z[int(np.argmin(abs(f)))] < -0.85)
    True

    See Also
    --------
    calc_rf_bandwidth : the width alone, from the envelope's transform.
    """
    if rephase_factor is None:
        rephase_factor = (
            0.0
            if getattr(rf, "use", None) == "refocusing"
            else -(rf.shape_dur - rf.center) / rf.shape_dur
        )

    freq_ppm = float(getattr(rf, "freq_ppm", 0.0) or 0.0)
    phase_ppm = float(getattr(rf, "phase_ppm", 0.0) or 0.0)
    freq_offset = float(rf.freq_offset)
    phase_offset = float(rf.phase_offset)
    if max(abs(freq_ppm), abs(phase_ppm)) > _np.finfo(float).eps:
        _warnings.warn(
            "sim_rf(): a ppm offset is read against the gamma and B0 of the "
            "default system",
            stacklevel=2,
        )
        system = _pp.Opts.default
        freq_offset += freq_ppm * 1e-6 * system.gamma * system.B0
        phase_offset += phase_ppm * 1e-6 * system.gamma * system.B0

    bandwidth = abs(_calc_rf_bandwidth(rf, cutoff=0.5, dw=df * 10.0, dt=10e-6)) + abs(
        freq_offset
    )
    if dt is None:
        dt = next(step for over, step in _RASTERS if bandwidth > over)

    t = (_np.arange(1, int(_np.round(rf.shape_dur / dt)) + 1) - 0.5) * dt
    f = (
        2
        * _np.pi
        * _np.linspace(
            freq_offset - bandwidth_multiplier * bandwidth / 2.0,
            freq_offset + bandwidth_multiplier * bandwidth / 2.0,
            int(max(1, _np.round(bandwidth / df))),
        )
    )
    envelope = (
        2
        * _np.pi
        * (
            _np.interp(t, rf.t, _np.real(rf.signal), left=0.0, right=0.0)
            + 1j * _np.interp(t, rf.t, _np.imag(rf.signal), left=0.0, right=0.0)
        )
    )
    envelope = envelope * _np.exp(1j * (phase_offset + 2 * _np.pi * freq_offset * t))

    elapsed = dt * t.size
    turns = (
        _about_z(f * elapsed * rephase_factor)
        @ _kernels.rotations(envelope / (2 * _np.pi), (f / (2 * _np.pi))[:, None], dt)
        @ _about_z(f * elapsed * prephase_factor)
    )

    mz_xy = turns[:, 0, 2] + 1j * turns[:, 1, 2]
    mz_z = turns[:, 2, 2]
    mx_xy = turns[:, 0, 0] + 1j * turns[:, 1, 0]
    my_xy = turns[:, 0, 1] + 1j * turns[:, 1, 1]

    return mz_z, mz_xy, f / (2 * _np.pi), (mx_xy + 1j * my_xy) / 2.0, mx_xy, my_xy
