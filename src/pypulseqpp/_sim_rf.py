"""What an RF pulse does to the magnetisation, across off-resonance."""

from __future__ import annotations

__all__ = ["bloch", "sim_rf"]

import warnings as _warnings

import numpy as _np
import pypulseq as _pp

from ._calc_rf_bandwidth import calc_rf_bandwidth as _calc_rf_bandwidth

#: Simulation raster against the pulse's bandwidth: a wider pulse is
#: integrated in finer steps, since the rotation per step is what the hard
#: pulse approximation is asked to hold over.
_RASTERS = ((2e4, 1e-6), (1e4, 2e-6), (4e3, 5e-6), (0.0, 10e-6))


def _quat_multiply(q, r):
    """Hamilton product of two ``(N, 4)`` scalar-first quaternion arrays."""
    scalar = (
        q[:, 0] * r[:, 0] - q[:, 1] * r[:, 1] - q[:, 2] * r[:, 2] - q[:, 3] * r[:, 3]
    )
    vector = q[:, :1] * r[:, 1:] + r[:, :1] * q[:, 1:] + _np.cross(q[:, 1:], r[:, 1:])
    return _np.column_stack((scalar, vector))


def _quat_conjugate(q):
    conjugated = q.copy()
    conjugated[:, 1:] *= -1.0
    return conjugated


def _applied(q, vector, count):
    """``q* v q`` for every row of ``q``, as a complex transverse pair."""
    m = _np.zeros((count, 4))
    m[:, 1:] = vector
    turned = _quat_multiply(_quat_conjugate(q), _quat_multiply(m, q))
    return turned[:, 1] + 1j * turned[:, 2], turned[:, 3]


def _free_precession(f, seconds: float, count: int):
    """Turn about z by what ``seconds`` of free precession comes to."""
    angle = -f * seconds
    return _np.column_stack(
        (_np.cos(angle / 2.0), _np.zeros((count, 2)), _np.sin(angle / 2.0))
    )


def bloch(b1_hz, bz_hz, dt: float, *, initial=None) -> _np.ndarray:
    """Integrate the Bloch equation over a hard-pulse approximation.

    Parameters
    ----------
    b1_hz : array_like
        Complex transverse field per time step, shape ``(T,)``, in Hz.
    bz_hz : array_like
        Longitudinal field, shape ``(P, T)`` or ``(P, 1)``, in Hz.
    dt : float
        Raster step, in seconds.
    initial : array_like, optional
        Starting magnetisation, ``(3,)``. Defaults to ``+z``.

    Returns
    -------
    numpy.ndarray
        Final magnetisation, shape ``(P, 3)``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp

    A hard pulse of 250 Hz held for 1 ms is a 90 degree flip; on resonance it
    takes ``+z`` onto ``-y``:

    >>> on_resonance = _np.zeros((1, 1))
    >>> pp.bloch(_np.full(1000, 250.0 + 0j), on_resonance, 1e-6).round(3)
    array([[ 0., -1.,  0.]])

    Twice the amplitude inverts it:

    >>> pp.bloch(_np.full(1000, 500.0 + 0j), on_resonance, 1e-6).round(3)
    array([[ 0., -0., -1.]])

    ``bz_hz`` carries one row per position, so a whole slice profile comes
    back at once:

    >>> offsets = _np.array([[0.0], [500.0], [-500.0]])
    >>> pp.bloch(_np.full(1000, 250.0 + 0j), offsets, 1e-6).round(3)
    array([[ 0.   , -1.   ,  0.   ],
           [ 0.773,  0.162,  0.614],
           [-0.773,  0.162,  0.614]])
    """
    b1_hz = _np.asarray(b1_hz, dtype=complex)
    bz_hz = _np.atleast_2d(_np.asarray(bz_hz, dtype=float))
    n_pos = bz_hz.shape[0]

    magnetisation = _np.zeros((n_pos, 3))
    magnetisation[:] = (
        _np.array([0.0, 0.0, 1.0])
        if initial is None
        else _np.asarray(initial, dtype=float)
    )

    two_pi_dt = 2.0 * _np.pi * dt
    for step in range(len(b1_hz)):
        omega = _np.empty((n_pos, 3))
        omega[:, 0] = two_pi_dt * b1_hz[step].real
        omega[:, 1] = two_pi_dt * b1_hz[step].imag
        omega[:, 2] = two_pi_dt * bz_hz[:, step if bz_hz.shape[1] > 1 else 0]

        angle = _np.linalg.norm(omega, axis=1)
        active = angle > 1e-15
        if not _np.any(active):
            continue
        axis = _np.zeros_like(omega)
        axis[active] = omega[active] / angle[active, None]

        cosine, sine = _np.cos(angle)[:, None], _np.sin(angle)[:, None]
        dot = _np.sum(axis * magnetisation, axis=1)[:, None]
        magnetisation = (
            magnetisation * cosine
            + _np.cross(axis, magnetisation) * sine
            + axis * dot * (1.0 - cosine)
        )
    return magnetisation


def sim_rf(
    rf,
    rephase_factor: float | None = None,
    prephase_factor: float = 0.0,
    *,
    df: float = 1.0,
    bandwidth_multiplier: float = 4.0,
    dt: float | None = None,
):
    """Simulate an RF pulse across off-resonance, as MATLAB's ``simRf`` does.

    The pulse is integrated as a train of hard pulses, one per raster step,
    and the rotation each step is accumulated as a quaternion; the whole
    accumulated rotation is applied once at the end, to three starting
    magnetisations at once. Relaxation is left out, which is what makes a
    single accumulated rotation valid -- over a pulse of a few milliseconds
    T1 and T2 do nothing a profile would show.

    Off-resonance stands in for position: through a slice-select gradient a
    frequency is a distance, so the profile against frequency is the slice
    profile.

    Parameters
    ----------
    rf : SimpleNamespace or RfEvent
        The RF event.
    rephase_factor : float, optional
        Free precession after the pulse, as a fraction of its duration. Zero
        for a refocusing pulse and ``-(shape_dur - center) / shape_dur``
        otherwise, which is the rephasing a slice-selective excitation is
        followed by -- without it the phase across the profile is the linear
        ramp the pulse leaves rather than the one the sequence plays.
    prephase_factor : float, optional
        The same, before the pulse.
    df : float, optional
        Spectral resolution, in Hz.
    bandwidth_multiplier : float, optional
        Width of the simulated axis, in pulse bandwidths.
    dt : float, optional
        Simulation raster, in seconds. Chosen from the bandwidth when
        omitted.

    Returns
    -------
    mz_z : numpy.ndarray
        ``Mz`` after the pulse, starting from ``+z``: the inversion or
        saturation profile.
    mz_xy : numpy.ndarray
        Complex ``Mxy`` after the pulse, starting from ``+z``: the excitation
        profile.
    f : numpy.ndarray
        The frequency axis, in Hz.
    ref_eff : numpy.ndarray
        Refocusing efficiency, complex: its magnitude is the refocused
        fraction and its phase the axis of the flip.
    mx_xy, my_xy : numpy.ndarray
        Complex ``Mxy`` starting from ``+x`` and from ``+y``.

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
    count = f.size

    envelope = (
        2
        * _np.pi
        * (
            _np.interp(t, rf.t, _np.real(rf.signal), left=0.0, right=0.0)
            + 1j * _np.interp(t, rf.t, _np.imag(rf.signal), left=0.0, right=0.0)
        )
    )
    envelope = envelope * _np.exp(1j * (phase_offset + 2 * _np.pi * freq_offset * t))

    q = _np.zeros((count, 4))
    q[:, 0] = 1.0
    q = _quat_multiply(q, _free_precession(f, dt * t.size * prephase_factor, count))

    for step in range(t.size):
        angle = -dt * _np.sqrt(abs(envelope[step]) ** 2 + f**2)
        magnitude = _np.abs(angle)
        axis = _np.column_stack(
            (
                _np.full(count, envelope[step].real),
                _np.full(count, envelope[step].imag),
                f,
            )
        )
        turning = magnitude > 0
        axis[turning] *= dt / magnitude[turning, None]
        q = _quat_multiply(
            q,
            _np.column_stack(
                (_np.cos(angle / 2.0), _np.sin(angle / 2.0)[:, None] * axis)
            ),
        )

    q = _quat_multiply(q, _free_precession(f, dt * t.size * rephase_factor, count))

    mz_xy, mz_z = _applied(q, (0.0, 0.0, 1.0), count)
    mx_xy, _ = _applied(q, (1.0, 0.0, 0.0), count)
    my_xy, _ = _applied(q, (0.0, 1.0, 0.0), count)

    return mz_z, mz_xy, f / (2 * _np.pi), (mx_xy + 1j * my_xy) / 2.0, mx_xy, my_xy
