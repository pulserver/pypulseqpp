"""Adiabatic pulses: upstream's two sweeps, with BIR-4 and GOIA-WURST beside them.

Hyperbolic-secant and WURST sweeps are PyPulseq's own and are built by it, so
a script asking for one writes the file it always did. BIR-4 and GOIA-WURST
follow SigPy's ``sigpy.mri.rf.adiabatic`` (Copyright (c) 2016, Frank Ong and
The Regents of the University of California; BSD 3-Clause, see
``LICENSES/SigPy-BSD-3-Clause.txt``), and take their amplitude the way
upstream's sweeps do: from the adiabaticity asked for where the frequency
sweep crosses zero.
"""

from __future__ import annotations

__all__ = ["ADIABATIC_PULSE_TYPES", "make_adiabatic_pulse"]

import math

import numpy as np

from . import _events
from ._opts import default_system

ADIABATIC_PULSE_TYPES = ("hypsec", "wurst", "bir4", "goia_wurst")

_UPSTREAM_TYPES = ("hypsec", "wurst")


def make_adiabatic_pulse(
    pulse_type: str,
    adiabaticity: float = 4,
    bandwidth: float = 40000,
    beta: float | None = None,
    delay: float = 0.0,
    duration: float = 10e-3,
    dwell: float | None = None,
    freq_offset: float = 0.0,
    max_grad: float | None = None,
    max_slew: float | None = None,
    n_fac: int | None = None,
    mu: float = 4.9,
    phase_offset: float = 0.0,
    return_gz: bool = False,
    slice_thickness: float = 0.0,
    system=None,
    use: str = "inversion",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    *,
    flip_angle: float = np.pi / 2,
    kappa: float = math.atan(20.0),
    gradient_modulation: float = 0.9,
    gradient_order: int = 4,
):
    """Make an adiabatic pulse, swept so that what it does is not what B1 is.

    Four sweeps:

    - ``"hypsec"``: hyperbolic-secant inversion (Baum, Tycko and Pines 1985).
    - ``"wurst"``: WURST inversion, a uniform-rate sweep under a smoothly
      truncated envelope (Kupce and Freeman 1995).
    - ``"bir4"``: a B1-insensitive rotation by ``flip_angle``: two BIR-1
      halves back to back, the flip set by the phase jumps between their
      quarters (Staewen et al. 1990). Selects no slice.
    - ``"goia_wurst"``: a slice-selective inversion whose gradient is
      modulated with the sweep, so every position in the slice sees the same
      adiabaticity (Andronesi et al. 2010). Returned with that gradient.

    The first two are PyPulseq's own factory. For every sweep the amplitude
    is the one that meets ``adiabaticity`` where the frequency sweep crosses
    zero: ``(gamma B1)^2 = adiabaticity * |d omega / dt|`` there.

    Parameters
    ----------
    pulse_type : {"hypsec", "wurst", "bir4", "goia_wurst"}
        Which sweep.
    adiabaticity : float, optional
        Margin over the adiabatic condition at the sweep's zero crossing.
    bandwidth : float, optional
        Full frequency sweep, in Hz. Unused by ``"hypsec"``, whose sweep is
        ``mu * beta``.
    beta : float, optional
        Envelope parameter: in 1/s for ``"hypsec"`` (800 by default), and
        dimensionless for ``"bir4"``, where it sets how steep the tanh
        quarters are (10 by default).
    delay : float, optional
        Delay before the pulse, in s.
    duration : float, optional
        Pulse duration, in s.
    dwell : float, optional
        RF raster, in s; the system's by default.
    freq_offset, phase_offset : float, optional
        Frequency (Hz) and phase (rad) offsets.
    max_grad, max_slew : float, optional
        Limits for the slice-selection gradient, in place of the system's.
    n_fac : int, optional
        Order of the envelope's truncation, ``1 - |cos(pi t / T)|^n_fac``:
        40 for ``"wurst"`` and 16 for ``"goia_wurst"`` by default.
    mu : float, optional
        ``"hypsec"`` sweep amplitude, in units of ``beta``.
    return_gz : bool, optional
        Also return the slice-selection gradient and its rephaser. Required
        by ``"goia_wurst"``, refused by ``"bir4"``.
    slice_thickness : float, optional
        Slice thickness, in m, when ``return_gz``.
    system : Opts, optional
        System limits.
    use : str, optional
        What the pulse is for, as Pulseq records it.
    freq_ppm, phase_ppm : float, optional
        Field-strength-relative offsets.
    flip_angle : float, optional
        ``"bir4"`` rotation, in rad.
    kappa : float, optional
        ``"bir4"`` frequency-sweep shape: ``tan(kappa * s) / tan(kappa)``
        over each quarter.
    gradient_modulation : float, optional
        ``"goia_wurst"``: how far the gradient dips mid-pulse, in ``[0, 1)``.
        It falls to ``1 - gradient_modulation`` of its peak.
    gradient_order : int, optional
        ``"goia_wurst"``: order of the gradient's modulation.

    Returns
    -------
    rf : SimpleNamespace
        The pulse.
    gz, gzr : SimpleNamespace, optional
        The slice-selection gradient and its rephaser, when ``return_gz``: a
        trapezoid for the first two sweeps, the modulated gradient for
        ``"goia_wurst"``.

    Raises
    ------
    ValueError
        If ``pulse_type`` is unknown, a BIR-4 pulse is asked for a gradient,
        a GOIA-WURST pulse is not, or the duration holds fewer than eight RF
        samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> rf = pp.make_adiabatic_pulse("bir4", flip_angle=np.pi / 2, duration=6e-3)
    >>> rf.type
    'rf'
    >>> rf, gz, gzr = pp.make_adiabatic_pulse(
    ...     "goia_wurst", bandwidth=10e3, duration=5e-3, return_gz=True, slice_thickness=0.01
    ... )
    >>> gz.channel
    'z'
    """
    if pulse_type in _UPSTREAM_TYPES:
        return _events.make_adiabatic_pulse(
            pulse_type,
            adiabaticity=adiabaticity,
            bandwidth=bandwidth,
            beta=800.0 if beta is None else beta,
            delay=delay,
            duration=duration,
            dwell=dwell,
            freq_offset=freq_offset,
            max_grad=max_grad,
            max_slew=max_slew,
            n_fac=40 if n_fac is None else n_fac,
            mu=mu,
            phase_offset=phase_offset,
            return_gz=return_gz,
            slice_thickness=slice_thickness,
            system=system,
            use=use,
            freq_ppm=freq_ppm,
            phase_ppm=phase_ppm,
        )
    if pulse_type not in ADIABATIC_PULSE_TYPES:
        raise ValueError(
            f"pulse_type must be one of {ADIABATIC_PULSE_TYPES}, got {pulse_type!r}"
        )
    if pulse_type == "bir4" and return_gz:
        raise ValueError(
            "a BIR-4 pulse selects no slice, so it has no gradient to return"
        )
    if pulse_type == "goia_wurst" and (not return_gz or slice_thickness <= 0):
        raise ValueError(
            "a GOIA-WURST pulse is designed with its gradient: pass return_gz=True "
            "and a slice_thickness"
        )

    system = default_system(system)
    dwell = system.rf_raster_time if dwell is None else dwell
    n_raw = round(duration / dwell)
    n = n_raw // 4 * 4
    if n < 8:
        raise ValueError("duration must hold at least eight RF samples")

    if pulse_type == "bir4":
        envelope, sweep = _bir4(
            n, 10.0 if beta is None else beta, kappa, flip_angle, np.pi * bandwidth
        )
        crossing = n // 2
        modulation = None
    else:
        envelope, sweep, modulation = _goia_wurst(
            n,
            duration,
            gradient_modulation,
            16 if n_fac is None else n_fac,
            gradient_order,
            bandwidth,
        )
        crossing = n // 4 + int(np.argmin(np.abs(sweep[n // 4 : 3 * n // 4])))

    phase = np.cumsum(sweep) * dwell
    amplitude, phase_at_crossing = _adiabatic_amplitude(
        envelope, sweep, phase, crossing, dwell, adiabaticity
    )
    signal = amplitude * envelope * np.exp(1j * (phase - phase_at_crossing))
    pad = n_raw - n
    signal = np.pad(signal, (pad // 2, pad - pad // 2))

    rf_arguments = {
        "signal": signal,
        "flip_angle": flip_angle if pulse_type == "bir4" else np.pi,
        "no_signal_scaling": True,
        "dwell": dwell,
        "freq_offset": freq_offset,
        "phase_offset": phase_offset,
        "system": system,
        "use": use,
        "freq_ppm": freq_ppm,
        "phase_ppm": phase_ppm,
    }
    if modulation is None:
        return _events.make_arbitrary_rf(delay=delay, **rf_arguments)

    # The sweep's ends sit at the slice's edges where the gradient is at its
    # peak, so the peak is the bandwidth over the thickness, and the pulse
    # starts where the gradient has ramped up to it.
    raster = system.grad_raster_time
    slew = max_slew or system.max_slew
    peak = bandwidth / slice_thickness
    n_ramp = max(1, math.ceil(peak / slew / raster - 1e-9))
    ramp_time = n_ramp * raster
    rf_delay = max(delay, system.rf_dead_time)
    gz_delay = 0.0
    if rf_delay > ramp_time:
        gz_delay = math.ceil((rf_delay - ramp_time) / raster - 1e-9) * raster
    rf_delay = gz_delay + ramp_time

    sample_times = (np.arange(n) + 0.5 + pad // 2) * dwell
    plateau = (np.arange(math.ceil(n_raw * dwell / raster - 1e-9)) + 0.5) * raster
    ramp = (np.arange(n_ramp) + 0.5) / n_ramp
    waveform = peak * np.concatenate(
        (ramp, np.interp(plateau, sample_times, modulation), ramp[::-1])
    )
    limits = {"max_grad": max_grad, "max_slew": max_slew}
    gz = _events.make_arbitrary_grad(
        "z", waveform, first=0.0, last=0.0, delay=gz_delay, system=system, **limits
    )
    rf = _events.make_arbitrary_rf(delay=rf_delay, **rf_arguments)

    centre = rf_delay + float(rf.center)
    times = gz_delay + (np.arange(waveform.size) + 0.5) * raster
    after = float(waveform[times > centre].sum()) * raster
    rephase = {key: value for key, value in limits.items() if value}
    gzr = _events.make_trapezoid("z", area=-after, system=system, **rephase)
    return rf, gz, gzr


def _adiabatic_amplitude(envelope, sweep, phase, index, dwell, adiabaticity):
    """Scale meeting ``adiabaticity`` where ``sweep`` crosses zero near ``index``.

    Returns the envelope's scale, in Hz, and the phase at the crossing. A
    crossing that falls on a sample takes the sweep's rate across it; one
    between samples is bracketed and interpolated, as upstream does.
    """
    if sweep[index] == 0.0:
        rate = abs(sweep[index + 1] - sweep[index - 1]) / (2.0 * dwell)
        at_crossing = abs(envelope[index])
        phase_there = phase[index]
    else:
        step = 1 if sweep[index] * sweep[index + 1] < 0 else -1
        here, there = sweep[index], sweep[index + step]
        at_crossing = abs(
            (envelope[index] * there - envelope[index + step] * here) / (there - here)
        )
        phase_there = (phase[index] * there - phase[index + step] * here) / (
            there - here
        )
        rate = abs(here - there) / dwell
    return math.sqrt(rate * adiabaticity) / (2.0 * math.pi * at_crossing), phase_there


def _bir4(n: int, beta: float, kappa: float, theta: float, dw0: float):
    """BIR-4 envelope (complex, carrying the flip's phase jumps) and sweep (rad/s)."""
    t = np.arange(n) / n
    q1, q2, q3 = n // 4, n // 2, 3 * n // 4
    envelope = np.concatenate(
        (
            np.tanh(beta * (1 - 4 * t[:q1])),
            np.tanh(beta * (4 * t[q1:q2] - 1)),
            np.tanh(beta * (3 - 4 * t[q2:q3])),
            np.tanh(beta * (4 * t[q3:] - 3)),
        )
    ).astype(np.complex128)
    envelope[q1:q3] *= np.exp(1j * (np.pi + theta / 2))
    sweep = (dw0 / np.tan(kappa)) * np.concatenate(
        (
            np.tan(kappa * 4 * t[:q1]),
            np.tan(kappa * (4 * t[q1:q2] - 2)),
            np.tan(kappa * (4 * t[q2:q3] - 2)),
            np.tan(kappa * (4 * t[q3:] - 4)),
        )
    )
    return envelope, sweep


def _goia_wurst(
    n: int, duration: float, f: float, n_b1: int, m_grad: int, bandwidth: float
):
    """GOIA-WURST envelope, sweep (rad/s) and gradient modulation (peak 1)."""
    t = np.arange(n) * duration / n
    s = np.abs(np.sin(np.pi / 2 * (2 * t / duration - 1)))
    envelope = 1 - s**n_b1
    modulation = (1 - f) + f * s**m_grad
    sweep = np.cumsum(envelope**2 / modulation) * duration / n
    sweep = modulation * (sweep - sweep[n // 2 + 1])
    sweep = sweep / np.max(np.abs(sweep)) * bandwidth / 2
    return envelope, 2 * np.pi * sweep, modulation
