"""Slab-encoding pulses (gSlider, Hadamard) and PINS multiband pulses."""

from __future__ import annotations

__all__ = ["make_gslider_pulse", "make_hadamard_pulse", "make_pins_pulse"]

import math

import numpy as np

from . import _events
from ._opts import default_system
from ._rf_pulses import DEFAULT_DURATION, _play_slr, _slr_sample_count
from ._slr import design_gslider, design_hadamard, design_slr

_LINEAR_PHASE_FILTERS = ("ls", "pm", "ms")


def make_gslider_pulse(
    flip_angle: float,
    num_subslices: int,
    subslice: int,
    *,
    subslice_phase: float = np.pi,
    duration: float = DEFAULT_DURATION,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    slice_thickness: float = 0.0,
    return_gz: bool = False,
    time_bw_product: float = 12.0,
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = True,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system=None,
    use: str = "excitation",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Design a gSlider slab pulse whose ``subslice`` carries ``subslice_phase``.

    One pulse per sub-slice, combined across acquisitions, resolves the slab
    into ``num_subslices`` sub-slices (Setsompop et al., Magn Reson Med
    79:141, 2018). The SLR design is exact for ``flip_angle``, so the pulse
    plays at its designed amplitude rather than being scaled by area.

    Parameters
    ----------
    subslice : int
        Encoded sub-slice, in ``[0, num_subslices)``, counted from the most
        negative position under a positive selection gradient.
    subslice_phase : float, optional
        Phase of the encoded sub-slice relative to the rest of the slab, in rad.
    slice_thickness : float, optional
        Whole slab, in m; required when ``return_gz``.

    Other parameters are as in :func:`make_slr_pulse`.

    Returns
    -------
    rf, or (rf, gz, gzr)
        The pulse, with its selection gradient and rephaser under ``return_gz``.

    Raises
    ------
    ValueError
        If ``subslice`` is out of range, or the time-bandwidth product does not
        fit the pulse's samples.
    """
    return _slab_pulse(
        lambda n: design_gslider(
            n,
            time_bw_product,
            num_subslices,
            subslice,
            phase=subslice_phase,
            flip_angle=flip_angle,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            cancel_alpha_phase=cancel_alpha_phase,
        ),
        flip_angle,
        duration=duration,
        dwell=dwell,
        time_bw_product=time_bw_product,
        slice_thickness=slice_thickness,
        return_gz=return_gz,
        system=system,
        delay=delay,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        max_grad=max_grad,
        max_slew=max_slew,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
    )


def make_hadamard_pulse(
    flip_angle: float,
    order: int,
    row: int,
    *,
    duration: float = DEFAULT_DURATION,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    slice_thickness: float = 0.0,
    return_gz: bool = False,
    time_bw_product: float = 12.0,
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = True,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system=None,
    use: str = "excitation",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Design a slab pulse whose sub-bands are signed by row ``row`` of a Hadamard matrix.

    Acquiring every row of an ``order`` x ``order`` Hadamard matrix and
    combining the images resolves the slab into ``order`` sub-slices (Souza
    et al., J Comput Assist Tomogr 12:1026, 1988). The SLR design is exact for
    ``flip_angle``, so the pulse plays at its designed amplitude.

    Parameters
    ----------
    order : int
        Sub-slices, a power of two.
    row : int
        Encoding, in ``[0, order)``; row 0 is the plain slab. Sub-slices are
        counted from the most negative position under a positive gradient.
    slice_thickness : float, optional
        Whole slab, in m; required when ``return_gz``.

    Other parameters are as in :func:`make_slr_pulse`.

    Returns
    -------
    rf, or (rf, gz, gzr)
        The pulse, with its selection gradient and rephaser under ``return_gz``.

    Raises
    ------
    ValueError
        If ``order`` is not a power of two, ``row`` is out of range, or the
        time-bandwidth product does not fit the pulse's samples.
    """
    return _slab_pulse(
        lambda n: design_hadamard(
            n,
            time_bw_product,
            order,
            row,
            flip_angle=flip_angle,
            passband_ripple=passband_ripple,
            stopband_ripple=stopband_ripple,
            cancel_alpha_phase=cancel_alpha_phase,
        ),
        flip_angle,
        duration=duration,
        dwell=dwell,
        time_bw_product=time_bw_product,
        slice_thickness=slice_thickness,
        return_gz=return_gz,
        system=system,
        delay=delay,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        max_grad=max_grad,
        max_slew=max_slew,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
    )


def _slab_pulse(
    design,
    flip_angle,
    *,
    duration,
    dwell,
    time_bw_product,
    slice_thickness,
    return_gz,
    system,
    **event,
):
    """Play ``design(n)``, a large-tip waveform in radians per sample, centred."""
    system = default_system(system)
    dwell = dwell or system.rf_raster_time
    if return_gz and slice_thickness <= 0:
        raise ValueError("slice_thickness must be > 0 when return_gz=True")
    return _play_slr(
        design(_slr_sample_count(duration, dwell)),
        flip_angle,
        designed=True,
        dwell=dwell,
        time_bw_product=time_bw_product,
        center_pos=0.5,
        return_gz=return_gz,
        slice_thickness=slice_thickness,
        system=system,
        **event,
    )


def make_pins_pulse(
    flip_angle: float,
    slice_thickness: float,
    slice_separation: float,
    *,
    time_bw_product: float = 4.0,
    pulse_type: str = "st",
    filter_type: str = "ls",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    max_b1: float = 18e-6,
    delay: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system=None,
    use: str = "excitation",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Design a PINS pulse exciting a slice every ``slice_separation`` along z.

    An SLR envelope for ``slice_thickness`` is played as hard subpulses with a
    z blip of area ``1 / slice_separation`` between each, so the profile
    repeats every ``slice_separation`` and the power does not depend on the
    number of slices (Norris et al., Magn Reson Med 66:1234, 2011). The
    envelope has ``time_bw_product * slice_separation / slice_thickness``
    subpulses, rounded to an even count, and is scaled to ``flip_angle`` by
    area. The RF centre is the pulse's midpoint for a linear-phase
    ``filter_type`` (``'ls'``, ``'pm'``, ``'ms'``) and the largest subpulse's
    centre otherwise.

    Parameters
    ----------
    slice_thickness, slice_separation : float
        In m.
    max_b1 : float, optional
        Peak B1, in T; each subpulse is the shortest, on the gradient raster,
        that stays within it.
    pulse_type, filter_type, time_bw_product, passband_ripple, stopband_ripple
        The envelope's SLR design, as in :func:`make_slr_pulse`.
    max_grad, max_slew : float, optional
        Limits for the blips and the rephaser, in place of the system's.

    Returns
    -------
    rf : SimpleNamespace
    gz : SimpleNamespace
        The blip train, an extended trapezoid with the same delay as ``rf``.
    gzr : SimpleNamespace
        Rephaser of minus the blip area after the RF centre.

    Raises
    ------
    ValueError
        If ``slice_separation`` does not exceed ``slice_thickness``, or the
        envelope would have fewer than eight subpulses.
    """
    system = default_system(system)
    if not 0 < slice_thickness < slice_separation:
        raise ValueError(
            "slice_separation must exceed slice_thickness, and both be > 0"
        )
    count = 2 * math.floor(
        math.ceil(time_bw_product * slice_separation / slice_thickness - 1e-9) / 2
    )
    if count < 8:
        raise ValueError(
            f"PINS needs at least 8 subpulses, and time_bw_product * "
            f"slice_separation / slice_thickness gives {count}"
        )
    envelope = design_slr(
        count,
        time_bw_product,
        pulse_type=pulse_type,
        filter_type=filter_type,
        passband_ripple=passband_ripple,
        stopband_ripple=stopband_ripple,
    )
    rotations = envelope * flip_angle / abs(envelope.sum())

    raster = system.grad_raster_time
    dwell = system.rf_raster_time
    limits = {"max_grad": max_grad, "max_slew": max_slew}
    limits = {key: value for key, value in limits.items() if value}
    blip = _events.make_trapezoid(
        "z", area=1.0 / slice_separation, system=system, **limits
    )
    blip_time = blip.rise_time + blip.flat_time + blip.fall_time
    peak = max_b1 * system.gamma
    hard = raster * max(
        1, math.ceil(np.abs(rotations).max() / (2 * np.pi * peak) / raster - 1e-9)
    )
    period = hard + blip_time
    total = count * hard + (count - 1) * blip_time

    times = (np.arange(round(total / dwell)) + 0.5) * dwell
    index = np.minimum((times // period).astype(int), count - 1)
    on = times - index * period < hard
    signal = np.where(on, rotations[index] / (2 * np.pi * hard), 0.0)
    if filter_type in _LINEAR_PHASE_FILTERS:
        centre = total / 2
    else:
        centre = int(np.argmax(np.abs(rotations))) * period + hard / 2
    start = raster * math.ceil(max(delay, system.rf_dead_time) / raster - 1e-9)
    rf = _events.make_arbitrary_rf(
        signal=signal,
        flip_angle=flip_angle,
        no_signal_scaling=True,
        dwell=dwell,
        delay=start,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        system=system,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        center=centre,
    )

    vertices, amplitudes = [0.0], [0.0]
    for k in range(count - 1):
        rise = (k + 1) * hard + k * blip_time
        corners = [rise, rise + blip.rise_time]
        levels = [0.0, blip.amplitude]
        if blip.flat_time > 0:
            corners.append(corners[-1] + blip.flat_time)
            levels.append(blip.amplitude)
        corners.append(corners[-1] + blip.fall_time)
        levels.append(0.0)
        vertices += corners
        amplitudes += levels
    vertices.append(total)
    amplitudes.append(0.0)
    vertices, amplitudes = np.asarray(vertices), np.asarray(amplitudes)
    gz = _events.make_extended_trapezoid(
        "z", times=vertices, amplitudes=amplitudes, system=system
    )
    gz.delay = start

    # The centre falls on a blip's apex, its flat top or between blips, where
    # the area accrued is linear in time, so interpolating it is exact.
    accrued = np.concatenate(
        ([0.0], np.cumsum(np.diff(vertices) * (amplitudes[1:] + amplitudes[:-1]) / 2))
    )
    after = accrued[-1] - np.interp(centre, vertices, accrued)
    gzr = _events.make_trapezoid("z", area=-after, system=system, **limits)
    return rf, gz, gzr
