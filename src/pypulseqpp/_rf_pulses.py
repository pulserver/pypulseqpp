"""RF pulse factories, each returning events rather than a module.

They follow :func:`pypulseq.make_sinc_pulse`'s shape: the pulse alone by
default, and ``(rf, gz, gz_reph)`` under ``return_gz=True`` where a selection
gradient makes sense.
"""

from __future__ import annotations

__all__ = [
    "make_2d_selective_pulse",
    "make_half_passages",
    "make_sigpy_pulse",
    "make_slr_pulse",
    "make_sms_pulse",
    "make_spsp_pulse",
]

from collections.abc import Sequence
from typing import Literal

import numpy as np

from . import _events
from ._angles import calc_uniform_angles
from ._band_phases import band_phases
from ._opts import default_system
from ._slr import NOMINAL_FLIP, design_slr

PulseType = Literal["st", "ex", "se", "inv", "sat"]
FilterType = Literal["ls", "pm", "min", "max", "ms"]

DEFAULT_DURATION = 4e-3
DEFAULT_TIME_BANDWIDTH_PRODUCT = 4.0


def _slr_sample_count(duration: float, dwell: float) -> int:
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if dwell <= 0:
        raise ValueError("dwell must be > 0")
    count = round(duration / dwell)
    if count < 4:
        raise ValueError("duration must span at least four RF samples")
    return count


def make_slr_pulse(
    flip_angle: float,
    *,
    duration: float = DEFAULT_DURATION,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    center_pos: float = 0.5,
    slice_thickness: float = 0.0,
    return_gz: bool = False,
    time_bw_product: float = DEFAULT_TIME_BANDWIDTH_PRODUCT,
    pulse_type: PulseType = "st",
    filter_type: FilterType = "ls",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = False,
    root_flip: bool = False,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    system=None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Design an RF pulse using the Shinnar-Le Roux algorithm.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    duration : float, optional
        Pulse duration (s).
    delay : float, optional
        Delay before the pulse (s).
    dwell : float, optional
        RF sample spacing (s); ``0`` uses ``system.rf_raster_time``.
    freq_offset, phase_offset : float, optional
        Frequency (Hz) and phase (rad) offsets.
    center_pos : float, optional
        Where the effective centre sits in the pulse, in ``[0, 1]``.
    slice_thickness : float, optional
        Slice thickness (m); required when ``return_gz``.
    return_gz : bool, optional
        Also return the selection gradient and its rephaser.
    time_bw_product : float, optional
        Time-bandwidth product.
    pulse_type : {'st', 'ex', 'se', 'inv', 'sat'}, optional
        Small-tip, excitation, spin-echo, inversion or saturation.
    filter_type : {'ls', 'pm', 'min', 'max', 'ms'}, optional
        FIR design method.
    passband_ripple, stopband_ripple : float, optional
        Ripple allowed in each band.
    cancel_alpha_phase : bool, optional
        Remove the SLR alpha polynomial's phase.
    root_flip : bool, optional
        Flip the roots of the SLR beta polynomial for the lowest peak B1 the
        same slice profile allows (Sharma, Lustig and Grissom, 2016). The
        profile's magnitude is unchanged and its phase is no longer linear, so
        it suits a refocusing or inversion pulse, and an excitation whose
        phase is refocused another way. Needs a ``pulse_type`` with a nominal
        flip, and searches every subset of the passband's roots. The pulse
        plays at the amplitude it was designed at, scaled by ``flip_angle``
        over that nominal flip, because its winding phase makes its area no
        measure of its flip.
    max_grad, max_slew : float, optional
        Override the system limits for the selection gradient.
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    freq_ppm, phase_ppm : float, optional
        Field-strength-relative offsets.

    Returns
    -------
    RfEvent, or tuple
        The pulse, or ``(rf, gz, gz_reph)`` when ``return_gz``.

    Raises
    ------
    ValueError
        If ``center_pos`` is outside ``[0, 1]``, ``return_gz`` is asked for
        without a positive ``slice_thickness``, or ``root_flip`` is asked of a
        small-tip pulse, alongside ``cancel_alpha_phase``, or of a passband
        with more roots than an exhaustive search can visit.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> rf, gz, gz_reph = pp.make_slr_pulse(
    ...     np.deg2rad(90), slice_thickness=5e-3, return_gz=True, system=system
    ... )
    >>> rf.type, gz.channel
    ('rf', 'z')

    See Also
    --------
    make_sinc_pulse : the windowed-sinc alternative.
    sim_rf : simulate the profile the design actually produces.
    """
    system = default_system(system)
    dwell = system.rf_raster_time if dwell == 0 else dwell
    if not 0.0 <= center_pos <= 1.0:
        raise ValueError("center_pos must lie in [0, 1]")
    if return_gz and slice_thickness <= 0:
        raise ValueError("slice_thickness must be > 0 when return_gz=True")

    n = _slr_sample_count(duration, dwell)
    waveform = design_slr(
        n,
        time_bw_product,
        pulse_type=pulse_type,
        filter_type=filter_type,
        passband_ripple=passband_ripple,
        stopband_ripple=stopband_ripple,
        cancel_alpha_phase=cancel_alpha_phase,
        root_flip=root_flip,
    )
    if root_flip:
        # A root-flipped pulse's phase winds through it, so its area is no
        # measure of its flip: it plays at the amplitude it was designed at,
        # scaled from the flip it was designed for.
        waveform = waveform * (flip_angle / NOMINAL_FLIP[pulse_type])
    return _play_slr(
        waveform,
        flip_angle,
        designed=root_flip,
        dwell=dwell,
        time_bw_product=time_bw_product,
        center_pos=center_pos,
        return_gz=return_gz,
        slice_thickness=slice_thickness,
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


def _play_slr(
    waveform,
    flip_angle,
    *,
    designed,
    dwell,
    time_bw_product,
    center_pos,
    return_gz,
    slice_thickness,
    system,
    **event,
):
    """Build an SLR waveform's event, and under ``return_gz`` its gradient and rephaser.

    A ``designed`` waveform is in radians per sample and plays as it is; any
    other is scaled until its area is ``flip_angle``.
    """
    duration = waveform.size * dwell
    result = _events.make_arbitrary_rf(
        signal=waveform / (2.0 * np.pi * dwell) if designed else waveform,
        flip_angle=flip_angle,
        no_signal_scaling=designed,
        dwell=dwell,
        return_gz=return_gz,
        slice_thickness=slice_thickness,
        bandwidth=time_bw_product / duration,
        time_bw_product=time_bw_product,
        system=system,
        center=center_pos * duration,
        **event,
    )
    if not return_gz:
        return result

    rf, gz = result
    flat_area = gz.amplitude * gz.flat_time
    ramp_area = gz.area - flat_area
    rephase_area = -flat_area * (1.0 - center_pos) - 0.5 * ramp_area
    return rf, gz, _events.make_trapezoid(channel="z", area=rephase_area, system=system)


#: The name PyPulseq's SigPy-backed factory went by. Same design, no SigPy.
make_sigpy_pulse = make_slr_pulse


def make_sms_pulse(
    rf,
    num_bands: int,
    band_offset: float,
    *,
    sideband_power: float | Sequence[float] = 1.0,
    phases: str | Sequence[float] | None = None,
):
    """Modulate one RF pulse into equispaced spectral bands.

    Multiplies the envelope by a sum of complex exponentials, so one designed
    pulse excites several slices at once. ``sideband_power`` is power relative
    to the on-resonance band rather than an amplitude multiplier, so the
    modulation uses its square root; a scalar applies to every off-resonance
    band, and one value per band gives asymmetric saturation.

    The offset list always contains 0 Hz. For an even band count the otherwise
    unpaired band goes on the positive-frequency side.

    Parameters
    ----------
    rf : RfEvent or SimpleNamespace
        Base pulse to modulate.
    num_bands : int
        Number of bands, counting the on-resonance one.
    band_offset : float
        Spacing between adjacent bands (Hz).
    sideband_power : float or sequence of float, optional
        Power of each off-resonance band relative to the on-resonance band.
    phases : {'quadratic', 'wong', 'malik'} or sequence of float, optional
        Per-band phase (rad), lowest frequency first, or a schedule that keeps
        the peak down: Grissom's quadratic one, Wong's optimised table (3 to
        16 bands) or Malik's Hermitian one (4 to 12 bands). ``None`` leaves
        every band in phase.

    Returns
    -------
    rf : RfEvent
        The modulated pulse.
    offsets : numpy.ndarray
        Band offsets applied (Hz).
    weights : numpy.ndarray
        Complex weight applied to each band.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> base = pp.make_sinc_pulse(flip_angle=np.deg2rad(30), duration=2e-3, system=system)
    >>> rf, offsets, weights = pp.make_sms_pulse(base, 3, 1000.0)
    >>> offsets.tolist()
    [-1000.0, 0.0, 1000.0]

    See Also
    --------
    sim_rf : check where the bands actually land.
    """
    offsets = _band_offsets(num_bands, band_offset)
    weights = _band_weights(len(offsets), sideband_power, phases)

    source = _events.as_namespace(rf)
    time = np.asarray(source.t, dtype=float)
    modulation = np.sum(
        weights[:, None] * np.exp(2j * np.pi * offsets[:, None] * time[None, :]), axis=0
    )

    banded = _events.convert(source)
    banded.signal = np.asarray(source.signal) * modulation
    return banded, offsets, weights


def _band_offsets(num_bands: int, band_offset: float) -> np.ndarray:
    num_bands = int(num_bands)
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    if band_offset <= 0 and num_bands > 1:
        raise ValueError("band_offset must be > 0 when num_bands > 1")
    # An even count cannot be both symmetric and include zero; keep the nearest
    # symmetric pairs and put the unmatched band on the positive side.
    indices = np.arange(-((num_bands - 1) // 2), num_bands // 2 + 1)
    return indices.astype(float) * float(band_offset)


def _band_weights(num_bands: int, sideband_power, phases) -> np.ndarray:
    if np.isscalar(sideband_power):
        powers = np.full(num_bands, float(sideband_power))
        powers[(num_bands - 1) // 2] = 1.0
    else:
        powers = np.asarray(sideband_power, dtype=float)
        if powers.shape != (num_bands,):
            raise ValueError("sideband_power must be a scalar or one value per band")
    if np.any(~np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("band powers must be finite and >= 0")
    if phases is None:
        phase = np.zeros(num_bands)
    elif isinstance(phases, str):
        phase = band_phases(num_bands, phases)
    else:
        phase = np.asarray(phases, dtype=float)
        if phase.shape != (num_bands,):
            raise ValueError("phases must be a schedule name or one value per band")
    return np.sqrt(powers) * np.exp(1j * phase)


def make_spsp_pulse(
    flip_angle: float,
    slice_thickness: float,
    spectral_bandwidth: float,
    *,
    freq_offset: float = 0.0,
    spatial_time_bandwidth_product: float = 4.0,
    spectral_time_bandwidth_product: float = 3.0,
    n_subpulses: int = 10,
    system=None,
    use: str = "excitation",
):
    """Design a spectral-spatial pulse on an alternating slice gradient.

    Both envelopes use SLR designs. The subpulse count is rounded up to an
    even number; a rephaser is returned only when the residual area is nonzero.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad).
    slice_thickness : float
        Slice thickness (m).
    spectral_bandwidth : float
        Spectral passband (Hz).
    freq_offset : float, optional
        Centre of the spectral passband (Hz).
    spatial_time_bandwidth_product : float, optional
        Time-bandwidth product of each spatial subpulse.
    spectral_time_bandwidth_product : float, optional
        Time-bandwidth product of the spectral envelope; sets the total
        duration as ``spectral_time_bandwidth_product / spectral_bandwidth``.
    n_subpulses : int, optional
        Number of subpulses (>= 4, rounded up to even).
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.

    Returns
    -------
    rf : RfEvent
        The spectral-spatial pulse.
    gz : GradEvent
        The alternating selection gradient.
    gz_reph : TrapEvent or None
        Its rephaser, or ``None`` when the train needs none: the lobes after
        the pulse's centre cancel whenever there is an even number of them.

    Raises
    ------
    ValueError
        If the requested selectivity exceeds the gradient or slew limit, or the
        spectral bandwidth is too wide for the subpulse count.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> rf, gz, gz_reph = pp.make_spsp_pulse(
    ...     np.deg2rad(30), slice_thickness=10e-3,
    ...     spectral_bandwidth=300.0, n_subpulses=12, system=system,
    ... )
    >>> rf.type, gz.channel
    ('rf', 'z')

    See Also
    --------
    make_slr_pulse : the design underneath both of its envelopes.
    sim_rf : check the spectral profile the design produces.
    """
    system = default_system(system)
    if slice_thickness <= 0 or spectral_bandwidth <= 0:
        raise ValueError("slice_thickness and spectral_bandwidth must be > 0")
    if n_subpulses < 4:
        raise ValueError("n_subpulses must be >= 4")
    if n_subpulses % 2:
        n_subpulses += 1

    grad_raster = system.grad_raster_time
    rf_raster = system.rf_raster_time
    total_duration = spectral_time_bandwidth_product / spectral_bandwidth
    lobe_duration = round(total_duration / (n_subpulses * grad_raster)) * grad_raster
    if lobe_duration < 8 * grad_raster:
        raise ValueError(
            "spectral bandwidth is too large for the requested subpulse count"
        )

    # Reserve a fifth of each lobe for its two ramps, on the gradient raster.
    ramp_time = max(grad_raster, round(0.1 * lobe_duration / grad_raster) * grad_raster)
    flat_time = lobe_duration - 2.0 * ramp_time
    if flat_time < 8 * rf_raster:
        raise ValueError("SPSP sublobes leave fewer than 8 RF samples")
    amplitude_hz_per_m = spatial_time_bandwidth_product / (flat_time * slice_thickness)
    if amplitude_hz_per_m > system.max_grad:
        raise ValueError("SPSP slice-selection amplitude exceeds system.max_grad")
    if amplitude_hz_per_m / ramp_time > system.max_slew * (1.0 + 1e-9):
        raise ValueError("SPSP slice-selection ramps exceed system.max_slew")

    # Design the spatial SLR pulse on the flat top, build one trapezoid, then
    # VERSE the RF onto its ramps: the time-bandwidth product survives and the
    # RF overlaps the gradient for the whole lobe.
    spatial_rf = make_slr_pulse(
        1.0,
        duration=flat_time,
        dwell=rf_raster,
        time_bw_product=spatial_time_bandwidth_product,
        pulse_type="st",
        system=system,
        use=use,
    )
    positive_lobe = _events.make_trapezoid(
        channel="z",
        amplitude=amplitude_hz_per_m,
        flat_time=flat_time,
        rise_time=ramp_time,
        fall_time=ramp_time,
        system=system,
    )
    spatial = _verse_to_trapezoid(spatial_rf.signal, positive_lobe, rf_raster)
    spectral = design_slr(
        n_subpulses, spectral_time_bandwidth_product, pulse_type="st", filter_type="ls"
    )

    shape = np.empty(n_subpulses * spatial.size, dtype=np.complex128)
    for index in range(n_subpulses):
        start = index * spatial.size
        subpulse = spatial if index % 2 == 0 else spatial[::-1]
        shape[start : start + spatial.size] = spectral[index] * subpulse

    rf = _events.make_arbitrary_rf(
        signal=shape,
        flip_angle=flip_angle,
        dwell=rf_raster,
        freq_offset=freq_offset,
        system=system,
        use=use,
    )
    gz = _alternating_extended_trapezoid(positive_lobe, n_subpulses, system)
    # The train and the pulse start together: each subpulse is played under the
    # lobe it was designed for, and a dead time that moved only one of them
    # would slide every subpulse off its own lobe.
    gz.delay = rf.delay
    # What has to be undone is the moment the train accrues after the pulse's
    # centre. Half the lobes fall there and they alternate, so an even number
    # of them cancels outright and an odd number leaves exactly one.
    lobes_after_centre = n_subpulses // 2
    gz_reph = (
        _events.make_trapezoid(channel="z", area=positive_lobe.area, system=system)
        if lobes_after_centre % 2
        else None
    )
    return rf, gz, gz_reph


def _sample_trapezoid(event, dwell: float) -> np.ndarray:
    """Sample a trapezoid at RF-raster midpoints."""
    duration = event.rise_time + event.flat_time + event.fall_time
    time = (np.arange(round(duration / dwell)) + 0.5) * dwell
    knots = np.array(
        [0.0, event.rise_time, event.rise_time + event.flat_time, duration]
    )
    values = np.array([0.0, event.amplitude, event.amplitude, 0.0])
    return np.interp(time, knots, values)


def _verse_to_trapezoid(signal: np.ndarray, event, dwell: float) -> np.ndarray:
    """VERSE an RF envelope designed on a flat gradient onto a trapezoid."""
    gradient = _sample_trapezoid(event, dwell)
    magnitude = np.abs(gradient)
    total = magnitude.sum()
    if total <= 0:
        raise ValueError("VERSE gradient has zero area")
    excitation_position = (np.cumsum(magnitude) - 0.5 * magnitude) / total
    source_position = (np.arange(len(signal)) + 0.5) / len(signal)
    source = np.asarray(signal, dtype=complex)
    interpolated = np.interp(
        excitation_position, source_position, source.real
    ) + 1j * np.interp(excitation_position, source_position, source.imag)
    return interpolated * magnitude / np.max(magnitude)


def _alternating_extended_trapezoid(event, count: int, system):
    """Concatenate alternating trapezoids without rasterising their ramps."""
    lobe_duration = event.rise_time + event.flat_time + event.fall_time
    times: list[float] = []
    amplitudes: list[float] = []
    for index in range(count):
        start = index * lobe_duration
        sign = 1.0 if index % 2 == 0 else -1.0
        knots = (
            start,
            start + event.rise_time,
            start + event.rise_time + event.flat_time,
            start + lobe_duration,
        )
        values = (0.0, sign * event.amplitude, sign * event.amplitude, 0.0)
        for knot, value in zip(knots, values, strict=True):
            if times and np.isclose(knot, times[-1], atol=1e-15):
                amplitudes[-1] = value
            else:
                times.append(knot)
                amplitudes.append(value)
    return _events.make_extended_trapezoid(
        channel="z",
        times=np.asarray(times),
        amplitudes=np.asarray(amplitudes),
        system=system,
    )


def make_2d_selective_pulse(
    flip_angle: float,
    fov: float,
    matrix: int,
    *,
    selective_size: float | Sequence[float] | None = None,
    target: np.ndarray | None = None,
    n_interleaves: int | None = None,
    axes: Sequence[str] = ("x", "y"),
    b1_maps=None,
    off_resonance=None,
    magnitude_only: bool = False,
    regularization: float = 0.0,
    system=None,
    use: str = "excitation",
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
):
    """Design a small-tip 2D-selective pulse on a spiral excitation trajectory.

    Each centre-out interleave is followed by an RF-off retrace to the origin.
    The envelope uses arc-length weighting, without an additional radial
    density factor. The closed trajectory requires no separate rephaser.

    Parameters
    ----------
    flip_angle : float
        Nominal flip angle (rad), reached at the centre of the excited region.
    fov : float
        Excitation field of view (m), square. Outside it the profile repeats:
        the trajectory samples excitation k-space at a finite pitch, so a
        second excited spot appears one ``fov`` away.
    matrix : int
        Excitation grid size, square. It sets how finely the profile is
        specified, and so how far out the trajectory has to reach.
    selective_size : float or sequence of float, optional
        Diameter (m) of the excited disc, one value or one per axis. Half the
        field of view by default.
    target : numpy.ndarray, optional
        Complex desired profile on the ``(matrix, matrix)`` grid, instead of a
        disc.
    n_interleaves : int, optional
        Spiral arms to play. The default is what covers excitation k-space at
        Nyquist. Fewer arms shorten the pulse and reduce the alias-free excitation FOV.
    axes : sequence of str, optional
        The two gradient channels the trajectory runs on.
    b1_maps : array_like, optional
        Complex B1+ per transmit channel, ``(num_channels, matrix, matrix)``,
        relative to nominal. When given, the pulse is designed in the
        spatial domain against them (Grissom et al., Magn Reson Med 56:620,
        2006) and returned as a pTx pulse, one waveform per channel, whose
        small-tip flip is ``flip_angle`` times the target; a single channel
        is a pulse tailored to that channel's B1.
    off_resonance : array_like, optional
        Off-resonance map, ``(matrix, matrix)``, in Hz, for the spatial-domain
        design.
    magnitude_only : bool, optional
        Fit only the target's magnitude in the spatial-domain design, leaving
        its phase free (magnitude least squares).
    regularization : float, optional
        Tikhonov weight on the waveforms' power in the spatial-domain design.
    system : pypulseq.Opts, optional
        System limits.
    use : str, optional
        Pulseq ``use`` tag.
    freq_offset, phase_offset : float, optional
        Event offsets in Hz and radians, respectively.

    Returns
    -------
    rf : RfEvent
        The pulse, sampled on the gradient raster; a pTx pulse when
        ``b1_maps`` is given.
    gradients : tuple of GradEvent
        One arbitrary gradient per axis, to be played in the pulse's block.
    rephasers : tuple of TrapEvent
        Empty tuple: the closed spiral trajectory needs no separate rephaser.

    Raises
    ------
    ValueError
        If a size is out of range, ``target`` does not match ``matrix``, or the
        requested profile has no response on the trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> rf, gradients, rephasers = pp.make_2d_selective_pulse(
    ...     np.deg2rad(30), fov=0.256, matrix=32, selective_size=0.04, system=system
    ... )
    >>> [gradient.channel for gradient in gradients], len(rephasers)
    (['x', 'y'], 0)

    Half the arms is half the pulse:

    >>> short, _, _ = pp.make_2d_selective_pulse(
    ...     np.deg2rad(30), fov=0.256, matrix=32, selective_size=0.04,
    ...     n_interleaves=8, system=system,
    ... )
    >>> round(float(short.shape_dur) / float(rf.shape_dur), 3)
    0.5

    See Also
    --------
    make_spsp_pulse : selective in one dimension and in frequency.
    sim_rf : check the profile a pulse actually produces.
    """
    # The compiled trajectory designer stays optional for every other pulse.
    from . import _arbgrad

    system = default_system(system)
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("axes must name two gradient channels")
    if fov <= 0 or int(matrix) < 2:
        raise ValueError("fov must be positive and matrix must be >= 2")

    native = _arbgrad.spiral(
        fov=fov,
        n_pix=int(matrix),
        slew_limit=system.max_slew * fov / int(matrix),
        grad_limit=system.max_grad * fov / int(matrix),
        dt=system.grad_raster_time,
    )
    arms = native.n_shots if n_interleaves is None else int(n_interleaves)
    if not 1 <= arms <= native.n_shots:
        raise ValueError(
            f"n_interleaves must be between 1 and the {native.n_shots} this trajectory "
            f"needs for Nyquist coverage, got {arms}"
        )
    interleaves = []
    active = []
    for angle in calc_uniform_angles(native.n_shots)[:arms]:
        cosine, sine = np.cos(angle), np.sin(angle)
        turned = (
            np.asarray(native.gradient)[:, :2]
            @ np.array([[cosine, -sine], [sine, cosine]]).T
        )
        interleaves.extend((turned, -turned[::-1]))
        active.extend((np.ones(len(turned), bool), np.zeros(len(turned), bool)))
    gradient = _arbgrad.to_gradient_tesla_per_meter(
        np.concatenate(interleaves, axis=0), fov, int(matrix), system.gamma
    )[:, :2]

    shape = (int(matrix), int(matrix))
    extent = (float(fov), float(fov))
    if selective_size is not None:
        selective_size = (
            (float(selective_size),) * 2
            if np.isscalar(selective_size)
            else tuple(float(value) for value in selective_size)
        )
        if len(selective_size) != 2:
            raise ValueError("selective_size must be a scalar or two values")

    dwell = system.grad_raster_time
    # Excitation k-space at a sample is the gradient moment still to come,
    # which is what puts the final phase in the right place.
    kspace = -np.cumsum((gradient * system.gamma)[::-1], axis=0)[::-1] * dwell
    desired, coordinates = _selective_target(shape, extent, selective_size, target)
    if b1_maps is None:
        weights = _small_tip_weights(desired, coordinates, kspace)
        weights *= np.r_[np.linalg.norm(np.diff(kspace, axis=0), axis=1), 0.0]
        weights[~np.concatenate(active)] = 0.0
        rf = _events.make_arbitrary_rf(
            signal=weights,
            flip_angle=flip_angle,
            dwell=dwell,
            freq_offset=freq_offset,
            phase_offset=phase_offset,
            system=system,
            use=use,
        )
    else:
        from ._ptx import _selective_waveforms, make_ptx_pulse

        waveforms = _selective_waveforms(
            flip_angle * desired,
            coordinates,
            kspace,
            np.concatenate(active),
            b1_maps,
            shape,
            off_resonance,
            magnitude_only,
            regularization,
            dwell,
        )
        rf = make_ptx_pulse(
            waveforms,
            dwell=dwell,
            freq_offset=freq_offset,
            phase_offset=phase_offset,
            system=system,
            use=use,
        )
    gradients = []
    rephasers = []
    for index, axis in enumerate(axes):
        waveform = np.ascontiguousarray(gradient[:, index] * system.gamma)
        if np.allclose(waveform, 0.0):
            continue
        event = _events.make_arbitrary_grad(
            channel=axis,
            waveform=waveform,
            first=0.0,
            last=0.0,
            # Sample n of the envelope was designed for sample n of the
            # trajectory, so the transmit dead time has to shift both or the
            # two come apart and the profile is not the one asked for.
            delay=float(rf.delay),
            system=system,
        )
        gradients.append(event)
        area = float(np.trapezoid(waveform, np.arange(len(waveform)) * dwell))
        if not np.isclose(area, 0.0, atol=1e-6):
            rephasers.append(
                _events.make_trapezoid(channel=axis, area=-area, system=system)
            )
    return rf, tuple(gradients), tuple(rephasers)


def _selective_target(matrix, fov, selective_size, target):
    axes = [
        (np.arange(n) - (n - 1) / 2.0) * (extent / n)
        for n, extent in zip(matrix, fov, strict=True)
    ]
    coordinates = np.stack(
        [axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=1
    )
    if target is not None:
        target = np.asarray(target, dtype=np.complex128)
        if target.shape != tuple(matrix):
            raise ValueError(
                f"target shape {target.shape} does not match matrix {matrix}"
            )
        return target.ravel(), coordinates

    if selective_size is None:
        selective_size = tuple(0.5 * extent for extent in fov)
    radius_squared = np.zeros(len(coordinates))
    for dim, diameter in enumerate(selective_size):
        if not 0 < diameter <= fov[dim]:
            raise ValueError("each selective_size entry must lie in (0, fov]")
        radius_squared += (coordinates[:, dim] / (0.5 * diameter)) ** 2
    return (radius_squared <= 1.0).astype(np.complex128), coordinates


def _small_tip_weights(target, coordinates, kspace):
    """Evaluate the target's Fourier transform along an excitation trajectory."""
    weights = np.empty(len(kspace), dtype=np.complex128)
    # Bound the temporary to about 64 MiB however long the trajectory runs.
    chunk = max(1, int(4_000_000 / max(1, len(coordinates))))
    for start in range(0, len(kspace), chunk):
        stop = min(start + chunk, len(kspace))
        phase = coordinates @ kspace[start:stop].T
        weights[start:stop] = target @ np.exp(-2j * np.pi * phase) / target.size
    if np.allclose(weights, 0.0):
        raise ValueError("the requested target has no response on this trajectory")
    return weights


def make_half_passages(
    duration: float,
    *,
    adiabaticity: float = 8,
    dwell: float = 10e-6,
    pulse_type: str = "hypsec",
    use: str = "preparation",
    system=None,
) -> tuple:
    """Build the adiabatic pair that tips magnetization down and stores it back.

    A half passage is one half of a full adiabatic sweep. Run from far
    off-resonance to on-resonance, it carries magnetization from ``+z`` into
    the transverse plane; run in reverse, it carries it back. Both are
    adiabatic, so above a threshold transmit amplitude neither depends on
    what the amplitude actually is -- which is why a T2 preparation uses them
    instead of a pair of hard 90s.

    The two are exact mirrors: the second is the first time-reversed and
    conjugated. That is what makes the phase the sweep accrues on the way down
    unwind on the way up, so what is stored is the magnetization's *magnitude*
    and not a phase that varied with transmit field.

    Parameters
    ----------
    duration : float
        Duration of each half passage, in s.
    adiabaticity : float, optional
        Sweep-rate margin over the adiabatic condition. The default is twice
        an inversion's, because a half passage has only half a sweep to
        converge in.
    dwell : float, optional
        RF raster, in s.
    pulse_type : str, optional
        Sweep family, as :func:`make_adiabatic_pulse` names them.
    use : str, optional
        What the pulses are for, as Pulseq records it.
    system : Opts, optional
        System limits.

    Returns
    -------
    down, up : SimpleNamespace
        The half passage that tips down, and the reverse one that stores what
        is left back on ``z``.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> down, up = pp.make_half_passages(4e-3)
    >>> bool(np.allclose(np.asarray(down.signal), np.conj(np.asarray(up.signal))[::-1]))
    True
    """
    system = default_system(system)
    full = _events.make_adiabatic_pulse(
        pulse_type=pulse_type,
        duration=2.0 * duration,
        dwell=dwell,
        adiabaticity=adiabaticity,
        system=system,
        use=use,
    )
    signal = np.asarray(full.signal)
    sweep = signal[: signal.size // 2]
    return tuple(
        _events.make_arbitrary_rf(
            signal=half,
            flip_angle=np.pi / 2.0,
            no_signal_scaling=True,
            dwell=dwell,
            system=system,
            use=use,
        )
        for half in (sweep, np.conj(sweep[::-1]))
    )
