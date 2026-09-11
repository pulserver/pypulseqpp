"""Parallel-transmit pulses: the multi-channel RF event, shims, and small-tip designs."""

from __future__ import annotations

__all__ = ["calc_rf_shim", "make_ptx_pulse", "make_spokes_pulse", "split_ptx_pulse"]

import math
from types import SimpleNamespace

import numpy as np
import pypulseq as _pp

from . import _events
from ._opts import default_system
from ._slr import design_slr


def make_ptx_pulse(
    signal,
    *,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    center: float | None = None,
    system=None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Make a dynamic pTx pulse from one waveform per transmit channel.

    The channels are stored one after another in a single arbitrary RF event
    over a shared time base, so the time shape restarts once per channel
    (Roos et al., Magn Reson Med 2025, doi:10.1002/mrm.30601). ``shape_dur``
    is one channel's duration, so an interpreter unaware of the layout still
    reads the right pulse length.

    Parameters
    ----------
    signal : array_like
        Complex waveforms, ``(num_channels, num_samples)``, in Hz. Played as
        given: nothing is scaled to a flip angle, since with several channels
        the flip depends on each channel's B1 map.
    dwell : float, optional
        Sample spacing, in s; ``system.rf_raster_time`` when zero.
    center : float, optional
        Centre of the pulse from its start, in s. Defaults to the centre of
        the peak of the channel-summed magnitude.

    Other parameters are as in :func:`make_arbitrary_rf`.

    Returns
    -------
    RfEvent
        The pulse; ``shape_dur`` is ``num_samples * dwell``.

    Raises
    ------
    ValueError
        If ``signal`` is not two-dimensional or holds no samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> rf = pp.make_ptx_pulse(np.full((8, 100), 250.0 + 0j))
    >>> pp.split_ptx_pulse(rf).shape
    (8, 100)
    """
    waveforms = np.asarray(signal, dtype=np.complex128)
    if waveforms.ndim != 2 or waveforms.shape[1] == 0:
        raise ValueError(
            "signal must be (num_channels, num_samples) with at least one sample"
        )
    system = default_system(system)
    dwell = dwell or system.rf_raster_time
    channels, samples = waveforms.shape
    times = (np.arange(samples) + 0.5) * dwell
    if center is None:
        center, _ = _pp.calc_rf_center(
            SimpleNamespace(signal=np.abs(waveforms).sum(axis=0), t=times)
        )

    rf = _pp.make_arbitrary_rf(
        signal=waveforms.reshape(-1),
        flip_angle=0.0,
        no_signal_scaling=True,
        delay=delay,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        system=system,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        center=float(center),
    )
    # The factory gives the flattened samples consecutive times; restart the
    # time base for each channel.
    rf.t = np.tile(times, channels)
    rf.shape_dur = samples * dwell
    return _events.convert(rf)


def split_ptx_pulse(rf) -> np.ndarray:
    """Return a pulse's waveforms, one row per transmit channel, in Hz.

    The channel count is the number of samples at the pulse's first sample
    time, as the reference interpreter reads it; a single-channel pulse comes
    back as one row.

    Raises
    ------
    ValueError
        If the time shape is not one time base repeated once per channel. The
        native checks instead treat such a pulse as a single channel.
    """
    times = np.asarray(rf.t, dtype=float)
    signal = np.asarray(rf.signal)
    channels = int(np.count_nonzero(times == times[0]))
    per_channel = times.size // channels
    if (
        per_channel * channels != times.size
        or signal.size != times.size
        or not np.array_equal(
            times.reshape(channels, per_channel),
            np.tile(times[:per_channel], (channels, 1)),
        )
    ):
        raise ValueError(
            "this pulse's time shape is not one time base repeated per channel"
        )
    return signal.reshape(channels, per_channel)


def calc_rf_shim(
    b1_maps,
    mask=None,
    *,
    target=1.0,
    regularization: float = 0.0,
    tolerance: float = 1e-6,
    rounds: int = 100,
) -> np.ndarray:
    """Return per-channel weights whose combined B1 field has magnitude ``target``.

    Magnitude least squares over ``mask``: the combined field
    ``sum_c w[c] * b1_maps[c]`` has free phase. The fit starts from the channel
    combination that delivers the most field over the mask.

    Parameters
    ----------
    b1_maps : array_like
        Complex B1+ per channel, ``(num_channels, *grid)``, in any unit shared
        by all channels.
    mask : array_like of bool, optional
        Where the magnitude is fitted, shaped like the grid. Defaults to every
        point where any map is non-zero.
    target : float or array_like, optional
        Magnitude wanted, in the unit of ``b1_maps``: a scalar or one value per
        grid point.
    regularization : float, optional
        Tikhonov weight on ``sum |w|^2``.
    tolerance : float, optional
        Relative change of the cost at which the fit stops.
    rounds : int, optional
        Maximum number of phase exchanges.

    Returns
    -------
    numpy.ndarray
        ``(num_channels,)`` complex weights, as :func:`make_rf_shim` takes them.

    Raises
    ------
    ValueError
        If ``mask`` or ``target`` does not match the grid, or the mask is empty.
    """
    from ._ext import ptx

    maps = np.asarray(b1_maps, dtype=np.complex128)
    grid = maps.shape[1:]
    inside = np.any(maps != 0, axis=0) if mask is None else np.asarray(mask, dtype=bool)
    if inside.shape != grid or not inside.any():
        raise ValueError(f"mask must be a non-empty boolean array of shape {grid}")
    wanted = np.broadcast_to(np.asarray(target, dtype=float), grid)[inside]
    return ptx.rf_shim(
        maps[:, inside],
        np.ascontiguousarray(wanted),
        np.ones(wanted.size),
        regularization,
        tolerance,
        rounds,
    )


def _selective_waveforms(
    target,
    coordinates,
    kspace,
    active,
    b1_maps,
    grid,
    off_resonance,
    magnitude_only,
    regularization,
    dwell,
):
    """Return ``(num_channels, samples)`` waveforms, in Hz, for a small-tip target.

    ``target`` is the transverse magnetisation wanted at ``coordinates`` (m),
    in rad; ``kspace`` is the gradient moment still to come after each sample,
    in cycles/m. Only samples in ``active`` are designed; the rest are zero.
    """
    from ._ext import ptx

    maps = np.asarray(b1_maps, dtype=np.complex128)
    if maps.ndim != 3 or maps.shape[1:] != tuple(grid):
        raise ValueError(f"b1_maps must be (num_channels, {grid[0]}, {grid[1]})")
    df = np.zeros(0)
    if off_resonance is not None:
        df = np.asarray(off_resonance, dtype=float)
        if df.shape != tuple(grid):
            raise ValueError(f"off_resonance must be shaped {tuple(grid)}")
        df = df.ravel()
    samples = kspace.shape[0]
    times = (np.arange(samples) + 0.5) * dwell
    played = ptx.design(
        np.ascontiguousarray(target),
        np.ones(target.size),
        maps.reshape(maps.shape[0], -1),
        np.ascontiguousarray(coordinates),
        np.ascontiguousarray(kspace[active]),
        np.ascontiguousarray(times[active]),
        samples * dwell,
        2.0 * np.pi * dwell,
        df,
        regularization,
        30,
        1e-6,
        10 if magnitude_only else 0,
    )
    waveforms = np.zeros((maps.shape[0], samples), dtype=np.complex128)
    waveforms[:, active] = played
    return waveforms


def make_spokes_pulse(
    flip_angle: float,
    b1_maps,
    fov,
    slice_thickness: float,
    *,
    n_spokes: int = 3,
    mask=None,
    resolution: float | None = None,
    time_bw_product: float = 4.0,
    regularization: float = 0.0,
    axes=("x", "y", "z"),
    delay: float = 0.0,
    system=None,
    use: str = "excitation",
):
    """Design a spokes pTx pulse: one slice excited at several in-plane k positions.

    Each spoke is a small-tip SLR sub-pulse on one lobe of an alternating
    slice gradient; in-plane blips between lobes move it through excitation
    k-space. Spoke positions are chosen greedily from a grid, and each
    channel's weight on each spoke by magnitude least squares against a
    uniform ``flip_angle`` over the mask (Grissom et al., Magn Reson Med
    68:1553, 2012).

    Parameters
    ----------
    flip_angle : float
        Flip wanted over the mask, in rad, in the small-tip model.
    b1_maps : array_like
        Complex B1+ per channel in the slice, ``(num_channels, nx, ny)``,
        relative: a map of ones reaches every point at nominal amplitude.
    fov : float or (float, float)
        In-plane extent the maps cover, in m, centred on the isocentre.
    slice_thickness : float
        In m.
    n_spokes : int, optional
        Number of spokes, one of them at the k-space centre.
    mask : array_like of bool, optional
        Where the flip is fitted, ``(nx, ny)``; everywhere by default.
    resolution : float, optional
        Finest in-plane scale the spokes may encode, in m: candidate positions
        span ``1 / resolution`` at a pitch of ``1 / fov``. An eighth of the
        smaller field of view by default.
    time_bw_product : float, optional
        Of each spoke's slice profile.
    regularization : float, optional
        Tikhonov weight on the spoke weights.
    axes : (str, str, str), optional
        The two in-plane gradient channels, then the slice-select one.
    delay : float, optional
        Delay before the pulse and its gradients, in s; raised to
        ``system.rf_dead_time`` and rounded up to the gradient raster.

    Returns
    -------
    rf : RfEvent
        A pTx pulse, one waveform per channel (see :func:`make_ptx_pulse`).
    gradients : tuple of GradEvent
        The slice lobes, then the in-plane blips, for the pulse's block.
    rephasers : tuple of TrapEvent
        The in-plane return to the k-space centre, then the slice rephaser,
        for the block after.

    Raises
    ------
    ValueError
        If the maps, mask or spoke count do not fit the grid.
    """
    from ._ext import ptx

    system = default_system(system)
    maps = np.asarray(b1_maps, dtype=np.complex128)
    if maps.ndim != 3:
        raise ValueError("b1_maps must be (num_channels, nx, ny)")
    grid = maps.shape[1:]
    extent = (float(fov),) * 2 if np.isscalar(fov) else tuple(float(v) for v in fov)
    inside = np.ones(grid, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if inside.shape != grid or not inside.any():
        raise ValueError(f"mask must be a non-empty boolean array of shape {grid}")
    coordinates = np.meshgrid(
        *[
            (np.arange(n) - (n - 1) / 2.0) * (e / n)
            for n, e in zip(grid, extent, strict=True)
        ],
        indexing="ij",
    )
    positions = np.column_stack([axis[inside] for axis in coordinates])
    reach = 1.0 / (min(extent) / 8.0 if resolution is None else float(resolution))
    pitches = [np.arange(-reach / 2, reach / 2 - 1 / e + 1e-9, 1 / e) for e in extent]
    candidates = np.array(
        [
            (a, b)
            for a in pitches[0]
            for b in pitches[1]
            if abs(a) > 1e-12 or abs(b) > 1e-12
        ]
    ).reshape(-1, 2)
    if not 1 <= int(n_spokes) <= len(candidates) + 1:
        raise ValueError(f"n_spokes must lie in [1, {len(candidates) + 1}]")
    where, amounts = ptx.spokes(
        np.ascontiguousarray(maps[:, inside]),
        np.ascontiguousarray(positions),
        np.full(len(positions), float(flip_angle)),
        np.ones(len(positions)),
        np.ascontiguousarray(candidates, dtype=float),
        int(n_spokes),
        regularization,
    )

    raster = system.grad_raster_time
    dwell = system.rf_raster_time
    # The minimum-time lobe whose plateau covers the slice's k-space width.
    area = time_bw_product / slice_thickness
    peak = min(system.max_grad, math.sqrt(system.max_slew * area / 2.0))
    flat_time = raster * math.ceil(area / peak / raster - 1e-9)
    lobe = _events.make_trapezoid(
        axes[2], flat_area=area, flat_time=flat_time, system=system
    )
    rise, fall = float(lobe.rise_time), float(lobe.fall_time)

    samples = round(flat_time / dwell)
    even = max(8, samples - samples % 2)
    sub = np.real(design_slr(even, time_bw_product, pulse_type="st"))
    if samples != even:
        sub = np.interp(
            (np.arange(samples) + 0.5) / samples, (np.arange(even) + 0.5) / even, sub
        )
    # Hz per radian of spoke weight.
    sub = sub / (2.0 * np.pi * dwell * sub.sum())

    moves = np.diff(np.vstack([where, np.zeros((1, 2))]), axis=0)
    z_times, z_levels = [0.0], [0.0]
    blip_vertices = [([0.0], [0.0]), ([0.0], [0.0])]
    plateaus = []
    start_of_lobe, sign = 0.0, 1.0
    for j in range(len(where)):
        if start_of_lobe > z_times[-1]:
            z_times.append(start_of_lobe)
            z_levels.append(0.0)
        plateau = start_of_lobe + rise
        plateaus.append(plateau)
        z_times += [plateau, plateau + flat_time, plateau + flat_time + fall]
        z_levels += [sign * lobe.amplitude, sign * lobe.amplitude, 0.0]
        end_of_lobe = plateau + flat_time + fall
        if j == len(where) - 1:
            start_of_lobe = end_of_lobe
            break
        blips = [
            _events.make_trapezoid(axes[a], area=moves[j, a], system=system)
            if abs(moves[j, a]) > 1e-9
            else None
            for a in (0, 1)
        ]
        lengths = [
            b.rise_time + b.flat_time + b.fall_time for b in blips if b is not None
        ]
        # Blips play between plateaus; the lobes part only if a blip needs room.
        gap = raster * math.ceil(
            max(0.0, max(lengths, default=0.0) - fall - rise) / raster - 1e-9
        )
        room = fall + gap + rise
        for a, b in enumerate(blips):
            if b is None:
                continue
            length = b.rise_time + b.flat_time + b.fall_time
            begin = (
                plateau
                + flat_time
                + raster * math.floor((room - length) / 2 / raster + 1e-9)
            )
            times, levels = blip_vertices[a]
            corners = [begin, begin + b.rise_time]
            heights = [0.0, b.amplitude]
            if b.flat_time > 0:
                corners.append(corners[-1] + b.flat_time)
                heights.append(b.amplitude)
            corners.append(corners[-1] + b.fall_time)
            heights.append(0.0)
            if corners[0] <= times[-1]:
                corners, heights = corners[1:], heights[1:]
            times += corners
            levels += heights
        start_of_lobe = end_of_lobe + gap
        sign = -sign
    total = start_of_lobe

    begin = raster * math.ceil(max(delay, system.rf_dead_time) / raster - 1e-9)
    waveforms = np.zeros((maps.shape[0], round(total / dwell)), dtype=np.complex128)
    for j, plateau in enumerate(plateaus):
        first = round(plateau / dwell)
        waveforms[:, first : first + samples] = amounts[:, j][:, None] * sub[None, :]
    centre = 0.5 * (plateaus[0] + plateaus[-1] + flat_time)
    rf = make_ptx_pulse(
        waveforms, delay=begin, dwell=dwell, center=centre, system=system, use=use
    )

    gradients = []
    z = _events.make_extended_trapezoid(
        axes[2],
        times=np.asarray(z_times),
        amplitudes=np.asarray(z_levels),
        system=system,
    )
    z.delay = begin
    gradients.append(z)
    for a, (times, levels) in enumerate(blip_vertices):
        if len(times) == 1:
            continue
        if times[-1] < total:
            times, levels = [*times, total], [*levels, 0.0]
        event = _events.make_extended_trapezoid(
            axes[a],
            times=np.asarray(times),
            amplitudes=np.asarray(levels),
            system=system,
        )
        event.delay = begin
        gradients.append(event)

    # What the last lobe winds after its centre, and the way back to k = 0.
    after = sign * (area / 2.0 + 0.5 * lobe.amplitude * fall)
    rephasers = [
        _events.make_trapezoid(axes[a], area=moves[-1, a], system=system)
        for a in (0, 1)
        if abs(moves[-1, a]) > 1e-9
    ]
    rephasers.append(_events.make_trapezoid(axes[2], area=-after, system=system))
    return rf, tuple(gradients), tuple(rephasers)
