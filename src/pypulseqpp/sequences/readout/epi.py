"""Echo-planar readouts: one blipped, ramp-sampled train per repetition."""

from __future__ import annotations

__all__ = ["EpiReadout2D", "EpiReadout3D"]

from typing import Any

import numpy as np

import pypulseqpp as pp

from ..._epi import make_epi_shot_offsets
from ..._gradients import _trapezoid
from .._module import SequenceModule
from ._common import (
    as_tuple,
    left_align_rephaser,
    present,
    solve_delay,
    solve_rephasing,
)

#: Fraction of ``max_grad`` the readout plateau may reach.
_READOUT_GRAD_MARGIN = 0.8


class _EpiReadout(SequenceModule):
    """One RF event followed by an EPI train with a fixed phase-encode pattern.

    The caller sets the shot origin by scaling the prewinders. Blips step
    between the lines of ``order`` and share the readout ramps; with
    ``flyback`` they play in the rewind gaps instead. TE is timed to the line
    ``te_line`` (the first by default), because which line samples the centre
    of k-space depends on the shot origin the caller sets.

    Navigator lines are read after the read prewinder and before the
    phase-encode prewinder, without blips, so they sample the centre line at
    alternating polarities; the train continues the alternation.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        Pulse that opens the repetition; a refocusing pulse gives the readout
        half of a spin-echo EPI.
    gz : GradEvent, default=None
        Selection gradient played with ``rf``.
    gz_reph : GradEvent, default=None
        Rephaser of ``gz``, left-aligned in the first block after the pulse.
    fov : float or sequence of float
        Field of view per encoded axis, readout first (m).
    matrix : int or sequence of int
        Matrix size per encoded axis, readout first.
    order : array_like, default=None
        Integer offsets from the shot origin, ``(etl,)`` or ``(etl, 2)``, one
        row per line. When given, ``etl`` to ``extent`` are ignored; ``None``
        generates the train with :func:`~pypulseqpp.make_epi_shot_offsets`.
    etl : int, default=None
        Lines per repetition. ``None`` is what one shot of ``scheme`` needs to
        cross the phase-encode matrix.
    scheme : {'linear', 'caipi', 'zigzag'}, default='linear'
        Ordering to generate. ``'linear'`` steps by ``segments * acceleration``
        every line within one partition; ``'caipi'`` adds the partition
        sawtooth of blipped-CAIPI; ``'zigzag'`` traverses up and down a
        phase-encode segment instead of across the matrix.
    acceleration : int, default=1
        Phase-encode undersampling ``Ry``: the number of lines a blip steps.
    segments : int, default=1
        Shots ``S`` the lattice is interleaved across. The blip becomes
        ``S * Ry``, which shortens the train.
    partition_acceleration : int, default=1
        Partition undersampling ``Rz``, the height of the CAIPI cycle.
        ``'caipi'`` only.
    caipi_shift : int, default=1
        Partitions the pattern climbs per acquired line. ``'caipi'`` only.
    extent : int, default=None
        Phase-encode lines one pass spans. Required by ``'zigzag'`` and
        rejected by the other schemes.
    te : float, default=None
        Echo time (s), from the RF centre to the echo of line ``te_line``.
        ``None`` is as short as possible. Consecutive echoes are ``esp``
        apart; ``echo_times`` lists them.
    te_line : float, default=0.0
        Line, counted from zero and possibly fractional, that ``te`` is timed
        to.
    tr : float, default=None
        Repetition time (s) of the whole module, including the longest echo
        shift. ``None`` is as short as possible.
    navigator_lines : int, default=0
        Lines read without blips between the read and phase-encode prewinders,
        for odd-even phase correction. Rejected with ``flyback``.
    echo_shifts : int, default=1
        Number of shots whose trains are delayed by successive multiples of
        ``esp / echo_shifts``. The module plays no shift: the caller lengthens
        ``wait_shift``.
    oversampling : float, default=1.0
        Readout oversampling: the sample spacing in k-space is
        ``1 / (oversampling * fov)``.
    readout_bandwidth_hz : float, default=500000.0
        Requested ADC sampling rate (Hz); ``bandwidth_hz`` is the rate achieved
        on the ADC and gradient rasters.
    ramp_sampling : bool, default=True
        Sample across the read ramps as well as the plateau, which shortens the
        echo spacing. False restricts the window to the plateau.
    flyback : bool, default=False
        Read every line in the same direction, rewinding between lines, instead
        of alternating polarity. Adds a rewind per line and removes the
        odd-even echo inconsistency of a bipolar train.
    spoiling_cycles : float, default=0.0
        Cycles of read-axis dephasing across ``voxel_size_m`` left at the end
        of the repetition.
    voxel_size_m : float, default=None
        Length the spoiling is counted over (m); ``None`` is the readout
        resolution.
    labels : sequence of str, default=None
        Names of the labels holding the shot origin of each encoded axis,
        phase encode first: two for a 3D train, one for 2D.
    trigger : TriggerEvent or DigitalOutputEvent, default=None
        Event armed on the prewinder block.


    Attributes
    ----------
    rf : RfEvent
        The excitation.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given, left-aligned in whichever block follows
        the pulse.
    gx_pre : TrapEvent
        Read prewinder, half a line lobe against the first line's polarity,
        right-aligned in the prewinder block so that it ends where the first
        line starts.
    gx : list of GradEvent
        One read lobe per line: alternating polarity for a blipped train, the
        same lobe every time for a flyback one.
    gx_navigator : list of GradEvent
        One read lobe per navigator line, the first positive; present only with
        ``navigator_lines``.
    gx_flyback : TrapEvent
        The rewind between two lines. ``flyback`` only.
    gy_blips, gz_blips : list
        One phase-encode event per line, ``etl`` long, with ``None`` where no
        step is taken. In a blipped train an entry carries the second half of
        the step into the line and the first half of the step to the next line
        out of it; in a flyback train it is the whole step, played in the
        rewind block after that line, so the last entry is ``None``.
        ``gz_blips`` is 3D only.
    gy_pre, gz_pre : TrapEvent
        Phase encodes at their largest step, to be scaled per shot,
        right-aligned with ``gx_pre``; with ``navigator_lines`` they play in a
        block of their own after the navigator. ``gz_pre`` is 3D only.
    gy_rew, gz_rew : TrapEvent
        The negated encodes, left-aligned in the last block of the repetition.
    gx_spoil : GradEvent
        Read-axis trapezoid in the last block: returns the read axis from the
        end of the last line to k = 0 and adds the spoiling.
    adc : AdcEvent
        The acquisition window, shared by every line.
    shot_labels : tuple of LabelSetEvent
        ``SET`` events on the prewinder block, one per name in ``labels``, for
        the caller to set to the shot origin. Always a tuple, so it can be
        unpacked into a block.
    line_labels : tuple of tuple of LabelIncEvent
        Per line, the ``INC`` events that step from the previous line to this
        one; empty for the first line, which ``shot_labels`` places.
    order : numpy.ndarray
        Integer ``(etl, 2)`` ``(ky, kz)`` offsets from the shot origin.
    wait_te : DelayEvent
        Present only when ``te`` exceeds the minimum.
    wait_shift : DelayEvent
        Echo-shift delay before the first train line, one block raster long;
        present only with ``echo_shifts`` above 1. The caller lengthens it by
        ``s * echo_shift_step`` on shot ``s`` and shortens ``wait_tr`` by the
        same amount.
    wait_tr : DelayEvent
        Present when ``tr`` exceeds the minimum, or when ``echo_shifts`` is
        above 1, in which case it holds the longest shift.
    etl : int
        Lines per repetition.
    esp : float
        Echo spacing (s).
    echo_times : numpy.ndarray
        Echo time of each line (s) from the RF centre, without echo shift.
    echo_time : float
        Echo time (s) of line ``te_line``, the time ``te`` specifies.
    echo_shift_step : float
        Echo shift between successive shots (s), ``esp / echo_shifts`` on the
        block raster; zero with one shift.
    bandwidth_hz : float
        Achieved ADC sampling rate ``1 / dwell`` (Hz).
    n_samples : int
        ADC samples per line.
    delta_kx : float
        K-space sample spacing along the readout (1/m).
    blip_span : float
        Duration of one phase-encode step (s), taken from the read ramps.

    Raises
    ------
    ValueError
        If a count is out of range, ``order`` is the wrong shape or steps
        outside the matrix, ``labels`` does not name one counter per encoded
        axis, ``te_line`` lies outside the train, a navigator is asked of a
        flyback train, or the requested TE or TR is shorter than the train can
        achieve.
    """

    #: 2 or 3. The only thing that separates the two shipped EPI readouts.
    _ndim = 2

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: Any,
        matrix: Any,
        order: Any = None,
        etl: int | None = None,
        scheme: str = "linear",
        acceleration: int = 1,
        segments: int = 1,
        partition_acceleration: int = 1,
        caipi_shift: int = 1,
        extent: int | None = None,
        te: float | None = None,
        te_line: float = 0.0,
        tr: float | None = None,
        navigator_lines: int = 0,
        echo_shifts: int = 1,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 500e3,
        ramp_sampling: bool = True,
        flyback: bool = False,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        ndim = self._ndim
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")
        if spoiling_cycles < 0:
            raise ValueError("spoiling_cycles must be >= 0")
        navigator_lines, echo_shifts = int(navigator_lines), int(echo_shifts)
        if navigator_lines < 0 or echo_shifts < 1:
            raise ValueError("navigator_lines must be >= 0 and echo_shifts >= 1")
        if navigator_lines and flyback:
            raise ValueError(
                "a flyback train reads every line the same way and has no odd-even "
                "inconsistency for a navigator to measure"
            )

        fov = as_tuple(fov, ndim, "fov")
        n = as_tuple(matrix, ndim, "matrix", int)
        if min(n) < 2 or min(fov) <= 0:
            raise ValueError("every matrix size must be >= 2 and every fov positive")

        if order is None:
            if etl is None:
                etl = -(-int(n[1]) // (int(acceleration) * int(segments)))
            order = make_epi_shot_offsets(
                etl,
                scheme=scheme,
                acceleration=acceleration,
                segments=segments,
                partition_acceleration=partition_acceleration,
                caipi_shift=caipi_shift,
                extent=extent,
            )
        order = _checked_order(order, n, ndim)
        etl = len(order)
        if not 0.0 <= te_line <= etl - 1:
            raise ValueError(f"te_line must lie in [0, {etl - 1}], got {te_line}")

        labels = tuple(labels or ())
        if labels and len(labels) != ndim - 1:
            raise ValueError(
                f"labels names the counter for each encoded axis after the readout, so a "
                f"{ndim}D train takes {ndim - 1}, got {len(labels)}"
            )

        raster = system.block_duration_raster
        # Oversampling widens the sampled read field of view; resolution stays.
        delta_k = (1.0 / (oversampling * fov[0]), *(1.0 / value for value in fov[1:]))

        # One blip window serves every step, so it is built for the largest
        # step in the pattern and every other is that shape scaled. Half of it
        # runs into a line and half out, so it has to be an even raster count.
        steps = np.diff(order, axis=0)
        blip_span = 0.0
        for axis in range(1, ndim):
            widest = int(np.max(np.abs(steps[:, axis - 1]))) if len(steps) else 0
            if widest:
                trial = pp.make_trapezoid(
                    channel="yz"[axis - 1], area=widest * delta_k[axis], system=system
                )
                blip_span = max(blip_span, pp.calc_duration(trial))
        blip_span = pp.ceil_to_raster(blip_span, 2.0 * system.grad_raster_time)

        # Sampling. A flyback line reads on its own and hands the step to the
        # rewind, so nothing rides its ramps and the pad is only what the
        # receiver needs.
        n_full = round(oversampling * n[0])
        k_width = n_full * delta_k[0]
        lobe_blip = 0.0 if flyback else blip_span
        dwell, sample_span = pp.calc_adc_timing(
            n_full,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=_min_sample_span(
                system, k_width, lobe_blip, ramp_sampling
            ),
        )
        lobe, echo_offset = _read_lobe(
            system, k_width, sample_span, lobe_blip, ramp_sampling
        )
        adc = pp.make_adc(
            num_samples=n_full,
            dwell=dwell,
            delay=echo_offset - 0.5 * sample_span,
            system=system,
        )

        # The prewinder is half a lobe against the first line, so the window is
        # centred on k = 0 whatever shape the ramps took.
        gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * _area(lobe), system=system)
        if flyback:
            # Every line reads the same way and a rewind between them puts the
            # read axis back where it started, carrying the step as it goes.
            gap_span = pp.ceil_to_raster(
                max(
                    pp.calc_duration(
                        pp.make_trapezoid(channel="x", area=-_area(lobe), system=system)
                    ),
                    blip_span,
                ),
                system.grad_raster_time,
            )
            gx_flyback = _trapezoid("x", -_area(lobe), system, gap_span)
            gx = [lobe] * etl
            esp = pp.calc_duration(lobe) + gap_span
            gy_blips = _gap_blips(system, "y", steps[:, 0], delta_k[1], gap_span, etl)
            gz_blips = (
                _gap_blips(system, "z", steps[:, 1], delta_k[2], gap_span, etl)
                if ndim == 3
                else [None] * etl
            )
        else:
            # The train continues the navigator's alternation.
            gx = [
                pp.scale_grad(lobe, 1.0 if (navigator_lines + line) % 2 == 0 else -1.0)
                for line in range(etl)
            ]
            esp = pp.calc_duration(lobe)
            gy_blips = _blip_events(
                system, "y", steps[:, 0], delta_k[1], blip_span, lobe, etl
            )
            gz_blips = (
                _blip_events(system, "z", steps[:, 1], delta_k[2], blip_span, lobe, etl)
                if ndim == 3
                else [None] * etl
            )

        gy_pre = pp.make_phase_encoding("y", fov[1] / n[1], system=system)
        gz_pre = (
            pp.make_phase_encoding("z", fov[2] / n[2], system=system)
            if ndim == 3
            else None
        )
        pre_span = pp.ceil_to_raster(
            max(
                pp.calc_duration(event) for event in (gx_pre, gy_pre, *present(gz_pre))
            ),
            raster,
        )
        gx_pre = _trapezoid("x", -0.5 * _area(lobe), system, pre_span)
        gy_pre = pp.make_phase_encoding(
            "y", fov[1] / n[1], system=system, duration=pre_span
        )
        if ndim == 3:
            gz_pre = pp.make_phase_encoding(
                "z", fov[2] / n[2], system=system, duration=pre_span
            )
        gy_rew = pp.scale_grad(gy_pre, -1.0)
        gz_rew = pp.scale_grad(gz_pre, -1.0) if ndim == 3 else None

        # The train ends half a lobe from the centre, on whichever side the
        # last line's polarity left it. The closing lobe travels back through
        # k = 0 and keeps going by whatever spoiling was asked for, so it never
        # degenerates to nothing however the two are balanced.
        if voxel_size_m is None:
            voxel_size_m = fov[0] / n[0]
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        gx_spoil = pp.make_trapezoid(
            channel="x",
            area=-float(np.sign(_area(gx[-1])))
            * (0.5 * abs(_area(lobe)) + spoiling_cycles / voxel_size_m),
            system=system,
        )

        gz_reph = left_align_rephaser(
            gz_reph, ("x", "y", "z"[: ndim - 2]), type(self).__name__
        )
        shot_labels = tuple(
            pp.make_label(type="SET", label=name, value=0) for name in labels
        )
        line_labels = tuple(
            tuple(
                pp.make_label(type="INC", label=name, value=int(steps[line - 1, axis]))
                for axis, name in enumerate(labels)
            )
            if line
            else ()
            for line in range(etl)
        )

        # The navigator reads the centre line at alternating polarities, so
        # the phase-encode prewinder moves to a block of its own after it.
        gx_navigator = [
            pp.scale_grad(lobe, 1.0 if line % 2 == 0 else -1.0)
            for line in range(navigator_lines)
        ]
        navigator_span = navigator_lines * _span(system, lobe)
        encode_span = (
            _span(system, gy_pre, *present(gz_pre)) if navigator_lines else 0.0
        )
        shift_span = raster if echo_shifts > 1 else 0.0
        echo_shift_step = (
            pp.round_to_raster(esp / echo_shifts, raster) if echo_shifts > 1 else 0.0
        )

        # Timing. The rephaser goes in whichever block follows the pulse, so it
        # runs straight off the selection lobe.
        exc_span = _span(system, rf, gz)
        rf_center = float(rf.delay) + float(rf.center)
        lead = navigator_span + encode_span + shift_span + te_line * esp
        wait, pre_span, echo_time = solve_rephasing(
            te,
            (exc_span - rf_center) + echo_offset + lead,
            pre_span,
            _span(system, gz_reph),
            system,
        )

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, *present(gz))
        if wait:
            wait_te = pp.make_delay(wait)
            self.seq.add_block(wait_te, *present(gz_reph))
        # A rephaser sharing the block can make it longer than the encodes were
        # stretched to; they are right-aligned so they still end at the first
        # line, while the rephaser stays left-aligned against the pulse.
        if navigator_lines:
            (gx_pre,) = pp.align(right=[gx_pre, pp.make_delay(pre_span)])[:-1]
        elif ndim == 3:
            gx_pre, gy_pre, gz_pre = pp.align(
                right=[gx_pre, gy_pre, gz_pre, pp.make_delay(pre_span)]
            )[:-1]
        else:
            gx_pre, gy_pre = pp.align(right=[gx_pre, gy_pre, pp.make_delay(pre_span)])[
                :-1
            ]
        encode = [gy_pre, *present(gz_pre), *shot_labels, *present(trigger)]
        prewinder = [gx_pre] if navigator_lines else [gx_pre, *encode]
        if not wait:
            prewinder += present(gz_reph)
        self.seq.add_block(*prewinder)
        for line in range(navigator_lines):
            self.seq.add_block(gx_navigator[line], adc)
        if navigator_lines:
            self.seq.add_block(*encode)
        if echo_shifts > 1:
            wait_shift = pp.make_delay(shift_span)
            self.seq.add_block(wait_shift)
        for line in range(etl):
            if flyback:
                self.seq.add_block(gx[line], adc, *line_labels[line])
                if line < etl - 1:
                    self.seq.add_block(
                        gx_flyback, *present(gy_blips[line]), *present(gz_blips[line])
                    )
            else:
                self.seq.add_block(
                    gx[line],
                    adc,
                    *present(gy_blips[line]),
                    *present(gz_blips[line]),
                    *line_labels[line],
                )
        self.seq.add_block(gx_spoil, gy_rew, *present(gz_rew))

        # The longest shift is held at the end of the repetition, so every shot
        # lasts the same once the loop moves its own shift forward.
        longest_shift = (echo_shifts - 1) * echo_shift_step
        tr_min = self.seq.duration()[0] + longest_shift
        tr_delay = solve_delay(tr, tr_min, "TR", system) + longest_shift
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            self.seq.add_block(wait_tr)

        self.register(
            gx=gx,
            gy_blips=gy_blips,
            shot_labels=shot_labels,
            line_labels=line_labels,
            order=order,
        )
        if ndim == 3:
            self.register(gz_blips=gz_blips)
        if navigator_lines:
            self.register(gx_navigator=gx_navigator)
        self.center = exc_span + wait + pre_span + echo_offset + lead
        self.echo_time = echo_time
        self.echo_times = echo_time + esp * (np.arange(etl) - te_line)
        self.echo_shift_step = echo_shift_step
        self.etl = etl
        self.esp = esp
        self.blip_span = blip_span
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_full
        self.delta_kx = delta_k[0]


def _checked_order(order: Any, n: tuple[int, ...], ndim: int) -> np.ndarray:
    """Ordering as ``(etl, 2)`` integer offsets, read-only."""
    values = np.asarray(order)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 1:
        raise ValueError("order must have shape (etl,) or (etl, 2)")
    if values.shape[1] == 1:
        values = np.column_stack((values[:, 0], np.zeros(len(values))))
    if values.shape[1] != 2:
        raise ValueError("order must have shape (etl,) or (etl, 2)")
    if not np.all(values == np.rint(values)):
        raise ValueError("order must contain integers: they index phase-encode lines")
    values = np.rint(values).astype(np.intp)
    if ndim == 2 and np.any(values[:, 1] != 0):
        raise ValueError(
            "order steps the partition axis, which a 2D train does not encode; use "
            "EpiReadout3D, whose second column is the partition offset"
        )
    for axis in range(1, ndim):
        span = int(np.ptp(values[:, axis - 1])) if len(values) > 1 else 0
        if span >= n[axis]:
            raise ValueError(
                f"order spans {span + 1} steps on the {'yz'[axis - 1]} axis, which does not "
                f"fit the {n[axis]} it is encoded over"
            )
    values.setflags(write=False)
    return values


def _read_lobe(
    system: pp.Opts,
    k_width: float,
    sample_span: float,
    blip_span: float,
    ramp_sampling: bool,
):
    """Return one line's read gradient and the time (s) from its start to the centre of the window.

    The lobe is padded symmetrically either side of the acquisition window, by
    enough to hold half a phase-encode step and never less than the receiver's
    own dead time. A step therefore runs entirely outside the samples it
    separates, and the window stays centred -- which is what lets the prewinder
    be half a lobe whatever shape the ramps took.
    """
    pad = _lobe_pad(system, blip_span)
    if ramp_sampling:
        # Over-drive the area so that, once the parts outside the window are
        # discounted, what remains inside it is still the full sweep -- then
        # scale, which lowers the amplitude and so cannot break a limit. The
        # first guess bounds the discount by the ramps; a pad wider than a ramp
        # eats into the plateau too, so the guess is checked and raised.
        total = sample_span + 2.0 * pad
        requested = _ramp_sampled_area(system, k_width, pad)
        for _ in range(8):
            wide = _trapezoid("x", requested, system, total)
            inside = _area_between(wide, pad, pad + sample_span)
            if inside >= k_width:
                break
            requested *= 1.05 * k_width / inside
        else:
            raise ValueError(
                "could not fit the readout sweep into its sampling window; lower "
                "readout_bandwidth_hz or oversampling"
            )
        return pp.scale_grad(wide, k_width / inside), 0.5 * total

    plateau = pp.make_trapezoid(
        channel="x", flat_area=k_width, flat_time=sample_span, system=system
    )
    rise = pp.ceil_to_raster(
        max(float(plateau.rise_time) + 0.5 * blip_span, system.adc_dead_time),
        system.grad_raster_time,
    )
    amplitude = float(plateau.amplitude)
    lobe = pp.make_extended_trapezoid(
        channel="x",
        times=np.array([0.0, rise, rise + sample_span, 2.0 * rise + sample_span]),
        amplitudes=np.array([0.0, amplitude, amplitude, 0.0]),
        system=system,
    )
    return lobe, rise + 0.5 * sample_span


def _lobe_pad(system: pp.Opts, blip_span: float) -> float:
    """Time (s) a read lobe runs on either side of its acquisition window."""
    return pp.ceil_to_raster(
        max(0.5 * blip_span, system.adc_dead_time), system.grad_raster_time
    )


def _ramp_sampled_area(system: pp.Opts, k_width: float, pad: float) -> float:
    """First area (1/m) asked of a ramp-sampled lobe sweeping ``k_width``.

    Ramps at ``max_slew`` that outlast the pads lose ``max_slew * pad**2 / 2``
    of area to each pad.
    """
    return k_width + pad**2 * system.max_slew


def _min_sample_span(
    system: pp.Opts, k_width: float, blip_span: float, ramp_sampling: bool
) -> float:
    """Shortest acquisition window (s) a read lobe sweeping ``k_width`` fits.

    A plateau reaches at most ``_READOUT_GRAD_MARGIN`` of ``max_grad``. A
    ramp-sampled lobe also has to hold its first area within the window and
    its pads, which the slew rate decides when the window is short. That
    bound is a whole number of rasters and is lowered by 1 ns, so that a
    window equal to it is not lost to rounding.
    """
    span = k_width / (_READOUT_GRAD_MARGIN * system.max_grad)
    if ramp_sampling:
        pad = _lobe_pad(system, blip_span)
        shortest = pp.make_trapezoid(
            channel="x", area=_ramp_sampled_area(system, k_width, pad), system=system
        )
        span = max(span, pp.calc_duration(shortest) - 2.0 * pad - 1e-9)
    return span


def _area_between(trapezoid: Any, start: float, stop: float) -> float:
    """Integrate a trapezoid over a time interval (s), returning area in 1/m."""
    rise, flat = float(trapezoid.rise_time), float(trapezoid.flat_time)
    fall, amplitude = float(trapezoid.fall_time), float(trapezoid.amplitude)
    times = np.array([0.0, rise, rise + flat, rise + flat + fall])
    amplitudes = np.array([0.0, amplitude, amplitude, 0.0])
    edges = np.unique(
        np.concatenate((times[(times > start) & (times < stop)], [start, stop]))
    )
    return float(np.trapezoid(np.interp(edges, times, amplitudes), edges))


def _blip_events(
    system: pp.Opts,
    channel: str,
    steps: np.ndarray,
    delta_k: float,
    blip_span: float,
    anchor: Any,
    etl: int,
) -> list:
    """Split each phase-encode step across adjacent readout ramps.

    The outgoing half plays on the earlier line; the incoming half plays on
    the next. All steps use blip_span, preserving constant echo spacing.
    Repeated flanking-step pairs share one event.
    """
    widest = int(np.max(np.abs(steps))) if len(steps) else 0
    if not widest or blip_span == 0.0:
        return [None] * etl
    template = _trapezoid(channel, widest * delta_k, system, blip_span)

    halves: dict[int, Any] = {}

    def split(step: int):
        if step not in halves:
            halves[step] = pp.split_gradient_at(
                grad=pp.scale_grad(template, step / widest),
                time_point=0.5 * blip_span,
                system=system,
            )
        return halves[step]

    flanked: dict[tuple[int, int], Any] = {}
    events = []
    for line in range(etl):
        leaves = int(steps[line]) if line < etl - 1 else 0
        enters = int(steps[line - 1]) if line else 0
        if (enters, leaves) not in flanked:
            flanked[(enters, leaves)] = _flank(
                system,
                split(enters)[1] if enters else None,
                split(leaves)[0] if leaves else None,
                anchor,
            )
        events.append(flanked[(enters, leaves)])
    return events


def _gap_blips(
    system: pp.Opts,
    channel: str,
    steps: np.ndarray,
    delta_k: float,
    gap_span: float,
    etl: int,
) -> list:
    """Return flyback-gap blips; entry j follows line j and the last is None."""
    widest = int(np.max(np.abs(steps))) if len(steps) else 0
    if not widest:
        return [None] * etl
    template = _trapezoid(channel, widest * delta_k, system, gap_span)
    scaled: dict[int, Any] = {}
    events = []
    for step in steps:
        step = int(step)
        if step and step not in scaled:
            scaled[step] = pp.scale_grad(template, step / widest)
        events.append(scaled.get(step))
    return [*events, None]


def _flank(system: pp.Opts, entering: Any, leaving: Any, anchor: Any):
    """Two half-steps flanking one line, as the single event that line plays."""
    if entering is None and leaving is None:
        return None
    if entering is None:
        return pp.align(right=[leaving], left=[anchor])[0]
    if leaving is None:
        return pp.align(left=[entering, anchor])[0]
    right, left, _ = pp.align(right=[leaving], left=[entering, anchor])
    return pp.add_gradients([left, right], system=system)


def _span(system: pp.Opts, *events: Any) -> float:
    events = [event for event in events if event is not None]
    if not events:
        return 0.0
    return pp.ceil_to_raster(pp.calc_duration(*events), system.block_duration_raster)


def _area(event: Any) -> float:
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


class EpiReadout2D(_EpiReadout):
    """A single- or multi-shot EPI train, frequency-encoded along x.

    ``fov`` and ``matrix`` take two values, readout first.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> excitation = design.SpatialSelectiveExcitation(system, 60.0, 3e-3)
    >>> epi = design.EpiReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=64, labels=("LIN",),
    ... )
    >>> epi.etl, int(epi.adc.num_samples)
    (64, 64)

    >>> half = design.EpiReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=64, segments=2,
    ... )
    >>> half.etl, int(half.order[1, 0])
    (32, 2)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
       excitation = sequences.SpatialSelectiveExcitation(system, 60.0, 3e-3)
       readout = sequences.EpiReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           fov=0.22, matrix=32, labels=("LIN",),
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _ndim = 2


class EpiReadout3D(_EpiReadout):
    """A slab-selective EPI train, phase-encoded along y and partition-encoded along z.

    ``fov`` and ``matrix`` take three values, readout first. The partition axis
    is blipped line by line, so a whole CAIPI pattern is played inside one
    train.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> epi = design.EpiReadout3D(
    ...     system, slab.rf, slab.gz, fov=(0.22, 0.22, 0.12), matrix=(64, 64, 32),
    ...     scheme="caipi", acceleration=2, segments=2, partition_acceleration=3,
    ...     labels=("LIN", "PAR"),
    ... )
    >>> epi.etl
    16

    >>> sorted(set(int(step) for step in epi.order[1:, 1] - epi.order[:-1, 1]))
    [-1, 2]

    One segment of a skipped-CAIPI train, and the lattice of
    ``(line, partition)`` views it reads:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       slab = sequences.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)
       readout = sequences.EpiReadout3D(
           system, slab.rf, slab.gz,
           fov=(0.22, 0.22, 0.12), matrix=(32, 32, 16),
           scheme="caipi", acceleration=2, segments=2, partition_acceleration=3,
           labels=("LIN", "PAR"),
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="yz", show_trajectory=False)
       plt.show()
    """

    _ndim = 3
