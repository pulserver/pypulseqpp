"""Balanced SSFP readouts: a fully rewound repetition with the echo at TR/2."""

from __future__ import annotations

__all__ = ["BssfpReadout2D", "BssfpReadout3D"]

import math
from typing import Any

import numpy as np

import pypulseqpp as pp

from .._module import SequenceModule
from ._common import as_tuple, present, solve_delay

#: Fraction of ``max_grad`` the readout plateau may reach.
_READOUT_GRAD_MARGIN = 0.8
#: Fraction of the selection amplitude a bridged slice axis is at where one
#: block of a slice meets the next, so that no join inside the slice has every
#: gradient at rest.
_BRIDGE_FRACTION = 0.0625


class _BssfpReadout(SequenceModule):
    """Balanced SSFP repetition with TE fixed at TR/2.

    Every axis is balanced between consecutive RF centres. The slice
    rephasers belong to that balance and are built here, so ``rf`` and ``gz``
    come from an excitation built with ``rephase=False``.

    The module's blocks are excitation, acquisition, rewind. The caller plays
    each shot as the previous shot's rewind, excitation and acquisition, and
    writes the transients: the half-flip pulse (``rf`` at half amplitude,
    opposite in phase to the first excitation) followed by ``wait_prep`` with
    ``gz_prep`` in place of the first rewind; and a last rewind block to ramp
    down.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        The excitation.
    gz : GradEvent, default=None
        Its selection gradient without a rephaser, played in the excitation
        block; ``None`` for a hard pulse.
    fov : float or sequence of float
        Field of view per encoded axis, readout first (m).
    matrix : int or sequence of int
        Matrix size per encoded axis, readout first.
    tr : float, default=None
        Repetition time (s). ``None`` is as short as possible; a longer TR is
        added evenly either side of the echo, so TE stays at TR/2.
    half_flip_prep : bool, default=True
        Build ``wait_prep`` for the half-flip preparation. Requires the
        acquisition block after the echo to be at least as long as the
        excitation block after the pulse centre. False when the caller ramps
        the signal in with dummy repetitions.
    oversampling : float, default=1.0
        Readout oversampling: the sample spacing in k-space is
        ``1 / (oversampling * fov)``.
    readout_bandwidth_hz : float, default=250000.0
        Requested ADC sampling rate (Hz); ``bandwidth_hz`` is the rate achieved
        on the ADC and gradient rasters.
    labels : sequence of str, default=None
        Labels set to 0 on the acquisition block, to be changed per shot by the
        caller.
    trigger : TriggerEvent or DigitalOutputEvent, default=None
        Event armed on the rewind block.
    bridge : bool, default=False
        Join the blocks of a slice on the slice axis, 2D with a trapezoidal
        ``gz`` only: the ramps of the selection plateau run through zero to
        a fraction of its amplitude where the excitation block meets the
        rewind before it and the acquisition after it, and the rephasers end
        and start there. No join between two repetitions then has every
        gradient at rest, so a scanner that prepares a segment between blocks
        at rest plays the slice as one. The timing is that of a slice axis
        that rests; where bridging would lengthen the repetition, the slice
        is left unbridged and ``bridged`` is False. The caller opens a bridged
        slice with ``gz_open`` in place of ``gz`` and closes it with
        ``gz_close`` in place of ``gz_rew``.


    Attributes
    ----------
    rf : RfEvent
        The excitation, replayed every repetition with the phase and amplitude
        the loop sets.
    gz : GradEvent
        Its selection gradient, if one was given.
    gx, gx_rew : GradEvent
        ``gx`` ramps up from zero, prephases and holds the readout plateau to
        the end of the acquisition block; ``gx_rew`` ramps down from the
        plateau and rewinds the read axis.
    gy_pre, gy_rew : TrapEvent
        In-plane encode at its largest step, and the same negated, to be scaled
        per shot.
    gz_pre, gz_rew : TrapEvent
        Slice rephasers: ``gz_pre`` in the acquisition block after the pulse,
        ``gz_rew`` in the rewind block before the next pulse. Absent for a
        hard pulse.
    bridged : bool
        Whether the blocks of a slice join on the slice axis.
    gz_open, gz_close : GradEvent
        Bridged only: the selection gradient of a slice's first excitation
        block, which opens from rest, and the rephaser of its last rewind,
        which ends at rest.
    gz_partition, gz_partition_rew : TrapEvent
        Partition encode at its largest step, and the same negated, sharing
        the window of ``gz_pre`` and ``gz_rew`` respectively. 3D only.
    adc : AdcEvent
        The acquisition window, spanning the readout plateau.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    prep_labels, train_labels, end_labels : list of LabelSetEvent
        ``ONCE`` set to 1, 0 and 2, for the half-flip block, the block after
        it and the final rewind.
    wait_prep : DelayEvent
        Delay after the half-flip block, which places the half-flip pulse
        half a repetition before the first excitation. Absent with
        ``half_flip_prep=False``.
    gz_prep : TrapEvent
        Slice rephaser played with ``wait_prep``: the two rephasers of a
        repetition in one lobe, so that the slice axis is balanced from the
        half-flip pulse to the first excitation. None without a selection
        gradient; absent with ``half_flip_prep=False``.
    tr, te : float
        Repetition and echo time (s); ``te`` is always half of ``tr``.
    bandwidth_hz : float
        Achieved ADC sampling rate (Hz).
    n_samples : int
        ADC samples per repetition.
    delta_kx : float
        K-space sample spacing along the readout (1/m).
    readout_duration : float
        Duration of the sampling window (s).

    Raises
    ------
    ValueError
        If a size or rate is out of range, the ``half_flip_prep`` condition
        fails, the requested TR is shorter than the module can achieve, or
        ``bridge`` is asked of a 3D readout or of a ``gz`` that is not a
        trapezoid.
    """

    #: 2 or 3. The only thing that separates the two shipped bSSFP readouts.
    _ndim = 2

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        *,
        fov: Any,
        matrix: Any,
        tr: float | None = None,
        half_flip_prep: bool = True,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
        bridge: bool = False,
    ) -> None:
        ndim = self._ndim
        if bridge and (ndim != 2 or gz is None or gz.type != "trap"):
            raise ValueError(
                "bridge joins the blocks of a slice on its selection gradient: it "
                "takes a 2D readout and a trapezoidal gz"
            )
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        fov = as_tuple(fov, ndim, "fov")
        n = as_tuple(matrix, ndim, "matrix", int)
        if min(n) < 2 or min(fov) <= 0:
            raise ValueError("every matrix size must be >= 2 and every fov positive")

        raster = system.block_duration_raster

        # A balanced readout is symmetric by construction -- the echo sits in
        # the middle of the plateau -- so there is no partial echo to ask for.
        delta_kx = 1.0 / (oversampling * fov[0])
        n_samples = round(oversampling * n[0])
        readout_area = n_samples * delta_kx
        dwell, readout_duration = pp.calc_adc_timing(
            n_samples,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=readout_area
            / (_READOUT_GRAD_MARGIN * system.max_grad),
        )
        shape = pp.make_trapezoid(
            channel="x",
            flat_area=readout_area,
            flat_time=readout_duration,
            system=system,
        )
        amplitude = float(shape.amplitude)
        # The plateau has to reach the end of its block, or the lobe that
        # follows has no plateau to leave from -- and the block runs on past
        # the last sample by the receiver's dead time.
        flat_span = pp.ceil_to_raster(readout_duration + system.adc_dead_time, raster)

        # What the read axis does outside the acquisition: down off the
        # plateau and back onto it, carrying the echo to k = 0 on the way.
        entry_area = -0.5 * amplitude * readout_duration
        gx_rew = pp.make_extended_trapezoid_area(
            area=-amplitude * (flat_span - 0.5 * readout_duration),
            channel="x",
            grad_start=amplitude,
            grad_end=0.0,
            system=system,
        )[0]
        entry_times, entry_amplitudes = _vertices(system, entry_area, 0.0, amplitude)

        rf_center = float(rf.delay) + float(rf.center)
        excite_span = _span(raster, rf, gz)
        prep_span = (
            0.5 * (flat_span + (flat_span - readout_duration)) + rf_center - excite_span
        )
        if half_flip_prep and prep_span < -1e-12:
            raise ValueError(
                f"the {flat_span * 1e3:.3f} ms acquisition block is too short beside the "
                f"{(excite_span - rf_center) * 1e3:.3f} ms that follows the pulse centre in "
                f"the excitation block, so no half-flip pulse can sit half a repetition before "
                f"the first excitation; lower readout_bandwidth_hz, shorten the pulse, or pass "
                f"half_flip_prep=False and ramp the train in with dummy repetitions"
            )

        # The rephasers are a pair that cancels the selection lobe outright.
        # Split so the one after the pulse unwinds what followed the pulse
        # centre, which is the half that has to be unwound by the echo.
        selection_area = _area(gz)
        post_area = -0.5 * selection_area
        pre_area = -(selection_area + post_area)

        # The z lobes of one window share their ramp, which is what makes a
        # rephaser and a partition encode add up to a single trapezoid instead
        # of an arbitrary waveform. The ramp is the one on which the largest
        # sum is fastest at the slew rate: a small sum is a short triangle, not
        # a lobe on the full-scale ramp max_grad / max_slew.
        step_z = 0.5 * n[2] / fov[2] if ndim == 3 else 0.0
        z_area = max(abs(pre_area), abs(post_area)) + step_z
        ramp = _z_ramp(system, z_area)
        z_floor = _z_floor(system, z_area, ramp, raster)
        # The half-flip gap holds both rephasers in one lobe, on its own ramp.
        prep_area = abs(pre_area + post_area)
        prep_ramp = _z_ramp(system, prep_area)
        prep_floor = (
            _z_floor(system, prep_area, prep_ramp, raster) - prep_span
            if half_flip_prep
            else 0.0
        )
        gz_open = gz_close = None
        if bridge:
            # The slice axis never rests where two blocks of a slice meet: the
            # ramps of the selection plateau run through zero to a fraction of
            # its amplitude on the other side, where the slew rate allows, so
            # the rephasers either side have less to unwind and stop short of
            # rest. Where it does not, they stop short of rest on the
            # plateau's side. The first excitation of a slice opens from rest.
            plateau = float(gz.amplitude)
            level = _BRIDGE_FRACTION * abs(plateau)
            sign = math.copysign(1.0, plateau)
            top = float(gz.delay) + float(gz.rise_time)
            end = top + float(gz.flat_time)
            step = system.grad_raster_time
            # The ramp up may start in the pulse's dead time, as early as it
            # must to reach the plateau from the far side within the slew rate.
            start = step * math.floor(
                (top - (abs(plateau) + level) / system.max_slew) / step + 1e-9
            )
            if start >= -1e-12:
                start = min(max(start, 0.0), float(gz.delay))
                lead = -sign * level
            else:
                start, lead = float(gz.delay), sign * level
            tail = (
                -sign * level
                if (abs(plateau) + level) / (excite_span - end) <= system.max_slew
                else sign * level
            )
            times = [0.0, start, top, end, excite_span]
            levels = [lead, lead, plateau, plateau, tail]
            opening = [0.0, float(gz.delay), top, end, excite_span]
            opened = [0.0, 0.0, plateau, plateau, tail]
            if start < 0.5 * step:
                times, levels = times[1:], levels[1:]
            if gz.delay < 0.5 * step:
                opening, opened = opening[1:], opened[1:]
            before, after = _areas_about(times, levels, rf_center)
            # Each rephaser is the fastest lobe between rest and the level the
            # excitation block opens or closes on; the half-flip gap holds
            # the two in one lobe.
            pre_times, pre_levels = _vertices(system, -before, 0.0, lead, "z")
            post_times, post_levels = _vertices(system, -after, tail, 0.0, "z")
            held = float(_vertices(system, -(before + after), tail, lead, "z")[0][-1])
            # A bridged slice shares its timing with one whose slice axis
            # rests, or is not bridged: the repetition is never longer for it.
            bridge = (
                max(float(pre_times[-1]), float(post_times[-1])) <= z_floor + 1e-12
                and held - prep_span <= prep_floor + 1e-12
            )
        if bridge:
            gz_open = pp.make_extended_trapezoid(
                channel="z", times=opening, amplitudes=opened, system=system
            )
            gz = pp.make_extended_trapezoid(
                channel="z", times=times, amplitudes=levels, system=system
            )
            post_area, pre_area = -after, -before
        y_floor = _span(
            raster, pp.make_phase_encoding("y", fov[1] / n[1], system=system)
        )
        encode_floor = max(
            pp.ceil_to_raster(float(entry_times[-1]), raster), y_floor, z_floor
        )
        rewind_floor = max(_span(raster, gx_rew, trigger), y_floor, z_floor, prep_floor)

        # TE = TR/2 puts the echo as far after the pulse centre as the next
        # pulse centre is after the echo, which is one equation between the two
        # windows either side of the excitation block.
        offset = (flat_span - readout_duration) + 2.0 * rf_center - excite_span
        rewind_span = pp.ceil_to_raster(
            max(rewind_floor, encode_floor - offset), raster
        )
        tr_min = 2.0 * rewind_span + offset + flat_span + excite_span
        rewind_span += pp.round_to_raster(
            0.5 * solve_delay(tr, tr_min, "TR", system), raster
        )
        encode_span = rewind_span + offset

        gy_pre = pp.make_phase_encoding(
            "y", fov[1] / n[1], system=system, duration=encode_span
        )
        gy_rew = pp.scale_grad(
            pp.make_phase_encoding(
                "y", fov[1] / n[1], system=system, duration=rewind_span
            ),
            -1.0,
        )
        if bridge:
            gz_pre = pp.make_extended_trapezoid(
                channel="z", times=post_times, amplitudes=post_levels, system=system
            )
            gz_rew = pp.make_extended_trapezoid(
                channel="z", times=pre_times, amplitudes=pre_levels, system=system
            )
            gz_rew.delay = rewind_span - float(pre_times[-1])
            # The rewind that ends a slice brings the slice axis to rest.
            gz_close = _lobe(system, pre_area, rewind_span, ramp)
        else:
            gz_pre = _lobe(system, post_area, encode_span, ramp)
            gz_rew = _lobe(system, pre_area, rewind_span, ramp)
        if ndim == 3:
            gz_partition = _lobe(system, step_z, encode_span, ramp)
            gz_partition_rew = _lobe(system, -step_z, rewind_span, ramp)
            self.register(gz_partition=gz_partition, gz_partition_rew=gz_partition_rew)

        gx = pp.make_extended_trapezoid(
            channel="x",
            times=np.append(entry_times, entry_times[-1] + flat_span),
            amplitudes=np.append(entry_amplitudes, amplitude),
            system=system,
        )
        gx.delay = encode_span - float(entry_times[-1])
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=encode_span, system=system
        )

        if half_flip_prep:
            wait_prep = pp.make_delay(prep_span + rewind_span)
            if bridge:
                gz_prep = pp.make_extended_trapezoid(
                    channel="z",
                    **_held_lobe(
                        system,
                        pre_area + post_area,
                        tail,
                        lead,
                        prep_span + rewind_span,
                    ),
                    system=system,
                )
            else:
                gz_prep = _lobe(
                    system, pre_area + post_area, prep_span + rewind_span, prep_ramp
                )

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]
        prep_labels = [pp.make_label(type="SET", label="ONCE", value=1)]
        train_labels = [pp.make_label(type="SET", label="ONCE", value=0)]
        end_labels = [pp.make_label(type="SET", label="ONCE", value=2)]

        # One repetition of the centre partition, laid out from the pulse so
        # that it starts and ends with every axis at rest.
        self.seq = pp.Sequence(system)
        if bridge:
            # Laid out as a slice of one repetition: opened from rest and
            # closed to it.
            self.seq.add_block(rf, gz_open)
            self.seq.add_block(gx, adc, gy_pre, gz_pre, *adc_labels)
            self.seq.add_block(gx_rew, gy_rew, *present(gz_close), *present(trigger))
            self.register(gz=gz, gz_rew=gz_rew, gz_close=gz_close)
        else:
            self.seq.add_block(rf, *present(gz))
            self.seq.add_block(gx, adc, gy_pre, *present(gz_pre), *adc_labels)
            self.seq.add_block(gx_rew, gy_rew, *present(gz_rew), *present(trigger))
        self.register(
            prep_labels=prep_labels,
            train_labels=train_labels,
            end_labels=end_labels,
            **({"wait_prep": wait_prep, "gz_prep": gz_prep} if half_flip_prep else {}),
        )

        self.bridged = bridge
        self.tr = rewind_span + excite_span + encode_span + flat_span
        self.te = 0.5 * self.tr
        self.center = excite_span + encode_span + 0.5 * readout_duration
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_samples
        # Balanced and never a partial echo, so k crosses zero at the midpoint.
        self.center_sample = n_samples // 2
        self.delta_kx = delta_kx
        self.readout_duration = readout_duration


def _area(event: Any) -> float:
    if event is None:
        return 0.0
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


def _span(raster: float, *events: Any) -> float:
    events = [event for event in events if event is not None]
    return pp.ceil_to_raster(pp.calc_duration(*events), raster) if events else 0.0


def _vertices(
    system: pp.Opts, area: float, grad_start: float, grad_end: float, channel: str = "x"
):
    _, times, amplitudes = pp.make_extended_trapezoid_area(
        area=area,
        channel=channel,
        grad_start=grad_start,
        grad_end=grad_end,
        system=system,
    )
    return np.asarray(times, dtype=float), np.asarray(amplitudes, dtype=float)


def _held_lobe(
    system: pp.Opts, area: float, start: float, end: float, window: float
) -> dict:
    """Return the vertices of a z lobe of ``area`` over ``window``, from ``start`` to ``end``.

    The fastest lobe, held at ``end`` to the end of the window; the lobe
    carries ``area`` less what the hold adds, which depends on its own
    duration, so the two are solved together.
    """
    step = system.grad_raster_time
    times, levels = _vertices(system, area, start, end, "z")
    for _ in range(32):
        duration = float(times[-1])
        times, levels = _vertices(
            system, area - end * (window - duration), start, end, "z"
        )
        if abs(float(times[-1]) - duration) < 0.5 * step:
            break
    else:
        raise ValueError(
            f"no lobe of {area:.1f} 1/m held to {window * 1e6:.0f} us was found"
        )
    hold = window - float(times[-1])
    if hold < -0.5 * step:
        raise ValueError(f"a lobe of {area:.1f} 1/m does not fit {window * 1e6:.0f} us")
    if hold > 0.5 * step:
        times, levels = np.append(times, window), np.append(levels, end)
    return {"times": times, "amplitudes": levels}


def _areas_about(times: Any, levels: Any, centre: float) -> tuple[float, float]:
    """Return the areas of a piecewise-linear waveform before and after ``centre``."""
    times = np.asarray(times, dtype=float)
    levels = np.asarray(levels, dtype=float)
    at = float(np.interp(centre, times, levels))
    early = times < centre
    before = np.trapezoid(np.append(levels[early], at), np.append(times[early], centre))
    after = np.trapezoid(
        np.append(at, levels[~early]), np.append(centre, times[~early])
    )
    return float(before), float(after)


def _lobe(system: pp.Opts, area: float, duration: float, ramp: float):
    """Return a z trapezoid of ``area`` over ``duration`` with ramps of ``ramp``, or None for zero area.

    Lobes over one window share their vertex times whatever their areas, so a
    rephaser and a partition encode can be added into one trapezoid.
    """
    if abs(area) < 1e-12:
        return None
    return pp.make_trapezoid(
        channel="z", area=area, duration=duration, rise_time=ramp, system=system
    )


def _z_ramp(system: pp.Opts, area: float) -> float:
    """Return the ramp of the fastest z trapezoid of ``area``: a triangle below full scale, else the full-scale ramp."""
    peak = min(system.max_grad, math.sqrt(abs(area) * system.max_slew))
    return max(
        pp.ceil_to_raster(peak / system.max_slew, system.grad_raster_time),
        system.grad_raster_time,
    )


def _z_floor(system: pp.Opts, area: float, ramp: float, raster: float) -> float:
    """Return the shortest window holding a z trapezoid of ``area`` on ramps of ``ramp``.

    ``area`` is a rephaser plus the largest partition step: the caller adds
    the two into one trapezoid, whose amplitude must stay within ``max_grad``
    and reach it no faster than ``max_slew`` allows over ``ramp``.
    """
    peak = min(system.max_grad, system.max_slew * ramp)
    return pp.ceil_to_raster(max(2.0 * ramp, ramp + abs(area) / peak), raster)


class BssfpReadout2D(_BssfpReadout):
    """Slice-selective balanced SSFP, phase-encoded along y.

    ``fov`` and ``matrix`` take two values, readout first.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(
    ...     system, 40.0, 5e-3, 1e-3, rephase=False
    ... )
    >>> readout = design.BssfpReadout2D(
    ...     system, excitation.rf, excitation.gz,
    ...     fov=(0.28, 0.28), matrix=(64, 64), readout_bandwidth_hz=50_000,
    ... )
    >>> readout.te == readout.tr / 2
    True

    One repetition and the line it reads. The net gradient area on every axis
    is zero over the repetition, the condition that defines a balanced
    acquisition:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       excitation = sequences.SpatialSelectiveExcitation(
           system, 40.0, 5e-3, 1e-3, rephase=False
       )
       readout = sequences.BssfpReadout2D(
           system, excitation.rf, excitation.gz,
           fov=(0.28, 0.28), matrix=(64, 64), readout_bandwidth_hz=50_000,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _ndim = 2


class BssfpReadout3D(_BssfpReadout):
    """Slab-selective balanced SSFP, encoded along y and z.

    ``fov`` and ``matrix`` take three values, readout first. Add each scaled
    partition encode to the slice rephaser whose window it shares
    (``gz_partition`` to ``gz_pre``, ``gz_partition_rew`` to ``gz_rew``), for
    example with :func:`pypulseqpp.add_gradients`, rather than playing it
    alone.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(
    ...     system, 25.0, 0.12, 1e-3, is_slab=True, rephase=False
    ... )
    >>> readout = design.BssfpReadout3D(
    ...     system, excitation.rf, excitation.gz,
    ...     fov=(0.28, 0.28, 0.12), matrix=(64, 64, 32),
    ...     readout_bandwidth_hz=50_000,
    ... )
    >>> readout.gz_partition.channel, readout.gz_pre.channel
    ('z', 'z')

    One repetition of the slab acquisition. The partition encode is
    published separately and is not played in the layout drawn here:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       excitation = sequences.SpatialSelectiveExcitation(
           system, 25.0, 0.12, 1e-3, is_slab=True, rephase=False
       )
       readout = sequences.BssfpReadout3D(
           system, excitation.rf, excitation.gz,
           fov=(0.28, 0.28, 0.12), matrix=(64, 64, 32), readout_bandwidth_hz=50_000,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _ndim = 3
