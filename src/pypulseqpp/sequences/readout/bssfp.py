"""Balanced SSFP readouts: a fully rewound repetition with the echo at TR/2."""

from __future__ import annotations

__all__ = ["BssfpReadout2D", "BssfpReadout3D"]

from typing import Any

import numpy as np

import pypulseqpp as pp

from .._module import SequenceModule
from ._common import as_tuple, present, solve_delay

#: Fraction of ``max_grad`` the readout plateau may reach.
_READOUT_GRAD_MARGIN = 0.8


class _BssfpReadout(SequenceModule):
    """Balanced SSFP repetition with TE fixed at TR/2.

    Every axis is balanced between consecutive RF centres. The slice
    rephasers are part of that balance and are built here, so pass an
    excitation built with ``rephase=False``.

    The module's blocks are excitation, acquisition, rewind. A loop plays each
    shot as the previous shot's rewind, excitation, acquisition, and writes
    the transients itself: the half-flip pulse (``rf`` at half amplitude,
    opposite in phase to the first excitation) then ``wait_prep``;
    ``wait_rewind`` with ``gz_rew`` in place of the first rewind; and a last
    rewind block to ramp down.

    Attributes
    ----------
    rf : RfEvent
        The excitation, replayed every repetition with the phase and amplitude
        the loop sets.
    gz : GradEvent
        Its selection gradient, if one was given.
    gx, gx_rew : GradEvent
        ``gx`` climbs from zero, prephases and holds the readout plateau to
        the end of the acquisition block; ``gx_rew`` leaves the plateau and
        rewinds the read axis.
    gy_pre, gy_rew : TrapEvent
        In-plane encode at its largest step, and the same negated, to be scaled
        per shot.
    gz_pre, gz_rew : TrapEvent
        The slice rephasers: ``gz_pre`` in the acquisition block after the
        pulse, ``gz_rew`` in the rewind block before the next one. Absent for
        a hard pulse.
    gz_partition, gz_partition_rew : TrapEvent
        Partition encode at its largest step, and the same negated, sharing
        the window of ``gz_pre`` and ``gz_rew`` respectively. 3D only.
    adc : AdcEvent
        The acquisition window, spanning the readout plateau.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    prep_labels, train_labels, end_labels : list of LabelSetEvent
        ``ONCE`` set to 1, 0 and 2, for the half-flip block, the block after
        it and the final rewind.
    wait_prep : DelayEvent
        Played after the half-flip block; with the ``wait_rewind`` block it
        puts the half-flip pulse half a repetition before the first
        excitation. Absent under ``half_flip_prep=False``.
    wait_rewind : DelayEvent
        The rewind block's duration, for the first repetition.
    tr, te : float
        Repetition and echo time (s); ``te`` is always half of ``tr``.
    bandwidth_hz : float
        Achieved ADC sampling rate (Hz).
    n_samples : int
        Samples per repetition.
    delta_kx : float
        Read-axis k-space step (1/m).
    readout_duration : float
        Sampling window (s).

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The excitation.
    gz : GradEvent, optional
        Its selection gradient without a rephaser, played in the excitation
        block. ``None`` for a hard pulse.
    fov : float or sequence of float
        Field of view (m), per encoded axis, readout first.
    matrix : int or sequence of int
        Matrix size, per encoded axis.
    tr : float, optional
        Repetition time (s). ``None`` is as short as possible; a longer one is
        padded evenly either side of the echo, so TE stays at TR/2.
    half_flip_prep : bool, optional
        Publish ``wait_prep``. This requires the acquisition block after the
        echo to be at least as long as the excitation block after the pulse
        centre. Turn it off to ramp the train in with dummy repetitions.
    oversampling : float, optional
        Read oversampling.
    readout_bandwidth_hz : float, optional
        Requested ADC sampling rate (Hz). ``bandwidth_hz`` reports the
        achieved raster-compatible rate.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.
    trigger : event, optional
        A trigger or digital output armed on the rewind block.

    Raises
    ------
    ValueError
        If a size or rate is out of range, the ``half_flip_prep`` condition
        fails, or the requested TR is shorter than the module can achieve.
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
    ) -> None:
        ndim = self._ndim
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

        # Every z lobe is built on the same ramp, which is what makes a
        # rephaser and a partition encode add up to a single trapezoid instead
        # of an arbitrary waveform.
        ramp = pp.ceil_to_raster(
            system.max_grad / system.max_slew, system.grad_raster_time
        )
        step_z = 0.5 * n[2] / fov[2] if ndim == 3 else 0.0
        floor = max(
            _z_floor(system, max(abs(pre_area), abs(post_area)) + step_z, ramp, raster),
            _span(raster, gx_rew, trigger),
            _span(raster, pp.make_phase_encoding("y", fov[1] / n[1], system=system)),
            pp.ceil_to_raster(float(entry_times[-1]), raster),
        )

        # TE = TR/2 puts the echo as far after the pulse centre as the next
        # pulse centre is after the echo, which is one equation between the two
        # windows either side of the excitation block.
        offset = (flat_span - readout_duration) + 2.0 * rf_center - excite_span
        rewind_span = pp.ceil_to_raster(max(floor, floor - offset), raster)
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
            wait_prep = pp.make_delay(prep_span)
        wait_rewind = pp.make_delay(rewind_span)

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]
        prep_labels = [pp.make_label(type="SET", label="ONCE", value=1)]
        train_labels = [pp.make_label(type="SET", label="ONCE", value=0)]
        end_labels = [pp.make_label(type="SET", label="ONCE", value=2)]

        # One repetition of the centre partition, laid out from the pulse so
        # that it starts and ends with every axis at rest.
        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, *present(gz))
        self.seq.add_block(gx, adc, gy_pre, *present(gz_pre), *adc_labels)
        self.seq.add_block(gx_rew, gy_rew, *present(gz_rew), *present(trigger))
        self.register(
            wait_rewind=wait_rewind,
            prep_labels=prep_labels,
            train_labels=train_labels,
            end_labels=end_labels,
            **({"wait_prep": wait_prep} if half_flip_prep else {}),
        )

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


def _vertices(system: pp.Opts, area: float, grad_start: float, grad_end: float):
    _, times, amplitudes = pp.make_extended_trapezoid_area(
        area=area, channel="x", grad_start=grad_start, grad_end=grad_end, system=system
    )
    return np.asarray(times, dtype=float), np.asarray(amplitudes, dtype=float)


def _lobe(system: pp.Opts, area: float, duration: float, ramp: float):
    """Z trapezoid of ``area`` over ``duration``, or ``None`` if there is none.

    The ramp is given rather than solved, so that two lobes over the same
    window share their vertex times whatever their areas -- which is what lets
    a rephaser and a partition encode be added into one trapezoid.
    """
    if abs(area) < 1e-12:
        return None
    return pp.make_trapezoid(
        channel="z", area=area, duration=duration, rise_time=ramp, system=system
    )


def _z_floor(system: pp.Opts, area: float, ramp: float, raster: float) -> float:
    """Shortest window that holds a z trapezoid of ``area`` on ramps of ``ramp``.

    The caller passes a rephaser plus the outermost partition step: the loop
    adds the two into one trapezoid, and their sum must stay within
    ``max_grad``.
    """
    return pp.ceil_to_raster(max(2.0 * ramp, ramp + area / system.max_grad), raster)


class BssfpReadout2D(_BssfpReadout):
    """Slice-selective balanced SSFP, phase-encoded along y.

    ``fov`` and ``matrix`` take two values, readout first. See
    :class:`~pypulseqpp.sequences.readout.bssfp._BssfpReadout` for the
    parameters and the loop the transients need.
    """

    _ndim = 2


class BssfpReadout3D(_BssfpReadout):
    """Slab-selective balanced SSFP, encoded along y and z.

    ``fov`` and ``matrix`` take three values, readout first. Add each scaled
    partition encode to the slice rephaser whose window it shares
    (``gz_partition`` to ``gz_pre``, ``gz_partition_rew`` to ``gz_rew``), for
    example with :func:`pypulseqpp.add_gradients`, rather than playing it
    alone.
    """

    _ndim = 3
