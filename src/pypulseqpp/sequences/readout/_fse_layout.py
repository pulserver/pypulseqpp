"""The packed FSE train: rewinder tails in the refocusing pulse's dead time.

All gradients are constructed once. Phase-encoding amplitudes change per
echo; shapes and areas at unit encoding are retained across every shot.
"""

from types import SimpleNamespace

import numpy as np

import pypulseqpp as pp


def _poly(gradient, span):
    if gradient.type == "trap":
        times = np.array(
            [
                0.0,
                gradient.rise_time,
                gradient.rise_time + gradient.flat_time,
                gradient.rise_time + gradient.flat_time + gradient.fall_time,
            ]
        )
        amplitudes = np.array([0.0, gradient.amplitude, gradient.amplitude, 0.0])
    else:
        times, amplitudes = np.asarray(gradient.tt), np.asarray(gradient.waveform)
    times = times + gradient.delay
    times = np.r_[0.0, times, span]
    amplitudes = np.r_[
        amplitudes[0] if gradient.delay == 0 else 0.0, amplitudes, amplitudes[-1]
    ]
    times, indices = np.unique(np.round(times, 12), return_index=True)
    return times, amplitudes[indices]


def _part(gradient, span, start, stop, *, duration=None, delay=0.0, system):
    times, amplitudes = _poly(gradient, span)
    inside = (times > start + 1e-12) & (times < stop - 1e-12)
    selected = np.r_[start, times[inside], stop]
    values = np.interp(selected, times, amplitudes)
    selected = selected - start + delay
    duration = stop - start + delay if duration is None else duration
    if delay:
        selected, values = np.r_[0.0, selected], np.r_[0.0, values]
    if duration > selected[-1] + 1e-12:
        selected, values = np.r_[selected, duration], np.r_[values, values[-1]]
    return pp.make_extended_trapezoid(
        gradient.channel, times=np.round(selected, 12), amplitudes=values, system=system
    )


class RfOverlapLayout:
    """The events of a packed FSE train, played one stage of an echo at a time.

    ``overlap`` (s) is the part of the refocusing pulse's dead time the
    previous echo's rewinders end in; zero plays them whole before it. A
    selective pulse (``fse.gz_ref``) holds the partition axis through its
    block, so the partition rewinder ends before it.
    """

    def __init__(self, fse, overlap, system):
        self.fse = fse
        self.overlap = overlap
        raster = system.block_duration_raster
        selective = fse.gz_ref is not None
        exspan = pp.calc_duration(fse.rf, *_present(fse.gz))
        leadspan = pp.ceil_to_raster(pp.calc_duration(fse.gx_pre), raster)
        refspan = pp.ceil_to_raster(
            pp.calc_duration(fse.rf_ref, *_present(fse.gz_ref)), raster
        )
        prespan = pp.calc_duration(fse.gx_bridge_pre, fse.gy_pre, *_present(fse.gz_pre))
        postspan = pp.calc_duration(
            fse.gx_bridge_post, fse.gy_rew, *_present(fse.gz_rew)
        )
        prespan = pp.round_to_raster(prespan, raster)
        postspan = pp.round_to_raster(postspan, raster)
        newpre = pp.round_to_raster(prespan - overlap, raster)
        headspan = pp.round_to_raster(postspan - overlap, raster)
        # An overlap as long as the prephaser leaves the excitation block none.
        self.excitation_x = None
        if leadspan - overlap > raster / 2:
            self.excitation_x = _part(
                fse.gx_pre,
                leadspan,
                0.0,
                leadspan - overlap,
                delay=exspan,
                system=system,
            )
        self.first_tail_x = self.tail_x = None
        if overlap > 0:
            self.first_tail_x = _part(
                fse.gx_pre,
                leadspan,
                leadspan - overlap,
                leadspan,
                duration=refspan,
                system=system,
            )
            self.tail_x = _part(
                fse.gx_bridge_post,
                postspan,
                headspan,
                postspan,
                duration=refspan,
                system=system,
            )
            self.head_x = _part(
                fse.gx_bridge_post, postspan, 0.0, headspan, system=system
            )
            pre_x = fse.gx_bridge_pre
            shifted = SimpleNamespace(
                type="grad",
                tt=np.asarray(pre_x.tt),
                waveform=np.asarray(pre_x.waveform),
                delay=pre_x.delay - overlap,
                channel="x",
            )
            self.pre_x = _part(shifted, newpre, 0.0, newpre, system=system)
        else:
            self.head_x, self.pre_x = fse.gx_bridge_post, fse.gx_bridge_pre
        self.pre, self.tail, self.head, self.final = [], [], [], []
        for axis, oldpre, oldpost in (
            ("y", fse.gy_pre, fse.gy_rew),
            ("z", fse.gz_pre, fse.gz_rew),
        ):
            if oldpre is None:
                for collection in (self.pre, self.tail, self.head, self.final):
                    collection.append(None)
                continue
            pre = pp.make_trapezoid(
                axis, area=float(oldpre.area), duration=newpre, system=system
            )
            if overlap > 0 and not (selective and axis == "z"):
                tail = _part(
                    oldpost,
                    postspan,
                    headspan,
                    postspan,
                    duration=refspan,
                    system=system,
                )
                head = _part(oldpost, postspan, 0.0, headspan, system=system)
            else:
                tail = None
                head = pp.make_trapezoid(
                    axis, area=float(oldpost.area), duration=headspan, system=system
                )
            # Distinct events allow every phase in the loop to be updated
            # without retaining amplitude changes from a previous phase.
            final = pp.scale_grad(oldpost, 1.0)
            for collection, event in (
                (self.pre, pre),
                (self.tail, tail),
                (self.head, head),
                (self.final, final),
            ):
                collection.append(
                    None
                    if event is None
                    else (
                        event,
                        float(event.amplitude),
                        float(event.first) if event.type == "grad" else 0.0,
                        float(event.last) if event.type == "grad" else 0.0,
                    )
                )
        self.first_tail = [
            None if item is None else pp.scale_grad(item[0], 0.0) for item in self.tail
        ]

    @staticmethod
    def encoded(collection, ky, kz):
        events = []
        for item, scale in zip(collection, (ky, kz), strict=True):
            if item is None:
                continue
            gradient, amplitude, first, last = item
            gradient.amplitude = amplitude * scale
            if gradient.type == "grad":
                gradient.first = first * scale
                gradient.last = last * scale
            events.append(gradient)
        return events

    def excitation(self, sequence, labels):
        sequence.add_block(
            self.fse.rf,
            *_present(self.excitation_x),
            *_present(self.fse.gz),
            *labels,
        )

    def refocusing(self, sequence, first, previous_ky, previous_kz):
        if first:
            tail = [event for event in self.first_tail if event is not None]
        else:
            tail = self.encoded(self.tail, previous_ky, previous_kz)
        sequence.add_block(
            self.fse.rf_ref,
            *_present(self.fse.gz_ref),
            *_present(self.first_tail_x if first else self.tail_x),
            *tail,
        )

    def encoding(self, sequence, ky, kz):
        sequence.add_block(self.pre_x, *self.encoded(self.pre, ky, kz))

    def rewinding(self, sequence, ky, kz, last):
        if last:
            sequence.add_block(
                self.fse.gx_bridge_post, *self.encoded(self.final, ky, kz)
            )
        else:
            sequence.add_block(self.head_x, *self.encoded(self.head, ky, kz))


def _present(event):
    return () if event is None else (event,)


class CrusherSplitLayout:
    """The events of a packed 2D FSE train: encodes and rewinders under the crushers.

    The selection gradient of the refocusing pulse is cut into three blocks:
    the crusher that leads into the pulse, the pulse with its plateau, and
    the crusher that leads out. The previous echo's read and phase rewinders
    play under the leading crusher and the next echo's read bridge and phase
    encode under the trailing one, so each phase-encode block holds a single
    lobe that the encode scales. The excitation's rephaser is added to the
    first leading crusher, and the read prephaser plays beside it.

    Every interval between a pulse centre and its echo is half of ``esp``,
    the first excitation-to-refocusing interval included; the blocks either
    side of the readout are padded to make it so. The refocusing pulse's
    ``delay`` is set to the system's dead time, its place in the pulse block.
    """

    def __init__(self, fse, system, *, flat_span, echo_offset, encode_resolution):
        raster = system.block_duration_raster
        self.fse = fse
        rf_ref, gz_ref = fse.rf_ref, fse.gz_ref
        dead, ringdown = system.rf_dead_time, system.rf_ringdown_time
        start = pp.round_to_raster(float(rf_ref.delay) - dead, raster)
        if start < -1e-12:
            raise ValueError("the refocusing pulse starts within the RF dead time")
        pulse_span = pp.ceil_to_raster(
            dead + float(rf_ref.shape_dur) + ringdown, raster
        )
        z_span = pp.ceil_to_raster(pp.calc_duration(gz_ref), raster)
        stop = start + pulse_span
        self.z_pulse = _part(gz_ref, z_span, start, min(stop, z_span), system=system)
        self.z_lead = (
            _part(gz_ref, z_span, 0.0, start, system=system) if start > 0 else None
        )
        self.z_trail = (
            _part(gz_ref, z_span, stop, z_span, system=system)
            if z_span - stop > raster / 2
            else None
        )
        plateau = float(np.interp(start, *_poly(gz_ref, z_span)))
        lead_area = 0.0 if self.z_lead is None else float(self.z_lead.area)
        rephase_area = 0.0 if fse.gz_reph is None else float(fse.gz_reph.area)
        self.z_first_lead = pp.make_extended_trapezoid_area(
            area=lead_area + rephase_area,
            channel="z",
            grad_start=0.0,
            grad_end=plateau,
            system=system,
        )[0]
        rf_ref.delay = dead
        centre = dead + float(rf_ref.center)

        unit_encode = pp.make_phase_encoding("y", encode_resolution, system=system)
        span = lambda *events: pp.ceil_to_raster(  # noqa: E731
            pp.calc_duration(*[e for e in events if e is not None]), raster
        )
        trail_min = span(self.z_trail, fse.gx_bridge_pre, unit_encode)
        lead_min = span(self.z_lead, fse.gx_bridge_post, unit_encode)
        first_min = span(self.z_first_lead, fse.gx_pre)
        excitation_span = span(fse.rf, fse.gz)
        excitation_tail = excitation_span - float(fse.rf.delay) - float(fse.rf.center)

        # The halves either side of the echo are sums of block spans and
        # fixed offsets, so they meet only to within a raster; the spans
        # nearest to equal halves, then the shortest, are taken.
        before = pulse_span - centre + echo_offset
        after = flat_span - echo_offset + centre
        trail_min = max(
            trail_min,
            pp.ceil_to_raster(
                excitation_tail + first_min + centre - before - 1e-12, raster
            ),
        )
        lead_min = max(
            lead_min,
            pp.ceil_to_raster(before + trail_min - after - 1e-12, raster),
        )
        best = None
        for i in range(12):
            for j in range(12):
                trail, lead = trail_min + i * raster, lead_min + j * raster
                half = before + trail
                first = pp.round_to_raster(half - excitation_tail - centre, raster)
                if first < first_min - 1e-12:
                    continue
                mismatch = abs(half - (after + lead)) + abs(
                    excitation_tail + first + centre - half
                )
                key = (
                    round(mismatch / raster, 6) > 0.5,
                    before + trail + after + lead,
                    mismatch,
                )
                if best is None or key < best[0]:
                    best = (key, trail, lead, first)
        if best is None:
            raise ValueError(
                "the excitation, its rephaser and the read prephaser do not fit "
                "in half an echo spacing of this packed train"
            )
        _, self.trail_span, self.lead_span, self.first_span = best
        self.pulse_span = pulse_span
        self.esp = self.pulse_span + self.trail_span + flat_span + self.lead_span
        self.first_echo = (
            excitation_span
            + self.first_span
            + pulse_span
            + self.trail_span
            + echo_offset
        )

        self.gy_encode = pp.make_phase_encoding(
            "y", encode_resolution, system=system, duration=self.trail_span
        )
        self.gy_rewind = pp.scale_grad(
            pp.make_phase_encoding(
                "y", encode_resolution, system=system, duration=self.lead_span
            ),
            -1.0,
        )
        self.gy_final = pp.scale_grad(unit_encode, -1.0)
        self.encode_amplitude = float(self.gy_encode.amplitude)
        self.rewind_amplitude = float(self.gy_rewind.amplitude)
        self.final_amplitude = float(self.gy_final.amplitude)

        self.x_trail = _shifted(fse.gx_bridge_pre, self.trail_span)
        self.x_first = _shifted(fse.gx_pre, self.first_span)
        self.z_lead_at_end = _shifted(self.z_lead, self.lead_span)
        self.z_first_at_end = _shifted(self.z_first_lead, self.first_span)

    def excitation(self, sequence, labels):
        sequence.add_block(self.fse.rf, *_present(self.fse.gz), *labels)

    def refocusing(self, sequence, first, previous_ky, previous_kz=0.0):  # noqa: ARG002
        if first:
            sequence.add_block(
                pp.make_delay(self.first_span),
                *_present(self.x_first),
                self.z_first_at_end,
            )
        else:
            self.gy_rewind.amplitude = self.rewind_amplitude * previous_ky
            sequence.add_block(
                pp.make_delay(self.lead_span),
                self.fse.gx_bridge_post,
                self.gy_rewind,
                *_present(self.z_lead_at_end),
            )
        sequence.add_block(self.fse.rf_ref, self.z_pulse)

    def encoding(self, sequence, ky, kz=0.0):  # noqa: ARG002
        self.gy_encode.amplitude = self.encode_amplitude * ky
        sequence.add_block(
            pp.make_delay(self.trail_span),
            self.x_trail,
            self.gy_encode,
            *_present(self.z_trail),
        )

    def rewinding(self, sequence, ky, kz, last):  # noqa: ARG002
        if last:
            self.gy_final.amplitude = self.final_amplitude * ky
            sequence.add_block(self.fse.gx_bridge_post, self.gy_final)


def _shifted(gradient, span):
    """Return ``gradient`` delayed to end at ``span``."""
    if gradient is None:
        return None
    moved = pp.scale_grad(gradient, 1.0)
    moved.delay = float(gradient.delay) + span - pp.calc_duration(gradient)
    return moved
