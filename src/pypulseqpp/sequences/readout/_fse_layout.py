"""The packed 3D FSE train: rewinder tails in the refocusing pulse's dead time.

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
    """The events of a packed 3D FSE train, played one stage of an echo at a time.

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
