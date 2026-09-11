"""2D gradient-echo EPI, multi-slice, written as a linked chain of sequences."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: What a shot of each kind of :meth:`Epi2DApp.kernel` plays.
KINDS = ("calibration", "navigator", "reference", "dummy", "image")


def sms_group_center(
    group: int, n_slices: int, n_bands: int, slice_step: float
) -> float:
    """Centre (m) of the band comb exciting slices ``group, group + n_groups, ...``."""
    n_groups = n_slices // n_bands
    first = (group - (n_slices - 1) / 2) * slice_step
    return first + (n_bands - 1) / 2 * n_groups * slice_step


class Epi2DApp(sequences.SequenceApp):
    """Multi-slice 2D gradient-echo EPI, single-shot or segmented, optionally multiband.

    The trains are ramp-sampled with the blips on the read ramps, and every
    acquisition carries ``REV`` for its read polarity. The imaging is preceded
    by linked prescans (:meth:`prescans`): an accelerated scan's
    ``calibration``, a low-resolution gradient echo per slice (``REF``); then
    the ``navigator``, per slice :attr:`NAVIGATOR_LINES` blip-nulled centre
    lines (``NAV``, ``REF``) and an opposite-phase-encode reference train
    (``SET = 1``). Under ``sms`` a :class:`~pypulseqpp.sequences.SmsExcitation`
    excites ``n_bands`` slices at once and a blipped-CAIPI train encodes the
    band in ``PAR``; its one prescan, ``calibration``, is a blip-nulled
    navigator at the centre group followed by the gradient-echo calibration.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.epi2D_sequence(n_x=32, n_y=16, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["EPIFactor"], seq.definitions["Name"]
    ([16.0], 'epi_2d')
    """

    NAME = "epi_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulses. The selection amplitude, which slice
    #: offsets are converted against, is ``TIME_BW_PRODUCT / (PULSE_DURATION *
    #: thickness)``.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Blip-nulled lines per navigator: odd, even, odd.
    NAVIGATOR_LINES = 3

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_gap: float = 0.0,
        flip_angle_deg: float = 70.0,
        te: float | None = None,
        tr: float | None = None,
        n_repetitions: int = 1,
        segments: int = 1,
        acceleration: int = 1,
        n_bands: int = 1,
        n_acs: int = 24,
        readout_bandwidth_hz: float = 500e3,
        opposite_reference: bool = True,
        slice_order: str = "interleaved",
        n_dummy: int = 2,
        n_gain_calibration_readouts: int | None = None,
        spoiling_cycles: float = 4.0,
        fat_saturation: bool = False,
        sms: bool = False,
    ) -> None:
        """Design the excitation, the train, the calibration and the shot order.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; ``(fov_x, fov_y)`` if a tuple.
        n_x : int, optional
            Readout samples.
        n_y : int, optional
            Phase-encode lines.
        n_slices : int, optional
            Number of slices, interleaved within each repetition.
        slice_thickness : float, optional
            Slice thickness, in metres.
        slice_gap : float, optional
            Gap between adjacent slices, in metres.
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        te : float or None, optional
            Echo time of the first line, in seconds. ``None`` is as short as
            possible.
        tr : float or None, optional
            Repetition time over one shot of one slice, in seconds. ``None``
            is as short as possible.
        n_repetitions : int, optional
            Volumes in the time series, each carrying its ``REP`` counter.
        segments : int, optional
            Interleaved shots the train is split into.
        acceleration : int, optional
            Uniform phase-encode undersampling factor. Above 1 the gradient-echo
            calibration is acquired.
        n_bands : int, optional
            Multiband factor: slices excited at once under ``sms``. It must
            divide ``n_slices``.
        n_acs : int, optional
            Calibration extent along y, in lines, of the gradient-echo
            calibration.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        opposite_reference : bool, optional
            Acquire the opposite-phase-encode reference train after each
            slice's navigator.
        slice_order : str, optional
            Order the slices are acquired in, as
            :func:`pypulseqpp.calc_traversal_order` accepts.
        n_dummy : int, optional
            Shots played without acquiring, per slice, before the first
            acquired one.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        spoiling_cycles : float, optional
            Cycles of dephasing left at the end of each shot.
        fat_saturation : bool, optional
            Saturate fat before every dummy and imaging shot.
        sms : bool, optional
            Excite ``n_bands`` slices at once with blipped-CAIPI encoding, when
            ``n_bands`` is above 1.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y)
        self.matrix = (n_x, n_y, n_slices)
        self.slice_thickness = slice_thickness
        self.n_repetitions, self.segments = n_repetitions, segments
        self.acceleration = acceleration
        self.n_dummy, self.opposite_reference = n_dummy, opposite_reference
        self.sms = sms and n_bands > 1
        self.n_bands = n_bands
        self.n_gain_calibration_readouts = (
            n_slices
            if n_gain_calibration_readouts is None
            else n_gain_calibration_readouts
        )
        slice_step = slice_thickness + slice_gap
        self.slab_thickness = n_slices * slice_step - slice_gap
        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * slice_step
        self.calibration_slices = [
            int(s) for s in pp.calc_traversal_order(n_slices, slice_order)
        ]

        single = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        train = {
            "segments": segments,
            "acceleration": acceleration,
            "te": te,
            "tr": tr,
            "readout_bandwidth_hz": readout_bandwidth_hz,
            "spoiling_cycles": spoiling_cycles,
        }
        if self.sms:
            if n_slices % n_bands:
                raise ValueError(
                    f"the slice count {n_slices} is not a multiple of the multiband "
                    f"factor {n_bands}"
                )
            n_groups = n_slices // n_bands
            band_spacing = n_groups * slice_step
            exc = sequences.SmsExcitation(
                system,
                flip_angle_deg,
                thickness_m=slice_thickness,
                slice_gap_m=band_spacing,
                n_bands=n_bands,
                duration_s=self.PULSE_DURATION,
                time_bw_product=self.TIME_BW_PRODUCT,
            )
            # The rephaser folded onto the selection lobe, so the prewinder
            # block holds no second z lobe.
            gz = pp.concatenate_gradients(exc.gz, exc.gz_reph, system=system)
            # The partition axis is the CAIPI band phase: the sawtooth's gz
            # blips give band j the modulation exp(i 2 pi ky j / n_bands).
            self.epi = sequences.EpiReadout3D(
                system,
                exc.rf,
                gz,
                **train,
                fov=(fov_x, fov_y, n_bands * band_spacing),
                matrix=(n_x, n_y, n_bands),
                scheme="caipi",
                partition_acceleration=n_bands,
                caipi_shift=1,
                labels=("LIN", "PAR"),
            )
            # Segment s starts s lines up, so its band phase starts s steps up
            # the sawtooth, wrapped into the band count: each start is a train.
            trains = {0: self.epi}
            for start in {segment % n_bands for segment in range(segments)} - {0}:
                order = self.epi.order.copy()
                order[:, 1] = (order[:, 1] + start) % n_bands
                trains[start] = sequences.EpiReadout3D(
                    system,
                    exc.rf,
                    gz,
                    **train,
                    order=order,
                    fov=(fov_x, fov_y, n_bands * band_spacing),
                    matrix=(n_x, n_y, n_bands),
                    labels=("LIN", "PAR"),
                )
            self.trains = [trains[segment % n_bands] for segment in range(segments)]
            self.gz_amplitude = float(exc.gz.amplitude)
            self.centers = [
                sms_group_center(g, n_slices, n_bands, slice_step)
                for g in range(n_groups)
            ]
            self.slices = [
                int(g) for g in pp.calc_traversal_order(n_groups, slice_order)
            ]
        else:
            self.epi = sequences.EpiReadout2D(
                system,
                single.rf,
                single.gz,
                single.gz_reph,
                **train,
                fov=self.fov,
                matrix=(n_x, n_y),
                labels=("LIN",),
            )
            self.trains = [self.epi] * segments
            self.gz_amplitude = float(single.gz.amplitude)
            self.centers = list(self.positions)
            self.slices = self.calibration_slices

        self.fatsat = (
            sequences.FatSaturation(
                system, voxel_size_m=min(fov_x / n_x, fov_y / n_y, slice_thickness)
            )
            if fat_saturation
            else None
        )

        # A gradient echo keeps EPI distortion out of the coil maps. An
        # accelerated scan calibrates from it, and a multiband scan always does.
        self.acs = (
            pp.calc_calibration_lines(n_y, n_acs)
            if self.sms or acceleration > 1
            else []
        )
        self.gre = None
        if self.acs:
            self.gre = sequences.LineReadout2D(
                system,
                single.rf,
                single.gz,
                single.gz_reph,
                fov=self.fov,
                matrix=(n_x, n_y),
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=spoiling_cycles,
                labels=("LIN",),
            )
            self.gre_gz_amplitude = float(single.gz.amplitude)

    def prescans(self) -> dict:
        """Return ``calibration`` (when there is one) and, without ``sms``, ``navigator``."""
        chain = {"calibration": self.calibrate} if self.sms or self.acs else {}
        if not self.sms:
            chain["navigator"] = self.navigate
        return chain

    def calibrate(self) -> None:
        """Play the gradient-echo calibration, after the navigator under ``sms``."""
        if self.sms:
            self.kernel(len(self.centers) // 2, None, "navigator")
        for s in self.calibration_slices:
            for line in self.acs:
                self.kernel(s, line, "calibration")
        name = (
            f"sms_{self.NAME}_calibration" if self.sms else f"{self.NAME}_calibration"
        )
        self._define(Name=name, **({"EchoSpacing": self.epi.esp} if self.sms else {}))

    def navigate(self) -> None:
        """Play each slice's navigator and opposite-phase-encode reference."""
        for s in self.slices:
            self.kernel(s, None, "navigator")
            if self.opposite_reference:
                self.kernel(s, 0, "reference")
        self._define(Name=f"{self.NAME}_navigator", EchoSpacing=self.epi.esp)

    def _define(self, **definitions) -> None:
        n_x, n_y, n_slices = self.matrix
        fov = [*self.fov, self.slice_thickness * n_slices]
        for key, value in {
            "FOV": fov,
            "Matrix": [n_x, n_y, n_slices],
            **definitions,
        }.items():
            self.seq.set_definition(key=key, value=value)

    def loop(self) -> None:
        """Play the dummy shots, then every repetition of the volume."""
        for _ in range(self.n_dummy):
            for s in self.slices:
                self.kernel(s, 0, "dummy")
        for repetition in range(self.n_repetitions):
            for segment in range(self.segments):
                for s in self.slices:
                    origin = segment * self.acceleration
                    self.kernel(s, origin, "image", repetition, segment)

    def kernel(
        self,
        s: int,
        origin: int | None,
        kind: str = "image",
        repetition: int = 0,
        segment: int = 0,
    ) -> None:
        """One shot of ``kind``, one of :data:`KINDS`, at slice (multiband group) ``s``.

        ``origin`` is the first line of the train, or the line a calibration
        shot acquires; ``None`` leaves the phase encode at the centre. An
        ``image`` shot plays the train of its ``segment``. The ``reference``
        train negates every phase-encode event, and each of its lines is
        labelled with the line it plays; the ``navigator`` plays
        :attr:`NAVIGATOR_LINES` lines without blips.
        """
        seq, sms = self.seq, self.sms
        n_y = self.matrix[1]
        if sms:
            flags = {
                "calibration": {"NAV": 0, "REF": 1, "SMS": 0, "SLC": s, "REV": 0},
                "navigator": {"NAV": 1, "REF": 0, "SMS": 0},
                "dummy": {"ONCE": 1},
                "image": {"NAV": 0, "REF": 0, "SMS": 1, "SLC": s, "REP": repetition},
            }[kind]
        else:
            flags = {
                "calibration": {"REF": 1, "SLC": s},
                "navigator": {"NAV": 1, "REF": 1, "SET": 0, "SLC": s},
                "reference": {"NAV": 0, "REF": 0, "SET": 1, "SLC": s},
                "dummy": {"ONCE": 1},
                "image": {"SLC": s, "REP": repetition},
            }[kind]
        if kind == "image" and self.n_dummy:
            flags["ONCE"] = 0
        labels = self.labels(**flags)

        if kind == "calibration":
            ro = self.gre
            ro.rf.freq_offset = self.gre_gz_amplitude * self.positions[s]
            ro.rf.phase_offset = -2 * np.pi * ro.rf.freq_offset * ro.rf.center
            ro.adc_labels.value = origin
            ky = (origin - n_y / 2) / (n_y / 2)
            seq.add_block(ro.rf, ro.gz, *labels)
            wait_te = getattr(ro, "wait_te", None)
            if wait_te is not None:
                seq.add_block(wait_te, ro.gz_reph)
                seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky))
            else:
                seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), ro.gz_reph)
            seq.add_block(ro.gx, ro.adc, ro.adc_labels)
            seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky))
            return

        epi = self.trains[segment] if kind == "image" else self.epi
        epi.rf.freq_offset = self.gz_amplitude * self.centers[s]
        epi.rf.phase_offset = -2 * np.pi * epi.rf.freq_offset * epi.rf.center
        acquire = kind != "dummy"
        blipped = kind != "navigator"
        sign = -1.0 if kind == "reference" else 1.0
        encoded = acquire and origin is not None
        n_lines = self.NAVIGATOR_LINES if kind == "navigator" else epi.etl
        gz_blips = getattr(epi, "gz_blips", [None] * epi.etl)

        if self.fatsat is not None and kind in ("dummy", "image"):
            for i, block in enumerate(self.fatsat.blocks):
                seq.add_block(*block, *(labels if i == 0 else ()))
            labels = []
        seq.add_block(epi.rf, epi.gz, *labels)

        # The slice rephaser runs straight off the selection lobe: in the TE
        # wait when there is one, in the prewinder block otherwise.
        rephaser = [g for g in (getattr(epi, "gz_reph", None),) if g is not None]
        wait_te = getattr(epi, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, *rephaser)
            rephaser = []
        ky = 0.0 if origin is None else sign * (origin - n_y / 2) / (n_y / 2)
        shot_labels, line_labels = (), [()] * n_lines
        if encoded and kind == "reference":
            # Every encode is negated, so line i plays -(origin + order_i),
            # which is line n_y - origin - order_i on the grid modulo n_y.
            offsets = epi.order[:, 0] - epi.order[0, 0]
            line_labels = [
                (pp.make_label("LIN", "SET", int(line)),)
                for line in (n_y - origin - offsets) % n_y
            ]
        elif encoded:
            epi.shot_labels[0].value = origin
            if sms:
                epi.shot_labels[1].value = int(epi.order[0, 1])
            shot_labels, line_labels = epi.shot_labels, epi.line_labels
        # A multiband shot's band phase starts at its train's first partition
        # offset, from kz = 0; the gz blips walk the CAIPI sawtooth from there.
        band_phase = (
            [pp.scale_grad(epi.gz_pre, epi.order[0, 1] / (self.n_bands / 2))]
            if sms
            else []
        )
        if blipped:
            seq.add_block(
                epi.gx_pre,
                pp.scale_grad(epi.gy_pre, ky),
                *band_phase,
                *rephaser,
                *shot_labels,
            )
        else:
            seq.add_block(epi.gx_pre, *rephaser)
        for line in range(n_lines):
            events = [epi.gx[line]]
            if acquire:
                events += [epi.adc, *self.labels(REV=line % 2)]
            if blipped:
                events += [
                    pp.scale_grad(blip, sign) if sign < 0 else blip
                    for blip in (epi.gy_blips[line], gz_blips[line])
                    if blip is not None
                ]
                events += line_labels[line]
            seq.add_block(*events)
        if blipped:
            seq.add_block(epi.gx_spoil, pp.scale_grad(epi.gy_rew, ky))
        else:
            seq.add_block(epi.gx_spoil)
        wait_tr = getattr(epi, "wait_tr", None)
        if wait_tr is not None:
            seq.add_block(wait_tr)

    def finalize(self) -> None:
        """Write the prescription and the train's echo timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV (compat=False: the lines are sampled on the ramps).
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": f"sms_{self.NAME}" if self.sms else self.NAME,
            "TE": float(self.epi.echo_times[len(self.epi.order) // 2]),
            "EchoSpacing": self.epi.esp,
            "EPIFactor": self.epi.etl,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        if self.sms:
            definitions["MultibandFactor"] = self.n_bands
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Epi2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="epi_2d.seq"))
