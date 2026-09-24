"""Balanced SSFP 2D Cartesian, multi-slice, with optional cardiac cine gating."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The cardiac gating modes ``gating`` selects from.
GATINGS = ("none", "retrospective", "prospective")


class Bssfp2DApp(sequences.SequenceApp):
    """Balanced SSFP 2D Cartesian: one complete train per slice, optionally cardiac-gated.

    Every axis returns to k = 0 between pulse centres and TE is TR/2
    (:class:`BssfpReadout2D`). Each slice's train is entered through a
    half-flip pulse half a TR ahead of the first excitation and opposite in
    phase to it; excitations then alternate between phase pi and 0, the ADC
    following. ``ONCE`` marks the half flip (1), the train (0) and the closing
    rewind (2), so the whole slice is one repetition of the scan. Slices play
    one after another, since a steady state does not survive interleaving,
    and a slice longer than :attr:`MAX_SLICE_DURATION` is refused.

    The phase-encode lines are played ``views_per_segment`` at a time. Under
    ``retrospective`` gating each segment is cycled for one heartbeat, the
    interpreter's ECG log binning it into cardiac phases; under
    ``prospective`` gating every heartbeat opens with a trigger event on
    :attr:`TRIGGER_CHANNEL`, then plays the segment once per cardiac phase;
    the first heartbeat's trigger precedes the half flip, so the whole train
    is timed from the R wave, and a slice held in one heartbeat is a triggered
    single shot.
    Acquisitions carry ``LIN``, ``SLC``, the segment as ``SEG`` and the cardiac
    phase, or the cycle within the heartbeat, as ``PHS``; calibration lines
    are marked ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.bssfp2D_sequence(
    ...     n_x=64, n_y=16, readout_bandwidth_hz=50e3, n_dummy=0
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "bssfp_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse. The selection amplitude, which slice
    #: offsets are converted against, is ``TIME_BW_PRODUCT / (PULSE_DURATION *
    #: thickness)``.
    PULSE_DURATION = 0.6e-3
    TIME_BW_PRODUCT = 4.0
    #: Physiological signal a prospective heartbeat waits for.
    TRIGGER_CHANNEL = "physio1"
    #: Longest a slice's train may last (s); the interpreter aborts beyond it.
    MAX_SLICE_DURATION = 15.0

    def init_sequence(
        self,
        fov_x: float = 300e-3,
        fov_y: float = 300e-3,
        n_x: int = 192,
        n_y: int = 192,
        n_slices: int = 1,
        slice_thickness: float = 6e-3,
        slice_spacing: float = 0.0,
        flip_angle_deg: float = 45.0,
        tr: float | None = None,
        readout_bandwidth_hz: float = 125e3,
        ry: int = 1,
        partial_fourier_y: float = 1.0,
        n_phases: int = 25,
        *,
        n_dummy: int = 10,
        readout_oversampling: float = 1.0,
        n_acs_y: int = 24,
        gating: str = "none",
        heart_rate_bpm: float = 60.0,
        views_per_segment: int = 12,
        trigger_delay: float = 0.0,
    ) -> None:
        """Design the balanced repetition, the slice positions and the segments.

        Parameters
        ----------
        fov_x, fov_y : float, default=0.3
            Field of view along the readout and the phase encode (m).
        n_x : int, default=192
            Readout matrix size.
        n_y : int, default=192
            Phase-encode matrix size.
        n_slices : int, default=1
            Number of slices, each acquired as its own complete train.
        slice_thickness : float, default=0.006
            Slice thickness (m).
        slice_spacing : float, default=0.0
            Gap between adjacent slices (m); zero is contiguous.
        flip_angle_deg : float, default=45.0
            Excitation flip angle (degrees).
        tr : float | None, default=None
            Repetition time (s); TE is TR/2. ``None`` is as short as possible.
        readout_bandwidth_hz : float, default=125000.0
            Requested receiver bandwidth (Hz). The half flip needs an
            acquisition block at least as long as the excitation's tail, which
            a lower bandwidth provides.
        ry : int, default=1
            Phase-encode undersampling: one line in every ``ry`` is acquired,
            the centre line among them.
        partial_fourier_y : float, default=1.0
            Fraction of the phase-encode extent acquired, in ``[0.75, 1]``.
        n_phases : int, default=25
            Cardiac phases a prospective heartbeat acquires.
        n_dummy : int, default=10
            Repetitions played without acquiring after the half flip, while
            the oscillating transient settles.
        readout_oversampling : float, default=1.0
            Readout oversampling factor, at least one.
        n_acs_y : int, default=24
            Fully sampled calibration lines at the centre of k-space, acquired
            ahead of the rest when ``ry > 1``.
        gating : {'none', 'retrospective', 'prospective'}, default='none'
            No gating; each segment cycled over one heartbeat; or each
            heartbeat triggered and acquired once per cardiac phase.
        heart_rate_bpm : float, default=60.0
            Nominal heart rate (beats per minute) the segments are timed to.
        views_per_segment : int, default=12
            Phase-encode lines per segment, one segment per heartbeat. Unused
            without gating.
        trigger_delay : float, default=0.0
            Wait after a prospective trigger before the first cardiac phase (s).

        Raises
        ------
        ValueError
            If ``gating`` is unknown, ``partial_fourier_y`` is outside
            ``[0.75, 1]``, a count is below one, a prospective heartbeat cannot
            hold its phases, a slice
            outlasts :attr:`MAX_SLICE_DURATION`, or the TR is shorter than the
            balanced repetition, or too short beside the pulse for the half
            flip.
        """
        if gating not in GATINGS:
            raise ValueError(f"gating must be one of {GATINGS}, got {gating!r}")
        if not 0.75 <= partial_fourier_y <= 1.0:
            raise ValueError(
                f"partial_fourier_y must lie in [0.75, 1], got {partial_fourier_y}"
            )
        for name, count in (
            ("ry", ry),
            ("n_phases", n_phases),
            ("views_per_segment", views_per_segment),
        ):
            if count < 1:
                raise ValueError(f"{name} must be at least 1, got {count}")

        system = self.system
        self.fov = (fov_x, fov_y)
        self.matrix = (n_x, n_y, n_slices)
        self.n_dummy, self.gating = n_dummy, gating
        self.n_phases, self.heart_rate_bpm = n_phases, heart_rate_bpm
        self.views_per_segment, self.trigger_delay = views_per_segment, trigger_delay
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            rephase=False,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        # Solving the balance and TE = TR/2 is the feasibility check: an
        # unreachable TR raises ValueError here.
        self.ro = sequences.BssfpReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            fov=self.fov,
            matrix=(n_x, n_y),
            tr=tr,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )
        self.nominal = self.ro.rf.amplitude
        self.trigger = pp.make_trigger(
            self.TRIGGER_CHANNEL, duration=trigger_delay, system=system
        )

        calibrating, imaging = pp.make_cartesian_axis_sampling(
            n_y, ry, n_acs_y, partial_fourier=partial_fourier_y
        )
        # The calibration block leads, so a reconstruction can estimate
        # coil sensitivities while the rest is still arriving.
        self.lines = [*calibrating, *imaging]
        self.calibration = set(calibrating)
        # A train is a list of (line, segment, phase) repetitions, None where a
        # heartbeat's trigger falls; the first heartbeat's is ahead of the half
        # flip, and so ahead of the dummies too.
        rr = 60.0 / heart_rate_bpm
        segments = (
            [self.lines]
            if gating == "none"
            else [
                self.lines[i : i + views_per_segment]
                for i in range(0, len(self.lines), views_per_segment)
            ]
        )
        self.train = [(None, 0, 0)] * n_dummy
        for s, segment in enumerate(segments):
            if gating == "none":
                self.train += [(line, 0, 0) for line in segment]
            elif gating == "retrospective":
                cycles = max(1, round(rr / (len(segment) * self.ro.tr)))
                self.train += [(line, s, c) for c in range(cycles) for line in segment]
            else:
                acquired = n_phases * len(segment) * self.ro.tr + trigger_delay
                if s == 0:
                    acquired += (0.5 + n_dummy) * self.ro.tr
                if acquired > rr + 1e-9:
                    raise ValueError(
                        f"{n_phases} cardiac phases of {len(segment)} lines take "
                        f"{acquired * 1e3:.0f} ms, longer than the "
                        f"{rr * 1e3:.0f} ms heartbeat"
                    )
                if s:
                    self.train.append(None)
                self.train += [
                    (line, s, f) for f in range(n_phases) for line in segment
                ]
        self.n_segments = len(segments)

        # The half flip takes half a TR and each repetition a TR, the last one's
        # closing rewind included; a prospective heartbeat lasts until the next
        # trigger.
        repetitions = sum(item is not None for item in self.train)
        self.slice_duration = (repetitions + 0.5) * self.ro.tr
        if gating == "prospective":
            self.slice_duration = self.n_segments * rr
        if self.slice_duration > self.MAX_SLICE_DURATION:
            raise ValueError(
                f"a slice's train lasts {self.slice_duration:.1f} s, longer than "
                f"the {self.MAX_SLICE_DURATION:.0f} s one repetition of the scan "
                f"may; raise ry or views_per_segment, or lower n_y"
            )
        slice_step = slice_thickness + slice_spacing
        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * slice_step
        self.slab_thickness = n_slices * slice_step - slice_spacing
        self.slice_gap = slice_step - self.exc.slice_thickness
        self.duration = n_slices * self.slice_duration

    def _ky(self, line: int | None) -> float:
        n_y = self.matrix[1]
        return 0.0 if line is None else (line - n_y // 2) / (n_y / 2)

    def loop(self) -> None:
        """Play each slice's whole train before the next slice's."""
        for s in range(self.matrix[2]):
            first = self.seq.num_blocks
            previous, shot, trigger = None, 0, False
            last = max(i for i, item in enumerate(self.train) if item is not None)
            for i, item in enumerate(self.train):
                if item is None:
                    trigger = True
                    continue
                line, segment, phase = item
                self.kernel(
                    s, shot, line, previous, segment, phase, trigger, last=i == last
                )
                previous, shot, trigger = self._ky(line), shot + 1, False
            if s == 0:
                self.plot_tr_start = first + 1
                self.plot_tr_size = self.seq.num_blocks - first

    def kernel(
        self,
        s: int,
        shot: int,
        line: int | None,
        previous_ky: float | None,
        segment: int = 0,
        phase: int = 0,
        trigger: bool = False,
        last: bool = False,
    ) -> None:
        """One repetition of slice ``s``: the rewind of the previous one, then excite and read.

        ``previous_ky`` is the fractional encode the rewind undoes; ``None``
        opens the slice's train with the half flip, after the trigger under
        ``prospective`` gating. ``trigger`` waits for the heartbeat between the
        rewind and the excitation, ``last`` closes the train with the final
        rewind, and ``line=None`` plays a dummy.

        Parameters
        ----------
        s : int
            Slice index, counting from 0 in play order.
        shot : int
            Repetition index within the slice's train, which sets the RF and
            ADC phase.
        line : int or None
            The phase-encode line to acquire, or None for a dummy.
        previous_ky : float or None
            The fractional encode the opening rewind undoes; None opens the
            train with the half flip.
        segment : int, default=0
            Value written to the SEG label.
        phase : int, default=0
            Value written to the PHS label.
        trigger : bool, default=False
            Wait for the heartbeat between the rewind and the excitation.
        last : bool, default=False
            Close the train with the final rewind.
        """
        ro, seq = self.ro, self.seq
        rf = ro.rf
        rf.freq_offset = self.exc.selection_amplitude * self.positions[s]
        centre_phase = -2 * np.pi * rf.freq_offset * rf.center

        if previous_ky is None:
            # Half the flip at phase 0, opposite to the first excitation's pi,
            # lands the magnetisation on the bisector the steady state
            # oscillates about. The module writes ONCE on this block and the
            # next, so the labels restart here as a ONCE through labels() would.
            self.restart_labels()
            rf.amplitude = 0.5 * self.nominal
            rf.phase_offset = centre_phase
            if self.gating == "prospective":
                seq.add_block(self.trigger, *ro.prep_labels)
                seq.add_block(rf, ro.gz)
            else:
                seq.add_block(rf, ro.gz, *ro.prep_labels)
            seq.add_block(ro.wait_prep, *ro.train_labels)
            rf.amplitude = self.nominal
            seq.add_block(ro.wait_rewind, ro.gz_rew)
        else:
            seq.add_block(ro.gx_rew, pp.scale_grad(ro.gy_rew, previous_ky), ro.gz_rew)
        if trigger:
            seq.add_block(self.trigger)

        alternation = np.pi * ((shot + 1) % 2)
        rf.phase_offset = alternation + centre_phase
        ro.adc.phase_offset = alternation
        seq.add_block(rf, ro.gz)

        ky = self._ky(line)
        if line is None:
            seq.add_block(ro.gx, pp.scale_grad(ro.gy_pre, ky), ro.gz_pre)
        else:
            labels = self.labels(
                LIN=line,
                SLC=s,
                SEG=segment,
                PHS=phase,
                IMA=line in self.calibration,
            )
            seq.add_block(
                ro.gx, ro.adc, pp.scale_grad(ro.gy_pre, ky), ro.gz_pre, *labels
            )
        if last:
            seq.add_block(
                ro.gx_rew, pp.scale_grad(ro.gy_rew, ky), ro.gz_rew, *ro.end_labels
            )

    def finalize(self) -> None:
        """Write the prescription, the gating and the k-space geometry as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": self.NAME,
            "TE": self.ro.te,
            "TR": self.ro.tr,
            "Gating": self.gating,
            "PlotTRsize": self.plot_tr_size,
            "PlotTRstart": self.plot_tr_start,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        if self.gating != "none":
            definitions.update(
                HeartRate=self.heart_rate_bpm,
                ViewsPerSegment=self.views_per_segment,
                NumSegments=self.n_segments,
            )
        if self.gating == "prospective":
            definitions.update(
                CardiacPhases=self.n_phases, TriggerDelay=self.trigger_delay
            )
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Bssfp2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="bssfp_2d.seq"))
