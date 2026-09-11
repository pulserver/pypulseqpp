"""Balanced SSFP 2D Cartesian, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Bssfp2DApp(sequences.SequenceApp):
    """Balanced SSFP 2D Cartesian: one complete train per slice.

    Every axis returns to k = 0 between pulse centres and TE is TR/2
    (:class:`BssfpReadout2D`). Each slice's train is entered through a
    half-flip pulse half a TR ahead of the first excitation and opposite in
    phase to it; excitations then alternate between phase pi and 0, the ADC
    following. ``ONCE`` marks the half-flip (1), the steady state (0) and the
    closing rewind (2). Slices play one after another, since a steady state
    does not survive interleaving.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.bssfp2D_sequence(
    ...     n_x=64, n_y=16, readout_bandwidth_hz=50e3, n_acs=0, n_dummy=0
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

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_gap: float = 0.0,
        flip_angle_deg: float = 45.0,
        tr: float | None = None,
        readout_bandwidth_hz: float = 125e3,
        partial_fourier: float = 1.0,
        acceleration: int = 1,
        n_acs: int = 24,
        n_dummy: int = 10,
        n_gain_calibration_readouts: int | None = None,
    ) -> None:
        """Design the balanced repetition, the slice positions and the line order.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; ``(fov_x, fov_y)`` if a tuple.
        n_x : int, optional
            Readout samples.
        n_y : int, optional
            Phase-encode steps.
        n_slices : int, optional
            Number of slices, each acquired as its own complete train.
        slice_thickness : float, optional
            Slice thickness, in metres.
        slice_gap : float, optional
            Gap between adjacent slices, in metres.
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        tr : float or None, optional
            Repetition time, in seconds; TE is always TR/2. ``None`` is as
            short as possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz. The half-flip pulse needs an
            acquisition block at least as long as the excitation's tail, which
            a lower bandwidth provides.
        partial_fourier : float, optional
            Fraction of the phase-encode extent acquired, in (0.5, 1].
        acceleration : int, optional
            Uniform phase-encode undersampling factor.
        n_acs : int, optional
            Fully sampled autocalibration lines, acquired ahead of the rest.
        n_dummy : int, optional
            Repetitions played without acquiring after the half-flip pulse,
            while the oscillating transient settles.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        """
        system = self.system
        self.fov = (fov, fov) if np.isscalar(fov) else tuple(fov)
        self.matrix = (n_x, n_y, n_slices)
        self.n_dummy = n_dummy
        self.n_gain_calibration_readouts = (
            n_slices
            if n_gain_calibration_readouts is None
            else n_gain_calibration_readouts
        )
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
            readout_bandwidth_hz=readout_bandwidth_hz,
        )
        self.nominal = self.ro.rf.amplitude

        self.lines = pp.calc_sampled_lines(
            n_y,
            acceleration,
            n_acs,
            order="calibration_first",
            partial_fourier=partial_fourier,
        )
        n_calibration = len(
            pp.calc_calibration_lines(n_y, n_acs, partial_fourier=partial_fourier)
        )
        # SEG splits the calibration block that leads the train from the rest.
        self.segment = {}
        for i, line in enumerate(self.lines):
            self.segment.setdefault(line, int(i >= n_calibration))
        acs_start = max(0, n_y // 2 - n_acs // 2)
        self.acs = range(acs_start, min(n_y, acs_start + n_acs))

        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_gap
        )
        self.slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
        self.slice_gap = slice_thickness + slice_gap - self.exc.slice_thickness
        # Half-flip, dummies, one repetition per line and the closing rewind.
        self.duration = n_slices * (n_dummy + len(self.lines) + 1.5) * self.ro.tr

    def _ky(self, line: int | None) -> float:
        n_y = self.matrix[1]
        return 0.0 if line is None else (line - n_y / 2) / (n_y / 2)

    def loop(self) -> None:
        """Play each slice's whole train before the next slice's."""
        shots = [None] * self.n_dummy + list(self.lines)
        for s in range(self.matrix[2]):
            previous = None
            for shot, line in enumerate(shots):
                self.kernel(s, shot, line, previous, last=shot == len(shots) - 1)
                previous = self._ky(line)

    def kernel(
        self,
        s: int,
        shot: int,
        line: int | None,
        previous_ky: float | None,
        last: bool = False,
    ) -> None:
        """One repetition of slice ``s``: the rewind of the previous one, then excite and read.

        ``previous_ky`` is the fractional encode the rewind undoes; ``None``
        opens the slice's train with the half-flip pulse. ``last`` closes it
        with the final rewind, and ``line=None`` plays a dummy.
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
            seq.add_block(rf, ro.gz, *ro.prep_labels)
            seq.add_block(ro.wait_prep, *ro.train_labels)
            rf.amplitude = self.nominal
            seq.add_block(ro.wait_rewind, ro.gz_rew)
        else:
            seq.add_block(ro.gx_rew, pp.scale_grad(ro.gy_rew, previous_ky), ro.gz_rew)

        alternation = np.pi * ((shot + 1) % 2)
        rf.phase_offset = alternation + centre_phase
        ro.adc.phase_offset = alternation
        seq.add_block(rf, ro.gz)

        ky = self._ky(line)
        if line is None:
            seq.add_block(ro.gx, pp.scale_grad(ro.gy_pre, ky), ro.gz_pre)
        else:
            labels = self.labels(
                LIN=line, SLC=s, IMA=line in self.acs, SEG=self.segment[line]
            )
            seq.add_block(
                ro.gx, ro.adc, pp.scale_grad(ro.gy_pre, ky), ro.gz_pre, *labels
            )
        if last:
            seq.add_block(
                ro.gx_rew, pp.scale_grad(ro.gy_rew, ky), ro.gz_rew, *ro.end_labels
            )

    def finalize(self) -> None:
        """Write the prescription and the k-space geometry as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": self.NAME,
            "TE": self.ro.te,
            "TR": self.ro.tr,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Bssfp2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="bssfp_2d.seq"))
