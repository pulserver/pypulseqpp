"""2D Cartesian spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Se2DApp(sequences.SequenceApp):
    """Multi-slice 2D Cartesian spin echo: one line per excitation.

    A slice-selective SLR 90, one SLR 180 between crushers half a TE later,
    and one frequency-encoded line at the echo. Both pulses are offset to the
    slice, each at its own selection gradient. Calibration lines precede
    imaging lines; slices one TR cannot hold are dealt into passes as
    :mod:`gre2D_sequence` deals them.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se2D_sequence(n_x=32, n_y=16, n_acs=0, te=15e-3, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulse. The
    #: refocusing selection amplitude, which slice offsets are converted
    #: against, is ``TIME_BW_PRODUCT / (PULSE_DURATION * thickness)``.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_gap: float = 0.0,
        slice_order: str = "interleaved",
        te: float | None = 15e-3,
        tr: float | None = 500e-3,
        readout_bandwidth_hz: float = 250e3,
        partial_echo: float = 1.0,
        partial_fourier: float = 1.0,
        acceleration: int = 1,
        n_acs: int = 24,
        n_dummy: int = 0,
        n_gain_calibration_readouts: int | None = None,
        crusher_cycles: float = 4.0,
        spoiling_cycles: float = 4.0,
    ) -> None:
        """Design the pulses, the readout, the slice passes and the line order.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; one value for both axes, or
            ``(fov_x, fov_y)``.
        n_x : int, optional
            Readout samples.
        n_y : int, optional
            Phase-encode steps.
        n_slices : int, optional
            Number of slices.
        slice_thickness : float, optional
            Slice thickness, in metres.
        slice_gap : float, optional
            Gap between adjacent slices, in metres.
        slice_order : str, optional
            Order the slices of one pass are excited in, as
            :func:`pypulseqpp.calc_traversal_order` accepts.
        te : float or None, optional
            Echo time, in seconds, excitation centre to echo, with the
            refocusing pulse at its midpoint. ``None`` is as short as possible.
        tr : float or None, optional
            Repetition time, in seconds, between successive excitations of one
            slice. ``None`` is as short as possible, and puts every slice in
            one pass.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        partial_echo : float, optional
            Fraction of the echo acquired, in (0.5, 1].
        partial_fourier : float, optional
            Fraction of the phase-encode extent acquired, in (0.5, 1].
        acceleration : int, optional
            Uniform phase-encode undersampling factor.
        n_acs : int, optional
            Fully sampled autocalibration lines at the centre of k-space,
            acquired ahead of the rest of the scan.
        n_dummy : int, optional
            Non-acquiring repetitions before the first line of each pass.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        crusher_cycles : float, optional
            Cycles of dephasing each crusher beside the refocusing pulse winds
            across one voxel.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the readout axis at the end of each
            repetition, counted across one voxel.
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
            90.0,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        self.ref = sequences.SpatialSelectiveRefocusing(
            system,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            spoiling_cycles=crusher_cycles,
        )
        # The refocusing gradient's amplitude is its crusher peak; the pulse
        # selects on the plateau between the crushers.
        self.ref_amplitude = self.TIME_BW_PRODUCT / (
            self.PULSE_DURATION * slice_thickness
        )
        self.ref_phase = float(self.ref.rf_ref.phase_offset)

        # TE is solved in two halves about the 180. The readout owns the second,
        # from the 180's centre to the echo; a delay before the 180 sets the
        # first, which is never shorter than the excitation's blocks allow.
        half_floor = self.exc.duration - self.exc.center + self.ref.center

        def readout(half_te: float | None):
            return sequences.LineReadout2D(
                system,
                self.ref.rf_ref,
                self.ref.gz,
                fov=self.fov,
                matrix=(n_x, n_y),
                te=half_te,
                partial_echo=partial_echo,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=spoiling_cycles,
            )

        te_min = 2 * max(readout(None).echo_time, half_floor)
        if te is not None and te < te_min - 1e-9:
            raise ValueError(
                f"TE {te * 1e3:.2f} ms is shorter than the {te_min * 1e3:.2f} ms "
                "the excitation and the readout admit"
            )
        half_te = (te_min if te is None else te) / 2
        self.ro = readout(half_te)
        self.raster = system.block_duration_raster
        wait = pp.round_to_raster(self.ro.echo_time - half_floor, self.raster)
        self.wait_half_te = pp.make_delay(wait) if wait > 0 else None
        self.echo_time = half_floor + wait + self.ro.echo_time

        # Every shot closes with a pure delay: one raster, and on the last slice
        # of a pass whatever is left of the TR, as gre2D_sequence closes them.
        shot = self.exc.duration + wait + self.ro.duration + self.raster
        per_pass = n_slices if tr is None else max(1, int(tr / shot + 1e-9))
        n_passes = -(-n_slices // per_pass)
        groups = [list(range(start, n_slices, n_passes)) for start in range(n_passes)]
        self.passes = [
            [group[i] for i in pp.calc_traversal_order(len(group), slice_order)]
            for group in groups
        ]
        cycle = tr if tr is not None else max(map(len, self.passes)) * shot
        self.pads = {
            size: pp.round_to_raster(cycle - size * shot, self.raster) + self.raster
            for size in {len(group) for group in self.passes}
        }
        if min(self.pads.values()) < self.raster:
            raise ValueError(
                f"TR {cycle * 1e3:.1f} ms is shorter than one repetition takes "
                f"({shot * 1e3:.1f} ms)"
            )
        # The last slice of a pass plays its pad in place of the closing raster.
        pass_time = {n: n * shot - self.raster + pad for n, pad in self.pads.items()}
        self.repetition_time = max(pass_time.values())

        self.lines = pp.calc_sampled_lines(
            n_y,
            acceleration,
            n_acs,
            order="calibration_first",
            partial_fourier=partial_fourier,
        )
        self.calibration = set(
            pp.calc_calibration_lines(n_y, n_acs, partial_fourier=partial_fourier)
        )
        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_gap
        )
        self.slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
        self.slice_gap = slice_thickness + slice_gap - self.exc.slice_thickness
        self.duration = (n_dummy + len(self.lines)) * sum(
            pass_time[len(group)] for group in self.passes
        )

    def loop(self) -> None:
        """Play each pass: its dummies, then every line at each of its slices."""
        lines = [None] * self.n_dummy + list(self.lines)
        for group in self.passes:
            for line in lines:
                for i, s in enumerate(group):
                    last = i == len(group) - 1
                    self.kernel(s, line, self.pads[len(group)] if last else self.raster)

    def kernel(self, s: int, line: int | None, pad: float) -> None:
        """One slice excitation at one line; ``line=None`` plays a dummy."""
        exc, ref, ro, seq = self.exc, self.ref, self.ro, self.seq
        position = self.positions[s]
        exc.rf.freq_offset = exc.gz.amplitude * position
        exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ref.rf_ref.freq_offset = self.ref_amplitude * position
        ref.rf_ref.phase_offset = (
            self.ref_phase - 2 * np.pi * ref.rf_ref.freq_offset * ref.rf_ref.center
        )

        if line is None:
            ky, labels = 0.0, self.labels(SLC=s, ONCE=1)
        else:
            n_y = self.matrix[1]
            ky = (line - n_y / 2) / (n_y / 2)
            calibrating = line in self.calibration
            labels = self.labels(
                SLC=s, LIN=line, IMA=calibrating, SEG=not calibrating, ONCE=0
            )

        seq.add_block(exc.rf, exc.gz, *labels)
        seq.add_block(exc.gz_reph)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        seq.add_block(ref.rf_ref, ref.gz)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky))
        seq.add_block(ro.gx, *([] if line is None else [ro.adc]))
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky))
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription and the k-space geometry as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": self.NAME,
            "TE": self.echo_time,
            "TR": self.repetition_time,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Se2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_2d.seq"))
