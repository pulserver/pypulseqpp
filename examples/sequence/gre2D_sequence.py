"""RF-spoiled 2D Cartesian gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Gre2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D Cartesian gradient echo.

    Calibration lines precede imaging lines. Slices that one TR cannot hold
    are dealt round-robin into passes, each played to its end before the next;
    every pass starts with its own dummy repetitions.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_x=32, n_y=16, n_acs=0, n_dummy=0, tr=None)
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 1.0], 'gre_2d')
    """

    NAME = "gre_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse. The selection amplitude, which slice
    #: offsets are converted against, is ``TIME_BW_PRODUCT / (PULSE_DURATION *
    #: thickness)``.
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
        flip_angle_deg: float = 12.0,
        te: float | None = 8e-3,
        tr: float | None = 250e-3,
        readout_bandwidth_hz: float = 250e3,
        partial_echo: float = 1.0,
        partial_fourier: float = 1.0,
        acceleration: int = 1,
        n_acs: int = 24,
        n_dummy: int = 16,
        rf_spoiling_increment_deg: float = 117.0,
        spoiling_cycles: float = 4.0,
    ) -> None:
        """Design the pulse, the readout, the slice passes and the line order.

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
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        te : float or None, optional
            Echo time, in seconds. ``None`` is as short as the readout admits.
        tr : float or None, optional
            Repetition time, in seconds, between successive excitations of one
            slice. ``None`` is as short as possible, and puts every slice in
            one pass.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz. What was achieved is on the
            readout module's ``bandwidth_hz``, and is generally lower.
        partial_echo : float, optional
            Fraction of the echo acquired, in (0.5, 1]. Truncates the samples
            before the echo, which shortens the minimum TE.
        partial_fourier : float, optional
            Fraction of the phase-encode extent acquired, in (0.5, 1].
            Truncates the lines before the centre, which shortens the scan.
        acceleration : int, optional
            Uniform phase-encode undersampling factor.
        n_acs : int, optional
            Fully sampled autocalibration lines at the centre of k-space,
            acquired ahead of the rest of the scan.
        n_dummy : int, optional
            Non-acquiring repetitions before the first line of each pass,
            which let the RF-spoiled transient settle.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the readout axis at the end of each
            repetition, counted across one voxel.
        """
        system = self.system
        self.fov = (fov, fov) if np.isscalar(fov) else tuple(fov)
        self.matrix = (n_x, n_y, n_slices)
        self.n_dummy = n_dummy
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        self.ro = sequences.LineReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            self.exc.gz_reph,
            fov=self.fov,
            matrix=(n_x, n_y),
            te=te,
            partial_echo=partial_echo,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
        )

        # Slices one TR cannot hold are dealt round-robin into passes, so the
        # slices of a pass sit a pass count apart and neighbours are never
        # excited back to back.
        self.raster = system.block_duration_raster
        shot = self.ro.duration + self.raster
        per_pass = n_slices if tr is None else max(1, int(tr / shot + 1e-9))
        n_passes = -(-n_slices // per_pass)
        groups = [list(range(start, n_slices, n_passes)) for start in range(n_passes)]
        self.passes = [
            [group[i] for i in pp.calc_traversal_order(len(group), slice_order)]
            for group in groups
        ]

        # Every shot closes with a pure delay: one raster, and on the last slice
        # of a pass whatever is left of the TR. Passes of different sizes then
        # differ in a duration, not a definition, so the scan is one repeating
        # shot whatever the slices divide into.
        cycle = tr if tr is not None else max(map(len, self.passes)) * shot
        self.pads = {
            size: pp.round_to_raster(cycle - size * shot, self.raster) + self.raster
            for size in {len(group) for group in self.passes}
        }
        if min(self.pads.values()) < self.raster:
            raise ValueError(
                f"the requested TR of {cycle * 1e3:.3f} ms is shorter than the "
                f"{max(self.pads) * shot * 1e3:.3f} ms the slices of a pass take"
            )
        # The last shot of a pass closes with its pad rather than the raster.
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
        self.spoiling_increment = np.deg2rad(rf_spoiling_increment_deg)
        self.duration = (n_dummy + len(self.lines)) * sum(
            pass_time[len(group)] for group in self.passes
        )

    def loop(self) -> None:
        """Play each pass: its dummies, then every line at each of its slices."""
        lines = [None] * self.n_dummy + list(self.lines)
        phases = iter(
            pp.make_rf_spoiling_schedule(
                len(lines) * self.matrix[2], increment=self.spoiling_increment
            )
        )
        for group in self.passes:
            for line in lines:
                for i, s in enumerate(group):
                    last = i == len(group) - 1
                    pad = self.pads[len(group)] if last else self.raster
                    self.kernel(s, line, next(phases), pad)

    def kernel(self, s: int, line: int | None, phase: float, pad: float) -> None:
        """One slice excitation at one line; ``line=None`` plays a dummy."""
        rf, gz, ro, seq = self.exc.rf, self.exc.gz, self.ro, self.seq
        rf.freq_offset = self.exc.selection_amplitude * self.positions[s]
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center
        ro.adc.phase_offset = phase

        if line is None:
            ky, labels = 0.0, self.labels(SLC=s, ONCE=1)
        else:
            n_y = self.matrix[1]
            ky = (line - n_y / 2) / (n_y / 2)
            calibrating = line in self.calibration
            labels = self.labels(
                SLC=s, LIN=line, IMA=calibrating, SEG=not calibrating, ONCE=0
            )

        seq.add_block(rf, gz, *labels)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, ro.gz_reph)
            seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky))
        else:
            seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), ro.gz_reph)
        seq.add_block(ro.gx, *([] if line is None else [ro.adc]))
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky))
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription and the k-space geometry as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV; which way the logical axes point is the
        # interpreter's business.
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Gre2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_2d.seq"))
