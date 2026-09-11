"""2D Cartesian fast spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Fse2DApp(sequences.SequenceApp):
    """Multi-slice 2D Cartesian fast spin echo: one CPMG train per excitation.

    Every echo is phase-encoded by its own trapezoid and unwound before the
    next refocusing pulse. The lines are dealt into trains in rolled linear
    order, so the centre of k-space is acquired at the echo nearest the
    requested effective TE. Every selective pulse is offset to its slice, each
    at its own selection gradient; slices are dealt into passes as
    :mod:`gre2D_sequence` deals them.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.fse2D_sequence(n_x=64, n_y=16, etl=4, te=20e-3, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "fse_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulses.
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
        etl: int = 8,
        te: float | None = 80e-3,
        tr: float | None = 2000e-3,
        readout_bandwidth_hz: float = 250e3,
        partial_fourier: float = 1.0,
        acceleration: int = 1,
        n_acs: int = 24,
        n_dummy: int = 0,
        n_gain_calibration_readouts: int | None = None,
        crusher_cycles: float = 4.0,
        readout_crusher_cycles: float = 0.0,
    ) -> None:
        """Design the train, the slice passes and the lines each train encodes.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; ``(fov_x, fov_y)`` if a tuple.
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
        etl : int, optional
            Echo train length: lines per excitation.
        te : float or None, optional
            Effective echo time, in seconds: the echo the centre of k-space is
            acquired at, rounded onto the train's echo grid. ``None`` is the
            first echo.
        tr : float or None, optional
            Repetition time, in seconds, between successive excitations of one
            slice. ``None`` is as short as possible, and puts every slice in
            one pass.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        partial_fourier : float, optional
            Fraction of the phase-encode extent acquired, in (0.5, 1].
        acceleration : int, optional
            Uniform phase-encode undersampling factor.
        n_acs : int, optional
            Fully sampled autocalibration lines at the centre of k-space.
        n_dummy : int, optional
            Trains played without acquiring before the first train of each
            pass.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        crusher_cycles : float, optional
            Cycles of dephasing each crusher beside a refocusing pulse winds
            across one voxel.
        readout_crusher_cycles : float, optional
            Read-axis crushing each side of every acquisition, in cycles across
            one voxel.
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
        self.fse = fse = sequences.FseReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            self.exc.gz_reph,
            rf_ref=self.ref.rf_ref,
            gz_ref=self.ref.gz,
            fov=self.fov,
            matrix=(n_x, n_y),
            etl=etl,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=readout_crusher_cycles,
        )
        self.ref_phase = float(fse.rf_ref.phase_offset)
        self.n_center = 0 if te is None else int(np.argmin(abs(fse.echo_times - te)))

        # Every train closes with a pure delay: one raster, and on the last
        # slice of a pass whatever is left of the TR, as gre2D_sequence closes
        # its shots.
        self.raster = system.block_duration_raster
        shot = fse.duration + self.raster
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
                f"TR {cycle * 1e3:.1f} ms is shorter than one train takes "
                f"({shot * 1e3:.1f} ms)"
            )
        # The last slice of a pass plays its pad in place of the closing raster.
        pass_time = {n: n * shot - self.raster + pad for n, pad in self.pads.items()}
        self.repetition_time = max(pass_time.values())

        lines = pp.calc_sampled_lines(
            n_y, acceleration, n_acs, partial_fourier=partial_fourier
        )
        self.trains = [
            [None if i is None else lines[i] for i in train]
            for train in pp.make_linear_order(
                lines, etl, center=(n_y / 2,), center_echo=self.n_center, pad=True
            )
        ]
        self.calibration = set(
            pp.calc_calibration_lines(n_y, n_acs, partial_fourier=partial_fourier)
        )
        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_gap
        )
        self.slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
        self.slice_gap = slice_thickness + slice_gap - self.exc.slice_thickness
        self.duration = (n_dummy + len(self.trains)) * sum(
            pass_time[len(group)] for group in self.passes
        )

    def loop(self) -> None:
        """Play each pass: its dummy trains, then every train at each of its slices."""
        blank = [None] * self.fse.etl
        trains = [(blank, True)] * self.n_dummy + [(t, False) for t in self.trains]
        for group in self.passes:
            for lines, dummy in trains:
                for i, s in enumerate(group):
                    last = i == len(group) - 1
                    pad = self.pads[len(group)] if last else self.raster
                    self.kernel(s, lines, pad, dummy=dummy)

    def kernel(
        self, s: int, lines: list[int | None], pad: float, dummy: bool = False
    ) -> None:
        """One echo train at slice ``s`` over ``lines``; ``None`` plays an echo unencoded."""
        fse, seq = self.fse, self.seq
        n_y = self.matrix[1]
        position = self.positions[s]
        # Each pulse selects at its own plateau; a crushed refocusing
        # gradient's amplitude is its crusher peak.
        fse.rf.freq_offset = self.exc.selection_amplitude * position
        fse.rf.phase_offset = -2 * np.pi * fse.rf.freq_offset * fse.rf.center
        fse.rf_ref.freq_offset = self.ref.selection_amplitude * position
        fse.rf_ref.phase_offset = (
            self.ref_phase - 2 * np.pi * fse.rf_ref.freq_offset * fse.rf_ref.center
        )

        seq.add_block(fse.rf, fse.gz, *self.labels(SLC=s, ONCE=dummy))
        seq.add_block(fse.gx_pre, fse.gz_reph)
        for echo, line in enumerate(lines):
            seq.add_block(fse.rf_ref, fse.gz_ref)
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(fse.wait_esp1)
            ky = 0.0 if line is None else (line - n_y / 2) / (n_y / 2)
            seq.add_block(fse.gx_bridge_pre, pp.scale_grad(fse.gy_pre, ky))
            if line is None:
                seq.add_block(fse.gx)
            else:
                labels = self.labels(LIN=line, IMA=line in self.calibration)
                seq.add_block(fse.gx, fse.adc, *labels)
            seq.add_block(fse.gx_bridge_post, pp.scale_grad(fse.gy_rew, ky))
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription, the echo timing and the k-space geometry."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": self.NAME,
            "TE": float(self.fse.echo_times[self.n_center]),
            "TR": self.repetition_time,
            "EchoSpacing": self.fse.esp,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.fse.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Fse2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="fse_2d.seq"))
