"""RF-spoiled 2D Cartesian gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._schedules import make_rf_spoiling_schedule


def sampled_lines(
    n: int, ry: int, n_acs_y: int, partial_fourier: float
) -> tuple[list[int], set[int]]:
    """Return the phase-encode lines in play order, and the calibration lines.

    The lattice keeps every line ``i`` with ``(i - n // 2) % ry == 0``, so the
    centre line is always acquired. Partial Fourier drops the lines before
    ``n - round(partial_fourier * n)``. The calibration block, centred on the
    same line, leads; a fully sampled scan has none.
    """
    first = n - round(partial_fourier * n)
    start = n // 2 - (n_acs_y if ry > 1 else 0) // 2
    stop = n // 2 + ((n_acs_y if ry > 1 else 0) + 1) // 2
    calibration = list(range(max(start, first), min(stop, n)))
    lattice = [
        i for i in range(first, n) if (i - n // 2) % ry == 0 and i not in calibration
    ]
    return calibration + lattice, set(calibration)


class Gre2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D Cartesian gradient echo.

    One line per repetition. Slices one TR cannot hold are dealt round-robin
    into packets, played one after another; within a packet the even slices
    are excited before the odd ones. Every packet starts with its own dummy
    repetitions, and the last slice of a packet waits out the rest of the TR.
    Under undersampling the calibration lines are acquired first, marked
    ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_x=32, n_y=16, tr=None)
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
    #: Quadratic RF spoiling phase increment (degrees), counted per slice.
    RF_SPOILING_INCREMENT_DEG = 117.0
    #: Dephasing left on the readout axis at the end of each repetition, in
    #: cycles across one voxel.
    SPOILING_CYCLES = 4.0

    def init_sequence(
        self,
        fov_x: float = 220e-3,
        fov_y: float = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_spacing: float = 0.0,
        flip_angle_deg: float = 12.0,
        te: float | None = 8e-3,
        tr: float | None = 250e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        *,
        n_dummy: int = 16,
        readout_oversampling: float = 2.0,
        n_acs_y: int = 24,
    ) -> None:
        """Design the pulse, the readout, the slice packets and the line order.

        Parameters
        ----------
        fov_x, fov_y : float, default=0.22
            Field of view along the readout and the phase encode (m).
        n_x : int, default=128
            Readout matrix size.
        n_y : int, default=128
            Phase-encode matrix size.
        n_slices : int, default=1
            Number of slices.
        slice_thickness : float, default=0.005
            Slice thickness (m).
        slice_spacing : float, default=0.0
            Gap between adjacent slices (m); zero is contiguous.
        flip_angle_deg : float, default=12.0
            Excitation flip angle (degrees).
        te : float | None, default=0.008
            Echo time (s). ``None`` is as short as the readout admits.
        tr : float | None, default=0.25
            Repetition time between successive excitations of one slice (s).
            ``None`` is as short as possible, and puts every slice in one
            packet.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz). What was achieved is the
            readout module's ``bandwidth_hz``.
        ry : int, default=1
            Phase-encode undersampling: one line in every ``ry`` is acquired,
            the centre line among them.
        partial_fourier_x : float, default=1.0
            Fraction of the echo acquired, in ``[0.75, 1]``. Truncates the
            samples before the echo, which shortens the minimum TE.
        partial_fourier_y : float, default=1.0
            Fraction of the phase-encode extent acquired, in ``[0.75, 1]``.
            Truncates the lines before the centre.
        n_dummy : int, default=16
            Non-acquiring repetitions before the first line of each packet.
        readout_oversampling : float, default=2.0
            Readout oversampling factor, at least one.
        n_acs_y : int, default=24
            Fully sampled calibration lines at the centre of k-space, acquired
            ahead of the rest when ``ry > 1``.

        Raises
        ------
        ValueError
            If a partial Fourier fraction is outside ``[0.75, 1]``, ``ry`` is
            below one, or the TR cannot hold one slice.
        """
        self.n_dummy = n_dummy
        for name, fraction in (
            ("partial_fourier_x", partial_fourier_x),
            ("partial_fourier_y", partial_fourier_y),
        ):
            if not 0.75 <= fraction <= 1.0:
                raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
        if ry < 1:
            raise ValueError(f"ry must be at least 1, got {ry}")

        system = self.system
        self.fov = (fov_x, fov_y)
        self.matrix = (n_x, n_y, n_slices)
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
            partial_echo=partial_fourier_x,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=self.SPOILING_CYCLES,
        )

        # Slices one TR cannot hold are dealt round-robin into packets, so the
        # slices of a packet sit a packet count apart and neighbours are never
        # excited back to back.
        self.raster = system.block_duration_raster
        shot = self.ro.duration + self.raster
        per_packet = n_slices if tr is None else max(1, int(tr / shot + 1e-9))
        n_packets = -(-n_slices // per_packet)
        packets = [range(start, n_slices, n_packets) for start in range(n_packets)]
        self.packets = [[*packet[::2], *packet[1::2]] for packet in packets]

        # Every shot closes with a pure delay: one raster, and on the last slice
        # of a packet whatever is left of the TR. Packets of different sizes
        # then differ in a duration, not a definition, so the scan is one
        # repeating shot whatever the slices divide into.
        cycle = tr if tr is not None else max(map(len, self.packets)) * shot
        self.pads = {
            size: pp.round_to_raster(cycle - size * shot, self.raster) + self.raster
            for size in {len(packet) for packet in self.packets}
        }
        if min(self.pads.values()) < self.raster:
            raise ValueError(
                f"the requested TR of {cycle * 1e3:.3f} ms is shorter than the "
                f"{shot * 1e3:.3f} ms one slice takes"
            )
        # The last shot of a packet closes with its pad rather than the raster.
        packet_time = {n: n * shot - self.raster + pad for n, pad in self.pads.items()}
        self.repetition_time = max(packet_time.values())

        self.lines, self.calibration = sampled_lines(
            n_y, ry, n_acs_y, partial_fourier_y
        )
        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_spacing
        )
        self.slab_thickness = (
            n_slices * (slice_thickness + slice_spacing) - slice_spacing
        )
        self.slice_gap = slice_thickness + slice_spacing - self.exc.slice_thickness
        self.duration = (self.n_dummy + len(self.lines)) * sum(
            packet_time[len(packet)] for packet in self.packets
        )

    def loop(self) -> None:
        """Play each packet: its dummies, then every line at each of its slices.

        Every slice sees the same line order, so the RF spoiling phase of a
        slice's ``k``-th excitation is the schedule's ``k``-th entry.
        """
        lines = [None] * self.n_dummy + list(self.lines)
        phases = make_rf_spoiling_schedule(
            len(lines), increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        )
        for packet in self.packets:
            for line, phase in zip(lines, phases, strict=True):
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    pad = self.pads[len(packet)] if last else self.raster
                    self.kernel(s, line, phase, pad)

    def kernel(self, s: int, line: int | None, phase: float, pad: float) -> None:
        """One excitation of slice ``s`` at one line; ``line=None`` plays a dummy."""
        rf, gz, ro, seq = self.exc.rf, self.exc.gz, self.ro, self.seq
        rf.freq_offset = self.exc.selection_amplitude * self.positions[s]
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center
        ro.adc.phase_offset = phase

        if line is None:
            ky, labels = 0.0, self.labels(SLC=s, ONCE=1)
        else:
            n_y = self.matrix[1]
            ky = (line - n_y // 2) / (n_y / 2)
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
