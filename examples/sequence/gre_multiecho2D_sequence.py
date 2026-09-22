"""RF-spoiled multi-echo 2D Cartesian gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._schedules import make_rf_spoiling_schedule


class GreMultiecho2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice multi-echo 2D Cartesian gradient echo.

    One line per repetition, read at ``n_echoes`` echo times: monopolar, with
    a flyback rewinder after every echo but the last, or bipolar, with even
    echoes read backwards. Each acquisition carries its echo index as
    ``ECO``. Slices one TR cannot hold are dealt round-robin into packets,
    played one after another; within a packet the even slices are excited
    before the odd ones. Every packet starts with its own dummy
    repetitions, and the last slice of a packet waits out the rest of the TR.
    Under undersampling the calibration lines are acquired first, marked
    ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_multiecho2D_sequence(
    ...     n_x=32, n_y=16, n_echoes=3, tr=None
    ... )
    >>> seq.check_timing()[0]
    True
    >>> len(seq.definitions["TE"]), seq.definitions["Name"]
    (3, 'gre_multiecho_2d')
    """

    NAME = "gre_multiecho_2d"
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
        te: float | None = None,
        tr: float | None = 250e-3,
        n_echoes: int = 4,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        *,
        n_dummy: int = 16,
        readout_oversampling: float = 2.0,
        n_acs_y: int = 24,
        echo_spacing: float | None = None,
        flyback: bool = True,
    ) -> None:
        """Design the pulse, the echo train, the slice packets and the line order.

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
        te : float | None, default=None
            First echo time (s). ``None`` is as short as the readout admits.
        tr : float | None, default=0.25
            Repetition time between successive excitations of one slice (s).
            ``None`` is as short as possible, and puts every slice in one
            packet.
        n_echoes : int, default=4
            Echoes per excitation.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz). What was achieved is the
            readout module's ``bandwidth_hz``.
        ry : int, default=1
            Phase-encode undersampling: one line in every ``ry`` is acquired,
            the centre line among them.
        partial_fourier_x : float, default=1.0
            Fraction of the echo acquired, in ``[0.75, 1]``. Truncates the
            samples before the echo, which shortens the minimum TE. A bipolar
            train needs a full echo.
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
        echo_spacing : float | None, default=None
            Echo spacing (s). ``None`` is as short as the readout admits; a
            longer one adds a delay after every echo but the last, following
            the flyback rewinder of a monopolar train.
        flyback : bool, default=True
            Monopolar echo train, rewound after every echo but the last so
            every echo is read the same way; off, a bipolar train. A bipolar
            train has a shorter echo spacing and reads even echoes backwards.

        Raises
        ------
        ValueError
            If a partial Fourier fraction is outside ``[0.75, 1]``, ``ry`` is
            below one, the echo spacing is shorter than the readout admits, a
            bipolar train is given a partial echo or an odd sample count, or
            the TR cannot hold one slice.
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
            n_echoes=n_echoes,
            flyback=flyback,
            echo_spacing=echo_spacing,
        )
        self.n_echoes, self.flyback = n_echoes, flyback
        self.echo_times = [
            self.ro.echo_time + i * self.ro.echo_spacing for i in range(n_echoes)
        ]

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

        calibrating, lattice = pp.calc_sampled_lines(
            n_y, ry, n_acs_y, partial_fourier=partial_fourier_y
        )
        # The calibration block leads, so a reconstruction can estimate
        # coil sensitivities while the rest is still arriving.
        self.lines = [*calibrating, *lattice]
        self.calibration = set(calibrating)
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
        """One excitation of slice ``s`` and its echo train at one line.

        ``line=None`` plays a dummy.
        """
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
        for echo in range(self.n_echoes):
            if echo and self.flyback:
                seq.add_block(ro.gx_flyback)
            if echo and hasattr(ro, "wait_esp"):
                seq.add_block(ro.wait_esp)
            lobe = ro.gx if self.flyback or echo % 2 == 0 else ro.gx_rev
            if line is None:
                seq.add_block(lobe)
            else:
                seq.add_block(lobe, ro.adc, *self.labels(ECO=echo))
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky))
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription, the echo times and the k-space geometry."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV; which way the logical axes point is the
        # interpreter's business.
        n_x, n_y, n_slices = self.matrix
        definitions = {
            "FOV": [*self.fov, self.slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "Name": self.NAME,
            "TE": self.echo_times,
            "TR": self.repetition_time,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreMultiecho2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_multiecho_2d.seq"))
