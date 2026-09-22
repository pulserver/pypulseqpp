"""RF-spoiled 2D PROPELLER gradient echo, multi-slice."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._schedules import make_rf_spoiling_schedule


class GrePropeller2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D PROPELLER gradient echo: one blade line per repetition.

    A blade is ``blade_width`` Cartesian lines centred on k = 0, and every
    blade is the same lines turned by a rotation extension. The Nyquist set is
    ``ceil(pi * n / (2 * blade_width))`` blades spread evenly over half a turn;
    every ``ry``-th of them is played, in order, each line by line. Slices are
    dealt into packets, and ordered within one, as :mod:`gre2D_sequence` deals
    them. Every acquisition carries its line as ``LIN``, its blade as ``SEG``
    and its slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_propeller2D_sequence(n=32, blade_width=8, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "gre_propeller_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Readout oversampling factor.
    READOUT_OVERSAMPLING = 2.0
    #: Quadratic RF spoiling phase increment (degrees), counted per slice.
    RF_SPOILING_INCREMENT_DEG = 117.0
    #: Dephasing left on the slice axis at the end of each repetition, in
    #: cycles across the slice.
    SPOILING_CYCLES = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n: int = 128,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_spacing: float = 0.0,
        flip_angle_deg: float = 12.0,
        te: float | None = 8e-3,
        tr: float | None = 250e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        *,
        n_dummy: int = 16,
        blade_width: int = 16,
    ) -> None:
        """Design the pulse, the readout, the slice packets and the blade angles.

        Parameters
        ----------
        fov : float, default=0.22
            Isotropic in-plane field of view (m).
        n : int, default=128
            In-plane matrix size.
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
            Requested receiver bandwidth (Hz).
        ry : int, default=1
            Angular undersampling: one blade in every ``ry`` of the Nyquist set
            is played.
        n_dummy : int, default=16
            Non-acquiring repetitions, at the first blade's angle, before each
            packet.
        blade_width : int, default=16
            Phase-encode lines per blade, at most ``n``.

        Raises
        ------
        ValueError
            If ``ry`` is below one, ``blade_width`` is outside ``[1, n]``, or
            the TR cannot hold one slice.
        """
        self.n_dummy = n_dummy
        if ry < 1:
            raise ValueError(f"ry must be at least 1, got {ry}")
        if not 1 <= blade_width <= n:
            raise ValueError(f"blade_width must lie in [1, {n}], got {blade_width}")

        system = self.system
        self.fov, self.matrix = fov, (n, n, n_slices)
        self.blade_width = blade_width
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
            fov=(fov, fov),
            matrix=(n, n),
            te=te,
            oversampling=self.READOUT_OVERSAMPLING,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )
        # The in-plane axes turn with the blade, so a spoiler on them would
        # point a different way from one blade to the next: it plays on z.
        self.gz_spoil, _, _ = pp.make_crusher(
            self.SPOILING_CYCLES, slice_thickness, "z", system=system
        )
        # A blade turned by half a turn is the same blade, which the Nyquist
        # set divides evenly.
        n_nyquist = math.ceil(np.pi * n / (2 * blade_width))
        self.angles = np.pi * np.arange(0, n_nyquist, ry) / n_nyquist
        self.rotations = [pp.make_rotation(float(angle)) for angle in self.angles]

        # Slices one TR cannot hold are dealt round-robin into packets, even
        # slices of a packet first. Every shot closes with a pure delay: one
        # raster, and on the last slice of a packet whatever is left of the TR.
        self.raster = system.block_duration_raster
        spoil = pp.ceil_to_raster(pp.calc_duration(self.gz_spoil), self.raster)
        shot = self.ro.duration + spoil + self.raster
        per_packet = n_slices if tr is None else max(1, int(tr / shot + 1e-9))
        n_packets = -(-n_slices // per_packet)
        packets = [range(start, n_slices, n_packets) for start in range(n_packets)]
        self.packets = [[*packet[::2], *packet[1::2]] for packet in packets]
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
        packet_time = {n: n * shot - self.raster + pad for n, pad in self.pads.items()}
        self.repetition_time = max(packet_time.values())

        self.views = [
            (blade, line)
            for blade in range(len(self.angles))
            for line in range(blade_width)
        ]
        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_spacing
        )
        self.slab_thickness = (
            n_slices * (slice_thickness + slice_spacing) - slice_spacing
        )
        self.slice_gap = slice_thickness + slice_spacing - self.exc.slice_thickness

    def loop(self) -> None:
        """Play each packet: its dummies, then every blade line at each of its slices."""
        views = [None] * self.n_dummy + self.views
        phases = make_rf_spoiling_schedule(
            len(views), increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        )
        for packet in self.packets:
            for view, phase in zip(views, phases, strict=True):
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    pad = self.pads[len(packet)] if last else self.raster
                    self.kernel(s, view, phase, pad)

    def kernel(
        self, s: int, view: tuple[int, int] | None, phase: float, pad: float
    ) -> None:
        """One excitation of slice ``s`` at ``(blade, line)``; ``None`` plays a dummy.

        A dummy plays the first blade's centre line without its ADC. Every
        block that drives an in-plane gradient carries the blade's rotation.

        Parameters
        ----------
        s : int
            Slice index, counting from 0.
        view : tuple of int or None
            The blade and the line within it, or None for a dummy.
        phase : float
            RF and ADC phase for this repetition, in radians.
        pad : float
            Delay closing the repetition, in s.
        """
        rf, gz, ro, seq = self.exc.rf, self.exc.gz, self.ro, self.seq
        rf.freq_offset = self.exc.selection_amplitude * self.positions[s]
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center
        ro.adc.phase_offset = phase

        if view is None:
            blade, ky = 0, 0.0
            labels = self.labels(SLC=s, ONCE=1)
        else:
            blade, line = view
            ky = (line - self.blade_width // 2) / (self.matrix[1] / 2)
            labels = self.labels(SLC=s, SEG=blade, LIN=line, ONCE=0)
        rotation = self.rotations[blade]
        gy_pre = pp.scale_grad(ro.gy_pre, ky)

        seq.add_block(rf, gz, *labels)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, ro.gz_reph)
            seq.add_block(ro.gx_pre, gy_pre, rotation)
        else:
            seq.add_block(ro.gx_pre, gy_pre, ro.gz_reph, rotation)
        seq.add_block(ro.gx, *([] if view is None else [ro.adc]), rotation)
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), rotation)
        seq.add_block(self.gz_spoil)
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription, the blade set and the timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": [self.fov, self.fov, self.slab_thickness],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "Trajectory": "propeller",
            "BladeWidth": self.blade_width,
            "NumBlades": len(self.angles),
            "kSpaceCenterLine": self.blade_width // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GrePropeller2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_propeller_2d.seq"))
