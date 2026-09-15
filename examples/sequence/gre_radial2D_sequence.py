"""RF-spoiled 2D radial gradient echo, multi-slice."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._schedules import make_rf_spoiling_schedule


class GreRadial2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D radial gradient echo: one full spoke per repetition.

    One spoke waveform is turned per shot by a rotation extension. The Nyquist
    set is ``ceil(pi / 2 * n)`` spokes spread evenly over half a turn, and
    every ``ry``-th of them is played, in order. Slices are dealt into packets,
    and ordered within one, as :mod:`gre2D_sequence` deals them. Every
    acquisition carries its spoke as ``LIN`` and its slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_radial2D_sequence(n=32, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "gre_radial_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Non-acquiring repetitions, at the first spoke's angle, before each packet.
    N_DUMMY = 16
    #: Quadratic RF spoiling phase increment (degrees), counted per slice.
    RF_SPOILING_INCREMENT_DEG = 117.0
    #: Dephasing left on the slice axis at the end of each repetition, in
    #: cycles across one voxel.
    SPOILING_CYCLES = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n: int = 128,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_spacing: float = 0.0,
        flip_angle_deg: float = 12.0,
        te: float | None = None,
        tr: float | None = 20e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        *,
        readout_oversampling: float = 2.0,
    ) -> None:
        """Design the pulse, the spoke, the slice packets and the spoke angles.

        Parameters
        ----------
        fov : float, optional
            Isotropic in-plane field of view (m).
        n : int, optional
            In-plane matrix size; a spoke reads it edge to edge.
        n_slices : int, optional
            Number of slices.
        slice_thickness : float, optional
            Slice thickness (m).
        slice_spacing : float, optional
            Gap between adjacent slices (m); zero is contiguous.
        flip_angle_deg : float, optional
            Excitation flip angle (degrees).
        te : float | None, optional
            Echo time to the spoke's centre crossing (s). ``None`` is as short
            as possible.
        tr : float | None, optional
            Repetition time between successive excitations of one slice (s).
            ``None`` is as short as possible, and puts every slice in one
            packet.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth (Hz).
        ry : int, optional
            Angular undersampling: one spoke in every ``ry`` of the Nyquist set
            is played.
        readout_oversampling : float, optional
            Readout oversampling factor, at least one.

        Raises
        ------
        ValueError
            If ``ry`` is below one, or the TR cannot hold one slice.
        """
        if ry < 1:
            raise ValueError(f"ry must be at least 1, got {ry}")

        system = self.system
        self.fov, self.matrix = fov, (n, n, n_slices)
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        self.ro = sequences.RadialReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            self.exc.gz_reph,
            fov=fov,
            matrix=n,
            te=te,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=self.SPOILING_CYCLES,
        )
        # A full spoke covers half a turn, which the Nyquist set divides evenly.
        n_nyquist = math.ceil(np.pi / 2 * n)
        self.angles = np.pi * np.arange(0, n_nyquist, ry) / n_nyquist
        self.rotations = [pp.make_rotation(float(angle)) for angle in self.angles]

        # Slices one TR cannot hold are dealt round-robin into packets, even
        # slices of a packet first. Every shot closes with a pure delay: one
        # raster, and on the last slice of a packet whatever is left of the TR.
        self.raster = system.block_duration_raster
        shot = self.ro.duration + self.raster
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

        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_spacing
        )
        self.slab_thickness = (
            n_slices * (slice_thickness + slice_spacing) - slice_spacing
        )
        self.slice_gap = slice_thickness + slice_spacing - self.exc.slice_thickness

    def loop(self) -> None:
        """Play each packet: its dummies, then every spoke at each of its slices."""
        spokes = [None] * self.N_DUMMY + list(range(len(self.angles)))
        phases = make_rf_spoiling_schedule(
            len(spokes), increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        )
        for packet in self.packets:
            for spoke, phase in zip(spokes, phases, strict=True):
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    pad = self.pads[len(packet)] if last else self.raster
                    self.kernel(s, spoke, phase, pad)

    def kernel(self, s: int, spoke: int | None, phase: float, pad: float) -> None:
        """One excitation of slice ``s`` reading ``spoke``; ``None`` plays a dummy.

        The readout's blocks after the pulse are played as it laid them out,
        each turned to the spoke's angle where it drives an in-plane gradient.
        """
        exc, ro, seq = self.exc, self.ro, self.seq
        exc.rf.freq_offset = exc.selection_amplitude * self.positions[s]
        exc.rf.phase_offset = phase - 2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ro.adc.phase_offset = phase

        acquire = spoke is not None
        if acquire:
            labels = self.labels(SLC=s, LIN=spoke, ONCE=0)
        else:
            labels = self.labels(SLC=s, ONCE=1)
        rotation = self.rotations[spoke if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *labels)
        for block in ro.blocks[1:]:
            events = [event for event in block if acquire or event is not ro.adc]
            if any(getattr(event, "channel", None) in ("x", "y") for event in events):
                events.append(rotation)
            seq.add_block(*events)
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription, the spoke set and the timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": [self.fov, self.fov, self.slab_thickness],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "Trajectory": "radial",
            "NumSpokes": len(self.angles),
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreRadial2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_radial_2d.seq"))
