"""RF-spoiled 2D spiral gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The spiral densities ``density`` selects from.
DENSITIES = ("constant", "variable", "dual")


class GreSpiral2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D spiral gradient echo: one interleaf per repetition.

    One solved outward interleaf is turned per shot by a rotation extension.
    ``n_shots`` interleaves, spread evenly over a full turn, sample the centre
    of k-space at Nyquist; every ``ry``-th of them is played, in order. Slices
    are dealt into packets, and ordered within one, as :mod:`gre2D_sequence`
    deals them. Every acquisition carries its interleaf as ``LIN`` and its
    slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_spiral2D_sequence(n=32, n_shots=4, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "gre_spiral_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Quadratic RF spoiling phase increment (degrees), counted per slice.
    RF_SPOILING_INCREMENT_DEG = 117.0
    #: Dephasing left on the slice axis at the end of each repetition, in
    #: cycles across one voxel.
    SPOILING_CYCLES = 4.0
    #: Exponent of the normalised radius, for variable density.
    VARIABLE_DENSITY_POWER = 2.0
    #: Normalised radius of the dual-density transition.
    TRANSITION_RADIUS = 0.5

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
        n_dummy: int = 16,
        n_shots: int = 16,
        density: str = "constant",
        periphery_undersampling: float = 2.0,
        transition_speed: float = 12.0,
    ) -> None:
        """Design the pulse, the interleaf, the slice packets and the angles.

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
        te : float | None, default=None
            Echo time to the start of the outward path (s). ``None`` is as
            short as possible.
        tr : float | None, default=0.02
            Repetition time between successive excitations of one slice (s).
            ``None`` is as short as possible, and puts every slice in one
            packet.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry : int, default=1
            Angular undersampling: one interleaf in every ``ry`` of the
            ``n_shots`` is played.
        n_dummy : int, default=16
            Non-acquiring repetitions, at the first interleaf's angle, before
            each packet.
        n_shots : int, default=16
            Interleaves that sample the centre of k-space at Nyquist.
        density : {'constant', 'variable', 'dual'}, default='constant'
            Constant pitch, a radial power-law transition to the periphery, or
            a logistic one.
        periphery_undersampling : float, default=2.0
            How much sparser the periphery is sampled than the centre, at
            least one. Unused at constant density.
        transition_speed : float, default=12.0
            Steepness of the dual-density transition.

        Raises
        ------
        ValueError
            If ``density`` is unknown, ``ry`` or ``periphery_undersampling`` is
            below one, or the TR cannot hold one slice.
        """
        self.n_dummy = n_dummy
        if density not in DENSITIES:
            raise ValueError(f"density must be one of {DENSITIES}, got {density!r}")
        if ry < 1:
            raise ValueError(f"ry must be at least 1, got {ry}")
        if periphery_undersampling < 1:
            raise ValueError(
                f"periphery_undersampling must be at least 1, got {periphery_undersampling}"
            )

        system = self.system
        self.fov, self.matrix = fov, (n, n, n_slices)
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        # The centre is designed for n_shots interleaves and the periphery for
        # proportionally more, which is what spreads an interleaf's turns there.
        shaped = {}
        if density != "constant":
            shaped = {
                "inner_design_interleaves": n_shots,
                "outer_design_interleaves": n_shots * periphery_undersampling,
                "variable_density_power": self.VARIABLE_DENSITY_POWER,
                "transition_radius": self.TRANSITION_RADIUS,
                "transition_speed": transition_speed,
            }
        self.ro = sequences.SpiralReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            self.exc.gz_reph,
            fov=fov,
            matrix=n,
            design_interleaves=n_shots,
            density=density,
            te=te,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=self.SPOILING_CYCLES,
            **shaped,
        )
        # An interleaf covers a full turn, which the n_shots divide evenly.
        self.angles = 2 * np.pi * np.arange(0, n_shots, ry) / n_shots
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
        """Play each packet: its dummies, then every interleaf at each of its slices."""
        arms = [None] * self.n_dummy + list(range(len(self.angles)))
        phases = pp.make_rf_spoiling_schedule(
            len(arms), increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        )
        for packet in self.packets:
            for arm, phase in zip(arms, phases, strict=True):
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    pad = self.pads[len(packet)] if last else self.raster
                    self.kernel(s, arm, phase, pad)

    def kernel(self, s: int, arm: int | None, phase: float, pad: float) -> None:
        """One excitation of slice ``s`` reading ``arm``; ``None`` plays a dummy.

        The readout's blocks after the pulse are played as it laid them out,
        each turned to the interleaf's angle where it drives an in-plane
        gradient.

        Parameters
        ----------
        s : int
            Slice index, counting from 0.
        arm : int or None
            The interleaf to read, or None for a dummy.
        phase : float
            RF and ADC phase for this repetition, in radians.
        pad : float
            Delay closing the repetition, in s.
        """
        exc, ro, seq = self.exc, self.ro, self.seq
        exc.rf.freq_offset = exc.selection_amplitude * self.positions[s]
        exc.rf.phase_offset = phase - 2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ro.adc.phase_offset = phase

        acquire = arm is not None
        if acquire:
            labels = self.labels(SLC=s, LIN=arm, ONCE=0)
        else:
            labels = self.labels(SLC=s, ONCE=1)
        rotation = self.rotations[arm if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *labels)
        for block in ro.blocks[1:]:
            events = [event for event in block if acquire or event is not ro.adc]
            if any(getattr(event, "channel", None) in ("x", "y") for event in events):
                events.append(rotation)
            seq.add_block(*events)
        seq.add_block(pp.make_delay(pad))

    def finalize(self) -> None:
        """Write the prescription, the interleaf set and the timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": [self.fov, self.fov, self.slab_thickness],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "Trajectory": "spiral",
            "NumArms": len(self.angles),
            "kSpaceCenterSample": self.ro.center_sample,
            "SlicePositions": self.positions.tolist(),
            "SliceThickness": self.exc.slice_thickness,
            "SliceGap": self.slice_gap,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreSpiral2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_spiral_2d.seq"))
