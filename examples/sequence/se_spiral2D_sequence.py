"""2D spiral spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The spiral densities ``density`` selects from.
DENSITIES = ("constant", "variable", "dual")


class SeSpiral2DApp(sequences.SequenceApp):
    """Multi-slice 2D spiral spin echo: one interleaf per excitation.

    A slice-selective SLR 90, one slice-selective SLR 180 between crushers
    half a TE later, and one outward interleaf starting at the echo, turned
    per shot by a rotation extension. The interleaves are designed and chosen
    as :mod:`gre_spiral2D_sequence` designs and chooses them, and the slices
    dealt into packets as :mod:`gre2D_sequence` deals them. Every acquisition
    carries its interleaf as ``LIN`` and its slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_spiral2D_sequence(n=32, n_shots=4, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_spiral_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Dephasing each crusher beside the refocusing pulse winds, in cycles
    #: across one voxel.
    CRUSHER_CYCLES = 4.0
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
        te: float | None = None,
        tr: float | None = 500e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        *,
        n_dummy: int = 0,
        n_shots: int = 16,
        density: str = "constant",
        periphery_undersampling: float = 2.0,
        transition_speed: float = 12.0,
    ) -> None:
        """Design the pulses, the interleaf, the slice packets and the angles.

        Parameters
        ----------
        fov : float, optional
            Isotropic in-plane field of view (m).
        n : int, optional
            In-plane matrix size.
        n_slices : int, optional
            Number of slices.
        slice_thickness : float, optional
            Slice thickness (m).
        slice_spacing : float, optional
            Gap between adjacent slices (m); zero is contiguous.
        te : float | None, optional
            Echo time (s), excitation centre to the start of the outward path,
            with the refocusing pulse at its midpoint. ``None`` is as short as
            possible.
        tr : float | None, optional
            Repetition time between successive excitations of one slice (s).
            ``None`` is as short as possible, and puts every slice in one
            packet.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth (Hz).
        ry : int, optional
            Angular undersampling: one interleaf in every ``ry`` of the
            ``n_shots`` is played.
        n_dummy : int, optional
            Non-acquiring repetitions, at the first interleaf's angle, before
            each packet.
        n_shots : int, optional
            Interleaves that sample the centre of k-space at Nyquist.
        density : {'constant', 'variable', 'dual'}, optional
            Constant pitch, a radial power-law transition to the periphery, or
            a logistic one.
        periphery_undersampling : float, optional
            How much sparser the periphery is sampled than the centre, at
            least one. Unused at constant density.
        transition_speed : float, optional
            Steepness of the dual-density transition.

        Raises
        ------
        ValueError
            If ``density`` is unknown, ``ry`` or ``periphery_undersampling`` is
            below one, or the TE or TR is shorter than the pulses and the
            readout take.
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
            90.0,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        # A nonselective 180 would invert every slice of the packet whichever
        # one it refocuses, so the refocusing pulse selects the slice too.
        self.ref = sequences.SpatialSelectiveRefocusing(
            system,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            spoiling_cycles=self.CRUSHER_CYCLES,
        )
        self.ref_phase = float(self.ref.rf_ref.phase_offset)

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

        # TE is solved in two halves about the 180. The readout owns the second,
        # from the 180's centre to the echo; a delay before the 180 sets the
        # first, which is never shorter than the excitation's blocks allow.
        half_floor = self.exc.duration - self.exc.center + self.ref.center

        def readout(half_te: float | None):
            return sequences.SpiralReadout2D(
                system,
                self.ref.rf_ref,
                self.ref.gz,
                fov=fov,
                matrix=n,
                design_interleaves=n_shots,
                density=density,
                te=half_te,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=self.SPOILING_CYCLES,
                **shaped,
            )

        te_min = 2 * max(readout(None).echo_time, half_floor)
        if te is not None and te < te_min - 1e-9:
            raise ValueError(
                f"TE {te * 1e3:.2f} ms is shorter than the {te_min * 1e3:.2f} ms "
                "the excitation and the readout admit"
            )
        # The 180 must sit midway, so both halves are solved on the block raster:
        # the excitation half waits a whole number of rasters, and the readout
        # is asked for the same span.
        self.raster = system.block_duration_raster
        target = (te_min if te is None else te) / 2 - half_floor
        wait = pp.ceil_to_raster(max(target, 0.0) - 1e-9, self.raster)
        self.ro = readout(half_floor + wait)
        self.wait_half_te = pp.make_delay(wait) if wait > 0 else None
        self.echo_time = 2 * (half_floor + wait)

        # An interleaf covers a full turn, which the n_shots divide evenly.
        self.angles = 2 * np.pi * np.arange(0, n_shots, ry) / n_shots
        self.rotations = [pp.make_rotation(float(angle)) for angle in self.angles]

        # Slices one TR cannot hold are dealt round-robin into packets, even
        # slices of a packet first. Every shot closes with a pure delay: one
        # raster, and on the last slice of a packet whatever is left of the TR.
        shot = self.exc.duration + wait + self.ro.duration + self.raster
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
                f"TR {cycle * 1e3:.1f} ms is shorter than one repetition takes "
                f"({shot * 1e3:.1f} ms)"
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
        for packet in self.packets:
            for arm in arms:
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    self.kernel(s, arm, self.pads[len(packet)] if last else self.raster)

    def kernel(self, s: int, arm: int | None, pad: float) -> None:
        """One spin echo of slice ``s`` reading ``arm``; ``None`` plays a dummy.

        The readout's blocks, from the refocusing pulse on, are played as it
        laid them out, each turned to the interleaf's angle where it drives
        an in-plane gradient.
        """
        exc, ref, ro, seq = self.exc, self.ref, self.ro, self.seq
        position = self.positions[s]
        # Each pulse selects at its own plateau; a crushed refocusing
        # gradient's amplitude is its crusher peak.
        exc.rf.freq_offset = exc.selection_amplitude * position
        exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ref.rf_ref.freq_offset = ref.selection_amplitude * position
        ref.rf_ref.phase_offset = (
            self.ref_phase - 2 * np.pi * ref.rf_ref.freq_offset * ref.rf_ref.center
        )

        acquire = arm is not None
        if acquire:
            labels = self.labels(SLC=s, LIN=arm, ONCE=0)
        else:
            labels = self.labels(SLC=s, ONCE=1)
        rotation = self.rotations[arm if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *labels)
        seq.add_block(exc.gz_reph)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        for block in ro.blocks:
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
            "TE": self.echo_time,
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


main = SeSpiral2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_spiral_2d.seq"))
