"""2D PROPELLER spin echo, multi-slice."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class SePropeller2DApp(sequences.SequenceApp):
    """Multi-slice 2D PROPELLER spin echo: one blade line per excitation.

    A slice-selective SLR 90, one slice-selective SLR 180 between crushers
    half a TE later, and one frequency-encoded line at the echo. A blade is
    ``blade_width`` lines centred on k = 0, turned by a rotation extension;
    blades are chosen as :mod:`gre_propeller2D_sequence` chooses them, and the
    slices dealt into packets as :mod:`gre2D_sequence` deals them. Every
    acquisition carries its line as ``LIN``, its blade as ``SEG`` and its
    slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_propeller2D_sequence(n=32, blade_width=8, te=None, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_propeller_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Readout oversampling factor.
    READOUT_OVERSAMPLING = 2.0
    #: Dephasing each crusher beside the refocusing pulse winds, in cycles
    #: across one voxel.
    CRUSHER_CYCLES = 4.0
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
        te: float | None = 15e-3,
        tr: float | None = 500e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        *,
        n_dummy: int = 0,
        blade_width: int = 16,
    ) -> None:
        """Design the pulses, the readout, the slice packets and the blade angles.

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
        te : float | None, default=0.015
            Echo time (s), excitation centre to echo, with the refocusing
            pulse at its midpoint. ``None`` is as short as possible.
        tr : float | None, default=0.5
            Repetition time between successive excitations of one slice (s).
            ``None`` is as short as possible, and puts every slice in one
            packet.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry : int, default=1
            Angular undersampling: one blade in every ``ry`` of the Nyquist set
            is played.
        n_dummy : int, default=0
            Non-acquiring repetitions, at the first blade's angle, before each
            packet.
        blade_width : int, default=16
            Phase-encode lines per blade, at most ``n``.

        Raises
        ------
        ValueError
            If ``ry`` is below one, ``blade_width`` is outside ``[1, n]``, or
            the TE or TR is shorter than the pulses and the readout take.
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
        # The in-plane axes turn with the blade, so a spoiler on them would
        # point a different way from one blade to the next: it plays on z.
        self.gz_spoil, _, _ = pp.make_crusher(
            self.SPOILING_CYCLES, slice_thickness, "z", system=system
        )

        # TE is solved in two halves about the 180. The readout owns the second,
        # from the 180's centre to the echo; a delay before the 180 sets the
        # first, which is never shorter than the excitation's blocks allow.
        half_floor = self.exc.duration - self.exc.center + self.ref.center

        def readout(half_te: float | None):
            return sequences.LineReadout2D(
                system,
                self.ref.rf_ref,
                self.ref.gz,
                fov=(fov, fov),
                matrix=(n, n),
                te=half_te,
                oversampling=self.READOUT_OVERSAMPLING,
                readout_bandwidth_hz=readout_bandwidth_hz,
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

        # A blade turned by half a turn is the same blade, which the Nyquist
        # set divides evenly.
        n_nyquist = math.ceil(np.pi * n / (2 * blade_width))
        self.angles = np.pi * np.arange(0, n_nyquist, ry) / n_nyquist
        self.rotations = [pp.make_rotation(float(angle)) for angle in self.angles]

        # Slices one TR cannot hold are dealt round-robin into packets, even
        # slices of a packet first. Every shot closes with a pure delay: one
        # raster, and on the last slice of a packet whatever is left of the TR.
        spoil = pp.ceil_to_raster(pp.calc_duration(self.gz_spoil), self.raster)
        shot = self.exc.duration + wait + self.ro.duration + spoil + self.raster
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
        self.duration = (self.n_dummy + len(self.views)) * sum(
            packet_time[len(packet)] for packet in self.packets
        )

    def loop(self) -> None:
        """Play each packet: its dummies, then every blade line at each of its slices."""
        views = [None] * self.n_dummy + self.views
        for packet in self.packets:
            for view in views:
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    self.kernel(
                        s, view, self.pads[len(packet)] if last else self.raster
                    )

    def kernel(self, s: int, view: tuple[int, int] | None, pad: float) -> None:
        """One spin echo of slice ``s`` at ``(blade, line)``; ``None`` plays a dummy.

        A dummy plays the first blade's centre line without its ADC. Every
        block that drives an in-plane gradient carries the blade's rotation.

        Parameters
        ----------
        s : int
            Slice index, counting from 0.
        view : tuple of int or None
            The blade and the line within it, or None for a dummy.
        pad : float
            Delay closing the repetition, in s.
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

        if view is None:
            blade, ky = 0, 0.0
            labels = self.labels(SLC=s, ONCE=1)
        else:
            blade, line = view
            ky = (line - self.blade_width // 2) / (self.matrix[1] / 2)
            labels = self.labels(SLC=s, SEG=blade, LIN=line, ONCE=0)
        rotation = self.rotations[blade]

        seq.add_block(exc.rf, exc.gz, *labels)
        seq.add_block(exc.gz_reph)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        seq.add_block(ref.rf_ref, ref.gz)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), rotation)
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
            "TE": self.echo_time,
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


main = SePropeller2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_propeller_2d.seq"))
