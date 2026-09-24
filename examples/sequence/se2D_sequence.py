"""2D Cartesian spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Se2DApp(sequences.SequenceApp):
    """Multi-slice 2D Cartesian spin echo: one line per excitation.

    A slice-selective SLR 90, one slice-selective SLR 180 between crushers
    half a TE later, and one frequency-encoded line at the echo. Both pulses
    are offset to the slice, each at its own selection gradient. Slices are
    dealt into packets, and ordered within one, as :mod:`gre2D_sequence` deals
    them; under undersampling the calibration lines are acquired first.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se2D_sequence(n_x=32, n_y=16, te=15e-3, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Dephasing each crusher beside the refocusing pulse winds, in cycles
    #: across one voxel.
    CRUSHER_CYCLES = 4.0
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
        te: float | None = 15e-3,
        tr: float | None = 500e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        *,
        n_dummy: int = 0,
        readout_oversampling: float = 2.0,
        n_acs_y: int = 24,
    ) -> None:
        """Design the pulses, the readout, the slice packets and the line order.

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
            Phase-encode undersampling: one line in every ``ry`` is acquired,
            the centre line among them.
        partial_fourier_x : float, default=1.0
            Fraction of the echo acquired, in ``[0.75, 1]``.
        partial_fourier_y : float, default=1.0
            Fraction of the phase-encode extent acquired, in ``[0.75, 1]``.
        n_dummy : int, default=0
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
            below one, or the TE or TR is shorter than the pulses and the
            readout take.
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
                partial_echo=partial_fourier_x,
                oversampling=readout_oversampling,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=self.SPOILING_CYCLES,
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

        # Slices one TR cannot hold are dealt round-robin into packets, even
        # slices of a packet first; every shot closes with a pure delay, as
        # gre2D_sequence closes them.
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
        # The last slice of a packet plays its pad in place of the closing raster.
        packet_time = {n: n * shot - self.raster + pad for n, pad in self.pads.items()}
        self.repetition_time = max(packet_time.values())

        calibrating, imaging = pp.make_cartesian_axis_sampling(
            n_y, ry, n_acs_y, partial_fourier=partial_fourier_y
        )
        # The calibration block leads, so a reconstruction can estimate
        # coil sensitivities while the rest is still arriving.
        self.lines = [*calibrating, *imaging]
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
        self.resolve(
            slice_thickness=self.exc.slice_thickness,
            slice_spacing=self.slice_gap,
            te=self.echo_time,
            tr=self.repetition_time,
            readout_bandwidth_hz=self.ro.bandwidth_hz,
        )

    def loop(self) -> None:
        """Play each packet: its dummies, then every line at each of its slices."""
        lines = [None] * self.n_dummy + list(self.lines)
        for packet in self.packets:
            for line in lines:
                for i, s in enumerate(packet):
                    last = i == len(packet) - 1
                    self.kernel(
                        s, line, self.pads[len(packet)] if last else self.raster
                    )

    def kernel(self, s: int, line: int | None, pad: float) -> None:
        """One excitation of slice ``s`` at one line; ``line=None`` plays a dummy.

        Parameters
        ----------
        s : int
            Slice index, counting from 0.
        line : int or None
            The phase-encode line to acquire, or None for a dummy.
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

        if line is None:
            ky, labels = 0.0, self.labels(SLC=s, ONCE=1)
        else:
            n_y = self.matrix[1]
            ky = (line - n_y // 2) / (n_y / 2)
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
