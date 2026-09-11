"""2D PROPELLER spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class SePropeller2DApp(sequences.SequenceApp):
    """Multi-slice 2D PROPELLER spin echo: one EPI blade per excitation.

    A slice-selective excitation and one refocusing pulse with crushers put
    the spin echo on the blade's central line, so ``te`` runs from the
    excitation centre to that line with the refocusing pulse at its midpoint.
    The blade straddles the centre of k-space and is turned per shot by a
    rotation extension. Slices one TR cannot hold are dealt round-robin into
    passes, and each slice of a pass is played at ``tr / len(pass)``. Every
    acquisition carries its line within the blade as ``LIN``, its slice as
    ``SLC`` and its blade as ``SEG``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_propeller2D_sequence(
    ...     n_x=32, blade_width=8, n_blades=4, te=None, tr=None
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_propeller_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulse. The
    #: selection amplitude of both, which slice offsets are converted against,
    #: is ``TIME_BW_PRODUCT / (PULSE_DURATION * thickness)``.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        blade_width: int = 16,
        n_blades: int | None = None,
        angle_scheme: str = "uniform",
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_gap: float = 0.0,
        slice_order: str = "interleaved",
        te: float | None = 80e-3,
        tr: float | None = 2000e-3,
        readout_bandwidth_hz: float = 250e3,
        crusher_cycles: float = 4.0,
        n_dummy: int = 0,
        n_gain_calibration_readouts: int | None = None,
    ) -> None:
        """Design the pulses, the blade, the spin-echo timing and the slice passes.

        Parameters
        ----------
        fov : float, optional
            Isotropic in-plane field of view, in metres.
        n_x : int, optional
            In-plane matrix size.
        blade_width : int, optional
            Phase-encode lines per blade.
        n_blades : int or None, optional
            Blades in the set. ``None`` is the smallest count that samples the
            rim of k-space at Nyquist.
        angle_scheme : str, optional
            ``'uniform'`` or ``'golden'``, spread over half a turn.
        n_slices : int, optional
            Number of slices.
        slice_thickness : float, optional
            Slice thickness, in metres.
        slice_gap : float, optional
            Gap between adjacent slices, in metres.
        slice_order : str, optional
            Order the slices of one pass are excited in, as
            :func:`pypulseqpp.calc_traversal_order` accepts.
        te : float or None, optional
            Effective echo time, excitation centre to the blade's central
            line, in seconds. ``None`` is as short as possible.
        tr : float or None, optional
            Repetition time between successive excitations of one slice, in
            seconds. ``None`` is as short as possible, and puts every slice in
            one pass.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        crusher_cycles : float, optional
            Cycles of dephasing each crusher beside the refocusing pulse winds.
        n_dummy : int, optional
            Non-acquiring blades, at the first blade's angle, before the first
            acquired blade of each pass.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        """
        system = self.system
        self.fov, self.matrix = fov, (n_x, n_x, n_slices)
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
        # The refocusing gradient is crushers and plateau in one waveform, so
        # its amplitude is the crusher peak; the plateau is the design's.
        self.refocusing_amplitude = self.TIME_BW_PRODUCT / (
            self.PULSE_DURATION * slice_thickness
        )

        def blade(first_line_te: float | None):
            return sequences.PropellerReadout2D(
                system,
                self.ref.rf_ref,
                self.ref.gz,
                fov=fov,
                matrix=n_x,
                blade_width=blade_width,
                n_blades=n_blades,
                scheme=angle_scheme,
                te=first_line_te,
                readout_bandwidth_hz=readout_bandwidth_hz,
            )

        # The blade's own te is its first line's, while the spin echo belongs
        # to its central line: the readout half is TE/2 less the lines before
        # the centre, and the excitation half is a wait.
        excitation_span = self.exc.seq.duration()[0]
        exc_center = self.exc.rf.delay + self.exc.rf.center
        half_te_floor = (excitation_span - exc_center) + self.ref.center
        probe = blade(None)
        centre_delta = float(probe.echo_times[blade_width // 2] - probe.echo_times[0])
        half_te = (
            max(probe.echo_time + centre_delta, half_te_floor) if te is None else te / 2
        )
        first_line_te = half_te - centre_delta
        if first_line_te < probe.echo_time - 1e-9:
            raise ValueError(
                f"TE {2 * half_te * 1e3:.1f} ms is shorter than the blade half "
                f"admits; the minimum is "
                f"{2 * (probe.echo_time + centre_delta) * 1e3:.1f} ms"
            )
        if half_te_floor > half_te + 1e-9:
            raise ValueError(
                f"TE {2 * half_te * 1e3:.1f} ms is shorter than the excitation "
                f"half admits; the minimum is {2 * half_te_floor * 1e3:.1f} ms"
            )
        self.echo_time = 2 * half_te
        self.blade = blade(first_line_te)
        raster = system.block_duration_raster
        wait = pp.round_to_raster(half_te - half_te_floor, raster)
        self.wait_half_te = pp.make_delay(wait) if wait > 0 else None
        self.gy_pre = pp.scale_grad(self.blade.gy_pre, self.blade.blade_start)
        # One event per distinct angle.
        made: dict[float, object] = {}
        self.rotations = [
            made.setdefault(float(a), pp.make_rotation(float(a)))
            for a in self.blade.blade_angles
        ]

        shot = excitation_span + max(wait, 0.0) + self.blade.seq.duration()[0]
        if tr is not None and tr < shot - 1e-9:
            raise ValueError(
                f"TR {tr * 1e3:.1f} ms is shorter than one blade takes "
                f"({shot * 1e3:.1f} ms)"
            )
        # Slices one TR cannot hold are dealt round-robin into passes, so
        # neighbours are never excited back to back.
        per_pass = n_slices if tr is None else max(1, int(tr / shot))
        n_passes = -(-n_slices // per_pass)
        groups = [list(range(start, n_slices, n_passes)) for start in range(n_passes)]
        self.passes = [
            [group[i] for i in pp.calc_traversal_order(len(group), slice_order)]
            for group in groups
        ]
        # Each slice of a pass closes with the wait that makes its shot tr / size.
        self.waits = {}
        for size in {len(group) for group in self.passes}:
            pad = 0.0 if tr is None else pp.round_to_raster(tr / size - shot, raster)
            self.waits[size] = pp.make_delay(pad) if pad > 0 else None
        self.repetition_time = max(
            size * (shot + (w.delay if w is not None else 0.0))
            for size, w in self.waits.items()
        )

        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_gap
        )
        self.slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap

    def loop(self) -> None:
        """Play each pass: its dummy blades, then every blade at each of its slices."""
        blades = [None] * self.n_dummy + list(range(self.blade.n_blades))
        for group in self.passes:
            wait = self.waits[len(group)]
            for index in blades:
                for s in group:
                    self.kernel(s, index, wait)

    def kernel(self, s: int, index: int | None, wait=None) -> None:
        """One spin echo of slice ``s`` reading blade ``index``; ``None`` plays a dummy.

        A dummy plays the first blade's orientation without acquiring.
        """
        exc, ref, blade, seq = self.exc, self.ref, self.blade, self.seq
        position = self.positions[s]
        exc.rf.freq_offset = exc.gz.amplitude * position
        exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ref.rf_ref.freq_offset = self.refocusing_amplitude * position
        ref.rf_ref.phase_offset = (
            np.pi / 2 - 2 * np.pi * ref.rf_ref.freq_offset * ref.rf_ref.center
        )

        acquire = index is not None
        once = {"ONCE": int(not acquire)} if self.n_dummy else {}
        if acquire:
            labels = self.labels(SLC=s, SEG=index, **once)
        else:
            labels = self.labels(SLC=s, **once)
        rotation = self.rotations[index if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *labels)
        seq.add_block(exc.gz_reph)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        seq.add_block(ref.rf_ref, ref.gz)
        wait_te = getattr(blade, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(blade.gx_pre, self.gy_pre, rotation)
        for line in range(blade.etl):
            blip = blade.gy_blips[line]
            acquired = (blade.adc, *self.labels(LIN=line)) if acquire else ()
            seq.add_block(
                blade.gx[line],
                *acquired,
                *([blip] if blip is not None else []),
                rotation,
            )
        seq.add_block(blade.gx_spoil, rotation)
        if wait is not None:
            seq.add_block(wait)

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
            "BladeWidth": self.blade.blade_width,
            "NumBlades": self.blade.n_blades,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = SePropeller2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_propeller_2d.seq"))
