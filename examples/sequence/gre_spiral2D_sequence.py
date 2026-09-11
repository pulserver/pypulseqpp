"""RF-spoiled 2D spiral gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The arm-angle schemes ``angle_scheme`` selects from. A spiral arm covers a
#: full turn, so both spread the arms over one.
ANGLE_SCHEMES = ("golden", "uniform")


def arm_angles(n_arms: int, scheme: str) -> np.ndarray:
    """In-plane rotation of every arm, in radians, over a full turn."""
    if scheme not in ANGLE_SCHEMES:
        raise ValueError(f"scheme must be one of {ANGLE_SCHEMES}, got {scheme!r}")
    if scheme == "golden":
        return np.asarray(pp.calc_golden_angles(n_arms, full_circle=True))
    return np.asarray(pp.calc_uniform_angles(n_arms))


class GreSpiral2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D spiral gradient echo: one interleave per repetition.

    One solved interleave is turned per shot by a rotation extension;
    ``use_rotation_ext=False`` writes every arm out as its own waveform
    instead. With ``n_echoes > 1`` each excitation reads the same arm at a
    series of echo times, and the readout labels them ``ECO``. Slices one TR
    cannot hold are dealt round-robin into passes, and each slice of a pass is
    played at ``tr / len(pass)``. Every acquisition carries its arm as ``LIN``
    and its slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_spiral2D_sequence(n_x=32, n_arms=4, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "gre_spiral_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse. The selection amplitude, which slice
    #: offsets are converted against, is ``TIME_BW_PRODUCT / (PULSE_DURATION *
    #: thickness)``.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        n_arms: int = 16,
        angle_scheme: str = "uniform",
        n_echoes: int = 1,
        n_slices: int = 1,
        slice_thickness: float = 5e-3,
        slice_gap: float = 0.0,
        slice_order: str = "interleaved",
        flip_angle_deg: float = 12.0,
        te: float | None = None,
        tr: float | None = 20e-3,
        readout_bandwidth_hz: float = 250e3,
        n_dummy: int = 16,
        n_gain_calibration_readouts: int | None = None,
        rf_spoiling_increment_deg: float = 117.0,
        spoiling_cycles: float = 4.0,
        use_rotation_ext: bool = True,
    ) -> None:
        """Design the pulse, the interleave, the slice passes and the arm angles.

        Parameters
        ----------
        fov : float, optional
            Isotropic in-plane field of view, in metres.
        n_x : int, optional
            In-plane matrix size.
        n_arms : int, optional
            Interleaves played, which is also the pitch the spiral is designed
            for.
        angle_scheme : str, optional
            One of :data:`ANGLE_SCHEMES`.
        n_echoes : int, optional
            Arms read per excitation, each the same interleave at a later echo
            time.
        n_slices : int, optional
            Number of slices.
        slice_thickness : float, optional
            Slice thickness, in metres.
        slice_gap : float, optional
            Gap between adjacent slices, in metres.
        slice_order : str, optional
            Order the slices of one pass are excited in, as
            :func:`pypulseqpp.calc_traversal_order` accepts.
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        te : float or None, optional
            Echo time to the start of the outward path, in seconds. ``None``
            is as short as possible.
        tr : float or None, optional
            Repetition time between successive excitations of one slice, in
            seconds. ``None`` is as short as possible, and puts every slice in
            one pass.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        n_dummy : int, optional
            Non-acquiring repetitions before the first arm of each pass.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the slice axis at the end of each
            repetition, counted across one voxel.
        use_rotation_ext : bool, optional
            Turn one stored interleave per shot with a rotation extension, or
            write every arm out as its own waveform for a reader that composes
            no rotation.
        """
        system = self.system
        self.fov, self.matrix = fov, (n_x, n_x, n_slices)
        self.n_dummy, self.angle_scheme = n_dummy, angle_scheme
        self.n_gain_calibration_readouts = (
            n_slices
            if n_gain_calibration_readouts is None
            else n_gain_calibration_readouts
        )
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slice_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        self.angles = arm_angles(n_arms, angle_scheme)
        # The readout writes ECO itself, since it plays the echo train.
        self.ro = sequences.SpiralReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            self.exc.gz_reph,
            fov=fov,
            matrix=n_x,
            design_interleaves=n_arms,
            te=te,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            n_echoes=n_echoes,
            explicit=not use_rotation_ext,
            angles=None if use_rotation_ext else self.angles,
        )
        self.echo_times = [
            self.ro.echo_time + i * self.ro.echo_spacing for i in range(n_echoes)
        ]
        # One event per distinct angle; an explicit arm is already turned.
        made: dict[float, object] = {}
        self.rotations = [
            made.setdefault(float(a), pp.make_rotation(float(a)))
            if use_rotation_ext
            else None
            for a in self.angles
        ]

        # Slices one TR cannot hold are dealt round-robin into passes, so
        # neighbours are never excited back to back.
        shot = self.ro.duration
        if tr is not None and tr < shot - 1e-9:
            raise ValueError(
                f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
                f"{shot * 1e3:.3f} ms one arm takes"
            )
        per_pass = n_slices if tr is None else max(1, int(tr / shot))
        n_passes = -(-n_slices // per_pass)
        groups = [list(range(start, n_slices, n_passes)) for start in range(n_passes)]
        self.passes = [
            [group[i] for i in pp.calc_traversal_order(len(group), slice_order)]
            for group in groups
        ]
        # Each slice of a pass closes with the wait that makes its shot tr / size.
        raster = system.block_duration_raster
        self.waits = {}
        for size in {len(group) for group in self.passes}:
            pad = 0.0 if tr is None else pp.round_to_raster(tr / size - shot, raster)
            self.waits[size] = pp.make_delay(pad) if pad > 0 else None
        self.repetition_time = max(
            size * (shot + (wait.delay if wait is not None else 0.0))
            for size, wait in self.waits.items()
        )

        self.positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
            slice_thickness + slice_gap
        )
        self.slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
        self.spoiling_increment = np.deg2rad(rf_spoiling_increment_deg)

    def loop(self) -> None:
        """Play each pass: its dummies, then every arm at each of its slices."""
        arms = [None] * self.n_dummy + list(range(len(self.angles)))
        phases = iter(
            pp.make_rf_spoiling_schedule(
                len(arms) * self.matrix[2], increment=self.spoiling_increment
            )
        )
        for group in self.passes:
            wait = self.waits[len(group)]
            for arm in arms:
                for s in group:
                    self.kernel(s, arm, next(phases), wait)

    def kernel(self, s: int, arm: int | None, phase: float, wait=None) -> None:
        """One excitation of slice ``s`` reading ``arm``; ``None`` plays a dummy.

        A dummy plays the first arm's orientation without its ADC or the
        readout's ``ECO`` labels. The arm's blocks are the readout's own, so
        the TE and echo spacing it solved are kept; every block driving an
        in-plane gradient carries the arm's rotation.
        """
        rf, gz, seq = self.exc.rf, self.exc.gz, self.seq
        rf.freq_offset = gz.amplitude * self.positions[s]
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center
        self.ro.adc.phase_offset = phase

        acquire = arm is not None
        once = {"ONCE": int(not acquire)} if self.n_dummy else {}
        if acquire:
            labels = self.labels(LIN=arm, SLC=s, **once)
        else:
            labels = self.labels(SLC=s, **once)
        index = arm if acquire else 0
        rotation = self.rotations[index]

        seq.add_block(rf, gz, *labels)
        for events in self.ro.arm(index)[1:]:
            played = [
                e
                for e in events
                if acquire
                or not (e is self.ro.adc or getattr(e, "type", "") == "labelset")
            ]
            turned = rotation is not None and any(
                getattr(e, "channel", None) in ("x", "y") for e in played
            )
            seq.add_block(*played, *([rotation] if turned else []))
        if wait is not None:
            seq.add_block(wait)

    def finalize(self) -> None:
        """Write the prescription, the arm set and the echo timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": [self.fov, self.fov, self.slab_thickness],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.echo_times,
            "TR": self.repetition_time,
            "Trajectory": "spiral",
            "NumArms": len(self.angles),
            "AngleScheme": self.angle_scheme,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.exc.slice_thickness,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreSpiral2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_spiral_2d.seq"))
