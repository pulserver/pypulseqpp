"""RF-spoiled 2D radial gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The spoke-angle schemes ``angle_scheme`` selects from.
ANGLE_SCHEMES = ("golden", "uniform")


def spoke_angles(n_spokes: int, scheme: str) -> np.ndarray:
    """In-plane angle of every spoke, in radians.

    ``golden`` steps by the diametric golden angle, so any prefix of the scan
    is near-uniform; ``uniform`` spreads the spokes over half a turn, which a
    full spoke covers.
    """
    if scheme not in ANGLE_SCHEMES:
        raise ValueError(f"scheme must be one of {ANGLE_SCHEMES}, got {scheme!r}")
    if scheme == "golden":
        return np.asarray(pp.calc_golden_angles(n_spokes))
    return np.asarray(pp.calc_uniform_angles(n_spokes, span=np.pi))


class GreRadial2DApp(sequences.SequenceApp):
    """RF-spoiled, multi-slice 2D radial gradient echo: one full spoke per repetition.

    The spoke's prephaser, traversal and rewinder are one waveform, turned per
    shot by a rotation extension; ``use_rotation_ext=False`` writes every spoke
    out as its own waveform instead. Slices one TR cannot hold are dealt
    round-robin into passes, and each slice of a pass is played at
    ``tr / len(pass)``. Every acquisition carries its spoke as ``LIN`` and its
    slice as ``SLC``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_radial2D_sequence(n_x=32, n_spokes=13, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "gre_radial_2d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        n_spokes: int | None = None,
        angle_scheme: str = "golden",
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
        """Design the pulse, the spoke, the slice passes and the spoke angles.

        Parameters
        ----------
        fov : float, optional
            Isotropic in-plane field of view, in metres.
        n_x : int, optional
            In-plane matrix size; a spoke reads this many samples edge to edge.
        n_spokes : int or None, optional
            Spokes to play. ``None`` is ``int(pi / 2 * n_x)``.
        angle_scheme : str, optional
            One of :data:`ANGLE_SCHEMES`.
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
            Echo time to the spoke's centre crossing, in seconds. ``None`` is
            as short as possible.
        tr : float or None, optional
            Repetition time between successive excitations of one slice, in
            seconds. ``None`` is as short as possible, and puts every slice in
            one pass.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        n_dummy : int, optional
            Non-acquiring repetitions before the first spoke of each pass.
        n_gain_calibration_readouts : int or None, optional
            Written as the ``NumGainCalibrationReadouts`` definition. ``None``
            is one per slice.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the slice axis at the end of each
            repetition, counted across one voxel.
        use_rotation_ext : bool, optional
            Turn one stored spoke per shot with a rotation extension, or write
            every spoke out as its own waveform for a reader that composes no
            rotation.
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
        count = int(np.pi / 2 * n_x) if n_spokes is None else int(n_spokes)
        self.angles = spoke_angles(count, angle_scheme)
        self.ro = sequences.RadialReadout2D(
            system,
            self.exc.rf,
            self.exc.gz,
            self.exc.gz_reph,
            fov=fov,
            matrix=n_x,
            te=te,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            explicit=not use_rotation_ext,
            angles=None if use_rotation_ext else self.angles,
        )
        # One event per distinct angle; an explicit spoke is already turned.
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
                f"{shot * 1e3:.3f} ms one spoke takes"
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

        # The readout carries the slice rephaser on its TE wait when the wait
        # is at least as long, and on the spoke otherwise.
        self.wait_te = getattr(self.ro, "wait_te", None)
        on_wait = self.wait_te is not None and self.wait_te.delay >= pp.calc_duration(
            self.ro.gz_reph
        )
        self.reph_on_wait = [self.ro.gz_reph] if on_wait else []
        self.reph_on_spoke = [] if on_wait else [self.ro.gz_reph]
        self.gz_spoil = getattr(self.ro, "gz_spoil", None)

    def loop(self) -> None:
        """Play each pass: its dummies, then every spoke at each of its slices."""
        spokes = [None] * self.n_dummy + list(range(len(self.angles)))
        phases = iter(
            pp.make_rf_spoiling_schedule(
                len(spokes) * self.matrix[2], increment=self.spoiling_increment
            )
        )
        for group in self.passes:
            wait = self.waits[len(group)]
            for spoke in spokes:
                for s in group:
                    self.kernel(s, spoke, next(phases), wait)

    def kernel(self, s: int, spoke: int | None, phase: float, wait=None) -> None:
        """One excitation of slice ``s`` reading ``spoke``; ``None`` plays a dummy.

        A dummy plays the first spoke's orientation without its ADC. The
        spoke block carries the spoke's rotation, and the slice rephaser when
        the TE wait cannot hold it.
        """
        exc, ro, seq = self.exc, self.ro, self.seq
        exc.rf.freq_offset = exc.selection_amplitude * self.positions[s]
        exc.rf.phase_offset = phase - 2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ro.adc.phase_offset = phase

        acquire = spoke is not None
        once = {"ONCE": int(not acquire)} if self.n_dummy else {}
        if acquire:
            labels = self.labels(LIN=spoke, SLC=s, **once)
        else:
            labels = self.labels(SLC=s, **once)
        index = spoke if acquire else 0
        rotation = self.rotations[index]
        gx, gy = ro.gx, ro.gy
        if isinstance(gx, list):  # explicit: one turned spoke per angle
            gx, gy = gx[index], gy[index]

        seq.add_block(exc.rf, exc.gz, *labels)
        if self.wait_te is not None:
            seq.add_block(self.wait_te, *self.reph_on_wait)
        seq.add_block(
            gx,
            gy,
            *([ro.adc] if acquire else []),
            *self.reph_on_spoke,
            *([rotation] if rotation is not None else []),
        )
        if self.gz_spoil is not None:
            seq.add_block(self.gz_spoil)
        if wait is not None:
            seq.add_block(wait)

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
            "AngleScheme": self.angle_scheme,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.exc.slice_thickness,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreRadial2DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_radial_2d.seq"))
