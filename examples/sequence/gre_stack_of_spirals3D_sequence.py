"""RF-spoiled 3D stack-of-spirals gradient echo."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The in-plane angle schemes ``angle_scheme`` selects from.
ANGLE_SCHEMES = ("golden", "uniform")


def arm_angles(n_arms: int, scheme: str) -> np.ndarray:
    """In-plane rotation of every interleave, in radians.

    ``golden`` increments by the full-turn golden angle ``2 * pi / phi**2``;
    ``uniform`` spaces the interleaves evenly over a full turn.
    """
    if scheme not in ANGLE_SCHEMES:
        raise ValueError(f"scheme must be one of {ANGLE_SCHEMES}, got {scheme!r}")
    if scheme == "golden":
        return np.asarray(pp.calc_golden_angles(n_arms, full_circle=True))
    return np.asarray(pp.calc_uniform_angles(n_arms))


def _at(events, index: int):
    return events[index] if isinstance(events, list) else events


def _present(*events) -> list:
    return [event for event in events if event is not None]


class GreStackOfSpirals3DApp(sequences.SequenceApp):
    """RF-spoiled 3D stack of spirals: spiral interleaves in-plane, Cartesian partitions along z.

    One solved interleave serves every shot: its in-plane angle is a rotation
    extension, or with ``use_rotation_ext=False`` a waveform of its own, and
    the partition is an amplitude on the encode pair. Every partition of an
    interleave is played before the next interleave. Acquisitions carry the
    interleave index as ``LIN`` and the partition as ``PAR``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_stack_of_spirals3D_sequence(
    ...     n_x=32, n_z=4, n_arms=8, n_dummy=0
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "gre_stack_of_spirals_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Fat methylene shift from water (ppm), converted against ``system.B0``
    #: when the spectral-spatial pulse is built.
    FAT_SHIFT_PPM = -3.4

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        n_z: int = 64,
        slab_thickness: float = 128e-3,
        n_arms: int = 16,
        angle_scheme: str = "uniform",
        flip_angle_deg: float = 12.0,
        te: float | None = None,
        tr: float | None = None,
        readout_bandwidth_hz: float = 250e3,
        n_dummy: int = 32,
        n_gain_calibration_readouts: int = 1,
        rf_spoiling_increment_deg: float = 117.0,
        spoiling_cycles: float = 4.0,
        partition_angle_offset_deg: float = 0.0,
        use_rotation_ext: bool = True,
        spsp: bool = False,
    ) -> None:
        """Design the slab pulse, the interleave and the angle of every shot.

        Parameters
        ----------
        fov : float, optional
            Isotropic in-plane field of view, in metres.
        n_x : int, optional
            In-plane matrix size.
        n_z : int, optional
            Number of partitions.
        slab_thickness : float, optional
            Excited slab thickness, in metres, which is also the field of view
            along z.
        n_arms : int, optional
            Interleaves per partition, which is also the designed pitch.
        angle_scheme : str, optional
            One of :data:`ANGLE_SCHEMES`.
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        te : float or None, optional
            Echo time, in seconds. ``None`` is as short as possible.
        tr : float or None, optional
            Repetition time, in seconds, one per excitation. ``None`` is as
            short as possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        n_dummy : int, optional
            Repetitions played without acquiring before the scan.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left at the end of each repetition, counted
            across one voxel.
        partition_angle_offset_deg : float, optional
            Angle added per partition step, in degrees. Nonzero turns an
            interleave further with each partition, staggering the sampling
            along kz.
        use_rotation_ext : bool, optional
            Turn one interleave per shot with a rotation extension, rather
            than write every shot out as its own waveform.
        spsp : bool, optional
            Excite with a spectral-spatial, slab- and water-selective pulse.
        """
        system = self.system
        self.fov = (fov, fov, slab_thickness)
        self.matrix = (n_x, n_x, n_z)
        self.n_dummy = n_dummy
        self.angle_scheme = angle_scheme
        self.partition_angle_offset_deg = partition_angle_offset_deg
        self.n_gain_calibration_readouts = n_gain_calibration_readouts
        self.spoiling_increment = np.deg2rad(rf_spoiling_increment_deg)

        if spsp:
            fat_offset_hz = self.FAT_SHIFT_PPM * 1e-6 * system.gamma * system.B0
            self.exc = sequences.SpspExcitation(
                system,
                flip_angle_deg,
                thickness_m=slab_thickness,
                spectral_bandwidth_hz=abs(fat_offset_hz),
                is_slab=True,
            )
        else:
            self.exc = sequences.SpatialSelectiveExcitation(
                system,
                flip_angle_deg,
                slab_thickness,
                duration_s=self.PULSE_DURATION,
                time_bw_product=self.TIME_BW_PRODUCT,
                is_slab=True,
            )

        self.angles = arm_angles(n_arms, angle_scheme)
        # One row per interleave, one column per partition when a partition
        # offset makes every (interleave, partition) a shot of its own, else
        # one column.
        step = np.deg2rad(partition_angle_offset_deg)
        columns = n_z if step else 1
        self.shot_angles = np.mod(
            self.angles[:, None] + step * np.arange(columns), 2 * np.pi
        )

        # The slab pulse carries its own rephaser, so the partition encode
        # block holds the only other z gradient.
        self.ro = ro = sequences.SpiralStackReadout(
            system,
            self.exc.rf,
            self.exc.gz,
            None,
            fov=fov,
            matrix=n_x,
            fov_z=slab_thickness,
            matrix_z=n_z,
            design_interleaves=n_arms,
            te=te,
            tr=tr,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            explicit=not use_rotation_ext,
            angles=None if use_rotation_ext else self.shot_angles.ravel(),
        )
        self.rotations = (
            [pp.make_rotation(float(angle)) for angle in self.shot_angles.ravel()]
            if use_rotation_ext
            else None
        )
        self.duration = (n_dummy + n_arms * n_z) * ro.duration

    def loop(self) -> None:
        """Play the dummies, then every partition of each interleave in turn."""
        n_z = self.matrix[2]
        phases = iter(
            pp.make_rf_spoiling_schedule(
                self.n_dummy + len(self.angles) * n_z,
                increment=self.spoiling_increment,
            )
        )
        for _ in range(self.n_dummy):
            self.kernel(None, 0, next(phases))
        for arm in range(len(self.angles)):
            for partition in range(n_z):
                self.kernel(arm, partition, next(phases))

    def kernel(self, arm: int | None, partition: int, phase: float) -> None:
        """One excitation, one interleave at one partition; ``arm=None`` plays a dummy.

        A dummy plays the first shot unencoded along z, without the ADC.
        """
        ro, seq = self.ro, self.seq
        n_z = self.matrix[2]
        ro.rf.phase_offset = phase
        ro.adc.phase_offset = phase

        once = self.labels(ONCE=int(arm is None)) if self.n_dummy else []
        if arm is None:
            shot, kz, acquisition = 0, 0.0, []
        else:
            columns = self.shot_angles.shape[1]
            shot = arm * columns + (partition if columns > 1 else 0)
            kz = (partition - n_z / 2) / (n_z / 2)
            acquisition = [ro.adc, *self.labels(LIN=arm, PAR=partition)]
        rotation = [] if self.rotations is None else [self.rotations[shot]]
        pre = _present(
            _at(getattr(ro, "gx_pre", None), shot),
            _at(getattr(ro, "gy_pre", None), shot),
        )
        rew = _present(
            _at(getattr(ro, "gx_rew", None), shot),
            _at(getattr(ro, "gy_rew", None), shot),
        )

        gz_reph = getattr(ro, "gz_reph", None)
        wait_te = getattr(ro, "wait_te", None)
        wait_pre = getattr(ro, "wait_pre", None)
        wait_rew = getattr(ro, "wait_rew", None)

        # The rephaser rides the first block after the pulse, and the rotation
        # only the blocks that drive an in-plane gradient.
        seq.add_block(ro.rf, ro.gz, *once)
        if wait_te is not None:
            seq.add_block(wait_te, *_present(gz_reph))
        if wait_pre is not None:
            seq.add_block(
                *pre,
                pp.scale_grad(ro.gz_pre, kz),
                *([] if wait_te is not None else _present(gz_reph)),
                wait_pre,
                *(rotation if pre else []),
            )
        seq.add_block(_at(ro.gx, shot), _at(ro.gy, shot), *acquisition, *rotation)
        if wait_rew is not None:
            seq.add_block(
                *rew,
                pp.scale_grad(ro.gz_rew, kz),
                *_present(getattr(ro, "gz_spoil", None)),
                wait_rew,
                *(rotation if rew else []),
            )
        wait_tr = getattr(ro, "wait_tr", None)
        if wait_tr is not None:
            seq.add_block(wait_tr)

    def finalize(self) -> None:
        """Write the prescription, the trajectory and its angles as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_z = self.matrix[2]
        definitions = {
            "FOV": list(self.fov),
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.ro.duration,
            "Trajectory": "stack_of_spirals",
            "NumArms": len(self.angles),
            "AngleScheme": self.angle_scheme,
            "PartitionAngleOffset": self.partition_angle_offset_deg,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreStackOfSpirals3DApp.main

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="gre_stack_of_spirals_3d.seq")
    )
