"""3D MPRAGE on a golden-angle stack of spirals."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


def stack_angles(
    angles: np.ndarray, n_z: int, partition_offset_deg: float
) -> tuple[np.ndarray, bool]:
    """In-plane angle (rad) of every distinct shot of a stack.

    Without a partition offset an arm plays at the same angle in every
    partition, one angle per arm. An offset turns it further with each
    partition, and every ``(arm, partition)`` pair is a shot of its own, at
    index ``arm * n_z + partition``. The flag says which.
    """
    angles = np.asarray(angles, dtype=float)
    step = np.deg2rad(float(partition_offset_deg))
    if not step:
        return angles, False
    return (angles[:, None] + step * np.arange(n_z)[None, :]).ravel(), True


class MprageStackOfSpirals3DApp(sequences.SequenceApp):
    """3D MPRAGE whose shots play spiral arms of one partition.

    Partitions are the outer loop: each inversion is followed by ``etl`` arms
    of one partition, the first echo at TI. Arms step by the golden angle, so
    any prefix of the arms covers the disc near-uniformly; a partition angle
    offset staggers them along kz. One arm is held and turned per shot by a
    rotation extension, or with ``use_rotation_ext=False`` every shot is
    written out as its own waveform. ``LIN`` carries the arm, ``PAR`` the
    partition and ``SEG`` the inversion train.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.mprage_stack_of_spirals3D_sequence(
    ...     n_x=32, n_z=8, n_arms=4, ti=100e-3, tr_outer=300e-3
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "mprage_stack_of_spirals_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab excitation.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Chemical shift of fat from water (ppm) that the spectral-spatial
    #: excitation suppresses, converted against ``system.B0``.
    FAT_SHIFT_PPM = -3.4

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        n_z: int = 64,
        slab_thickness: float = 128e-3,
        n_arms: int = 16,
        design_interleaves: int | None = None,
        flip_angle_deg: float = 9.0,
        ti: float = 900e-3,
        tr_outer: float = 2000e-3,
        te: float | None = None,
        readout_bandwidth_hz: float = 250e3,
        n_dummy: int = 1,
        n_gain_calibration_readouts: int = 1,
        rf_spoiling_increment_deg: float = 117.0,
        spoiling_cycles: float = 4.0,
        etl: int | None = None,
        angular_increment_deg: float | None = None,
        partition_angle_offset_deg: float = 0.0,
        use_rotation_ext: bool = True,
        spsp: bool = False,
    ) -> None:
        """Design the arms, their rotations and the shot timing around them.

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
            Arms per partition, stepping by ``angular_increment_deg``.
        design_interleaves : int or None, optional
            The pitch the spiral is designed for. ``None`` matches ``n_arms``.
        flip_angle_deg : float, optional
            Readout excitation flip angle, in degrees.
        ti : float, optional
            Inversion time, in seconds, from the inversion pulse's centre to
            the echo of the train's first arm.
        tr_outer : float, optional
            Inversion-to-inversion interval, in seconds.
        te : float or None, optional
            Readout echo time, in seconds. ``None`` is as short as possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        n_dummy : int, optional
            Whole shots played without acquiring before the first acquired
            one, so the scan measures the magnetisation's steady state.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left at the end of each inner repetition.
        etl : int or None, optional
            Arms per inversion train; it must divide ``n_arms``, so every train
            samples its arms at the same points of the recovery. ``None`` is
            one train per partition.
        angular_increment_deg : float or None, optional
            Angle between successive arms, in degrees. ``None`` is the golden
            angle.
        partition_angle_offset_deg : float, optional
            Angle added per partition step, in degrees, staggering the arms
            along kz.
        use_rotation_ext : bool, optional
            Turn one stored arm per shot with a rotation extension, rather than
            writing every shot out as its own waveform.
        spsp : bool, optional
            Excite with a spectral-spatial, slab- and water-selective pulse.
        """
        system = self.system
        self.fov = (fov, fov, slab_thickness)
        self.matrix = (n_x, n_x, n_z)
        self.ti, self.tr_outer = ti, tr_outer
        self.n_dummy = n_dummy
        self.n_gain_calibration_readouts = n_gain_calibration_readouts
        self.spoiling_increment = np.deg2rad(rf_spoiling_increment_deg)
        self.use_rotation_ext = use_rotation_ext
        self.partition_angle_offset_deg = partition_angle_offset_deg

        self.etl = n_arms if etl is None else int(etl)
        if self.etl < 1 or n_arms % self.etl:
            raise ValueError(
                f"etl must divide the {n_arms} arms evenly, and {self.etl} does not; "
                "a short final train would sample its arms at a different point of "
                "the recovery than the others"
            )
        self.n_arms = n_arms

        self.inv = sequences.InversionPreparation(
            system, voxel_size_m=min(fov / n_x, slab_thickness / n_z)
        )
        if spsp:
            fat_offset_hz = self.FAT_SHIFT_PPM * 1e-6 * system.gamma * system.B0
            self.exc = sequences.SpspExcitation(
                system,
                flip_angle_deg,
                thickness_m=slab_thickness,
                spectral_bandwidth_hz=abs(fat_offset_hz),
                freq_offset_hz=0.0,
                is_slab=True,
            )
        else:
            self.exc = sequences.SpatialSelectiveExcitation(
                system,
                flip_angle_deg,
                slab_thickness,
                duration_s=self.PULSE_DURATION,
                is_slab=True,
                time_bw_product=self.TIME_BW_PRODUCT,
            )

        if angular_increment_deg is None:
            angles = np.asarray(pp.calc_golden_angles(n_arms, full_circle=True))
            self.angular_increment = float(
                np.diff(pp.calc_golden_angles(2, full_circle=True))[0]
            )
        else:
            self.angular_increment = np.deg2rad(float(angular_increment_deg))
            angles = np.mod(np.arange(n_arms) * self.angular_increment, 2 * np.pi)
        self.angle_scheme = "golden" if angular_increment_deg is None else "uniform"
        shot_angles, self.staggered = stack_angles(
            angles, n_z, partition_angle_offset_deg
        )
        shot_angles = np.mod(shot_angles, 2 * np.pi)

        self.ro = sequences.SpiralStackReadout(
            system,
            self.exc.rf,
            self.exc.gz,
            None,
            fov=fov,
            matrix=n_x,
            fov_z=slab_thickness,
            matrix_z=n_z,
            design_interleaves=n_arms
            if design_interleaves is None
            else design_interleaves,
            explicit=not use_rotation_ext,
            angles=None if use_rotation_ext else shot_angles,
            te=te,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
        )
        # One rotation event per distinct angle, reused by every shot at it.
        self.rotations = (
            [pp.make_rotation(float(angle)) for angle in shot_angles]
            if use_rotation_ext
            else [None] * len(shot_angles)
        )
        self.inner_tr = self.ro.duration

        # From the inversion centre to the first arm's echo: the rest of the
        # inversion module, the TI wait, and the first repetition up to its echo.
        raster = system.block_duration_raster
        rf_prep = self.inv.rf_prep
        inversion_tail = self.inv.duration - (rf_prep.delay + rf_prep.center)
        ti_floor = (
            inversion_tail + self.ro.rf.delay + self.ro.rf.center + self.ro.echo_time
        )
        if ti < ti_floor:
            raise ValueError(
                f"TI {ti * 1e3:.0f} ms is shorter than the inversion tail and the "
                f"first arm's echo time admit; the minimum is {ti_floor * 1e3:.0f} ms"
            )
        self.wait_ti = pp.make_delay(
            max(raster, pp.round_to_raster(ti - ti_floor, raster))
        )

        body = self.inv.duration + self.wait_ti.delay + self.etl * self.inner_tr
        if tr_outer < body:
            raise ValueError(
                f"TR {tr_outer * 1e3:.0f} ms is shorter than one segment takes "
                f"({body * 1e3:.0f} ms)"
            )
        self.wait_recovery = pp.make_delay(
            max(raster, pp.round_to_raster(tr_outer - body, raster))
        )
        self.duration = n_z * (n_arms // self.etl) * (body + self.wait_recovery.delay)

    def loop(self) -> None:
        """Play the dummy shots, then every partition's trains of arms."""
        etl, n_z = self.etl, self.matrix[2]
        n_shots = self.n_dummy + n_z * self.n_arms // etl
        phases = iter(
            pp.make_rf_spoiling_schedule(
                n_shots * etl, increment=self.spoiling_increment
            ).reshape(n_shots, etl)
        )
        for _ in range(self.n_dummy):
            self.kernel(range(etl), 0, next(phases), acquire=False, flags={"ONCE": 1})
        once = {"ONCE": 0} if self.n_dummy else {}
        train = 0
        for partition in range(n_z):
            for start in range(0, self.n_arms, etl):
                arms = range(start, start + etl)
                self.kernel(arms, partition, next(phases), flags={**once, "SEG": train})
                train += 1

    def kernel(
        self,
        arms: range,
        partition: int,
        phases: np.ndarray,
        acquire: bool = True,
        flags: dict[str, int] | None = None,
    ) -> None:
        """One inversion-prepared shot: ``arms`` of one partition.

        ``phases`` are the RF-spoiling phases (rad) of its repetitions, and
        ``flags`` the labels the shot carries on its inversion.
        """
        inv, ro, seq = self.inv, self.ro, self.seq
        n_z = self.matrix[2]
        kz = (partition - n_z / 2) / (n_z / 2)

        seq.add_block(inv.rf_prep, *self.labels(**(flags or {})))
        seq.add_block(inv.gz_spoil)
        seq.add_block(self.wait_ti)
        for arm, phase in zip(arms, phases, strict=True):
            shot = arm * n_z + partition if self.staggered else arm
            rotation = self.rotations[shot]
            # The slab sits at isocentre, so the excitation has no frequency
            # offset and its phase is the spoiling phase alone.
            ro.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            # The arm's blocks as the readout laid them out, with the partition
            # encode scaled, the rotation on every block driving x or y, and a
            # dummy's ADC dropped.
            for block in ro.arm(shot):
                events = [
                    pp.scale_grad(e, kz) if e is ro.gz_pre or e is ro.gz_rew else e
                    for e in block
                    if acquire or e is not ro.adc
                ]
                if rotation is not None and any(
                    getattr(e, "channel", "") in ("x", "y") for e in events
                ):
                    events.append(rotation)
                if acquire and any(e is ro.adc for e in block):
                    events += self.labels(LIN=arm, PAR=partition)
                seq.add_block(*events)
        seq.add_block(self.wait_recovery)

    def finalize(self) -> None:
        """Write the prescription, the shot timing and the arm scheme as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": list(self.fov),
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.tr_outer,
            "TI": self.ti,
            "InnerTR": self.inner_tr,
            "Trajectory": "stack_of_spirals",
            "NumArms": self.n_arms,
            "AngleScheme": self.angle_scheme,
            "AngularIncrement": np.rad2deg(self.angular_increment),
            "PartitionAngleOffset": self.partition_angle_offset_deg,
            "EchoTrainLength": self.etl,
            "ExplicitArms": 0 if self.use_rotation_ext else 1,
            "kSpaceCenterPartition": self.matrix[2] // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = MprageStackOfSpirals3DApp.main

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="mprage_stack_of_spirals_3d.seq")
    )
