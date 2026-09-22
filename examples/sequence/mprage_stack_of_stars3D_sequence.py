"""3D MPRAGE on a stack of stars: one partition of radial spokes per inversion."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._schedules import make_rf_spoiling_schedule

#: How far each partition turns its spokes, as a fraction of the half turn a
#: spoke covers: not at all, by the golden ratio ``1 / phi``, or by the tiny
#: golden angle ``1 / (phi + 1)``.
PARTITION_SHIFTS = {
    "none": 0.0,
    "golden": 2 / (1 + math.sqrt(5)),
    "tiny_golden": 2 / (3 + math.sqrt(5)),
}


def golden_order(n: int) -> list[int]:
    """Return a play order of ``n`` evenly spread angles whose every prefix is spread too.

    The ``t``-th angle played is the rank of ``t / phi`` (mod 1) among the
    first ``n`` such values, so the angles played by any time lie close to a
    golden-angle set and a train binned in time still covers the half turn.
    """
    positions = (np.arange(n) * (math.sqrt(5) - 1) / 2) % 1.0
    return np.argsort(np.argsort(positions, kind="stable"), kind="stable").tolist()


class MprageStackOfStars3DApp(sequences.SequenceApp):
    """3D MPRAGE on a stack of stars: one inversion per partition, then its spokes.

    Each shot is the inversion, a wait that puts the first spoke's excitation
    centre at TI, one :class:`RadialStackReadout` repetition per spoke of one
    partition, and a recovery that makes every inversion-to-inversion interval
    the TR. Partitions are played in order. The Nyquist set is
    ``ceil(pi / 2 * n)`` spokes spread evenly over half a turn, and every
    ``ry``-th of them is played in a golden order (:func:`golden_order`), so a
    train binned in time still covers the half turn; one spoke waveform serves
    every repetition, turned by a rotation extension. Acquisitions carry the
    spoke as ``LIN``, the partition as ``PAR`` and the place in the train as
    ``ECO``; under partition undersampling the central ``n_acs_z`` partitions
    are acquired too, marked ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.mprage_stack_of_stars3D_sequence(
    ...     n=32, n_z=4, ti=100e-3, tr=500e-3
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "mprage_stack_of_stars_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab-selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Duration of the nonselective hard pulse (s).
    HARD_PULSE_DURATION = 0.5e-3
    #: Fat methylene shift from water (ppm), converted against ``system.B0``
    #: when the spectral-spatial pulse is built.
    FAT_SHIFT_PPM = -3.4
    #: Quadratic RF spoiling phase increment (degrees).
    RF_SPOILING_INCREMENT_DEG = 117.0
    #: Dephasing left on the partition axis at the end of each repetition, in
    #: cycles across one voxel.
    SPOILING_CYCLES = 4.0
    #: Pacing of one three-plane navigator (s), and the most one recovery
    #: carries: each takes a little longitudinal magnetisation from the volume
    #: the recovery is restoring.
    NAVIGATOR_TR = 100e-3
    NAVIGATOR_COUNT = 5

    def init_sequence(
        self,
        fov: float = 256e-3,
        n: int = 256,
        fov_z: float = 176e-3,
        n_z: int = 176,
        flip_angle_deg: float = 9.0,
        te: float | None = None,
        esp: float | None = None,
        ti: float | None = 900e-3,
        tr: float | None = 2300e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        rz: int = 1,
        partial_fourier_z: float = 1.0,
        *,
        n_dummy: int = 1,
        excitation: str = "slab",
        partition_angle_shift: str = "golden",
        n_acs_z: int = 16,
        readout_oversampling: float = 2.0,
        navigator: bool = False,
    ) -> None:
        """Design the inversion, the spoke, the angles and the shot timing.

        Parameters
        ----------
        fov : float, default=0.256
            Isotropic in-plane field of view (m).
        n : int, default=256
            In-plane matrix size; a spoke reads it edge to edge.
        fov_z : float, default=0.176
            Field of view along the partitions (m). The slab excited is
            ``fov_z`` thick.
        n_z : int, default=176
            Number of partitions.
        flip_angle_deg : float, default=9.0
            Readout excitation flip angle (degrees).
        te : float | None, default=None
            Echo time to the spoke's centre crossing (s). ``None`` is as short
            as possible.
        esp : float | None, default=None
            Spacing of successive spoke excitations (s). ``None`` is as short
            as the readout admits.
        ti : float | None, default=0.9
            Inversion time (s), from the inversion pulse's centre to the first
            spoke's excitation. ``None`` is as short as the inversion module
            admits.
        tr : float | None, default=2.3
            Inversion-to-inversion interval (s). ``None`` leaves one raster of
            recovery after the train.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry : int, default=1
            Angular undersampling: one spoke in every ``ry`` of the Nyquist set
            is played.
        rz : int, default=1
            Partition undersampling: one partition in every ``rz`` is
            acquired, the centre one among them.
        partial_fourier_z : float, default=1.0
            Fraction of the partition extent acquired, in ``[0.75, 1]``.
        n_dummy : int, default=1
            Whole shots played without acquiring before the first.
        excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        partition_angle_shift : {'none', 'golden', 'tiny_golden'}, default='golden'
            How far each partition turns the spokes past the previous one, as
            :data:`PARTITION_SHIFTS` names the fractions of a half turn.
        n_acs_z : int, default=16
            Fully sampled calibration partitions at the centre, acquired when
            ``rz > 1``.
        readout_oversampling : float, default=2.0
            Readout oversampling factor, at least one.
        navigator : bool, default=False
            Play three-plane spiral navigators in the recovery after each
            shot, as many as it holds up to :attr:`NAVIGATOR_COUNT`.

        Raises
        ------
        ValueError
            If ``excitation`` or ``partition_angle_shift`` is unknown, an
            undersampling factor is below one, ``partial_fourier_z`` is outside
            ``[0.75, 1]``, or the TE, the spacing, the TI or the TR is shorter
            than the shot takes.
        """
        self.n_dummy = n_dummy
        if excitation not in sequences.EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {sequences.EXCITATIONS}, got {excitation!r}"
            )
        if partition_angle_shift not in PARTITION_SHIFTS:
            raise ValueError(
                f"partition_angle_shift must be one of {tuple(PARTITION_SHIFTS)}, "
                f"got {partition_angle_shift!r}"
            )
        if ry < 1 or rz < 1:
            raise ValueError(f"ry and rz must be at least 1, got {ry} and {rz}")
        if not 0.75 <= partial_fourier_z <= 1.0:
            raise ValueError(
                f"partial_fourier_z must lie in [0.75, 1], got {partial_fourier_z}"
            )

        system = self.system
        self.fov, self.matrix, self.fov_z = fov, (n, n, n_z), fov_z
        self.excitation = excitation
        self.partition_angle_shift = partition_angle_shift
        self.inv = sequences.InversionPreparation(
            system, voxel_size_m=min(fov / n, fov_z / n_z)
        )
        self.exc = sequences.make_excitation(
            system,
            excitation,
            flip_angle_deg,
            fov_z,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            hard_duration_s=self.HARD_PULSE_DURATION,
            fat_shift_ppm=self.FAT_SHIFT_PPM,
        )
        self.gz = getattr(self.exc, "gz", None)
        self.ro = ro = sequences.RadialStackReadout(
            system,
            self.exc.rf,
            self.gz,
            fov=fov,
            matrix=n,
            fov_z=fov_z,
            matrix_z=n_z,
            te=te,
            tr=esp,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=self.SPOILING_CYCLES,
        )
        self.esp = ro.duration

        # A full spoke covers half a turn, which the Nyquist set divides evenly.
        n_nyquist = math.ceil(np.pi / 2 * n)
        self.span = np.pi
        self.angles = self.span * np.arange(0, n_nyquist, ry) / n_nyquist
        self.spokes = golden_order(len(self.angles))
        self.shift = PARTITION_SHIFTS[partition_angle_shift] * self.span
        calibrating, lattice = pp.calc_sampled_lines(
            n_z, rz, n_acs_z, partial_fourier=partial_fourier_z
        )
        self.partitions = sorted({*calibrating, *lattice})
        self.calibration = set(calibrating)
        self._rotations: dict[float, object] = {}

        raster = system.block_duration_raster
        inversion_tail = self.inv.duration - (
            self.inv.rf_prep.delay + self.inv.rf_prep.center
        )
        ti_floor = inversion_tail + ro.rf.delay + ro.rf.center
        if ti is None:
            ti = ti_floor + raster
        elif ti < ti_floor + raster - 1e-9:
            raise ValueError(
                f"the requested TI of {ti * 1e3:.3f} ms is shorter than the "
                f"{(ti_floor + raster) * 1e3:.3f} ms the inversion takes"
            )
        self.wait_ti = pp.make_delay(pp.round_to_raster(ti - ti_floor, raster))
        self.ti = ti
        body = self.inv.duration + self.wait_ti.delay + len(self.spokes) * self.esp
        if tr is None:
            tr = body + raster
        recovery = tr - body
        if recovery < raster - 1e-9:
            raise ValueError(
                f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
                f"{(body + raster) * 1e3:.3f} ms one shot takes"
            )
        self.repetition_time = tr
        # Navigators ride in the recovery, where they cost no scan time.
        self.navigator, self.n_navigators, navigating = None, 0, 0.0
        if navigator:
            self.navigator = sequences.SpiralNavigator(
                system, navigator_tr=self.NAVIGATOR_TR
            )
            self.n_navigators = self.navigator.fit(
                recovery - raster, "auto", limit=self.NAVIGATOR_COUNT
            )
            navigating = self.n_navigators * self.navigator.duration
        self.wait_recovery = pp.make_delay(
            pp.round_to_raster(recovery - navigating, raster)
        )
        self.duration = (n_dummy + len(self.partitions)) * (
            body + navigating + self.wait_recovery.delay
        )

    def rotation(self, spoke: int, partition: int):
        """Return the rotation extension turning ``spoke`` at ``partition``."""
        angle = float((self.angles[spoke] + partition * self.shift) % self.span)
        key = round(angle, 12)
        if key not in self._rotations:
            self._rotations[key] = pp.make_rotation(angle)
        return self._rotations[key]

    def loop(self) -> None:
        """Play the dummy shots, then every acquired partition in order."""
        shots = [None] * self.n_dummy + self.partitions
        n = len(self.spokes)
        phases = make_rf_spoiling_schedule(
            len(shots) * n, increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        ).reshape(len(shots), n)
        for partition, shot_phases in zip(shots, phases, strict=True):
            self.kernel(partition, shot_phases)

    def kernel(self, partition: int | None, phases) -> None:
        """One inversion-prepared shot of every spoke at ``partition``.

        ``partition=None`` plays a dummy at the centre partition. ``phases``
        are the RF-spoiling phases (rad) of its repetitions. The readout's
        blocks after the pulse are played as it laid them out, with the
        partition encode scaled and every block that drives an in-plane
        gradient turned to the spoke's angle.
        """
        inv, ro, seq = self.inv, self.ro, self.seq
        n_z = self.matrix[2]
        acquire = partition is not None
        z = n_z // 2 if partition is None else partition
        kz = (z - n_z // 2) / (n_z / 2)
        encoded = {
            id(ro.gz_pre): pp.scale_grad(ro.gz_pre, kz),
            id(ro.gz_rew): pp.scale_grad(ro.gz_rew, kz),
        }
        if acquire:
            flags = {"ONCE": 0} if self.n_dummy else {}
            flags.update(PAR=z, IMA=z in self.calibration)
        else:
            flags = {"ONCE": 1}

        seq.add_block(inv.rf_prep, *self.labels(**flags))
        seq.add_block(inv.gz_spoil)
        seq.add_block(self.wait_ti)
        for echo, (spoke, phase) in enumerate(zip(self.spokes, phases, strict=True)):
            self.exc.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            rotation = self.rotation(spoke, z)
            seq.add_block(self.exc.rf, *([] if self.gz is None else [self.gz]))
            for block in ro.blocks[1:]:
                events = [
                    encoded.get(id(event), event)
                    for event in block
                    if acquire or event is not ro.adc
                ]
                if acquire and any(event is ro.adc for event in block):
                    events += self.labels(LIN=spoke, ECO=echo)
                if any(getattr(e, "channel", None) in ("x", "y") for e in events):
                    events.append(rotation)
                seq.add_block(*events)
        for _ in range(self.n_navigators):
            for block in self.navigator.blocks:
                seq.add_block(*block)
        seq.add_block(self.wait_recovery)

    def finalize(self) -> None:
        """Write the prescription, the trajectory and the shot timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_z = self.matrix[2]
        definitions = {
            "FOV": [self.fov, self.fov, self.fov_z],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "TI": self.ti,
            "EchoSpacing": self.esp,
            "EchoTrainLength": len(self.spokes),
            "Trajectory": "stack_of_stars",
            "Excitation": self.excitation,
            "NumSpokes": len(self.angles),
            "PartitionAngleShift": self.partition_angle_shift,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov_z,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = MprageStackOfStars3DApp.main

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="mprage_stack_of_stars_3d.seq")
    )
