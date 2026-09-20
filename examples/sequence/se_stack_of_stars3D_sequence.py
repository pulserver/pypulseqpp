"""3D stack-of-stars spin echo."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The excitations ``excitation`` selects from.
EXCITATIONS = ("nonselective", "slab", "spsp")

#: How far each partition turns its tilts, as a fraction of the half turn a
#: spoke covers: not at all, by the golden ratio ``1 / phi``, or by the tiny
#: golden angle ``1 / (phi + 1)``.
PARTITION_SHIFTS = {
    "none": 0.0,
    "golden": 2 / (1 + math.sqrt(5)),
    "tiny_golden": 2 / (3 + math.sqrt(5)),
}


def make_excitation(app, flip_angle_deg: float, kind: str, thickness: float):
    """Build the excitation ``kind`` names, from the application's pulse settings."""
    system = app.system
    if kind == "nonselective":
        # An even number of block rasters puts the pulse centre on the raster,
        # which a spin echo needs to place its 180 midway.
        raster = system.block_duration_raster
        duration = 2 * raster * math.ceil(app.HARD_PULSE_DURATION / (2 * raster) - 1e-9)
        return sequences.NonSelectiveExcitation(
            system, flip_angle_deg, duration_s=duration
        )
    if kind == "slab":
        return sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            thickness,
            duration_s=app.PULSE_DURATION,
            time_bw_product=app.TIME_BW_PRODUCT,
            is_slab=True,
        )
    fat_offset_hz = app.FAT_SHIFT_PPM * 1e-6 * system.gamma * system.B0
    return sequences.SpspExcitation(
        system,
        flip_angle_deg,
        thickness_m=thickness,
        spectral_bandwidth_hz=abs(fat_offset_hz),
        is_slab=True,
    )


def sampled_partitions(
    n: int, rz: int, n_acs_z: int, partial_fourier: float
) -> tuple[list[int], list[int]]:
    """Return the calibration partitions and the other partitions acquired.

    The lattice keeps every partition ``z`` with ``(z - n // 2) % rz == 0``, so
    the centre partition is always acquired; partial Fourier drops those before
    ``n - round(partial_fourier * n)``. Under undersampling the ``n_acs_z``
    partitions centred on the same one are acquired in full; a fully sampled
    scan has none. Both lists are in order.
    """
    first = n - round(partial_fourier * n)
    size = n_acs_z if rz > 1 else 0
    calibration = list(
        range(max(n // 2 - size // 2, first), min(n // 2 + (size + 1) // 2, n))
    )
    lattice = [
        z for z in range(first, n) if (z - n // 2) % rz == 0 and z not in calibration
    ]
    return calibration, lattice


class SeStackOfStars3DApp(sequences.SequenceApp):
    """3D stack-of-stars spin echo: one spoke at one partition per excitation.

    A 90 from the selected excitation, a nonselective 180 between crushers
    half a TE later, and one spoke through the centre of the partition's plane
    at the echo. Spokes and partitions are chosen and ordered as
    :mod:`gre_stack_of_stars3D_sequence` chooses them. Acquisitions carry the
    spoke as ``LIN`` and the partition as ``PAR``.

    Under partition undersampling the central ``n_acs_z`` partitions are
    acquired in full at every tilt, ahead of the rest, and marked ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_stack_of_stars3D_sequence(n=32, n_z=4, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_stack_of_stars_3d"
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
    #: Dephasing each crusher beside the refocusing pulse winds, in cycles
    #: across one voxel.
    CRUSHER_CYCLES = 4.0
    #: Dephasing left on the partition axis at the end of each repetition, in
    #: cycles across one voxel.
    SPOILING_CYCLES = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n: int = 128,
        fov_z: float = 128e-3,
        n_z: int = 64,
        te: float | None = None,
        tr: float | None = 100e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        rz: int = 1,
        partial_fourier_z: float = 1.0,
        *,
        n_dummy: int = 0,
        excitation: str = "slab",
        partition_angle_shift: str = "none",
        n_acs_z: int = 16,
        readout_oversampling: float = 2.0,
    ) -> None:
        """Design the pulses, the spoke, the spoke angles and the partitions.

        Parameters
        ----------
        fov : float, default=0.22
            Isotropic in-plane field of view (m).
        n : int, default=128
            In-plane matrix size; a spoke reads it edge to edge.
        fov_z : float, default=0.128
            Field of view along the partitions (m). The slab excited is
            ``fov_z`` thick.
        n_z : int, default=64
            Number of partitions.
        te : float | None, default=None
            Echo time (s), excitation centre to the spoke's centre crossing,
            with the refocusing pulse at its midpoint. ``None`` is as short as
            possible.
        tr : float | None, default=0.1
            Repetition time (s), one per excitation. ``None`` is as short as
            possible.
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
        n_dummy : int, default=0
            Non-acquiring repetitions, at the first spoke's angle and the
            centre partition, before the scan.
        excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        partition_angle_shift : {'none', 'golden', 'tiny_golden'}, default='none'
            How far each partition turns the spokes past the previous one, as
            :data:`PARTITION_SHIFTS` names the fractions of a half turn.
        n_acs_z : int, default=16
            Fully sampled calibration partitions at the centre, acquired at
            every tilt ahead of the rest when ``rz > 1``.
        readout_oversampling : float, default=2.0
            Readout oversampling factor, at least one.

        Raises
        ------
        ValueError
            If ``excitation`` or ``partition_angle_shift`` is unknown, an
            undersampling factor is below one, ``partial_fourier_z`` is outside
            ``[0.75, 1]``, or the TE or TR is shorter than the pulses and the
            readout take.
        """
        self.n_dummy = n_dummy
        if excitation not in EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {EXCITATIONS}, got {excitation!r}"
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
        self.fov, self.matrix = fov, (n, n, n_z)
        self.fov_z = fov_z
        self.excitation = excitation
        self.partition_angle_shift = partition_angle_shift
        self.exc = make_excitation(self, 90.0, excitation, fov_z)
        self.gz = getattr(self.exc, "gz", None)
        self.ref = sequences.NonSelectiveRefocusing(
            system, spoiling_cycles=self.CRUSHER_CYCLES
        )

        # TE is solved in two halves about the 180. The readout owns the second
        # but counts it through a block holding the 180 alone, where the
        # refocusing module plays the 180 and then its crusher: the readout's
        # half is shortened by what that crusher adds.
        raster = system.block_duration_raster
        rf = self.ref.rf_ref
        rf_span = pp.ceil_to_raster(pp.calc_duration(rf), raster)
        crusher = self.ref.duration - self.ref.center - (rf_span - rf.delay - rf.center)
        half_floor = self.exc.duration - self.exc.center + self.ref.center

        def readout(half_te: float | None):
            return sequences.RadialStackReadout(
                system,
                rf,
                None,
                fov=fov,
                matrix=n,
                fov_z=fov_z,
                matrix_z=n_z,
                te=None if half_te is None else half_te - crusher,
                oversampling=readout_oversampling,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=self.SPOILING_CYCLES,
            )

        te_min = 2 * max(readout(None).echo_time + crusher, half_floor)
        if te is not None and te < te_min - 1e-9:
            raise ValueError(
                f"TE {te * 1e3:.2f} ms is shorter than the {te_min * 1e3:.2f} ms "
                "the excitation and the readout admit"
            )
        # The 180 must sit midway, so both halves are solved on the block raster:
        # the excitation half waits a whole number of rasters, and the readout
        # is asked for the same span.
        target = (te_min if te is None else te) / 2 - half_floor
        wait = pp.ceil_to_raster(max(target, 0.0) - 1e-9, raster)
        self.ro = readout(half_floor + wait)
        self.wait_half_te = pp.make_delay(wait) if wait > 0 else None
        self.echo_time = 2 * (half_floor + wait)

        # A full spoke covers half a turn, which the Nyquist set divides evenly.
        n_nyquist = math.ceil(np.pi / 2 * n)
        self.span = np.pi
        self.angles = self.span * np.arange(0, n_nyquist, ry) / n_nyquist
        self.shift = PARTITION_SHIFTS[partition_angle_shift] * self.span
        calibration, lattice = sampled_partitions(n_z, rz, n_acs_z, partial_fourier_z)
        self.calibration = set(calibration)
        self.partitions = sorted([*calibration, *lattice])
        # The calibration partitions lead, at every tilt, then the rest.
        self.views = [
            (spoke, z)
            for partitions in (calibration, lattice)
            for spoke in range(len(self.angles))
            for z in partitions
        ]
        self._rotations: dict[float, object] = {}

        # The readout's first block, the 180 alone, is not played.
        length = (
            self.exc.duration + wait + self.ref.duration + self.ro.duration - rf_span
        )
        self.wait_tr = None
        if tr is not None:
            if tr < length - 1e-9:
                raise ValueError(
                    f"TR {tr * 1e3:.1f} ms is shorter than one repetition takes "
                    f"({length * 1e3:.1f} ms)"
                )
            pad = pp.round_to_raster(tr - length, raster)
            if pad > 0:
                self.wait_tr = pp.make_delay(pad)
                length += pad
        self.repetition_time = length
        self.duration = (self.n_dummy + len(self.views)) * length

    def rotation(self, spoke: int, partition: int):
        """Return the rotation extension turning ``spoke`` at ``partition``."""
        angle = float((self.angles[spoke] + partition * self.shift) % self.span)
        key = round(angle, 12)
        if key not in self._rotations:
            self._rotations[key] = pp.make_rotation(angle)
        return self._rotations[key]

    def loop(self) -> None:
        """Play the dummies, then every acquired partition of each spoke in turn."""
        for view in [None] * self.n_dummy + self.views:
            self.kernel(view)

    def kernel(self, view: tuple[int, int] | None) -> None:
        """One spin echo, one spoke at one partition; ``None`` plays a dummy.

        After the refocusing module, the readout's blocks are played as it laid
        them out, with the partition encode scaled and every block that drives
        an in-plane gradient turned to the shot's angle.
        """
        ro, seq = self.ro, self.seq
        n_z = self.matrix[2]
        if view is None:
            spoke, partition = 0, n_z // 2
            labels = self.labels(ONCE=1)
        else:
            spoke, partition = view
            labels = self.labels(
                LIN=spoke, PAR=partition, IMA=partition in self.calibration, ONCE=0
            )
        kz = (partition - n_z // 2) / (n_z / 2)
        encoded = {
            id(ro.gz_pre): pp.scale_grad(ro.gz_pre, kz),
            id(ro.gz_rew): pp.scale_grad(ro.gz_rew, kz),
        }
        rotation = self.rotation(spoke, partition)

        seq.add_block(self.exc.rf, *([] if self.gz is None else [self.gz]), *labels)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        for block in self.ref.blocks:
            seq.add_block(*block)
        for block in ro.blocks[1:]:
            events = [
                encoded.get(id(event), event)
                for event in block
                if view is not None or event is not ro.adc
            ]
            if any(getattr(event, "channel", None) in ("x", "y") for event in events):
                events.append(rotation)
            seq.add_block(*events)
        if self.wait_tr is not None:
            seq.add_block(self.wait_tr)

    def finalize(self) -> None:
        """Write the prescription, the trajectory and its angles as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_z = self.matrix[2]
        definitions = {
            "FOV": [self.fov, self.fov, self.fov_z],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.echo_time,
            "TR": self.repetition_time,
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


main = SeStackOfStars3DApp.main

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="se_stack_of_stars_3d.seq")
    )
