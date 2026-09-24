"""3D stack-of-blades (PROPELLER) spin echo."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: How far each partition turns its tilts, as a fraction of the half turn a
#: blade covers: not at all, by the golden ratio ``1 / phi``, or by the tiny
#: golden angle ``1 / (phi + 1)``.
PARTITION_SHIFTS = {
    "none": 0.0,
    "golden": 2 / (1 + math.sqrt(5)),
    "tiny_golden": 2 / (3 + math.sqrt(5)),
}


class SeStackOfBlades3DApp(sequences.SequenceApp):
    """3D stack-of-blades spin echo: one blade line at one partition per excitation.

    A 90 from the selected excitation, a nonselective 180 between crushers
    half a TE later, and one frequency-encoded line at the echo. Blades, lines
    and partitions are chosen and ordered as
    :mod:`gre_stack_of_blades3D_sequence` chooses them. Acquisitions carry the
    line as ``LIN``, the blade as ``SEG`` and the partition as ``PAR``.

    Under partition undersampling the central ``n_acs_z`` partitions are
    acquired in full at every tilt, ahead of the rest, and marked ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_stack_of_blades3D_sequence(
    ...     n=32, n_z=4, blade_width=8, tr=None
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_stack_of_blades_3d"
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
    #: Readout oversampling factor.
    READOUT_OVERSAMPLING = 2.0
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
        blade_width: int = 16,
    ) -> None:
        """Design the pulses, the readout, the blade angles and the partitions.

        Parameters
        ----------
        fov : float, default=0.22
            Isotropic in-plane field of view (m).
        n : int, default=128
            In-plane matrix size.
        fov_z : float, default=0.128
            Field of view along the partitions (m). The slab excited is
            ``fov_z`` thick.
        n_z : int, default=64
            Number of partitions.
        te : float | None, default=None
            Echo time (s), excitation centre to echo, with the refocusing
            pulse at its midpoint. ``None`` is as short as possible.
        tr : float | None, default=0.1
            Repetition time (s), one per excitation. ``None`` is as short as
            possible.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry : int, default=1
            Angular undersampling: one blade in every ``ry`` of the Nyquist set
            is played.
        rz : int, default=1
            Partition undersampling: one partition in every ``rz`` is
            acquired, the centre one among them.
        partial_fourier_z : float, default=1.0
            Fraction of the partition extent acquired, in ``[0.75, 1]``.
        n_dummy : int, default=0
            Non-acquiring repetitions, at the first blade's centre line and the
            centre partition, before the scan.
        excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        partition_angle_shift : {'none', 'golden', 'tiny_golden'}, default='none'
            How far each partition turns the blades past the previous one, as
            :data:`PARTITION_SHIFTS` names the fractions of a half turn.
        n_acs_z : int, default=16
            Fully sampled calibration partitions at the centre, acquired at
            every tilt ahead of the rest when ``rz > 1``.
        blade_width : int, default=16
            Phase-encode lines per blade, at most ``n``.

        Raises
        ------
        ValueError
            If ``excitation`` or ``partition_angle_shift`` is unknown, an
            undersampling factor is below one, ``blade_width`` is outside
            ``[1, n]``, ``partial_fourier_z`` is outside ``[0.75, 1]``, or the
            TE or TR is shorter than the pulses and the readout take.
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
        if not 1 <= blade_width <= n:
            raise ValueError(f"blade_width must lie in [1, {n}], got {blade_width}")
        if not 0.75 <= partial_fourier_z <= 1.0:
            raise ValueError(
                f"partial_fourier_z must lie in [0.75, 1], got {partial_fourier_z}"
            )

        system = self.system
        self.fov, self.matrix = fov, (n, n, n_z)
        self.fov_z = fov_z
        self.blade_width = blade_width
        self.excitation = excitation
        self.partition_angle_shift = partition_angle_shift
        self.exc = sequences.make_excitation(
            system,
            excitation,
            90.0,
            fov_z,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            hard_duration_s=self.HARD_PULSE_DURATION,
            fat_shift_ppm=self.FAT_SHIFT_PPM,
        )
        self.gz = getattr(self.exc, "gz", None)
        self.ref = sequences.NonSelectiveRefocusing(
            system, spoiling_cycles=self.CRUSHER_CYCLES
        )
        # The in-plane axes turn with the blade, so a spoiler on them would
        # point a different way from one blade to the next: it plays on z.
        self.gz_spoil, _, _ = pp.make_crusher(
            self.SPOILING_CYCLES, fov_z / n_z, "z", system=system
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
            return sequences.LineReadout3D(
                system,
                rf,
                None,
                fov=(fov, fov, fov_z),
                matrix=self.matrix,
                te=None if half_te is None else half_te - crusher,
                oversampling=self.READOUT_OVERSAMPLING,
                readout_bandwidth_hz=readout_bandwidth_hz,
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

        # A blade turned by half a turn is the same blade, which the Nyquist
        # set divides evenly.
        n_nyquist = math.ceil(np.pi * n / (2 * blade_width))
        self.span = np.pi
        self.angles = self.span * np.arange(0, n_nyquist, ry) / n_nyquist
        self.shift = PARTITION_SHIFTS[partition_angle_shift] * self.span
        calibration, imaging = pp.make_cartesian_axis_sampling(
            n_z, rz, n_acs_z, partial_fourier=partial_fourier_z
        )
        self.calibration = set(calibration)
        self.partitions = sorted([*calibration, *imaging])
        # The calibration partitions lead, at every tilt, then the rest.
        self.views = [
            (blade, line, z)
            for partitions in (calibration, imaging)
            for blade in range(len(self.angles))
            for line in range(blade_width)
            for z in partitions
        ]
        self._rotations: dict[float, object] = {}

        # The readout's first block, the 180 alone, is not played; the
        # partition-axis spoiler closes the repetition.
        length = (
            self.exc.duration
            + wait
            + self.ref.duration
            + self.ro.duration
            - rf_span
            + pp.ceil_to_raster(pp.calc_duration(self.gz_spoil), raster)
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
        self.resolve(
            te=self.echo_time,
            tr=self.repetition_time,
            readout_bandwidth_hz=self.ro.bandwidth_hz,
        )

    def rotation(self, blade: int, partition: int):
        """Return the rotation extension turning ``blade`` at ``partition``.

        Parameters
        ----------
        blade : int
            The blade to turn to.
        partition : int
            The partition it is played at, which advances the angle.

        Returns
        -------
        object
            The rotation extension, shared between shots at the same angle.
        """
        angle = float((self.angles[blade] + partition * self.shift) % self.span)
        key = round(angle, 12)
        if key not in self._rotations:
            self._rotations[key] = pp.make_rotation(angle)
        return self._rotations[key]

    def loop(self) -> None:
        """Play the dummies, then every acquired partition of each blade line in turn."""
        for view in [None] * self.n_dummy + self.views:
            self.kernel(view)

    def kernel(self, view: tuple[int, int, int] | None) -> None:
        """One spin echo at ``(blade, line, partition)``; ``None`` plays a dummy.

        Every block that drives an in-plane gradient carries the shot's
        rotation, which turns about z and leaves the partition encode alone.

        Parameters
        ----------
        view : tuple of int or None
            The blade, the line within it and the partition to acquire, or
            None for a dummy.
        """
        ro, seq = self.ro, self.seq
        n, n_z = self.matrix[1:]
        if view is None:
            blade, line, partition = 0, self.blade_width // 2, n_z // 2
            labels = self.labels(ONCE=1)
        else:
            blade, line, partition = view
            labels = self.labels(
                LIN=line,
                SEG=blade,
                PAR=partition,
                IMA=partition in self.calibration,
                ONCE=0,
            )
        ky = (line - self.blade_width // 2) / (n / 2)
        kz = (partition - n_z // 2) / (n_z / 2)
        rotation = self.rotation(blade, partition)

        seq.add_block(self.exc.rf, *([] if self.gz is None else [self.gz]), *labels)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        for block in self.ref.blocks:
            seq.add_block(*block)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            ro.gx_pre,
            pp.scale_grad(ro.gy_pre, ky),
            pp.scale_grad(ro.gz_pre, kz),
            rotation,
        )
        seq.add_block(ro.gx, *([] if view is None else [ro.adc]), rotation)
        seq.add_block(
            ro.gx_spoil,
            pp.scale_grad(ro.gy_rew, ky),
            pp.scale_grad(ro.gz_rew, kz),
            rotation,
        )
        seq.add_block(self.gz_spoil)
        if self.wait_tr is not None:
            seq.add_block(self.wait_tr)

    def finalize(self) -> None:
        """Write the prescription, the blade set and the timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_z = self.matrix[2]
        definitions = {
            "FOV": [self.fov, self.fov, self.fov_z],
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": self.echo_time,
            "TR": self.repetition_time,
            "Trajectory": "stack_of_blades",
            "Excitation": self.excitation,
            "BladeWidth": self.blade_width,
            "NumBlades": len(self.angles),
            "PartitionAngleShift": self.partition_angle_shift,
            "kSpaceCenterLine": self.blade_width // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov_z,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = SeStackOfBlades3DApp.main

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="se_stack_of_blades_3d.seq")
    )
