"""3D MPRAGE: inversion-prepared, RF-spoiled Cartesian gradient echo, one partition per inversion."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._masks import make_poisson_disc_mask
from pypulseqpp._schedules import make_rf_spoiling_schedule

#: The excitations ``excitation`` selects from.
EXCITATIONS = ("nonselective", "slab", "spsp")

#: The line orders ``ordering`` selects from.
ORDERINGS = ("radial", "shuffling")


def make_excitation(app, flip_angle_deg: float, kind: str, thickness: float):
    """Build the excitation ``kind`` names, from the application's pulse settings."""
    system = app.system
    if kind == "nonselective":
        # An even number of block rasters puts the pulse centre on the raster.
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


def sampled_views(
    shape: tuple[int, int],
    acceleration: tuple[int, int],
    caipi_shift: int,
    calibration: tuple[int, int],
    partial_fourier: tuple[float, float],
    elliptical_acs: bool,
    *,
    shuffling: bool = False,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Return the ``(line, partition)`` views sampled, and the calibration views.

    Only views inside the ellipse inscribed in the ``ny x nz`` grid are
    sampled, and partial Fourier drops the lines and partitions before the
    centre. The regular set keeps a CAIPIRINHA lattice holding the centre
    view: lines with ``(y - ny // 2) % ry == 0``, and in the ``j``-th of them
    from the centre the partitions with ``(z - nz // 2 - caipi_shift * j) % rz
    == 0``. The ``shuffling`` set is a variable-density Poisson-disc draw at
    ``ry * rz``. Under undersampling the ``n_acs_y x n_acs_z`` calibration
    region, centred on the centre view, is sampled whole: a rectangle, or under
    ``elliptical_acs`` the ellipse inscribed in it.
    """
    (ny, nz), (ry, rz) = shape, acceleration
    first_y = ny - round(partial_fourier[0] * ny)
    first_z = nz - round(partial_fourier[1] * nz)
    n_acs_y, n_acs_z = calibration if ry * rz > 1 else (0, 0)

    def inside(y: int, z: int, extent_y: int, extent_z: int) -> bool:
        # Offsets from the centre view, the one the encodes scale to zero.
        dy, dz = (y - ny // 2) / extent_y, (z - nz // 2) / extent_z
        return dy * dy + dz * dz <= 0.25

    region = {
        (y, z)
        for y in range(
            max(ny // 2 - n_acs_y // 2, first_y), min(ny // 2 + (n_acs_y + 1) // 2, ny)
        )
        for z in range(
            max(nz // 2 - n_acs_z // 2, first_z), min(nz // 2 + (n_acs_z + 1) // 2, nz)
        )
        if not elliptical_acs or inside(y, z, n_acs_y, n_acs_z)
    }
    if shuffling:
        mask = (
            make_poisson_disc_mask(
                (ny, nz), float(ry * rz), calib=(n_acs_y, n_acs_z), seed=seed
            )
            if ry * rz > 1
            else np.ones((ny, nz), dtype=bool)
        )
        kept = {(int(y), int(z)) for y, z in np.argwhere(mask)}
    else:
        kept = {
            (y, z)
            for y in range(ny)
            if (y - ny // 2) % ry == 0
            for z in range(nz)
            if (z - nz // 2 - caipi_shift * ((y - ny // 2) // ry)) % rz == 0
        }
    views = {
        (y, z)
        for y, z in kept
        if y >= first_y and z >= first_z and inside(y, z, ny, nz)
    }
    return sorted(views | region), region


def order_lines(lines: list[int], centre: int, ordering: str, rng) -> list[int]:
    """Return one partition's lines in play order.

    ``radial`` plays them centre-out, the lower of two equally distant lines
    first; ``shuffling`` in a random order.
    """
    if ordering == "shuffling":
        return [lines[i] for i in rng.permutation(len(lines))]
    return sorted(lines, key=lambda y: (abs(y - centre), y))


class Mprage3DApp(sequences.SequenceApp):
    """3D MPRAGE: one inversion per partition, then a train of spoiled low-flip lines.

    Each shot is the inversion, a wait that puts the first line's excitation
    at TI, one :class:`LineReadout3D` repetition per line of one partition,
    and a recovery that makes every inversion-to-inversion interval the TR.
    Partitions are played in order. Only views inside the inscribed ky-kz
    ellipse are sampled, so partitions hold different numbers of lines; every
    shot plays as many repetitions as the fullest partition, and the ones it
    has no line for play without acquiring. ``ordering`` plays a partition's
    lines centre-out, so the centre line is read at TI, or shuffled for a
    time-resolved reconstruction. Each acquisition carries its line, its
    partition and its place in the train as ``LIN``, ``PAR`` and ``ECO``;
    calibration views are marked ``IMA``, and under wave-CAIPI the
    calibration region is first acquired wave-free in shots of its own,
    marked ``REF``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.mprage3D_sequence(
    ...     n_x=32, n_y=16, n_z=8, ti=100e-3, tr=300e-3
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "mprage_3d"
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
    #: Dephasing left on the readout axis at the end of each repetition, in
    #: cycles across one voxel.
    SPOILING_CYCLES = 4.0
    #: Seed of the shuffled line order and of its Poisson-disc views.
    SHUFFLE_SEED = 0
    #: Pacing of one three-plane navigator (s), and the most one recovery
    #: carries: each takes a little longitudinal magnetisation from the volume
    #: the recovery is restoring.
    NAVIGATOR_TR = 100e-3
    NAVIGATOR_COUNT = 5

    def init_sequence(
        self,
        fov_x: float = 256e-3,
        fov_y: float = 256e-3,
        fov_z: float = 176e-3,
        n_x: int = 256,
        n_y: int = 256,
        n_z: int = 176,
        flip_angle_deg: float = 9.0,
        te: float | None = None,
        esp: float | None = None,
        ti: float | None = 900e-3,
        tr: float | None = 2300e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        rz: int = 1,
        caipi_shift: int = 0,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        partial_fourier_z: float = 1.0,
        *,
        n_dummy: int = 1,
        excitation: str = "slab",
        readout_oversampling: float = 2.0,
        n_acs_y: int = 24,
        n_acs_z: int = 16,
        elliptical_acs: bool = False,
        ordering: str = "radial",
        wave: str = "both",
        wave_cycles: int = 8,
        wave_amplitude: float = 0.0,
        navigator: bool = False,
    ) -> None:
        """Design the inversion, the readout, the shot timing and the partition trains.

        Parameters
        ----------
        fov_x, fov_y, fov_z : float, optional
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        n_x, n_y, n_z : int, optional
            Matrix size along the readout, the phase encode and the partition
            encode.
        flip_angle_deg : float, optional
            Readout excitation flip angle (degrees).
        te : float | None, optional
            Echo time of every line (s). ``None`` is as short as the readout
            admits.
        esp : float | None, optional
            Spacing of successive line excitations (s). ``None`` is as short
            as the readout admits.
        ti : float | None, optional
            Inversion time (s), from the inversion pulse's centre to the first
            line's excitation. ``None`` is as short as the inversion module
            admits.
        tr : float | None, optional
            Inversion-to-inversion interval (s). ``None`` leaves one raster of
            recovery after the train.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth (Hz).
        ry, rz : int, optional
            Undersampling along the phase and the partition encode.
        caipi_shift : int, optional
            Partitions the lattice climbs per acquired line, in ``[0, rz)``.
            Unused by ``shuffling``.
        partial_fourier_x : float, optional
            Fraction of the echo acquired, in ``[0.75, 1]``.
        partial_fourier_y, partial_fourier_z : float, optional
            Fraction of the phase- and partition-encode extent acquired, in
            ``[0.75, 1]``.
        n_dummy : int, optional
            Whole shots played without acquiring before the first.
        excitation : {'slab', 'nonselective', 'spsp'}, optional
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        readout_oversampling : float, optional
            Readout oversampling factor, at least one.
        n_acs_y, n_acs_z : int, optional
            Extent of the fully sampled calibration region along the phase
            and the partition encode, when undersampled.
        elliptical_acs : bool, optional
            Make the calibration region the ellipse inscribed in the
            ``n_acs_y x n_acs_z`` rectangle rather than the rectangle.
        ordering : {'radial', 'shuffling'}, optional
            Line order within a partition: centre-out, with the views on the
            CAIPIRINHA lattice; or shuffled, with the views a variable-density
            Poisson-disc draw.
        wave : {'phase', 'partition', 'both'}, optional
            Wave-CAIPI channels: a sine on y, a cosine on z, or both. With
            wave-encoding gradients the calibration region is acquired again
            first without them, marked ``REF``, and no wave-encoded view is
            marked ``IMA``.
        wave_cycles : int, optional
            Wave periods across the sampling window; zero plays no wave.
        wave_amplitude : float, optional
            Requested peak wave-encoding gradient amplitude (T/m); zero, the
            default, plays no wave. The slew rate may lower it.
        navigator : bool, optional
            Play three-plane spiral navigators in the recovery after each
            shot, as many as it holds up to :attr:`NAVIGATOR_COUNT`.

        Raises
        ------
        ValueError
            If ``excitation`` or ``ordering`` is unknown, a partial Fourier
            fraction is outside ``[0.75, 1]``, an undersampling factor is below
            one, ``caipi_shift`` is outside ``[0, rz)``, ``wave`` names no wave
            mode, or the TE, the spacing, the TI or the TR is shorter than the
            shot takes.
        """
        self.n_dummy = n_dummy
        if excitation not in EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {EXCITATIONS}, got {excitation!r}"
            )
        if ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
        for name, fraction in (
            ("partial_fourier_x", partial_fourier_x),
            ("partial_fourier_y", partial_fourier_y),
            ("partial_fourier_z", partial_fourier_z),
        ):
            if not 0.75 <= fraction <= 1.0:
                raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
        if ry < 1 or rz < 1:
            raise ValueError(f"ry and rz must be at least 1, got {ry} and {rz}")
        if not 0 <= caipi_shift < rz:
            raise ValueError(f"caipi_shift must lie in [0, {rz}), got {caipi_shift}")

        system = self.system
        self.fov = (fov_x, fov_y, fov_z)
        self.matrix = (n_x, n_y, n_z)
        self.excitation, self.ordering = excitation, ordering
        self.inv = sequences.InversionPreparation(
            system, voxel_size_m=min(fov_x / n_x, fov_z / n_z)
        )
        self.exc = make_excitation(self, flip_angle_deg, excitation, fov_z)
        self.gz = getattr(self.exc, "gz", None)
        self.ro = ro = sequences.LineReadout3D(
            system,
            self.exc.rf,
            self.gz,
            fov=self.fov,
            matrix=self.matrix,
            te=te,
            tr=esp,
            partial_echo=partial_fourier_x,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=self.SPOILING_CYCLES,
            wave=wave,
            wave_cycles=wave_cycles,
            wave_amplitude=wave_amplitude,
        )
        self.waves = [
            gradient
            for gradient in (getattr(ro, "gy_wave", None), getattr(ro, "gz_wave", None))
            if gradient is not None
        ]
        self.no_waves = [pp.scale_grad(g, 0.0) for g in self.waves]
        self.esp = ro.duration

        # One shot per partition, every shot as long as the fullest partition.
        views, self.calibration = sampled_views(
            (n_y, n_z),
            (ry, rz),
            caipi_shift,
            (n_acs_y, n_acs_z),
            (partial_fourier_y, partial_fourier_z),
            elliptical_acs,
            shuffling=ordering == "shuffling",
            seed=self.SHUFFLE_SEED,
        )
        rng = np.random.default_rng(self.SHUFFLE_SEED)
        by_partition: dict[int, list[int]] = {}
        for y, z in views:
            by_partition.setdefault(z, []).append(y)
        self.shots = [
            (z, order_lines(lines, n_y // 2, ordering, rng))
            for z, lines in sorted(by_partition.items())
        ]
        # A wave-encoded view calibrates nothing, so with the wave on the
        # calibration region is acquired again wave-free, in shots of its own.
        self.reference = []
        if self.waves:
            for z, lines in self.shots:
                region = [y for y in lines if (y, z) in self.calibration]
                if region:
                    self.reference.append(
                        (z, order_lines(region, n_y // 2, "radial", rng))
                    )
        self.n_readouts = max(len(lines) for _, lines in self.shots)

        # From the inversion centre to the first excitation centre: the rest
        # of the inversion module, the TI wait and the pulse's place in its
        # block.
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

        body = self.inv.duration + self.wait_ti.delay + self.n_readouts * self.esp
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
        self.duration = (n_dummy + len(self.reference) + len(self.shots)) * (
            body + navigating + self.wait_recovery.delay
        )

    def loop(self) -> None:
        """Play the dummy shots, the wave-free reference shots, then every partition."""
        n = self.n_readouts
        blank = (self.matrix[2] // 2, [None] * n)
        shots = [(blank, "dummy")] * self.n_dummy
        shots += [(shot, "reference") for shot in self.reference]
        shots += [(shot, "image") for shot in self.shots]
        phases = make_rf_spoiling_schedule(
            len(shots) * n, increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        ).reshape(len(shots), n)
        for ((partition, lines), kind), shot_phases in zip(shots, phases, strict=True):
            self.kernel(partition, lines, shot_phases, kind)

    def kernel(
        self, partition: int, lines: list[int | None], phases, kind: str = "image"
    ) -> None:
        """One inversion-prepared shot over ``lines`` of ``partition``.

        ``phases`` are the RF-spoiling phases (rad) of its repetitions. A
        ``dummy`` shot acquires nothing, a ``reference`` shot plays without
        the wave, and a repetition past the end of ``lines``, or at ``None``,
        acquires nothing either.
        """
        inv, ro, seq = self.inv, self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        waves = self.no_waves if kind == "reference" else self.waves
        wait_te = getattr(ro, "wait_te", None)
        wait_tr = getattr(ro, "wait_tr", None)
        lines = list(lines) + [None] * (self.n_readouts - len(lines))
        if kind == "dummy":
            flags = {"ONCE": 1}
        else:
            flags = {"ONCE": 0} if self.n_dummy else {}
            if self.waves:
                flags["REF"] = int(kind == "reference")

        seq.add_block(inv.rf_prep, *self.labels(**flags))
        seq.add_block(inv.gz_spoil)
        seq.add_block(self.wait_ti)
        kz = (partition - n_z // 2) / (n_z / 2)
        for echo, (line, phase) in enumerate(zip(lines, phases, strict=True)):
            # The slab sits at isocentre, so the excitation has no frequency
            # offset and its phase is the spoiling phase alone.
            ro.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            ky = 0.0 if line is None else (line - n_y // 2) / (n_y / 2)
            seq.add_block(ro.rf, *([] if self.gz is None else [self.gz]))
            if wait_te is not None:
                seq.add_block(wait_te)
            seq.add_block(
                ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
            )
            if kind != "dummy" and line is not None:
                calibrating = not self.waves and (line, partition) in self.calibration
                labels = self.labels(LIN=line, PAR=partition, ECO=echo, IMA=calibrating)
                seq.add_block(ro.gx, *waves, ro.adc, *labels)
            else:
                seq.add_block(ro.gx, *waves)
            seq.add_block(
                ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
            )
            if wait_tr is not None:
                seq.add_block(wait_tr)
        for _ in range(self.n_navigators):
            for block in self.navigator.blocks:
                seq.add_block(*block)
        seq.add_block(self.wait_recovery)

    def finalize(self) -> None:
        """Write the prescription, the shot timing and the ordering as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "TI": self.ti,
            "EchoSpacing": self.esp,
            "EchoTrainLength": self.n_readouts,
            "Excitation": self.excitation,
            "ViewOrdering": self.ordering,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Mprage3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="mprage_3d.seq"))
