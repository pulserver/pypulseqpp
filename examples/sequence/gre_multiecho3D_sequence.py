"""RF-spoiled multi-echo 3D Cartesian gradient echo."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._schedules import make_rf_spoiling_schedule

#: The excitations ``excitation`` selects from.
EXCITATIONS = ("nonselective", "slab", "spsp")


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


def sampled_views(
    shape: tuple[int, int],
    acceleration: tuple[int, int],
    caipi_shift: int,
    calibration: tuple[int, int],
    partial_fourier: tuple[float, float],
    elliptical_sampling: bool = False,
    elliptical_acs: bool = False,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Return the ``(line, partition)`` pairs in play order, and the calibration pairs.

    Lines keep ``(y - ny // 2) % ry == 0``. The partitions of the ``j``-th
    line from the centre keep ``(z - nz // 2 - caipi_shift * j) % rz == 0``,
    so the centre pair is always acquired and the partition lattice climbs
    ``caipi_shift`` per acquired line. Partial Fourier drops the lines and the
    partitions before the centre, and ``elliptical_sampling`` the pairs outside
    the ellipse inscribed in the ``ny x nz`` grid. Under undersampling the
    ``n_acs_y x n_acs_z`` calibration region, centred on the same pair, leads
    and is acquired whole: a rectangle, or under ``elliptical_acs`` the ellipse
    inscribed in it. Both parts run line by line, each line's partitions in
    order.
    """
    (ny, nz), (ry, rz) = shape, acceleration
    first_y = ny - round(partial_fourier[0] * ny)
    first_z = nz - round(partial_fourier[1] * nz)
    n_acs_y, n_acs_z = calibration if ry * rz > 1 else (0, 0)

    def inside(y: int, z: int, extent_y: int, extent_z: int) -> bool:
        # Offsets from the centre pair, the one the encodes scale to zero.
        dy, dz = (y - ny // 2) / extent_y, (z - nz // 2) / extent_z
        return dy * dy + dz * dz <= 0.25

    lines_acs = range(
        max(ny // 2 - n_acs_y // 2, first_y), min(ny // 2 + (n_acs_y + 1) // 2, ny)
    )
    partitions_acs = range(
        max(nz // 2 - n_acs_z // 2, first_z), min(nz // 2 + (n_acs_z + 1) // 2, nz)
    )
    region = [
        (y, z)
        for y in lines_acs
        for z in partitions_acs
        if not elliptical_acs or inside(y, z, n_acs_y, n_acs_z)
    ]
    calibrating = set(region)
    lattice = [
        (y, z)
        for y in range(first_y, ny)
        if (y - ny // 2) % ry == 0
        for z in range(first_z, nz)
        if (z - nz // 2 - caipi_shift * ((y - ny // 2) // ry)) % rz == 0
        and (not elliptical_sampling or inside(y, z, ny, nz))
        and (y, z) not in calibrating
    ]
    return region + lattice, calibrating


class GreMultiecho3DApp(sequences.SequenceApp):
    """RF-spoiled multi-echo 3D Cartesian gradient echo.

    One line per repetition, read at ``n_echoes`` echo times: monopolar, with
    a flyback rewinder after every echo but the last, or bipolar, with even
    echoes read backwards. Each acquisition carries its echo index as
    ``ECO``. Lines are phase-encoded along y and z over a CAIPIRINHA lattice
    that always holds the centre of k-space, optionally cropped to an ellipse.
    Under undersampling the fully sampled calibration region, a rectangle or
    an ellipse, leads, marked ``IMA``; under wave-CAIPI it is first acquired
    wave-free and marked ``REF``. Lines are played in order, each line's
    partitions one after another.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_multiecho3D_sequence(n_x=32, n_y=16, n_z=4, n_echoes=3)
    >>> seq.check_timing()[0]
    True
    >>> len(seq.definitions["TE"]), seq.definitions["Name"]
    (3, 'gre_multiecho_3d')
    """

    NAME = "gre_multiecho_3d"
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

    def init_sequence(
        self,
        fov_x: float = 220e-3,
        fov_y: float = 220e-3,
        fov_z: float = 128e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 64,
        flip_angle_deg: float = 12.0,
        te: float | None = None,
        tr: float | None = None,
        n_echoes: int = 4,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        rz: int = 1,
        caipi_shift: int = 0,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        partial_fourier_z: float = 1.0,
        *,
        n_dummy: int = 64,
        excitation: str = "slab",
        readout_oversampling: float = 2.0,
        n_acs_y: int = 24,
        n_acs_z: int = 16,
        elliptical_sampling: bool = True,
        elliptical_acs: bool = False,
        wave: str = "both",
        wave_cycles: int = 8,
        wave_amplitude: float = 0.0,
        echo_spacing: float | None = None,
        flyback: bool = True,
    ) -> None:
        """Design the excitation, the echo train and the ``(line, partition)`` order.

        Parameters
        ----------
        fov_x : float, default=0.22
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        fov_y : float, default=0.22
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        fov_z : float, default=0.128
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        n_x : int, default=128
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_y : int, default=128
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_z : int, default=64
            Matrix size along the readout, the phase encode and the partition
            encode.
        flip_angle_deg : float, default=12.0
            Excitation flip angle (degrees).
        te : float | None, default=None
            First echo time (s). ``None`` is as short as the readout admits.
        tr : float | None, default=None
            Repetition time (s), one per excitation. ``None`` is as short as
            possible.
        n_echoes : int, default=4
            Echoes per excitation.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry, rz : int, default=1
            Undersampling along the phase and the partition encode.
        caipi_shift : int, default=0
            Partitions the lattice climbs per acquired line, in ``[0, rz)``.
        partial_fourier_x : float, default=1.0
            Fraction of the echo acquired, in ``[0.75, 1]``. A bipolar train
            needs a full echo.
        partial_fourier_y, partial_fourier_z : float, default=1.0
            Fraction of the phase- and partition-encode extent acquired, in
            ``[0.75, 1]``.
        n_dummy : int, default=64
            Non-acquiring repetitions before the first view.
        excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        readout_oversampling : float, default=2.0
            Readout oversampling factor, at least one.
        n_acs_y : int, default=24
            Extent of the fully sampled calibration region along the phase
            and the partition encode, acquired ahead of the rest when
            undersampled.
        n_acs_z : int, default=16
            Extent of the fully sampled calibration region along the phase
            and the partition encode, acquired ahead of the rest when
            undersampled.
        elliptical_sampling : bool, default=True
            Acquire only the views inside the ellipse inscribed in the
            ``n_y x n_z`` grid; off, the whole grid. The calibration region is
            acquired whole either way.
        elliptical_acs : bool, default=False
            Make the calibration region the ellipse inscribed in the
            ``n_acs_y x n_acs_z`` rectangle rather than the rectangle.
        wave : {'phase', 'partition', 'both'}, default='both'
            Wave-CAIPI channels: a sine on y, a cosine on z, or both. With
            wave-encoding gradients the calibration region is acquired again
            first without them, marked ``REF``, and no wave-encoded view is
            marked ``IMA``.
        wave_cycles : int, default=8
            Wave periods across the sampling window; zero plays no wave.
        wave_amplitude : float, default=0.0
            Requested peak wave-encoding gradient amplitude (T/m); zero, the
            default, plays no wave. The slew rate may lower it.
        echo_spacing : float | None, default=None
            Echo spacing (s). ``None`` is as short as the readout admits; a
            longer one adds a delay after every echo but the last, following
            the flyback rewinder of a monopolar train.
        flyback : bool, default=True
            Monopolar echo train, rewound after every echo but the last so
            every echo is read the same way; off, a bipolar train. A bipolar
            train has a shorter echo spacing and reads even echoes backwards.

        Raises
        ------
        ValueError
            If ``excitation`` is unknown, a partial Fourier fraction is outside
            ``[0.75, 1]``, an undersampling factor is below one,
            ``caipi_shift`` is outside ``[0, rz)``, ``wave`` names no wave
            mode, the TE, TR or echo spacing is shorter than the readout takes,
            or a bipolar train is given a partial echo or an odd sample count.
        """
        self.n_dummy = n_dummy
        if excitation not in EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {EXCITATIONS}, got {excitation!r}"
            )
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

        self.fov = (fov_x, fov_y, fov_z)
        self.matrix = (n_x, n_y, n_z)
        self.excitation = excitation
        self.exc = make_excitation(self, flip_angle_deg, excitation, fov_z)
        self.gz = getattr(self.exc, "gz", None)
        self.ro = sequences.LineReadout3D(
            self.system,
            self.exc.rf,
            self.gz,
            fov=self.fov,
            matrix=self.matrix,
            te=te,
            tr=tr,
            partial_echo=partial_fourier_x,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=self.SPOILING_CYCLES,
            wave=wave,
            wave_cycles=wave_cycles,
            wave_amplitude=wave_amplitude,
            n_echoes=n_echoes,
            flyback=flyback,
            echo_spacing=echo_spacing,
        )
        self.n_echoes, self.flyback = n_echoes, flyback
        self.echo_times = [
            self.ro.echo_time + i * self.ro.echo_spacing for i in range(n_echoes)
        ]
        self.views, self.calibration = sampled_views(
            (n_y, n_z),
            (ry, rz),
            caipi_shift,
            (n_acs_y, n_acs_z),
            (partial_fourier_y, partial_fourier_z),
            elliptical_sampling,
            elliptical_acs,
        )
        # A wave-encoded view calibrates nothing, so with the wave on the
        # calibration region is acquired again wave-free ahead of the scan.
        self.waves = [
            gradient
            for gradient in (
                getattr(self.ro, "gy_wave", None),
                getattr(self.ro, "gz_wave", None),
            )
            if gradient is not None
        ]
        self.no_waves = [pp.scale_grad(g, 0.0) for g in self.waves]
        self.reference = self.views[: len(self.calibration)] if self.waves else []
        self.repetition_time = self.ro.duration
        self.duration = (
            self.n_dummy + len(self.reference) + len(self.views)
        ) * self.ro.duration

    def loop(self) -> None:
        """Play the dummies, the wave-free reference views, then every view."""
        views = [None] * self.n_dummy + self.reference + list(self.views)
        references = [False] * len(views)
        references[self.n_dummy : self.n_dummy + len(self.reference)] = [True] * len(
            self.reference
        )
        phases = make_rf_spoiling_schedule(
            len(views), increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        )
        for view, reference, phase in zip(views, references, phases, strict=True):
            self.kernel(view, phase, reference)

    def kernel(
        self, view: tuple[int, int] | None, phase: float, reference: bool = False
    ) -> None:
        """One excitation and its echo train.

        ``view=None`` plays a dummy; ``reference`` plays the view wave-free.
        """
        rf, ro, seq = self.exc.rf, self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        rf.phase_offset = phase
        ro.adc.phase_offset = phase

        if view is None:
            ky = kz = 0.0
            labels = self.labels(ONCE=1)
        else:
            line, partition = view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            if self.waves:
                flags = {"REF": reference, "IMA": False, "SEG": not reference}
            else:
                calibrating = view in self.calibration
                flags = {"IMA": calibrating, "SEG": not calibrating}
            labels = self.labels(LIN=line, PAR=partition, **flags, ONCE=0)
        waves = self.no_waves if reference else self.waves

        seq.add_block(rf, *([] if self.gz is None else [self.gz]), *labels)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
        )
        for echo in range(self.n_echoes):
            if echo and self.flyback:
                seq.add_block(ro.gx_flyback)
            if echo and hasattr(ro, "wait_esp"):
                seq.add_block(ro.wait_esp)
            lobe = ro.gx if self.flyback or echo % 2 == 0 else ro.gx_rev
            if view is None:
                seq.add_block(lobe, *waves)
            else:
                seq.add_block(lobe, *waves, ro.adc, *self.labels(ECO=echo))
        seq.add_block(
            ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
        )
        wait_tr = getattr(ro, "wait_tr", None)
        if wait_tr is not None:
            seq.add_block(wait_tr)

    def finalize(self) -> None:
        """Write the prescription, the echo times and the k-space geometry."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": self.echo_times,
            "TR": self.repetition_time,
            "Excitation": self.excitation,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreMultiecho3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_multiecho_3d.seq"))
