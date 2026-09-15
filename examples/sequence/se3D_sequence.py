"""3D Cartesian spin echo."""

from __future__ import annotations

import math
import sys

import pypulseqpp as pp
from pypulseqpp import cli, sequences

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
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Return the ``(line, partition)`` pairs in play order, and the calibration pairs.

    Lines keep ``(y - ny // 2) % ry == 0``. The partitions of the ``j``-th
    line from the centre keep ``(z - nz // 2 - caipi_shift * j) % rz == 0``,
    so the centre pair is always acquired and the partition lattice climbs
    ``caipi_shift`` per acquired line. Partial Fourier drops the lines and the
    partitions before the centre. Under undersampling the calibration
    rectangle, centred on the same pair, leads. Both parts run line by line,
    each line's partitions in order.
    """
    (ny, nz), (ry, rz) = shape, acceleration
    first_y = ny - round(partial_fourier[0] * ny)
    first_z = nz - round(partial_fourier[1] * nz)
    cy, cz = calibration if ry * rz > 1 else (0, 0)
    lines = range(max(ny // 2 - cy // 2, first_y), min(ny // 2 + (cy + 1) // 2, ny))
    partitions = range(
        max(nz // 2 - cz // 2, first_z), min(nz // 2 + (cz + 1) // 2, nz)
    )
    rectangle = [(y, z) for y in lines for z in partitions]
    inside = set(rectangle)
    lattice = [
        (y, z)
        for y in range(first_y, ny)
        if (y - ny // 2) % ry == 0
        for z in range(first_z, nz)
        if (z - nz // 2 - caipi_shift * ((y - ny // 2) // ry)) % rz == 0
        and (y, z) not in inside
    ]
    return rectangle + lattice, inside


class Se3DApp(sequences.SequenceApp):
    """3D Cartesian spin echo: one ``(line, partition)`` view per excitation.

    A 90 from the selected excitation, a nonselective 180 between crushers
    half a TE later, and one line phase-encoded along y and z at the echo. The
    views are chosen and ordered as :mod:`gre3D_sequence` chooses them.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se3D_sequence(n_x=32, n_y=8, n_z=4, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_3d"
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
    #: Non-acquiring repetitions before the first view.
    N_DUMMY = 0
    #: Dephasing each crusher beside the refocusing pulse winds, in cycles
    #: across one voxel.
    CRUSHER_CYCLES = 4.0
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
        te: float | None = None,
        tr: float | None = 100e-3,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        rz: int = 1,
        caipi_shift: int = 0,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        partial_fourier_z: float = 1.0,
        *,
        excitation: str = "slab",
        readout_oversampling: float = 2.0,
        n_acs: int = 24,
        n_acs_z: int = 16,
    ) -> None:
        """Design the pulses, the readout and the ``(line, partition)`` order.

        Parameters
        ----------
        fov_x, fov_y, fov_z : float, optional
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        n_x, n_y, n_z : int, optional
            Matrix size along the readout, the phase encode and the partition
            encode.
        te : float | None, optional
            Echo time (s), excitation centre to echo, with the refocusing
            pulse at its midpoint. ``None`` is as short as possible.
        tr : float | None, optional
            Repetition time (s), one per excitation. ``None`` is as short as
            possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth (Hz).
        ry, rz : int, optional
            Undersampling along the phase and the partition encode.
        caipi_shift : int, optional
            Partitions the lattice climbs per acquired line, in ``[0, rz)``.
        partial_fourier_x : float, optional
            Fraction of the echo acquired, in ``[0.75, 1]``.
        partial_fourier_y, partial_fourier_z : float, optional
            Fraction of the phase- and partition-encode extent acquired, in
            ``[0.75, 1]``.
        excitation : {'slab', 'nonselective', 'spsp'}, optional
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        readout_oversampling : float, optional
            Readout oversampling factor, at least one.
        n_acs, n_acs_z : int, optional
            Extent of the fully sampled calibration rectangle along the phase
            and the partition encode, acquired ahead of the rest when
            undersampled.

        Raises
        ------
        ValueError
            If ``excitation`` is unknown, a partial Fourier fraction is outside
            ``[0.75, 1]``, an undersampling factor is below one, ``caipi_shift``
            is outside ``[0, rz)``, or the TE or TR is shorter than the pulses
            and the readout take.
        """
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

        system = self.system
        self.fov = (fov_x, fov_y, fov_z)
        self.matrix = (n_x, n_y, n_z)
        self.excitation = excitation
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
        rf_tail = pp.ceil_to_raster(pp.calc_duration(rf), raster) - rf.delay - rf.center
        crusher = self.ref.duration - self.ref.center - rf_tail
        half_floor = self.exc.duration - self.exc.center + self.ref.center

        def readout(half_te: float | None):
            return sequences.LineReadout3D(
                system,
                rf,
                None,
                fov=self.fov,
                matrix=self.matrix,
                te=None if half_te is None else half_te - crusher,
                partial_echo=partial_fourier_x,
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

        # The readout's first block, the 180 alone, is not played.
        length = (
            self.exc.duration
            + wait
            + self.ref.duration
            + self.ro.duration
            - pp.ceil_to_raster(pp.calc_duration(rf), raster)
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

        self.views, self.calibration = sampled_views(
            (n_y, n_z),
            (ry, rz),
            caipi_shift,
            (n_acs, n_acs_z),
            (partial_fourier_y, partial_fourier_z),
        )
        self.duration = (self.N_DUMMY + len(self.views)) * length

    def loop(self) -> None:
        """Play the dummies, then every view."""
        for view in [None] * self.N_DUMMY + list(self.views):
            self.kernel(view)

    def kernel(self, view: tuple[int, int] | None) -> None:
        """One spin echo encoding ``(line, partition)``; ``None`` plays a dummy."""
        ro, seq = self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        if view is None:
            ky = kz = 0.0
            labels = self.labels(ONCE=1)
        else:
            line, partition = view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            calibrating = view in self.calibration
            labels = self.labels(
                LIN=line, PAR=partition, IMA=calibrating, SEG=not calibrating, ONCE=0
            )

        seq.add_block(self.exc.rf, *([] if self.gz is None else [self.gz]), *labels)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        for block in self.ref.blocks:
            seq.add_block(*block)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
        )
        seq.add_block(ro.gx, *([] if view is None else [ro.adc]))
        seq.add_block(
            ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
        )
        if self.wait_tr is not None:
            seq.add_block(self.wait_tr)

    def finalize(self) -> None:
        """Write the prescription and the k-space geometry as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": self.echo_time,
            "TR": self.repetition_time,
            "Excitation": self.excitation,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Se3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_3d.seq"))
