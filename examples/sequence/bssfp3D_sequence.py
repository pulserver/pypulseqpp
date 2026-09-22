"""Balanced SSFP, 3D Cartesian, with phase cycling."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

EXCITATIONS = ("nonselective", "slab")


class Bssfp3DApp(sequences.SequenceApp):
    """Balanced SSFP, 3D Cartesian: one train per phase cycle, each opened by a half flip.

    Every repetition returns all three gradient moments to zero, with TE at
    TR/2 (:class:`BssfpReadout3D`). Only views inside the inscribed ky-kz
    ellipse are sampled, lines in order and each line's partitions back and
    forth. Phase cycle ``k`` of ``n_phase_cycles`` steps the RF and receiver
    phase by ``pi + 2 pi k / n_phase_cycles`` per repetition, which shifts the
    off-resonance bands; two cycles is CISS. Each cycle is a file of its own,
    a repeating unit from its first excitation, and a ``catalyst`` file ahead
    of it plays the half flip half a repetition before that excitation, at the
    phase the repetition before the first would have had. The files are
    chained in that order (:meth:`prescans`), so the scanner plays them as one
    table. Acquisitions carry their line, partition and cycle as ``LIN``,
    ``PAR`` and ``SET``, and calibration views are marked ``IMA``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.bssfp3D_sequence(n_x=64, n_y=16, n_z=4)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "bssfp_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab-selective pulse.
    PULSE_DURATION = 1.0e-3
    TIME_BW_PRODUCT = 2.0
    #: Duration of the nonselective hard pulse (s).
    HARD_PULSE_DURATION = 0.5e-3

    def init_sequence(
        self,
        fov_x: float = 220e-3,
        fov_y: float = 220e-3,
        fov_z: float = 128e-3,
        n_x: int = 256,
        n_y: int = 256,
        n_z: int = 128,
        flip_angle_deg: float = 45.0,
        tr: float | None = None,
        readout_bandwidth_hz: float = 125e3,
        ry: int = 1,
        rz: int = 1,
        caipi_shift: int = 0,
        partial_fourier_y: float = 1.0,
        partial_fourier_z: float = 1.0,
        *,
        excitation: str = "nonselective",
        readout_oversampling: float = 2.0,
        n_acs_y: int = 24,
        n_acs_z: int = 16,
        elliptical_acs: bool = False,
        n_phase_cycles: int = 1,
    ) -> None:
        """Design the pulse, the balanced repetition, the views and the phase cycles.

        Parameters
        ----------
        fov_x : float, default=0.22
            Field of view along the readout, the phase encode and the
            partition encode (m). A slab excited is ``fov_z`` thick.
        fov_y : float, default=0.22
            Field of view along the readout, the phase encode and the
            partition encode (m). A slab excited is ``fov_z`` thick.
        fov_z : float, default=0.128
            Field of view along the readout, the phase encode and the
            partition encode (m). A slab excited is ``fov_z`` thick.
        n_x : int, default=256
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_y : int, default=256
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_z : int, default=128
            Matrix size along the readout, the phase encode and the partition
            encode.
        flip_angle_deg : float, default=45.0
            Excitation flip angle (degrees).
        tr : float | None, default=None
            Repetition time (s); TE is TR/2. ``None`` is as short as possible.
        readout_bandwidth_hz : float, default=125000.0
            Requested receiver bandwidth (Hz). The half flip needs an
            acquisition block at least as long as the excitation's tail, which
            a lower bandwidth provides.
        ry, rz : int, default=1
            Undersampling along the phase and the partition encode.
        caipi_shift : int, default=0
            Partitions the lattice climbs per acquired line, in ``[0, rz)``.
        partial_fourier_y, partial_fourier_z : float, default=1.0
            Fraction of the phase- and partition-encode extent acquired, in
            ``[0.75, 1]``.
        excitation : {'nonselective', 'slab'}, default='nonselective'
            A hard pulse, or a slab-selective SLR pulse whose rephasers the
            balanced readout builds.
        readout_oversampling : float, default=2.0
            Readout oversampling factor, at least one.
        n_acs_y : int, default=24
            Extent of the fully sampled calibration region along the phase
            and the partition encode, when undersampled.
        n_acs_z : int, default=16
            Extent of the fully sampled calibration region along the phase
            and the partition encode, when undersampled.
        elliptical_acs : bool, default=False
            Make the calibration region the ellipse inscribed in the
            ``n_acs_y x n_acs_z`` rectangle rather than the rectangle.
        n_phase_cycles : int, default=1
            Trains acquired, each with its own RF phase increment.

        Raises
        ------
        ValueError
            If ``excitation`` is unknown, a partial Fourier fraction is outside
            ``[0.75, 1]``, a count is below one, ``caipi_shift`` is outside
            ``[0, rz)``, or the TR is shorter than the repetition, or too short
            beside the pulse for the half flip.
        """
        if excitation not in EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {EXCITATIONS}, got {excitation!r}"
            )
        for name, fraction in (
            ("partial_fourier_y", partial_fourier_y),
            ("partial_fourier_z", partial_fourier_z),
        ):
            if not 0.75 <= fraction <= 1.0:
                raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
        for name, count in (("ry", ry), ("rz", rz), ("n_phase_cycles", n_phase_cycles)):
            if count < 1:
                raise ValueError(f"{name} must be at least 1, got {count}")
        if not 0 <= caipi_shift < rz:
            raise ValueError(f"caipi_shift must lie in [0, {rz}), got {caipi_shift}")

        system = self.system
        self.fov = (fov_x, fov_y, fov_z)
        self.matrix = (n_x, n_y, n_z)
        self.excitation = excitation
        if excitation == "slab":
            exc = sequences.SpatialSelectiveExcitation(
                system,
                flip_angle_deg,
                fov_z,
                duration_s=self.PULSE_DURATION,
                time_bw_product=self.TIME_BW_PRODUCT,
                rephase=False,
            )
            rf, gz = exc.rf, exc.gz
        else:
            # An even number of block rasters puts the pulse centre on the raster.
            raster = system.block_duration_raster
            hard = (
                2 * raster * math.ceil(self.HARD_PULSE_DURATION / (2 * raster) - 1e-9)
            )
            rf, gz = (
                sequences.NonSelectiveExcitation(
                    system, flip_angle_deg, duration_s=hard
                ).rf,
                None,
            )
        self.ro = ro = sequences.BssfpReadout3D(
            system,
            rf,
            gz,
            fov=self.fov,
            matrix=self.matrix,
            tr=tr,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )
        self.gz = getattr(ro, "gz", None)
        self.nominal = ro.rf.amplitude
        # The slice rephaser and the partition encode share one ramp and one
        # window, so their sum is the partition lobe scaled.
        self.z_pre = getattr(ro, "gz_pre", None)
        self.z_rew = getattr(ro, "gz_rew", None)

        calibrating, lattice = pp.calc_sampled_pairs(
            (n_y, n_z),
            (ry, rz),
            (n_acs_y, n_acs_z),
            caipi_shift=caipi_shift,
            partial_fourier=(partial_fourier_y, partial_fourier_z),
            elliptical=True,
            elliptical_acs=elliptical_acs,
        )
        # A balanced sequence holds its steady state only while the encoding
        # changes gently, so the views are played in order rather than with the
        # calibration rectangle pulled to the front.
        self.views = sorted({*calibrating, *lattice})
        self.calibration = set(calibrating)
        self.increments = [
            np.pi + 2 * np.pi * k / n_phase_cycles for k in range(n_phase_cycles)
        ]
        self.duration = n_phase_cycles * (len(self.views) + 1) * ro.tr

    def prescans(self) -> dict:
        """Return every file but the last cycle: each cycle's catalyst, then the cycle."""
        chain = {}
        for k in range(len(self.increments)):
            chain[f"catalyst_{k}"] = lambda k=k: self.catalyst(k)
            if k < len(self.increments) - 1:
                chain[f"cycle_{k}"] = lambda k=k: self.cycle(k)
        return chain

    def loop(self) -> None:
        """Play the last phase cycle."""
        self.cycle(len(self.increments) - 1)

    def catalyst(self, k: int) -> None:
        """Play cycle ``k``'s half flip, half a repetition before its first excitation."""
        ro, seq = self.ro, self.seq
        ro.rf.amplitude = 0.5 * self.nominal
        # The first excitation is at the increment itself, so the one before it
        # would have been at zero.
        ro.rf.phase_offset = 0.0
        seq.add_block(ro.rf, *([] if self.gz is None else [self.gz]))
        ro.rf.amplitude = self.nominal
        seq.add_block(ro.wait_prep)
        seq.add_block(ro.wait_rewind, *([] if self.z_rew is None else [self.z_rew]))
        self._define(Name=f"{self.NAME}_catalyst", PhaseCycle=k)

    def cycle(self, k: int) -> None:
        """Play every view of phase cycle ``k``."""
        increment = self.increments[k]
        for n, view in enumerate(self.views):
            self.kernel(view, (n + 1) * increment % (2 * np.pi), k)
        self._define(
            Name=self.NAME,
            PhaseCycle=k,
            PhaseIncrement=float(np.rad2deg(increment)),
        )

    def _z(self, lobe, rephaser, kz: float):
        """Return ``lobe`` scaled to ``kz``, plus the ``rephaser`` sharing its window."""
        amplitude = kz * lobe.amplitude
        if rephaser is not None:
            amplitude += rephaser.amplitude
        return pp.scale_grad(lobe, amplitude / lobe.amplitude)

    def kernel(self, view: tuple[int, int], phase: float, cycle: int = 0) -> None:
        """One balanced repetition at ``(line, partition)``: excite, read, rewind."""
        ro, seq = self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        line, partition = view
        ky = (line - n_y // 2) / (n_y / 2)
        kz = (partition - n_z // 2) / (n_z / 2)
        ro.rf.phase_offset = phase
        ro.adc.phase_offset = phase
        labels = self.labels(
            LIN=line, PAR=partition, SET=cycle, IMA=view in self.calibration
        )
        seq.add_block(ro.rf, *([] if self.gz is None else [self.gz]))
        seq.add_block(
            ro.gx,
            ro.adc,
            pp.scale_grad(ro.gy_pre, ky),
            self._z(ro.gz_partition, self.z_pre, kz),
            *labels,
        )
        seq.add_block(
            ro.gx_rew,
            pp.scale_grad(ro.gy_rew, ky),
            self._z(ro.gz_partition_rew, self.z_rew, kz),
        )

    def _define(self, **definitions) -> None:
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_z = self.matrix
        for key, value in {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "TE": self.ro.te,
            "TR": self.ro.tr,
            "Excitation": self.excitation,
            "NumPhaseCycles": len(self.increments),
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
            **definitions,
        }.items():
            self.seq.set_definition(key=key, value=value)


main = Bssfp3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="bssfp_3d.seq"))
