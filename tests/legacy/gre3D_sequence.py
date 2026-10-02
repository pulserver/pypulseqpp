"""RF-spoiled 3D Cartesian gradient echo."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Gre3DApp(sequences.SequenceApp):
    """RF-spoiled 3D Cartesian gradient echo.

    One line per repetition, phase-encoded along y and z over a CAIPIRINHA
    lattice that always contains the centre of k-space, optionally cropped to an
    ellipse. Under undersampling the fully sampled calibration region, a
    rectangle or an ellipse, leads, marked ``IMA``; under wave-CAIPI it is
    first acquired wave-free and marked ``REF``. Lines are played in order,
    each line's partitions one after another.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre3D_sequence(n_x=32, n_y=16, n_z=4)
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 4.0], 'gre_3d')
    """

    NAME = "gre_3d"
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
    ) -> None:
        """Design the excitation, the readout and the ``(line, partition)`` order.

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
            Echo time (s). ``None`` is as short as the readout admits.
        tr : float | None, default=None
            Repetition time (s), one per excitation. ``None`` is as short as
            possible.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry, rz : int, default=1
            Undersampling along the phase and the partition encode.
        caipi_shift : int, default=0
            CAIPIRINHA shift: partitions by which the lattice is displaced per
            acquired line, in ``[0, rz)``.
        partial_fourier_x : float, default=1.0
            Fraction of the echo acquired, in ``[0.75, 1]``.
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

        Raises
        ------
        ValueError
            If ``excitation`` is unknown, a partial Fourier fraction is outside
            ``[0.75, 1]``, an undersampling factor is below one,
            ``caipi_shift`` is outside ``[0, rz)``, ``wave`` names no wave
            mode, or the TE or TR is shorter than the readout takes.
        """
        self.n_dummy = n_dummy
        if excitation not in sequences.EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {sequences.EXCITATIONS}, got {excitation!r}"
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
        self.exc = sequences.make_excitation(
            self.system,
            excitation,
            flip_angle_deg,
            fov_z,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            hard_duration_s=self.HARD_PULSE_DURATION,
            fat_shift_ppm=self.FAT_SHIFT_PPM,
        )
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
        )
        calibrating, imaging = pp.make_cartesian_plane_sampling(
            (n_y, n_z),
            (ry, rz),
            (n_acs_y, n_acs_z),
            caipi_shift=caipi_shift,
            partial_fourier=(partial_fourier_y, partial_fourier_z),
            elliptical=elliptical_sampling,
            elliptical_acs=elliptical_acs,
        )
        # The calibration rectangle leads, so a reconstruction can
        # estimate coil sensitivities while the rest is still arriving.
        self.views = [*calibrating, *imaging]
        self.calibration = set(calibrating)
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
        self.resolve(
            te=self.ro.echo_time,
            tr=self.repetition_time,
            readout_bandwidth_hz=self.ro.bandwidth_hz,
            wave_amplitude=self.ro.wave_amplitude,
        )

    def loop(self) -> None:
        """Play the dummies, the wave-free reference views, then every view."""
        views = [None] * self.n_dummy + self.reference + list(self.views)
        references = [False] * len(views)
        references[self.n_dummy : self.n_dummy + len(self.reference)] = [True] * len(
            self.reference
        )
        phases = pp.make_rf_spoiling_schedule(
            len(views), increment=np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
        )
        for view, reference, phase in zip(views, references, phases, strict=True):
            self.kernel(view, phase, reference)

    def kernel(
        self, view: tuple[int, int] | None, phase: float, reference: bool = False
    ) -> None:
        """One excitation at ``(line, partition)``.

        ``view=None`` plays a dummy; ``reference`` plays the view wave-free.

        Parameters
        ----------
        view : tuple of int or None
            The phase-encode line and the partition to acquire, or None for a
            dummy.
        phase : float
            RF and ADC phase for this repetition, in radians.
        reference : bool, default=False
            Play the view wave-free.
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
        seq.add_block(ro.gx, *waves, *([] if view is None else [ro.adc]))
        seq.add_block(
            ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
        )
        wait_tr = getattr(ro, "wait_tr", None)
        if wait_tr is not None:
            seq.add_block(wait_tr)

    def finalize(self) -> None:
        """Write the prescription and the k-space geometry as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": self.ro.echo_time,
            "TR": self.repetition_time,
            "Excitation": self.excitation,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Gre3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_3d.seq"))
