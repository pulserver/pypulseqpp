"""RF-spoiled multi-echo 3D Cartesian gradient echo, slab-selective."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class GreMultiecho3DApp(sequences.SequenceApp):
    """RF-spoiled, slab-selective multi-echo 3D Cartesian gradient echo.

    Every repetition phase-encodes one ``(ky, kz)`` pair over a CAIPIRINHA
    lattice and reads it at ``n_echoes`` echo times, monopolar (a flyback
    between echoes) or bipolar (even echoes read backwards). Each acquisition
    carries its echo index as ``ECO``. The fully sampled calibration rectangle
    leads the scan as segment 0 with ``IMA`` set; under a wave corkscrew it is
    first acquired again wave-free and marked ``REF``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_multiecho3D_sequence(
    ...     n_x=32, n_y=16, n_z=4, n_echoes=3, n_acs=0, n_acs_z=0, n_dummy=0
    ... )
    >>> seq.check_timing()[0]
    True
    >>> len(seq.definitions["TE"])
    3
    """

    NAME = "gre_multiecho_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab-selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 64,
        n_echoes: int = 4,
        monopolar: bool = True,
        slab_thickness: float = 128e-3,
        flip_angle_deg: float = 12.0,
        te: float | None = None,
        tr: float | None = None,
        readout_bandwidth_hz: float = 250e3,
        partial_fourier: float = 1.0,
        partial_fourier_z: float = 1.0,
        acceleration: int = 1,
        acceleration_z: int = 1,
        caipi_shift: int = 0,
        elliptical: bool = True,
        n_acs: int = 24,
        n_acs_z: int = 16,
        n_dummy: int = 64,
        n_gain_calibration_readouts: int = 1,
        rf_spoiling_increment_deg: float = 117.0,
        spoiling_cycles: float = 4.0,
        wave: str | None = None,
        wave_cycles: int = 8,
        wave_amplitude: float = 8e-3,
    ) -> None:
        """Design the slab pulse, the echo train and the ``(line, partition)`` order.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; one value for both axes, or
            ``(fov_x, fov_y)``.
        n_x : int, optional
            Readout samples.
        n_y : int, optional
            Phase-encode steps.
        n_z : int, optional
            Partition-encode steps.
        n_echoes : int, optional
            Echoes per repetition.
        monopolar : bool, optional
            Rewind between echoes so every one is read the same way; ``False``
            alternates the readout sign.
        slab_thickness : float, optional
            Excited slab thickness, in metres, which is also the field of view
            along z.
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        te : float or None, optional
            First echo time, in seconds; the rest follow at the echo spacing.
            ``None`` is as short as the readout admits.
        tr : float or None, optional
            Repetition time, in seconds, one per excitation. ``None`` is as
            short as possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        partial_fourier : float, optional
            Fraction of the phase-encode extent acquired along y, in (0.5, 1].
        partial_fourier_z : float, optional
            Fraction of the partition-encode extent acquired along z, in
            (0.5, 1].
        acceleration : int, optional
            Uniform phase-encode undersampling factor along y.
        acceleration_z : int, optional
            Uniform partition-encode undersampling factor along z.
        caipi_shift : int, optional
            CAIPIRINHA shift along kz per sampled-ky block, below
            ``acceleration_z``. ``0`` is a plain lattice.
        elliptical : bool, optional
            Keep only the pairs inside the inscribed ky-kz ellipse.
        n_acs : int, optional
            Calibration extent along y, in lines.
        n_acs_z : int, optional
            Calibration extent along z, in partitions.
        n_dummy : int, optional
            Non-acquiring repetitions before the first pair.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the readout axis at the end of each
            repetition, counted across one voxel.
        wave : {'phase', 'partition', 'both'} or None, optional
            Wave-CAIPI corkscrew under every echo. The calibration rectangle
            is then acquired again without it.
        wave_cycles : int, optional
            Wave periods across one echo's readout.
        wave_amplitude : float, optional
            Peak wave gradient, in T/m; a ceiling the slew rate may lower.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y, slab_thickness)
        self.matrix = (n_x, n_y, n_z)
        self.n_echoes, self.monopolar = n_echoes, monopolar
        self.n_dummy, self.wave = n_dummy, wave
        self.n_gain_calibration_readouts = n_gain_calibration_readouts
        self.spoiling_increment = np.deg2rad(rf_spoiling_increment_deg)

        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slab_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            is_slab=True,
        )
        self.ro = ro = sequences.LineReadout3D(
            system,
            self.exc.rf,
            self.exc.gz,
            fov=self.fov,
            matrix=self.matrix,
            te=te,
            tr=tr,
            n_echoes=n_echoes,
            flyback=monopolar,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            wave=wave,
            wave_cycles=wave_cycles,
            wave_amplitude=wave_amplitude,
        )
        self.wave_events = [
            g
            for g in (getattr(ro, "gy_wave", None), getattr(ro, "gz_wave", None))
            if g is not None
        ]
        spacing = pp.calc_duration(ro.gx)
        if monopolar and n_echoes > 1:
            spacing += pp.calc_duration(ro.gx_flyback)
        self.echo_times = [ro.echo_time + i * spacing for i in range(n_echoes)]

        self.pairs, self.n_calibration = pp.calc_sampled_pairs(
            (n_y, n_z),
            (acceleration, acceleration_z),
            (n_acs, n_acs_z),
            partial_fourier=(partial_fourier, partial_fourier_z),
            caipi_shift=caipi_shift,
            elliptical=elliptical,
            order="calibration_first",
        )
        # A wave-encoded line calibrates nothing, so with the wave on the
        # calibration rectangle is acquired again wave-free ahead of the scan.
        self.reference = self.pairs[: self.n_calibration] if wave is not None else []
        self.repetition_time = ro.duration
        self.duration = (n_dummy + len(self.reference) + len(self.pairs)) * ro.duration

    def loop(self) -> None:
        """Play the dummies, the wave-free reference pairs, then every pair."""
        phases = iter(
            pp.make_rf_spoiling_schedule(
                self.n_dummy + len(self.reference) + len(self.pairs),
                increment=self.spoiling_increment,
            )
        )
        for _ in range(self.n_dummy):
            self.kernel(None, next(phases), {"ONCE": 1})
        for view in self.reference:
            flags = {"ONCE": 0, "REF": 1, "IMA": 0, "SEG": 0}
            self.kernel(view, next(phases), flags, wave=0.0)
        for index, view in enumerate(self.pairs):
            if self.wave is not None:
                flags = {"ONCE": 0, "REF": 0, "IMA": 0, "SEG": 1}
            else:
                calibrating = index < self.n_calibration
                flags = {"ONCE": 0, "IMA": calibrating, "SEG": not calibrating}
            self.kernel(view, next(phases), flags)

    def kernel(
        self,
        view: tuple[int, int] | None,
        phase: float,
        flags: dict[str, int],
        wave: float = 1.0,
    ) -> None:
        """One excitation and its echo train at ``(line, partition)``.

        ``view=None`` plays a dummy. ``flags`` are labels carried on the
        excitation with ``LIN`` and ``PAR``; ``ECO`` rides each acquisition.
        ``wave`` scales the corkscrew.
        """
        rf, gz, ro, seq = self.exc.rf, self.exc.gz, self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        # The slab sits at the isocentre of the logical frame.
        rf.freq_offset = 0.0
        rf.phase_offset = phase
        ro.adc.phase_offset = phase

        if view is None:
            ky = kz = 0.0
            labels = self.labels(**flags)
        else:
            line, partition = view
            ky, kz = (line - n_y / 2) / (n_y / 2), (partition - n_z / 2) / (n_z / 2)
            labels = self.labels(LIN=line, PAR=partition, **flags)
        corkscrew = [pp.scale_grad(g, wave) for g in self.wave_events]

        seq.add_block(rf, gz, *labels)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
        )
        for echo in range(self.n_echoes):
            if self.monopolar and echo:
                seq.add_block(ro.gx_flyback)
            lobe = ro.gx if self.monopolar or echo % 2 == 0 else ro.gx_rev
            if view is None:
                seq.add_block(lobe, *corkscrew)
            else:
                seq.add_block(lobe, ro.adc, *corkscrew, *self.labels(ECO=echo))
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
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.exc.slice_thickness,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = GreMultiecho3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_multiecho_3d.seq"))
