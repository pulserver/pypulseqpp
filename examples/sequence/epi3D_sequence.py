"""3D gradient-echo EPI, slab-selective, with its prescan in the same sequence."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: What a shot of each kind of :meth:`Epi3DApp.kernel` plays.
KINDS = ("calibration", "navigator", "reference", "dummy", "image")


class Epi3DApp(sequences.SequenceApp):
    """Slab-selective 3D gradient-echo EPI: one train per ``(segment, shell)``.

    A shell is one partition for a plain stack of trains, or, under
    ``acceleration_z`` above 1, a band of ``acceleration_z`` partitions the
    CAIPI sawtooth walks within the train. Every acquisition carries ``REV``
    for its read polarity and its partition as ``PAR``. The prescan precedes
    the imaging under ``ONCE = 1``: an undersampled scan first acquires a
    Cartesian gradient-echo calibration over the central ``n_acs x n_acs_z``
    rectangle (``REF``); then :attr:`NAVIGATOR_LINES` blip-nulled lines at the
    centre partition (``NAV``), an opposite-phase-encode reference train
    (``SET = 1``), and the dummy shots. ``spsp`` excites water only.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.epi3D_sequence(n_x=32, n_y=16, n_z=4, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 4.0], 'epi_3d')
    """

    NAME = "epi_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Fat's chemical shift from water (ppm), converted against ``system.B0``
    #: into the spectral passband of the ``spsp`` excitation.
    FAT_SHIFT_PPM = -3.4
    #: Blip-nulled lines per navigator: odd, even, odd.
    NAVIGATOR_LINES = 3

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 32,
        slab_thickness: float = 96e-3,
        flip_angle_deg: float = 20.0,
        te: float | None = None,
        tr: float | None = None,
        n_repetitions: int = 1,
        segments: int = 1,
        acceleration: int = 1,
        acceleration_z: int = 1,
        caipi_shift: int = 1,
        partial_fourier: float = 1.0,
        partial_fourier_z: float = 1.0,
        n_acs: int = 24,
        n_acs_z: int = 16,
        readout_bandwidth_hz: float = 500e3,
        opposite_reference: bool = True,
        partition_order: str = "center_out",
        n_dummy: int = 2,
        n_gain_calibration_readouts: int = 1,
        spoiling_cycles: float = 4.0,
        spsp: bool = False,
    ) -> None:
        """Design the slab excitation, the trains, the calibration and the shell order.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; ``(fov_x, fov_y)`` if a tuple.
        n_x : int, optional
            Readout samples.
        n_y : int, optional
            Phase-encode lines.
        n_z : int, optional
            Partitions.
        slab_thickness : float, optional
            Excited slab thickness, in metres, which is also the field of view
            along z.
        flip_angle_deg : float, optional
            Excitation flip angle, in degrees.
        te : float or None, optional
            Echo time of the first line, in seconds. ``None`` is as short as
            possible.
        tr : float or None, optional
            Repetition time over one shot, in seconds. ``None`` is as short as
            possible.
        n_repetitions : int, optional
            Volumes in the time series, each carrying its ``REP`` counter.
        segments : int, optional
            Interleaved shots per shell.
        acceleration : int, optional
            Uniform phase-encode undersampling factor along y.
        acceleration_z : int, optional
            Partition undersampling factor along z. Above 1 each shot walks a
            shell of ``acceleration_z`` partitions with the CAIPI sawtooth.
        caipi_shift : int, optional
            Partitions the CAIPI sawtooth climbs per acquired line, used when
            ``acceleration_z`` is above 1.
        partial_fourier : float, optional
            Fraction of the phase-encode extent acquired along y, in
            ``(0.5, 1]``. Shortens the train and drops its leading lines.
        partial_fourier_z : float, optional
            Fraction of the shells acquired along z, in ``(0.5, 1]``. Drops
            the leading shells.
        n_acs : int, optional
            Calibration extent along y, in lines.
        n_acs_z : int, optional
            Calibration extent along z, in partitions.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        opposite_reference : bool, optional
            Acquire the opposite-phase-encode reference train after the
            navigator.
        partition_order : str, optional
            Order the shells are encoded in, as
            :func:`pypulseqpp.calc_traversal_order` accepts.
        n_dummy : int, optional
            Shots played without acquiring before the first acquired one.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        spoiling_cycles : float, optional
            Cycles of dephasing left at the end of each shot.
        spsp : bool, optional
            Excite with a spectral-spatial, slab- and water-selective pulse.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y, slab_thickness)
        self.matrix = (n_x, n_y, n_z)
        self.n_repetitions, self.segments = n_repetitions, segments
        self.acceleration, self.acceleration_z = acceleration, acceleration_z
        self.n_dummy, self.opposite_reference = n_dummy, opposite_reference
        self.n_gain_calibration_readouts = n_gain_calibration_readouts

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
                time_bw_product=self.TIME_BW_PRODUCT,
                is_slab=True,
            )

        # Partial Fourier along y keeps the trailing whole lattice steps of the
        # train, so the retained lattice stays aligned; along z it drops the
        # leading shells.
        etl_full = -(-n_y // (acceleration * segments))
        etl = max(1, round(partial_fourier * etl_full))
        self.first_y = max(0, n_y - etl * acceleration * segments)
        n_shells = n_z // acceleration_z
        first_shell = n_shells - max(1, round(partial_fourier_z * n_shells))
        kept = list(range(first_shell, n_shells))
        self.shells = [
            kept[i] for i in pp.calc_traversal_order(len(kept), partition_order)
        ]

        train = {
            "fov": self.fov,
            "matrix": self.matrix,
            "scheme": "caipi" if acceleration_z > 1 else "linear",
            "segments": segments,
            "acceleration": acceleration,
            "partition_acceleration": acceleration_z,
            "caipi_shift": caipi_shift,
            "te": te,
            "tr": tr,
            "readout_bandwidth_hz": readout_bandwidth_hz,
            "spoiling_cycles": spoiling_cycles,
            "labels": ("LIN", "PAR"),
        }
        self.epi = sequences.EpiReadout3D(
            system, self.exc.rf, self.exc.gz, etl=etl, **train
        )
        # The navigator and the reference play the full-length train.
        self.ref_epi = (
            sequences.EpiReadout3D(system, self.exc.rf, self.exc.gz, **train)
            if etl != etl_full
            else self.epi
        )

        undersampled = (
            acceleration > 1
            or acceleration_z > 1
            or partial_fourier < 1.0
            or partial_fourier_z < 1.0
        )
        acs_y = pp.calc_calibration_lines(n_y, n_acs)
        acs_z = pp.calc_calibration_lines(n_z, n_acs_z)
        # Partitions outer, lines inner.
        self.acs = [(y, z) for z in acs_z for y in acs_y] if undersampled else []
        self.gre = None
        if self.acs:
            self.gre = sequences.LineReadout3D(
                system,
                self.exc.rf,
                self.exc.gz,
                fov=self.fov,
                matrix=self.matrix,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=spoiling_cycles,
                labels=("LIN", "PAR"),
            )

    def loop(self) -> None:
        """Play the prescan under ``ONCE = 1``, then every repetition of the volume."""
        az = self.acceleration_z
        for view in self.acs:
            self.kernel(view, "calibration")
        self.kernel(None, "navigator")
        if self.opposite_reference:
            self.kernel((0, self.matrix[2] // 2), "reference")
        for _ in range(self.n_dummy):
            self.kernel((self.first_y, self.shells[0] * az), "dummy")
        for repetition in range(self.n_repetitions):
            for segment in range(self.segments):
                for shell in self.shells:
                    origin = (self.first_y + segment * self.acceleration, shell * az)
                    self.kernel(origin, "image", repetition)

    def kernel(
        self, origin: tuple[int, int] | None, kind: str = "image", repetition: int = 0
    ) -> None:
        """One shot of ``kind``, one of :data:`KINDS`.

        ``origin`` is the ``(line, partition)`` the train starts from, or the
        one a calibration shot acquires; ``None`` leaves both encodes at the
        centre. The ``reference`` train negates the phase-encode prewinder and
        blips; the ``navigator`` plays :attr:`NAVIGATOR_LINES` lines without
        blips.
        """
        seq = self.seq
        n_y, n_z = self.matrix[1:]
        flags = {
            "calibration": {"ONCE": 1, "NAV": 0, "REF": 1, "SET": 0, "REV": 0},
            "navigator": {"ONCE": 1, "NAV": 1, "REF": 1, "SET": 0},
            "reference": {"ONCE": 1, "NAV": 0, "REF": 0, "SET": 1},
            "dummy": {"ONCE": 1},
            "image": {"ONCE": 0, "NAV": 0, "REF": 0, "SET": 0, "REP": repetition},
        }[kind]
        labels = self.labels(**flags)
        line, partition = (n_y / 2, n_z / 2) if origin is None else origin
        sign = -1.0 if kind == "reference" else 1.0
        ky = sign * (line - n_y / 2) / (n_y / 2)
        kz = (partition - n_z / 2) / (n_z / 2)

        if kind == "calibration":
            ro = self.gre
            ro.adc_labels[0].value, ro.adc_labels[1].value = line, partition
            seq.add_block(ro.rf, ro.gz, *labels)
            wait_te = getattr(ro, "wait_te", None)
            if wait_te is not None:
                seq.add_block(wait_te)
            seq.add_block(
                ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
            )
            seq.add_block(ro.gx, ro.adc, *ro.adc_labels)
            seq.add_block(
                ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
            )
            wait_tr = getattr(ro, "wait_tr", None)
            if wait_tr is not None:
                seq.add_block(wait_tr)
            return

        epi = self.epi if kind in ("dummy", "image") else self.ref_epi
        acquire = kind != "dummy"
        blipped = kind != "navigator"
        encoded = acquire and origin is not None
        n_lines = self.NAVIGATOR_LINES if kind == "navigator" else epi.etl

        seq.add_block(epi.rf, epi.gz, *labels)
        wait_te = getattr(epi, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        if encoded:
            epi.shot_labels[0].value, epi.shot_labels[1].value = line, partition
        seq.add_block(
            epi.gx_pre,
            pp.scale_grad(epi.gy_pre, ky),
            pp.scale_grad(epi.gz_pre, kz),
            *(epi.shot_labels if encoded else ()),
        )
        for i in range(n_lines):
            events = [epi.gx[i]]
            if acquire:
                events += [epi.adc, *self.labels(REV=i % 2)]
            if blipped:
                events += [
                    pp.scale_grad(blip, sign) if sign < 0 else blip
                    for blip in (epi.gy_blips[i], epi.gz_blips[i])
                    if blip is not None
                ]
            if encoded:
                events += epi.line_labels[i]
            seq.add_block(*events)
        seq.add_block(
            epi.gx_spoil, pp.scale_grad(epi.gy_rew, ky), pp.scale_grad(epi.gz_rew, kz)
        )
        wait_tr = getattr(epi, "wait_tr", None)
        if wait_tr is not None:
            seq.add_block(wait_tr)

    def finalize(self) -> None:
        """Write the prescription and the train's echo timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV (compat=False: the lines are sampled on the ramps).
        definitions = {
            "FOV": list(self.fov),
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": float(self.epi.echo_times[len(self.epi.order) // 2]),
            "EchoSpacing": self.epi.esp,
            "EPIFactor": self.epi.etl,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Epi3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="epi_3d.seq"))
