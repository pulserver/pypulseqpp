"""Balanced SSFP, 3D Cartesian, non-selective."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Bssfp3DApp(sequences.SequenceApp):
    """Balanced SSFP, 3D Cartesian: a hard pulse and a balanced line readout per TR.

    Every repetition returns all three gradient moments to zero, and the RF
    and ADC phases alternate by π. A half-flip pulse of the opposite phase,
    half a TR before the first full flip, prepares the steady state;
    ``n_dummy`` unacquired repetitions follow, then the calibration pairs lead
    the scan.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.bssfp3D_sequence(
    ...     n_x=64, n_y=16, n_z=4, n_acs=0, n_acs_z=0, n_dummy=0
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "bssfp_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: Duration of the hard pulse (s).
    PULSE_DURATION = 0.5e-3

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 64,
        slab_thickness: float = 128e-3,
        flip_angle_deg: float = 45.0,
        tr: float | None = None,
        readout_bandwidth_hz: float = 125e3,
        partial_fourier: float = 1.0,
        partial_fourier_z: float = 1.0,
        acceleration: int = 1,
        acceleration_z: int = 1,
        caipi_shift: int = 0,
        elliptical: bool = True,
        n_acs: int = 24,
        n_acs_z: int = 16,
        n_dummy: int = 10,
        n_gain_calibration_readouts: int = 1,
    ) -> None:
        """Design the pulses, the balanced readout and the ``(line, partition)`` order.

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
        slab_thickness : float, optional
            Field of view along z, in metres. The pulse is non-selective, so
            the partition encode alone sets it.
        flip_angle_deg : float, optional
            Flip angle of every pulse of the train, in degrees.
        tr : float or None, optional
            Repetition time, in seconds. ``None`` is as short as the readout
            admits.
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
            CAIPIRINHA shift along kz per sampled-ky block. ``0`` is a plain
            lattice.
        elliptical : bool, optional
            Keep only the pairs inside the inscribed ky-kz ellipse.
        n_acs : int, optional
            Calibration extent along y, in lines.
        n_acs_z : int, optional
            Calibration extent along z, in partitions.
        n_dummy : int, optional
            Unacquired repetitions after the half-flip preparation.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y, slab_thickness)
        self.matrix = (n_x, n_y, n_z)
        self.n_dummy = n_dummy
        self.n_gain_calibration_readouts = n_gain_calibration_readouts

        self.exc = sequences.NonSelectiveExcitation(
            system, flip_angle_deg, self.PULSE_DURATION
        )
        self.half = sequences.NonSelectiveExcitation(
            system, flip_angle_deg / 2, self.PULSE_DURATION
        )
        self.ro = sequences.LineReadout3D(
            system,
            self.exc.rf,
            fov=self.fov,
            matrix=self.matrix,
            tr=tr,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=0.0,
        )
        self.repetition_time = self.ro.duration

        # Half a TR from the centre of the half flip to that of the first full
        # flip, to the block raster.
        lead = pp.calc_duration(self.half.rf) - self.half.center + self.exc.center
        pad = pp.round_to_raster(
            self.repetition_time / 2 - lead, system.block_duration_raster
        )
        if pad < 0:
            raise ValueError(
                f"half the TR, {self.repetition_time * 500:.3f} ms, is shorter than "
                f"the {lead * 1e3:.3f} ms between the centres of two pulses"
            )
        self.preparation_wait = pp.make_delay(pad) if pad > 0 else None

        self.pairs, self.n_calibration = pp.calc_sampled_pairs(
            (n_y, n_z),
            (acceleration, acceleration_z),
            (n_acs, n_acs_z),
            partial_fourier=(partial_fourier, partial_fourier_z),
            caipi_shift=caipi_shift,
            elliptical=elliptical,
            order="calibration_first",
        )
        self.duration = (
            pp.calc_duration(self.half.rf)
            + pad
            + (n_dummy + len(self.pairs)) * self.repetition_time
        )

    def loop(self) -> None:
        """Play the half-flip preparation, the dummies, then every pair."""
        self.half.rf.phase_offset = 0.0
        self.seq.add_block(self.half.rf, *self.labels(ONCE=1))
        if self.preparation_wait is not None:
            self.seq.add_block(self.preparation_wait)
        views = [None] * self.n_dummy + list(self.pairs)
        for k, view in enumerate(views):
            calibrating = k - self.n_dummy < self.n_calibration
            # The first full flip is opposite in phase to the half flip, and the
            # rest alternate.
            self.kernel(view, np.pi * ((k + 1) % 2), calibrating)

    def kernel(
        self, view: tuple[int, int] | None, phase: float, calibrating: bool = False
    ) -> None:
        """One balanced repetition at ``(line, partition)``; ``view=None`` is a dummy."""
        rf, ro, seq = self.exc.rf, self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        rf.phase_offset = phase
        ro.adc.phase_offset = phase

        if view is None:
            ky = kz = 0.0
            labels = self.labels(ONCE=1)
        else:
            line, partition = view
            ky, kz = (line - n_y / 2) / (n_y / 2), (partition - n_z / 2) / (n_z / 2)
            labels = self.labels(
                LIN=line, PAR=partition, IMA=calibrating, SEG=not calibrating, ONCE=0
            )

        seq.add_block(rf, *labels)
        seq.add_block(
            ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
        )
        seq.add_block(ro.gx, *([] if view is None else [ro.adc]))
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
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.fov[2],
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Bssfp3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="bssfp_3d.seq"))
