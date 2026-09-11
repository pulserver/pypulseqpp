"""3D Cartesian spin echo, slab-selective."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Se3DApp(sequences.SequenceApp):
    """3D Cartesian spin echo: one (ky, kz) view per excitation.

    A slab-selective SLR 90, one SLR 180 between crushers half a TE later,
    and one line phase-encoded along y and z at the echo. The calibration
    rectangle leads the traversal; regular undersampling lays a CAIPIRINHA
    lattice with a selectable kz shift per ky block.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se3D_sequence(n_x=32, n_y=8, n_z=4, n_acs=0, n_acs_z=0, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "se_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design shared by the excitation and the refocusing pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 64,
        slab_thickness: float = 128e-3,
        te: float | None = 15e-3,
        tr: float | None = 100e-3,
        readout_bandwidth_hz: float = 250e3,
        partial_echo: float = 1.0,
        partial_fourier: float = 1.0,
        partial_fourier_z: float = 1.0,
        acceleration: int = 1,
        acceleration_z: int = 1,
        caipi_shift: int = 0,
        elliptical: bool = True,
        n_acs: int = 24,
        n_acs_z: int = 16,
        n_dummy: int = 0,
        n_gain_calibration_readouts: int = 1,
        crusher_cycles: float = 4.0,
        spoiling_cycles: float = 4.0,
    ) -> None:
        """Design the pulses, the readout and the (ky, kz) traversal.

        Parameters
        ----------
        fov : float or tuple of float, optional
            In-plane field of view, in metres; ``(fov_x, fov_y)`` if a tuple.
        n_x : int, optional
            Readout samples.
        n_y : int, optional
            Phase-encode steps.
        n_z : int, optional
            Partition-encode steps.
        slab_thickness : float, optional
            Excited slab thickness, in metres, which is also the field of view
            along z.
        te : float or None, optional
            Echo time, in seconds, excitation centre to echo, with the
            refocusing pulse at its midpoint. ``None`` is as short as possible.
        tr : float or None, optional
            Repetition time, in seconds, one per excitation. ``None`` is as
            short as possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        partial_echo : float, optional
            Fraction of the echo acquired, in (0.5, 1].
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
            Keep only the views inside the inscribed ky-kz ellipse.
        n_acs : int, optional
            Calibration extent along y, in lines.
        n_acs_z : int, optional
            Calibration extent along z, in partitions.
        n_dummy : int, optional
            Repetitions played without acquiring before the first view.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        crusher_cycles : float, optional
            Cycles of dephasing each crusher beside the refocusing pulse winds
            across one voxel.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the readout axis at the end of each
            repetition, counted across one voxel.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y, slab_thickness)
        self.matrix = (n_x, n_y, n_z)
        self.n_dummy = n_dummy
        self.n_gain_calibration_readouts = n_gain_calibration_readouts

        # A slab excitation carries its rephaser inside its selection gradient,
        # and the slab sits at the isocentre: neither pulse is offset.
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            90.0,
            slab_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            is_slab=True,
        )
        self.ref = sequences.SpatialSelectiveRefocusing(
            system,
            slab_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            spoiling_cycles=crusher_cycles,
        )

        # TE is solved in two halves about the 180, as se2D_sequence solves it.
        half_floor = self.exc.duration - self.exc.center + self.ref.center

        def readout(half_te: float | None):
            return sequences.LineReadout3D(
                system,
                self.ref.rf_ref,
                self.ref.gz,
                fov=self.fov,
                matrix=self.matrix,
                te=half_te,
                partial_echo=partial_echo,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=spoiling_cycles,
            )

        te_min = 2 * max(readout(None).echo_time, half_floor)
        if te is not None and te < te_min - 1e-9:
            raise ValueError(
                f"TE {te * 1e3:.2f} ms is shorter than the {te_min * 1e3:.2f} ms "
                "the excitation and the readout admit"
            )
        self.ro = readout((te_min if te is None else te) / 2)
        raster = system.block_duration_raster
        wait = pp.round_to_raster(self.ro.echo_time - half_floor, raster)
        self.wait_half_te = pp.make_delay(wait) if wait > 0 else None
        self.echo_time = half_floor + wait + self.ro.echo_time

        length = self.exc.duration + wait + self.ro.duration
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

        self.views, self.n_calibration = pp.calc_sampled_pairs(
            (n_y, n_z),
            (acceleration, acceleration_z),
            (n_acs, n_acs_z),
            partial_fourier=(partial_fourier, partial_fourier_z),
            caipi_shift=caipi_shift,
            elliptical=elliptical,
            order="calibration_first",
        )
        self.duration = (n_dummy + len(self.views)) * length

    def loop(self) -> None:
        """Play the dummies, then every view, calibration rectangle first."""
        for _ in range(self.n_dummy):
            self.kernel(None)
        for index, view in enumerate(self.views):
            self.kernel(view, calibrating=index < self.n_calibration)

    def kernel(self, view: tuple[int, int] | None, calibrating: bool = False) -> None:
        """One excitation encoding ``(line, partition)``; ``None`` plays a dummy."""
        ro, seq = self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        if view is None:
            ky = kz = 0.0
            labels = self.labels(ONCE=1)
        else:
            line, partition = view
            ky, kz = (line - n_y / 2) / (n_y / 2), (partition - n_z / 2) / (n_z / 2)
            labels = self.labels(
                LIN=line, PAR=partition, IMA=calibrating, SEG=not calibrating, ONCE=0
            )

        seq.add_block(self.exc.rf, self.exc.gz, *labels)
        if self.wait_half_te is not None:
            seq.add_block(self.wait_half_te)
        seq.add_block(self.ref.rf_ref, self.ref.gz)
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
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.exc.slice_thickness,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Se3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_3d.seq"))
