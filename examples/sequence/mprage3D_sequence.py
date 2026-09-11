"""3D MPRAGE: inversion-prepared, segmented, RF-spoiled Cartesian gradient echo."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The view orderings ``ordering`` selects from, each a pypulseqpp echo-train
#: ordering of the same name.
ORDERINGS = ("linear", "centric", "radial", "radial_adaptive", "shuffling")


def order_views(
    views: list[tuple[int, int]],
    etl: int,
    n_center: int,
    ordering: str,
    grid: tuple[int, int],
    *,
    seed: int = 0,
) -> list[list[tuple[int, int] | None]]:
    """Deal ``(line, partition)`` views into trains of ``etl``.

    ``None`` pads a train position with nothing left to encode. The orderings
    rank by radius, so the views are ranked in fractional k-space about the
    centre of ``grid``.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    coords = (np.asarray(views, dtype=float) - np.divide(grid, 2)) / np.asarray(grid)
    if ordering == "shuffling":
        trains = pp.make_shuffling_order(coords, etl, seed=seed, pad=True)
    elif ordering == "radial":
        trains = pp.make_radial_order(coords, etl, center=(0.0, 0.0), pad=True)
    else:
        make = getattr(pp, f"make_{ordering}_order")
        trains = make(coords, etl, center=(0.0, 0.0), center_echo=n_center, pad=True)
    return [[None if i is None else views[i] for i in train] for train in trains]


def _acs_range(n: int, n_acs: int) -> range:
    return range(max(0, n // 2 - n_acs // 2), min(n, n // 2 + -(-n_acs // 2)))


class Mprage3DApp(sequences.SequenceApp):
    """3D MPRAGE: one adiabatic inversion per segment of spoiled low-flip lines.

    Each shot is the inversion, a wait that puts the segment's
    centre-of-k-space view at TI, a train of :class:`LineReadout3D`
    repetitions, and a recovery wait that makes every inversion-to-inversion
    interval the outer TR. The ``(ky, kz)`` views are dealt into segments by
    one of :data:`ORDERINGS`; each acquisition carries its within-segment
    index as ``ECO``. With ``wave`` set, the calibration rectangle is acquired
    again, wave-free, in shots of its own marked ``REF``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.mprage3D_sequence(
    ...     n_x=32, n_y=16, n_z=8, views_per_segment=16, ti=100e-3, tr_outer=300e-3
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "mprage_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab excitation.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Pacing of one three-plane navigator (s), and the most one recovery
    #: carries when their count is ``"auto"``: each takes a little
    #: longitudinal magnetisation from the volume the recovery is restoring.
    NAVIGATOR_TR = 100e-3
    NAVIGATOR_COUNT = 5

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 64,
        slab_thickness: float = 128e-3,
        flip_angle_deg: float = 9.0,
        ti: float = 900e-3,
        tr_outer: float = 2000e-3,
        views_per_segment: int = 64,
        ordering: str = "linear",
        te: float | None = None,
        tr: float | None = None,
        readout_bandwidth_hz: float = 250e3,
        acceleration: int = 1,
        acceleration_z: int = 1,
        caipi_shift: int = 0,
        elliptical: bool = True,
        n_acs: int = 24,
        n_acs_z: int = 16,
        shuffle_seed: int = 0,
        n_dummy: int = 1,
        n_gain_calibration_readouts: int = 1,
        rf_spoiling_increment_deg: float = 117.0,
        spoiling_cycles: float = 4.0,
        navigator: bool = False,
        n_navigators: int | str = "auto",
        wave: str | None = None,
        wave_cycles: int = 8,
        wave_amplitude: float = 8e-3,
    ) -> None:
        """Design the inversion, the readout, the shot timing and the segments.

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
        flip_angle_deg : float, optional
            Readout excitation flip angle, in degrees.
        ti : float, optional
            Inversion time, in seconds, from the inversion pulse's centre to
            the excitation of the segment's centre-of-k-space view.
        tr_outer : float, optional
            Inversion-to-inversion interval, in seconds.
        views_per_segment : int, optional
            Views acquired per inversion.
        ordering : str, optional
            How views are dealt into segments; one of :data:`ORDERINGS`.
        te : float or None, optional
            Readout echo time, in seconds. ``None`` is as short as possible.
        tr : float or None, optional
            Inner repetition time, in seconds, one per readout. ``None`` is as
            short as possible.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        acceleration : int, optional
            Uniform phase-encode undersampling factor along y.
        acceleration_z : int, optional
            Uniform partition-encode undersampling factor along z.
        caipi_shift : int, optional
            CAIPIRINHA shift along kz per sampled-ky block, for the regular
            orderings. ``0`` is a plain lattice.
        elliptical : bool, optional
            Keep only the views inside the inscribed ky-kz ellipse.
        n_acs : int, optional
            Calibration extent along y, in lines.
        n_acs_z : int, optional
            Calibration extent along z, in partitions.
        shuffle_seed : int, optional
            Seed of the shuffling permutation and its Poisson-disc set.
        n_dummy : int, optional
            Whole shots played without acquiring before the first acquired
            one, so the scan measures the magnetisation's steady state.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        rf_spoiling_increment_deg : float, optional
            Quadratic RF spoiling phase increment, in degrees.
        spoiling_cycles : float, optional
            Cycles of dephasing left on the readout axis at the end of each
            inner repetition, counted across one voxel.
        navigator : bool, optional
            Play three-plane spiral navigators in the recovery after each
            segment.
        n_navigators : int or str, optional
            Navigators per recovery; ``"auto"`` fits as many as it holds, up to
            ``NAVIGATOR_COUNT``.
        wave : {'phase', 'partition', 'both'} or None, optional
            Wave-CAIPI corkscrew under every readout.
        wave_cycles : int, optional
            Wave periods across the readout.
        wave_amplitude : float, optional
            Peak wave gradient, in T/m; a ceiling the slew rate may lower.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y, slab_thickness)
        self.matrix = (n_x, n_y, n_z)
        self.ti, self.tr_outer, self.ordering = ti, tr_outer, ordering
        self.views_per_segment = views_per_segment
        self.n_dummy, self.wave = n_dummy, wave
        self.n_gain_calibration_readouts = n_gain_calibration_readouts
        self.spoiling_increment = np.deg2rad(rf_spoiling_increment_deg)

        self.inv = sequences.InversionPreparation(
            system, voxel_size_m=min(fov_x / n_x, slab_thickness / n_z)
        )
        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            slab_thickness,
            duration_s=self.PULSE_DURATION,
            is_slab=True,
            time_bw_product=self.TIME_BW_PRODUCT,
        )
        self.ro = ro = sequences.LineReadout3D(
            system,
            self.exc.rf,
            self.exc.gz,
            fov=self.fov,
            matrix=self.matrix,
            te=te,
            tr=tr,
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
        self.inner_tr = ro.duration

        # Regular orderings sample the CAIPIRINHA lattice around its calibration
        # rectangle; shuffling draws a variable-density Poisson-disc set, which
        # needs acceleration to thin, so at R = 1 it takes the full grid.
        if ordering == "shuffling" and acceleration * acceleration_z > 1:
            mask = pp.make_poisson_disc_mask(
                (n_y, n_z),
                float(acceleration * acceleration_z),
                calib=(n_acs, n_acs_z),
                seed=shuffle_seed,
            )
            views = [(int(y), int(z)) for y, z in np.argwhere(mask)]
        elif ordering == "shuffling":
            views = [(y, z) for z in range(n_z) for y in range(n_y)]
        else:
            views, _ = pp.calc_sampled_pairs(
                (n_y, n_z),
                (acceleration, acceleration_z),
                (n_acs, n_acs_z),
                caipi_shift=caipi_shift,
                elliptical=elliptical,
                order="ascending",
            )
        # TI is defined at the centre view, placed mid-segment so the transient
        # splits evenly around it; radial pins it to the first view.
        self.n_center = 0 if ordering == "radial" else views_per_segment // 2
        self.segments = order_views(
            views,
            views_per_segment,
            self.n_center,
            ordering,
            (n_y, n_z),
            seed=shuffle_seed,
        )
        self.acs = (_acs_range(n_y, n_acs), _acs_range(n_z, n_acs_z))
        # A wave-encoded line calibrates nothing, so with the wave on the
        # calibration rectangle is acquired again wave-free, in shots of its own.
        self.calibration_views = (
            [v for v in views if v[0] in self.acs[0] and v[1] in self.acs[1]]
            if wave is not None
            else []
        )

        # From the inversion centre to the centre view's excitation centre:
        # the rest of the inversion module, the TI wait, the repetitions before
        # that view, and the excitation's place in its block.
        raster = system.block_duration_raster
        inversion_tail = self.inv.duration - (
            self.inv.rf_prep.delay + self.inv.rf_prep.center
        )
        ti_floor = (
            inversion_tail + self.n_center * self.inner_tr + ro.rf.delay + ro.rf.center
        )
        if ti < ti_floor:
            raise ValueError(
                f"TI {ti * 1e3:.0f} ms is shorter than the inversion tail and the "
                f"{self.n_center} repetitions before the centre view admit; the "
                f"minimum is {ti_floor * 1e3:.0f} ms"
            )
        self.wait_ti = pp.make_delay(
            max(raster, pp.round_to_raster(ti - ti_floor, raster))
        )

        body = (
            self.inv.duration + self.wait_ti.delay + views_per_segment * self.inner_tr
        )
        recovery = tr_outer - body
        if recovery < 0:
            raise ValueError(
                f"TR {tr_outer * 1e3:.0f} ms is shorter than one segment takes "
                f"({body * 1e3:.0f} ms)"
            )
        # Navigators ride in the recovery, where they cost no scan time.
        self.navigator, self.n_navigators, navigating = None, 0, 0.0
        if navigator:
            self.navigator = sequences.SpiralNavigator(
                system, navigator_tr=self.NAVIGATOR_TR
            )
            self.n_navigators = self.navigator.fit(
                recovery, n_navigators, limit=self.NAVIGATOR_COUNT
            )
            navigating = self.n_navigators * self.navigator.duration
        self.wait_recovery = pp.make_delay(
            max(raster, pp.round_to_raster(recovery - navigating, raster))
        )

        n_calibration_shots = -(-len(self.calibration_views) // views_per_segment)
        self.duration = (len(self.segments) + n_calibration_shots) * (
            body + navigating + self.wait_recovery.delay
        )

    def loop(self) -> None:
        """Play the dummy shots, the wave-free calibration shots, then the segments."""
        vps = self.views_per_segment
        blank = [None] * vps
        cal = self.calibration_views
        calibration = [
            (cal[i : i + vps] + blank)[:vps] for i in range(0, len(cal), vps)
        ]
        n_shots = self.n_dummy + len(calibration) + len(self.segments)
        phases = iter(
            pp.make_rf_spoiling_schedule(
                n_shots * vps, increment=self.spoiling_increment
            ).reshape(n_shots, vps)
        )
        for _ in range(self.n_dummy):
            self.kernel(blank, next(phases), acquire=False, flags={"ONCE": 1})
        once = {"ONCE": 0} if self.n_dummy else {}
        for views in calibration:
            self.kernel(views, next(phases), wave=0.0, flags={**once, "REF": 1})
        reference = {"REF": 0} if self.wave is not None else {}
        for views in self.segments:
            self.kernel(views, next(phases), flags={**once, **reference})

    def kernel(
        self,
        views: list[tuple[int, int] | None],
        phases: np.ndarray,
        acquire: bool = True,
        wave: float = 1.0,
        flags: dict[str, int] | None = None,
    ) -> None:
        """One inversion-prepared shot over ``views``; ``None`` plays a line unencoded.

        ``phases`` are the RF-spoiling phases (rad) of its repetitions,
        ``wave`` scales the corkscrew, and ``flags`` are the labels the shot
        carries on its inversion.
        """
        inv, ro, seq = self.inv, self.ro, self.seq
        n_y, n_z = self.matrix[1:]
        acs_y, acs_z = self.acs
        corkscrew = [pp.scale_grad(g, wave) for g in self.wave_events]
        wait_te = getattr(ro, "wait_te", None)
        wait_tr = getattr(ro, "wait_tr", None)

        seq.add_block(inv.rf_prep, *self.labels(**(flags or {})))
        seq.add_block(inv.gz_spoil)
        seq.add_block(self.wait_ti)
        for echo, (view, phase) in enumerate(zip(views, phases, strict=True)):
            # The slab sits at isocentre, so the excitation has no frequency
            # offset and its phase is the spoiling phase alone.
            ro.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            line, partition = (n_y / 2, n_z / 2) if view is None else view
            ky, kz = (line - n_y / 2) / (n_y / 2), (partition - n_z / 2) / (n_z / 2)

            seq.add_block(ro.rf, ro.gz)
            if wait_te is not None:
                seq.add_block(wait_te)
            seq.add_block(
                ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
            )
            if acquire and view is not None:
                labels = self.labels(
                    LIN=line,
                    PAR=partition,
                    IMA=line in acs_y and partition in acs_z,
                    SEG=0,
                    ECO=echo,
                )
                seq.add_block(ro.gx, ro.adc, *corkscrew, *labels)
            else:
                seq.add_block(ro.gx, *corkscrew)
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
            "TR": self.tr_outer,
            "TI": self.ti,
            "InnerTR": self.inner_tr,
            "ViewOrdering": self.ordering,
            "ViewsPerSegment": self.views_per_segment,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.ro.center_sample,
            "SliceThickness": self.exc.slice_thickness,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Mprage3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="mprage_3d.seq"))
