"""3D Cartesian fast spin echo, slab-selective, with selectable view ordering."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The view orderings ``ordering`` selects from, each a pypulseqpp echo-train
#: ordering of the same name.
ORDERINGS = ("linear", "centric", "radial", "radial_adaptive", "shuffling")


def traps_flip_schedule(
    etl: int,
    n_center: int,
    *,
    alpha_min_deg: float = 60.0,
    alpha_center_deg: float = 100.0,
    alpha_max_deg: float = 160.0,
    n_down: int = 6,
) -> np.ndarray:
    """TRAPS-style refocusing flips, in degrees, one per echo.

    Piecewise linear between Busse's control points (MRM 2008;60:640):
    ``alpha_max`` at the first echo, ``alpha_min`` from echo ``n_down``,
    ``alpha_center`` at echo ``n_center`` and ``alpha_max`` again at the last.
    """
    bottom = min(max(1, n_down), max(1, etl - 1))
    echoes, flips = [0], [alpha_max_deg]
    if bottom < etl:
        echoes, flips = [*echoes, bottom], [*flips, alpha_min_deg]
    if n_center > bottom:
        echoes, flips = [*echoes, n_center], [*flips, alpha_center_deg]
    if etl - 1 > echoes[-1]:
        echoes, flips = [*echoes, etl - 1], [*flips, alpha_max_deg]
    return np.interp(np.arange(etl), echoes, flips)


def order_views(
    views: list[tuple[int, int]],
    etl: int,
    n_center: int,
    ordering: str,
    grid: tuple[int, int],
    *,
    seed: int = 0,
) -> list[list[tuple[int, int] | None]]:
    """Deal ``(line, partition)`` views into trains, one per echo.

    ``None`` pads an echo with nothing left to encode. The orderings rank by
    radius, so the views are ranked in fractional k-space about the centre of
    ``grid``.
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


class Fse3DApp(sequences.SequenceApp):
    """3D Cartesian fast spin echo: one CPMG train per excitation over a (ky, kz) grid.

    The view ordering maps the train's signal modulation into k-space; every
    acquisition carries its echo index as ``ECO``. Regular orderings sample a
    CAIPIRINHA lattice around a fully sampled calibration rectangle, and
    ``shuffling`` a variable-density Poisson-disc set. The optional variable
    refocusing train follows :func:`traps_flip_schedule`, and the angles played
    are written as ``RefocusingFlipAngles``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.fse3D_sequence(n_x=64, n_y=16, n_z=4, etl=4, te=20e-3, tr=None)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "fse_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Pacing of one three-plane navigator (s), and the most one TR wait
    #: carries when their count is ``"auto"``: each takes a little
    #: longitudinal magnetisation from the volume the wait is restoring.
    NAVIGATOR_TR = 100e-3
    NAVIGATOR_COUNT = 5

    def init_sequence(
        self,
        fov: float | tuple[float, float] = 220e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 64,
        slab_thickness: float = 128e-3,
        etl: int = 32,
        te: float | None = 100e-3,
        tr: float | None = 1000e-3,
        ordering: str = "radial_adaptive",
        variable_flip: bool = True,
        alpha_min_deg: float = 60.0,
        alpha_center_deg: float = 100.0,
        alpha_max_deg: float = 160.0,
        readout_bandwidth_hz: float = 250e3,
        acceleration: int = 1,
        acceleration_z: int = 1,
        caipi_shift: int = 0,
        elliptical: bool = True,
        n_acs: int = 24,
        n_acs_z: int = 16,
        n_dummy: int = 0,
        shuffle_seed: int = 0,
        n_gain_calibration_readouts: int = 1,
        crusher_cycles: float = 4.0,
        readout_crusher_cycles: float = 0.0,
        navigator: bool = False,
        n_navigators: int | str = "auto",
        wave: str | None = None,
        wave_cycles: int = 8,
        wave_amplitude: float = 8e-3,
    ) -> None:
        """Design the train, its flip schedule and the view order that fills it.

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
        etl : int, optional
            Echo train length: views per excitation.
        te : float or None, optional
            Effective echo time, in seconds, rounded onto the echo grid.
            Ignored by ``radial``, which puts the centre on the first echo,
            and meaningless under ``shuffling``. ``None`` is the first echo.
        tr : float or None, optional
            Repetition time, in seconds, one per train. ``None`` is as short
            as the train admits.
        ordering : str, optional
            One of :data:`ORDERINGS`.
        variable_flip : bool, optional
            Play the refocusing train of :func:`traps_flip_schedule` rather
            than constant 180s.
        alpha_min_deg, alpha_center_deg, alpha_max_deg : float, optional
            The variable train's control points, in degrees.
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
        n_dummy : int, optional
            Trains played without acquiring before the first.
        shuffle_seed : int, optional
            Seed of the shuffling permutation and its Poisson-disc set.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        crusher_cycles : float, optional
            Cycles of dephasing each crusher beside a refocusing pulse winds.
        readout_crusher_cycles : float, optional
            Read-axis crushing each side of every acquisition, in cycles.
        navigator : bool, optional
            Play three-plane spiral navigators in the TR wait after each train.
        n_navigators : int or str, optional
            Navigators per wait; ``"auto"`` fits as many as the wait holds.
        wave : {'phase', 'partition', 'both'} or None, optional
            Wave-CAIPI corkscrew under every readout. The calibration
            rectangle is then acquired again without it.
        wave_cycles : int, optional
            Wave periods across the readout.
        wave_amplitude : float, optional
            Peak wave gradient, in T/m; a ceiling the slew rate may lower.
        """
        system = self.system
        fov_x, fov_y = (fov, fov) if np.isscalar(fov) else fov
        self.fov = (fov_x, fov_y, slab_thickness)
        self.matrix = (n_x, n_y, n_z)
        self.n_dummy, self.wave, self.ordering = n_dummy, wave, ordering
        self.n_gain_calibration_readouts = n_gain_calibration_readouts

        self.exc = sequences.SpatialSelectiveExcitation(
            system,
            90.0,
            slab_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            is_slab=True,
        )
        refocusing = sequences.SpatialSelectiveRefocusing(
            system,
            slab_thickness,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            spoiling_cycles=crusher_cycles,
        )
        self.fse = fse = sequences.FseReadout3D(
            system,
            self.exc.rf,
            self.exc.gz,
            rf_ref=refocusing.rf_ref,
            gz_ref=refocusing.gz,
            fov=self.fov,
            matrix=self.matrix,
            etl=etl,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=readout_crusher_cycles,
            wave=wave,
            wave_cycles=wave_cycles,
            wave_amplitude=wave_amplitude,
        )
        self.nominal = fse.rf_ref.amplitude
        self.wave_events = [
            g
            for g in (getattr(fse, "gy_wave", None), getattr(fse, "gz_wave", None))
            if g
        ]

        radial = ordering == "radial" or te is None
        self.n_center = 0 if radial else int(np.argmin(abs(fse.echo_times - te)))
        self.flips = (
            traps_flip_schedule(
                etl,
                self.n_center,
                alpha_min_deg=alpha_min_deg,
                alpha_center_deg=alpha_center_deg,
                alpha_max_deg=alpha_max_deg,
            )
            if variable_flip
            else np.full(etl, 180.0)
        )

        # Navigators ride in the TR wait after each train, where they cost no
        # scan time; what is left of the wait closes the repetition.
        length = fse.duration
        self.navigator, self.n_navigators = None, 0
        if navigator:
            self.navigator = sequences.SpiralNavigator(
                system, navigator_tr=self.NAVIGATOR_TR
            )
            window = 0.0 if tr is None else max(tr - length, 0.0)
            self.n_navigators = self.navigator.fit(
                window, n_navigators, limit=self.NAVIGATOR_COUNT
            )
            length += self.n_navigators * self.navigator.duration
        self.wait_tr = None
        if tr is not None:
            if tr < length - 1e-9:
                raise ValueError(
                    f"TR {tr * 1e3:.1f} ms is shorter than one train takes "
                    f"({length * 1e3:.1f} ms)"
                )
            pad = pp.round_to_raster(tr - length, system.block_duration_raster)
            if pad > 0:
                self.wait_tr = pp.make_delay(pad)
                length += pad
        self.repetition_time = length

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
        self.trains = order_views(
            views, etl, self.n_center, ordering, (n_y, n_z), seed=shuffle_seed
        )

        self.acs = (
            set(pp.calc_calibration_lines(n_y, n_acs)),
            set(pp.calc_calibration_lines(n_z, n_acs_z)),
        )
        # A wave-encoded line calibrates nothing, so with the wave on the
        # calibration rectangle is acquired again wave-free, as trains of its own.
        self.calibration_views = (
            [v for v in views if v[0] in self.acs[0] and v[1] in self.acs[1]]
            if wave is not None
            else []
        )
        n_calibration_trains = -(-len(self.calibration_views) // etl)
        self.duration = (n_dummy + len(self.trains) + n_calibration_trains) * length

    def loop(self) -> None:
        """Play the dummy trains, the wave-free calibration trains, then the imaging trains."""
        etl = self.fse.etl
        blank = [None] * etl
        for _ in range(self.n_dummy):
            self.kernel(blank, acquire=False, flags={"ONCE": 1})
        segment = 0
        if self.wave is not None:
            views = self.calibration_views
            for start in range(0, len(views), etl):
                train = (views[start : start + etl] + blank)[:etl]
                self.kernel(train, wave=0.0, flags={"ONCE": 0, "REF": 1, "SEG": 0})
            segment = 1
        for train in self.trains:
            self.kernel(train, flags={"ONCE": 0, "REF": 0, "SEG": segment})

    def kernel(
        self,
        views: list[tuple[int, int] | None],
        acquire: bool = True,
        wave: float = 1.0,
        flags: dict[str, int] | None = None,
    ) -> None:
        """One echo train over ``views``; ``None`` plays an echo unencoded.

        ``wave`` scales the corkscrew, and ``flags`` are the labels the train
        carries on its excitation.
        """
        fse, seq = self.fse, self.seq
        n_y, n_z = self.matrix[1:]
        acs_y, acs_z = self.acs
        corkscrew = [pp.scale_grad(g, wave) for g in self.wave_events]

        seq.add_block(fse.rf, fse.gz, *self.labels(**(flags or {})))
        seq.add_block(fse.gx_pre)
        for echo, view in enumerate(views):
            fse.rf_ref.amplitude = self.nominal * self.flips[echo] / 180.0
            seq.add_block(fse.rf_ref, fse.gz_ref)
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(fse.wait_esp1)
            line, partition = (n_y / 2, n_z / 2) if view is None else view
            ky, kz = (line - n_y / 2) / (n_y / 2), (partition - n_z / 2) / (n_z / 2)
            seq.add_block(
                fse.gx_bridge_pre,
                pp.scale_grad(fse.gy_pre, ky),
                pp.scale_grad(fse.gz_pre, kz),
            )
            if acquire and view is not None:
                calibrating = self.wave is None and line in acs_y and partition in acs_z
                labels = self.labels(LIN=line, PAR=partition, ECO=echo, IMA=calibrating)
                seq.add_block(fse.gx, fse.adc, *corkscrew, *labels)
            else:
                seq.add_block(fse.gx, *corkscrew)
            seq.add_block(
                fse.gx_bridge_post,
                pp.scale_grad(fse.gy_rew, ky),
                pp.scale_grad(fse.gz_rew, kz),
            )
        for _ in range(self.n_navigators):
            for block in self.navigator.blocks:
                seq.add_block(*block)
        if self.wait_tr is not None:
            seq.add_block(self.wait_tr)
        fse.rf_ref.amplitude = self.nominal

    def finalize(self) -> None:
        """Write the prescription, echo timing and flip train as definitions."""
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": float(self.fse.echo_times[self.n_center]),
            "TR": self.repetition_time,
            "EchoSpacing": self.fse.esp,
            "EchoTrainLength": self.fse.etl,
            "ViewOrdering": self.ordering,
            "RefocusingFlipAngles": list(self.flips),
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.fse.center_sample,
            "SliceThickness": self.exc.slice_thickness,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Fse3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="fse_3d.seq"))
