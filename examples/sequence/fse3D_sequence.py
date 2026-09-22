"""3D Cartesian fast spin echo, with optionally designed and individually parameterized echo trains."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp._masks import make_shuffling_order

EXCITATIONS = ("nonselective", "slab")

#: The view orders ``ordering`` selects from.
ORDERINGS = ("radial", "shuffling")

#: The refocusing trains ``flip_modulation`` selects from.
MODULATIONS = ("constant", "optimized")


def cubic(fraction):
    """Return the smooth step ``3 u**2 - 2 u**3`` the parameters move along.

    Parameters
    ----------
    fraction : array-like
        Where along the transition, from 0 at the centre to 1 at the
        periphery.

    Returns
    -------
    numpy.ndarray
        The step, over the same range.
    """
    fraction = np.asarray(fraction, dtype=float)
    return 3 * fraction**2 - 2 * fraction**3


def shot_parameters(
    n_views: int, etl: int, etl_periphery: int, tr: float, tr_periphery: float
) -> tuple[list[int], list[float], np.ndarray]:
    """Return every shot's train length and TR, and its place in the transition.

    Shot ``s`` of ``n`` sits at ``u = s / (n - 1)`` and takes its train length
    and TR a cubic step (:func:`cubic`) of the way from the centre values to
    the periphery ones (Buonincontri et al., ISMRM 2025, abstract 566-05-007,
    Fig. 1). ``n`` is the fewest shots whose trains hold every view.

    Parameters
    ----------
    n_views : int
        Views the shots have to hold between them.
    etl : int
        Echo train length of the centre shot.
    etl_periphery : int
        Echo train length of the periphery shot.
    tr : float
        Repetition time of the centre shot, in s.
    tr_periphery : float
        Repetition time of the periphery shot, in s.

    Returns
    -------
    lengths : list of int
        Each shot's train length.
    times : list of float
        Each shot's repetition time, in s.
    place : numpy.ndarray
        Each shot's place along the transition, from 0 to 1.
    """
    n = max(1, -(-n_views // max(etl, etl_periphery)))
    while True:
        place = np.arange(n) / (n - 1) if n > 1 else np.zeros(1)
        step = cubic(place)
        lengths = [round(etl + (etl_periphery - etl) * c) for c in step]
        if sum(lengths) >= n_views:
            break
        n += 1
    times = [tr + (tr_periphery - tr) * c for c in step]
    return lengths, times, place


def deal_trains(
    coords: np.ndarray,
    lengths: list[int],
    place: np.ndarray,
    te_echo: int,
    etl_max: int,
) -> list[list[int | None]]:
    """Deal views into trains by adaptive radial reordering.

    Every ``(shot, echo)`` slot a train plays is ranked by its distance from
    the prescription: how far the echo is from the TE echo, and how far the
    shot is along the transition. The views are ranked by radius. Both are cut
    into sections of one slot per shot, the innermost views filling the
    nearest slots, and within a section the views, sorted by angle, fill the
    slots in shot order (Buonincontri et al., abstract 566-05-007, Fig. 2D);
    the centre of k-space, which has no angle, takes the nearest slot.
    With identical trains the shot term vanishes and a section is one echo, so
    this is the radial ordering folded about the TE echo.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n_views, 2)`` phase-encode and partition coordinates of each view,
        centred on k-space.
    lengths : list of int
        Each shot's train length.
    place : numpy.ndarray
        Each shot's place along the transition, from 0 to 1.
    te_echo : int
        The echo the prescribed echo time falls on, counting from 0.
    etl_max : int
        Echoes the longest train holds.

    Returns
    -------
    list of list of int or None
        One ``etl_max``-long train per shot, holding view indices; None where
        an echo plays unencoded and unacquired.
    """
    n_shots = len(lengths)
    radius = np.hypot(coords[:, 0], coords[:, 1])
    angle = np.arctan2(coords[:, 1], coords[:, 0])
    views = sorted(range(len(coords)), key=lambda i: (radius[i], angle[i]))
    span = max(etl_max - 1, 1)
    slots = sorted(
        ((s, e) for s in range(n_shots) for e in range(lengths[s])),
        key=lambda slot: (
            math.hypot(abs(slot[1] - te_echo) / span, place[slot[0]]),
            slot[1],
            slot[0],
        ),
    )
    trains: list[list[int | None]] = [[None] * etl_max for _ in range(n_shots)]
    # The centre of k-space, which has no angle, takes the nearest slot.
    if len(views) and radius[views[0]] == 0.0:
        shot, echo = slots.pop(0)
        trains[shot][echo] = views.pop(0)
    for start in range(0, len(views), n_shots):
        section = sorted(views[start : start + n_shots], key=lambda i: angle[i])
        # The last section may hold fewer views than it has slots.
        for (s, e), view in zip(
            sorted(slots[start : start + n_shots]), section, strict=False
        ):
            trains[s][e] = view
    return trains


def design_trains(app, lengths, times, place, te_echo, etl_max, esp) -> np.ndarray:
    """Design the refocusing angles (degrees) of every shot, ``(shots, etl_max)``.

    A train falls from its maximum angle to its minimum over the first five
    echoes, rises to the prescribed angle at the TE echo and returns to the
    maximum at its end (Busse et al., Magn Reson Med 2008;60:640); a TE echo
    among the first five holds the prescribed angle up to it and falls to the
    minimum after. The minimum and maximum are designed with torchsim against
    the sharpness of the periphery of k-space, the contrast at the centre
    between the tissues ``DESIGN_CONTRAST`` names, and the RF power of the
    starting trains. Individually parameterized trains have a minimum and a
    maximum at each end, and every shot takes them a cubic step of the way
    along its transition. Angles past a shot's own train length are zero.

    Parameters
    ----------
    app : Fse3DApp
        The application being designed, read for its prescribed angle,
        design tissues and simulation parameters.
    lengths : list of int
        Each shot's train length.
    times : list of float
        Each shot's repetition time, in s.
    place : numpy.ndarray
        Each shot's place along the transition, from 0 to 1.
    te_echo : int
        The echo the prescribed echo time falls on, counting from 0.
    etl_max : int
        Echoes the longest train holds.
    esp : float
        Echo spacing, in s.

    Returns
    -------
    numpy.ndarray
        ``(shots, etl_max)`` refocusing angles in degrees, zero past a shot's
        own train length.

    Raises
    ------
    ImportError
        If torchsim is not installed.
    """
    try:
        import torch
        from torchsim.optim import Bounded, SequenceDesign
        from torchsim.simulators import FSESimulator
    except ImportError as error:
        raise ImportError(
            "flip_modulation='optimized' designs the trains with torchsim; "
            "install it with pip install 'pypulseqpp[design]'"
        ) from error

    dtype = torch.float64
    angle = float(app.refocusing_angle_deg)
    echo = torch.arange(1, etl_max + 1, dtype=dtype)
    length = torch.tensor(lengths, dtype=dtype)[:, None]
    tr_ms = torch.tensor(times, dtype=dtype)[:, None] * 1e3
    step = torch.as_tensor(cubic(place), dtype=dtype)[:, None]
    acquired = (echo <= length).to(dtype)
    centre = float(te_echo + 1)
    one = torch.ones_like(length)

    def smooth(knots, values):
        """Smooth steps between successive ``(knot, value)`` pairs, held beyond them."""
        out = values[-1] * one
        for k in reversed(range(len(knots) - 1)):
            span = (echo - knots[k]) / (knots[k + 1] - knots[k]).clamp_min(1e-3)
            span = span.clamp(0.0, 1.0)
            ramp = values[k] + (values[k + 1] - values[k]) * span.square() * (
                3 - 2 * span
            )
            out = torch.where(echo <= knots[k + 1], ramp, out)
        return out

    def trains(low, high):
        if centre > 5.0:
            knots = [one, 5.0 * one, centre * one, length]
            values = [high, low, angle * one, high]
        else:
            fallen = torch.minimum((centre + 4.0) * one, length)
            knots = [centre * one, fallen, length]
            values = [angle * one, low, high]
        return smooth(knots, values) * acquired

    simulator = FSESimulator(
        ESP=esp * 1e3,
        states=app.DESIGN_STATES,
        T1=list(app.DESIGN_T1_MS),
        T2=list(app.DESIGN_T2_MS),
    )
    bright, dark = app.DESIGN_CONTRAST

    def measure(centre_low, centre_high, edge_low, edge_high):
        low = centre_low + (edge_low - centre_low) * step
        high = centre_high + (edge_high - centre_high) * step
        flip = trains(low, high)
        signal = simulator.simulate(flip=flip, TR=tr_ms).abs()
        pair = acquired[:, None, :-1] * acquired[:, None, 1:]
        slope = (torch.diff(signal, dim=-1) * pair).square().sum(-1)
        energy = (signal * acquired[:, None, :]).square().sum(-1).clamp_min(1e-12)
        blur = (length / (2 * torch.pi) * (slope / energy).sqrt()).mean(-1)
        contrast = signal[:, bright, te_echo] - signal[:, dark, te_echo]
        power = ((flip / 180.0).square() * acquired).sum(-1) / (tr_ms[:, 0] * 1e-3)
        return flip, blur, contrast, power

    # Each control angle moves between its limits, the midpoint to start; one
    # with no room to move is held at its limit.
    lowest = min(app.DESIGN_MIN_DEG, 0.5 * angle)
    limits = {"low": (lowest, angle), "high": (angle, 180.0)}
    free, fixed = {}, {}
    for end in ("centre", "edge") if app.individual else ("centre",):
        for kind, (lower, upper) in limits.items():
            if upper - lower > 1.0:
                start = torch.tensor([0.5 * (lower + upper)], dtype=dtype)
                free[f"{end}_{kind}"] = Bounded(start, lower, upper)
            else:
                fixed[f"{end}_{kind}"] = torch.tensor([upper], dtype=dtype)

    def complete(values):
        values = {**fixed, **values}
        for kind in limits:
            values.setdefault(f"edge_{kind}", values[f"centre_{kind}"])
        return values

    start = complete({name: bound.initial for name, bound in free.items()})
    budget = measure(**start)[3].mean()
    weight = step[:, 0]
    outer = weight if app.individual else torch.ones_like(weight)
    inner = 1.0 - weight

    def cost(**values):
        _, blur, contrast, power = measure(**complete(values))
        return (
            (outer * blur).sum() / outer.sum()
            - app.DESIGN_CONTRAST_WEIGHT * (inner * contrast).sum() / inner.sum()
            + app.DESIGN_POWER_WEIGHT * torch.relu(power.mean() / budget - 1.0)
        )

    if free:
        result = SequenceDesign(cost, **free).minimize(
            iterations=app.DESIGN_ITERATIONS, learning_rate=app.DESIGN_LEARNING_RATE
        )
        start = complete(result.parameters)
    with torch.no_grad():
        flip = measure(**start)[0]
    return flip.numpy(force=True)


class Fse3DApp(sequences.SequenceApp):
    """3D Cartesian fast spin echo: one CPMG train per excitation over a (ky, kz) grid.

    Only views inside the inscribed ky-kz ellipse are sampled. ``radial``
    deals them by adaptive radial reordering (:func:`deal_trains`), so the
    centre of k-space is read at the TE echo; ``shuffling`` deals a
    variable-density Poisson-disc set randomly across echoes (Tamir et al.,
    Magn Reson Med 2017;77:180), for a time-resolved reconstruction. Every
    acquisition carries its line, partition and echo as ``LIN``, ``PAR`` and
    ``ECO``, and calibration views are marked ``IMA``; under wave-CAIPI the
    calibration region is first acquired wave-free in trains of its own,
    marked ``REF``.

    The refocusing angle is constant, or, under ``optimized``, the angle at the
    TE echo of a train :func:`design_trains` shapes around it. Giving the
    periphery of k-space its own TR or train length individually
    parameterizes the trains (Buonincontri et al., ISMRM 2025, abstract
    566-05-007): every shot moves from the centre values to the periphery ones
    along :func:`shot_parameters`, and the view order is radial. Every shot
    plays the longest train; past its own length a shot's refocusing pulses
    have zero amplitude and nothing is acquired, and its closing delay makes
    its TR, so the scan stays one repeating train.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.fse3D_sequence(
    ...     n_x=32, n_y=16, n_z=8, etl=8, te=20e-3, tr=300e-3
    ... )
    >>> seq.check_timing()[0]
    True
    """

    NAME = "fse_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab-selective excitation and refocusing.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Duration of the nonselective hard pulses (s).
    HARD_PULSE_DURATION = 0.5e-3
    #: Dephasing each crusher beside a refocusing pulse winds, in cycles
    #: across one voxel; a nonselective train crushes on the readout axis.
    CRUSHER_CYCLES = 4.0
    #: Seed of the shuffled order and of its Poisson-disc views.
    SHUFFLE_SEED = 0
    #: Pacing of one three-plane navigator (s), and the most one TR wait
    #: carries: each takes a little longitudinal magnetisation from the volume
    #: the wait is restoring.
    NAVIGATOR_TR = 100e-3
    NAVIGATOR_COUNT = 5
    #: Tissues the ``optimized`` trains are designed for (ms): cartilage,
    #: muscle and synovial fluid.
    DESIGN_T1_MS = (1200.0, 1420.0, 3600.0)
    DESIGN_T2_MS = (35.0, 30.0, 250.0)
    #: The two tissues, brighter first, whose difference at the TE echo is the
    #: contrast the design keeps, and how much it weighs against sharpness.
    DESIGN_CONTRAST = (2, 0)
    DESIGN_CONTRAST_WEIGHT = 12.0
    #: How much RF power beyond the starting trains' costs.
    DESIGN_POWER_WEIGHT = 20.0
    #: Lowest refocusing angle a designed train may reach (degrees).
    DESIGN_MIN_DEG = 20.0
    #: Configuration-state count, iterations and step of the design.
    DESIGN_STATES = 12
    DESIGN_ITERATIONS = 25
    DESIGN_LEARNING_RATE = 0.3

    def init_sequence(
        self,
        fov_x: float = 160e-3,
        fov_y: float = 160e-3,
        fov_z: float = 160e-3,
        n_x: int = 320,
        n_y: int = 240,
        n_z: int = 240,
        te: float | None = 28e-3,
        tr: float = 1800e-3,
        etl: int = 45,
        refocusing_angle_deg: float = 180.0,
        readout_bandwidth_hz: float = 250e3,
        ry: int = 1,
        rz: int = 1,
        caipi_shift: int = 0,
        partial_fourier_x: float = 1.0,
        partial_fourier_y: float = 1.0,
        partial_fourier_z: float = 1.0,
        *,
        n_dummy: int = 0,
        excitation: str = "slab",
        readout_oversampling: float = 2.0,
        esp: float | None = None,
        n_acs_y: int = 24,
        n_acs_z: int = 16,
        elliptical_acs: bool = False,
        ordering: str = "radial",
        flip_modulation: str = "constant",
        tr_periphery: float | None = None,
        etl_periphery: int | None = None,
        wave: str = "both",
        wave_cycles: int = 8,
        wave_amplitude: float = 0.0,
        navigator: bool = False,
    ) -> None:
        """Design the train, its refocusing angles, the shots and the views they read.

        Parameters
        ----------
        fov_x, fov_y, fov_z : float, default=0.16
            Field of view along the readout, the phase encode and the
            partition encode (m). A slab excited is ``fov_z`` thick.
        n_x : int, default=320
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_y : int, default=240
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_z : int, default=240
            Matrix size along the readout, the phase encode and the partition
            encode.
        te : float | None, default=0.028
            Effective echo time (s): the echo the centre of k-space is read
            at, rounded onto the echo grid. ``None`` is the first echo.
            Meaningless under ``shuffling``.
        tr : float, default=1.8
            Repetition time (s), one per train; at the centre of k-space when
            the trains are individually parameterized.
        etl : int, default=45
            Echo train length; at the centre of k-space when the trains are
            individually parameterized.
        refocusing_angle_deg : float, default=180.0
            Refocusing flip angle (degrees): of every pulse under
            ``constant``, at the TE echo under ``optimized``.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz).
        ry, rz : int, default=1
            Undersampling along the phase and the partition encode.
        caipi_shift : int, default=0
            Partitions the lattice climbs per acquired line, in ``[0, rz)``.
            Unused by ``shuffling``.
        partial_fourier_x : float, default=1.0
            Fraction of the echo acquired, in ``[0.75, 1]``.
        partial_fourier_y, partial_fourier_z : float, default=1.0
            Fraction of the phase- and partition-encode extent acquired, in
            ``[0.75, 1]``.
        n_dummy : int, default=0
            Trains played without acquiring before the first.
        excitation : {'slab', 'nonselective'}, default='slab'
            Slab-selective excitation and refocusing, or hard pulses with
            readout-axis crushers, which shorten the echo spacing.
        readout_oversampling : float, default=2.0
            Readout oversampling factor, at least one.
        esp : float | None, default=None
            Echo spacing (s). ``None`` is as short as the train admits.
        n_acs_y : int, default=24
            Extent of the fully sampled calibration region along the phase
            and the partition encode, when undersampled.
        n_acs_z : int, default=16
            Extent of the fully sampled calibration region along the phase
            and the partition encode, when undersampled.
        elliptical_acs : bool, default=False
            Make the calibration region the ellipse inscribed in the
            ``n_acs_y x n_acs_z`` rectangle rather than the rectangle.
        ordering : {'radial', 'shuffling'}, default='radial'
            Adaptive radial reordering on the CAIPIRINHA lattice, or a
            shuffled Poisson-disc set.
        flip_modulation : {'constant', 'optimized'}, default='constant'
            Constant refocusing angles, or trains designed with torchsim,
            which the ``design`` extra installs.
        tr_periphery : float | None, default=None
            Repetition time (s) at the periphery of k-space. ``None`` is
            ``tr``.
        etl_periphery : int | None, default=None
            Echo train length at the periphery of k-space. ``None`` is
            ``etl``.
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
        navigator : bool, default=False
            Play three-plane spiral navigators after each train, as many as
            the shortest TR holds up to :attr:`NAVIGATOR_COUNT`.

        Raises
        ------
        ValueError
            If ``excitation``, ``ordering`` or ``flip_modulation`` is unknown, a
            partial Fourier fraction is outside ``[0.75, 1]``, a count is below
            one, ``caipi_shift`` is outside ``[0, rz)``, ``wave`` names no wave
            mode, individually parameterized trains are shuffled, a train ends
            before the TE echo, or a TR is shorter than the longest train.
        ImportError
            If ``optimized`` is asked for without torchsim.
        """
        self.n_dummy = n_dummy
        for name, value, allowed in (
            ("excitation", excitation, EXCITATIONS),
            ("ordering", ordering, ORDERINGS),
            ("flip_modulation", flip_modulation, MODULATIONS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        for name, fraction in (
            ("partial_fourier_x", partial_fourier_x),
            ("partial_fourier_y", partial_fourier_y),
            ("partial_fourier_z", partial_fourier_z),
        ):
            if not 0.75 <= fraction <= 1.0:
                raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
        etl_periphery = etl if etl_periphery is None else etl_periphery
        for name, count in (
            ("ry", ry),
            ("rz", rz),
            ("etl", etl),
            ("etl_periphery", etl_periphery),
        ):
            if count < 1:
                raise ValueError(f"{name} must be at least 1, got {count}")
        if not 0 <= caipi_shift < rz:
            raise ValueError(f"caipi_shift must lie in [0, {rz}), got {caipi_shift}")
        system = self.system
        self.fov = (fov_x, fov_y, fov_z)
        self.matrix = (n_x, n_y, n_z)
        self.excitation, self.ordering = excitation, ordering
        self.flip_modulation = flip_modulation
        self.refocusing_angle_deg = refocusing_angle_deg
        self.etl, self.etl_periphery = etl, etl_periphery
        etl_max = max(etl, etl_periphery)

        if excitation == "slab":
            self.exc = sequences.SpatialSelectiveExcitation(
                system,
                90.0,
                fov_z,
                duration_s=self.PULSE_DURATION,
                time_bw_product=self.TIME_BW_PRODUCT,
                is_slab=True,
            )
            refocusing = sequences.SpatialSelectiveRefocusing(
                system,
                fov_z,
                duration_s=self.PULSE_DURATION,
                time_bw_product=self.TIME_BW_PRODUCT,
                spoiling_cycles=self.CRUSHER_CYCLES,
            )
            rf_ref, gz_ref, read_crushing = refocusing.rf_ref, refocusing.gz, 0.0
        else:
            # An even number of block rasters puts each pulse centre on the raster.
            raster = system.block_duration_raster
            hard = (
                2 * raster * math.ceil(self.HARD_PULSE_DURATION / (2 * raster) - 1e-9)
            )
            self.exc = sequences.NonSelectiveExcitation(system, 90.0, duration_s=hard)
            rf_ref = sequences.NonSelectiveRefocusing(system, duration_s=hard).rf_ref
            gz_ref, read_crushing = None, self.CRUSHER_CYCLES
        self.gz = getattr(self.exc, "gz", None)
        self.fse = fse = sequences.FseReadout3D(
            system,
            self.exc.rf,
            self.gz,
            rf_ref=rf_ref,
            gz_ref=gz_ref,
            fov=self.fov,
            matrix=self.matrix,
            etl=etl_max,
            esp=esp,
            partial_echo=partial_fourier_x,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=read_crushing,
            wave=wave,
            wave_cycles=wave_cycles,
            wave_amplitude=wave_amplitude,
        )
        self.nominal = fse.rf_ref.amplitude
        self.gz_ref = getattr(fse, "gz_ref", None)
        self.waves = [
            gradient
            for gradient in (
                getattr(fse, "gy_wave", None),
                getattr(fse, "gz_wave", None),
            )
            if gradient is not None
        ]
        self.no_waves = [pp.scale_grad(g, 0.0) for g in self.waves]
        self.te_echo = 0 if te is None else int(np.argmin(np.abs(fse.echo_times - te)))
        if min(etl, etl_periphery) <= self.te_echo:
            raise ValueError(
                f"a train of {min(etl, etl_periphery)} echoes ends before echo "
                f"{self.te_echo + 1}, which the TE of {te * 1e3:.1f} ms falls on"
            )

        minimum_tr = fse.duration + system.block_duration_raster
        tr = minimum_tr if tr is None else tr
        tr_periphery = tr if tr_periphery is None else tr_periphery
        self.individual = tr_periphery != tr or etl_periphery != etl
        if self.individual and ordering != "radial":
            raise ValueError(
                "individually parameterized trains need the radial order, which "
                "places each view by its distance from the prescription"
            )
        self.repetition_time, self.tr_periphery = tr, tr_periphery

        calibrating, lattice = pp.calc_sampled_pairs(
            (n_y, n_z),
            (ry, rz),
            (n_acs_y, n_acs_z),
            caipi_shift=caipi_shift,
            partial_fourier=(partial_fourier_y, partial_fourier_z),
            elliptical=True,
            elliptical_acs=elliptical_acs,
            shuffling=ordering == "shuffling",
            seed=self.SHUFFLE_SEED,
        )
        self.calibration = set(calibrating)
        views = sorted({*calibrating, *lattice})
        coords = (np.asarray(views, dtype=float) - [n_y // 2, n_z // 2]) / [n_y, n_z]
        self.lengths, self.times, place = shot_parameters(
            len(views), etl, etl_periphery, tr, tr_periphery
        )
        raster = system.block_duration_raster
        self.times = [pp.round_to_raster(time, raster) for time in self.times]
        if not self.individual:
            # Identical trains sit nowhere along a transition.
            place = np.zeros_like(place)
        if ordering == "shuffling":
            order = make_shuffling_order(coords, etl, seed=self.SHUFFLE_SEED, pad=True)
            self.lengths = [etl] * len(order)
            self.times = [tr] * len(order)
            place = np.zeros(len(order))
        else:
            order = deal_trains(coords, self.lengths, place, self.te_echo, etl_max)
        self.trains = [
            [None if i is None else views[i] for i in train] for train in order
        ]

        # Refocusing angles, one row per shot; past its own length a shot's
        # pulses play at zero amplitude.
        played = np.arange(etl_max)[None, :] < np.asarray(self.lengths)[:, None]
        if flip_modulation == "optimized":
            self.flips = design_trains(
                self, self.lengths, self.times, place, self.te_echo, etl_max, fse.esp
            )
        else:
            self.flips = refocusing_angle_deg * played.astype(float)

        # A wave-encoded view calibrates nothing, so with the wave on the
        # calibration region is acquired again wave-free, in trains of its own
        # played as the first shot is.
        self.reference = []
        if self.waves:
            region = [v for v in views if v in self.calibration]
            for start in range(0, len(region), self.lengths[0]):
                chunk = region[start : start + self.lengths[0]]
                self.reference.append(chunk + [None] * (etl_max - len(chunk)))

        # Navigators ride after each train, where they cost no scan time; the
        # closing delay makes every shot's own TR.
        shortest = min(self.times)
        if shortest < fse.duration + raster - 1e-9:
            raise ValueError(
                f"the TR of {shortest * 1e3:.1f} ms is shorter than the "
                f"{(fse.duration + raster) * 1e3:.1f} ms the longest train takes"
            )
        self.navigator, self.n_navigators, navigating = None, 0, 0.0
        if navigator:
            self.navigator = sequences.SpiralNavigator(
                system, navigator_tr=self.NAVIGATOR_TR
            )
            self.n_navigators = self.navigator.fit(
                shortest - fse.duration - raster, "auto", limit=self.NAVIGATOR_COUNT
            )
            navigating = self.n_navigators * self.navigator.duration
        self.closing = [
            pp.round_to_raster(time - fse.duration - navigating, raster)
            for time in self.times
        ]
        self.duration = (n_dummy + len(self.reference)) * tr + sum(self.times)

    def loop(self) -> None:
        """Play the dummy trains, the wave-free reference trains, then every shot."""
        blank = [None] * self.fse.etl
        for _ in range(self.n_dummy):
            self.kernel(blank, self.flips[0], self.closing[0], "dummy")
        for views in self.reference:
            self.kernel(views, self.flips[0], self.closing[0], "reference")
        for views, flips, closing in zip(
            self.trains, self.flips, self.closing, strict=True
        ):
            self.kernel(views, flips, closing)

    def kernel(self, views, flips, closing: float, kind: str = "image") -> None:
        """One echo train over ``views`` at refocusing angles ``flips`` (degrees).

        ``None`` plays an echo unencoded and unacquired, and ``closing`` is the
        delay after the train. A ``dummy`` train acquires nothing and a
        ``reference`` train plays without the wave.

        Parameters
        ----------
        views : sequence
            One view per echo, as ``(line, partition)``; None plays an echo
            unencoded and unacquired.
        flips : sequence of float
            One refocusing angle per echo, in degrees.
        closing : float
            Delay after the train, in s.
        kind : str, default="image"
            ``"image"``, ``"dummy"`` or ``"reference"``.
        """
        fse, seq = self.fse, self.seq
        n_y, n_z = self.matrix[1:]
        waves = self.no_waves if kind == "reference" else self.waves
        if kind == "dummy":
            flags = {"ONCE": 1}
        else:
            flags = {"ONCE": 0} if self.n_dummy else {}
            if self.waves:
                flags["REF"] = int(kind == "reference")

        seq.add_block(
            fse.rf, *([] if self.gz is None else [self.gz]), *self.labels(**flags)
        )
        seq.add_block(fse.gx_pre)
        for echo, (view, flip) in enumerate(zip(views, flips, strict=True)):
            fse.rf_ref.amplitude = self.nominal * float(flip) / 180.0
            seq.add_block(fse.rf_ref, *([] if self.gz_ref is None else [self.gz_ref]))
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(fse.wait_esp1)
            line, partition = (n_y // 2, n_z // 2) if view is None else view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            seq.add_block(
                fse.gx_bridge_pre,
                pp.scale_grad(fse.gy_pre, ky),
                pp.scale_grad(fse.gz_pre, kz),
            )
            if kind != "dummy" and view is not None:
                calibrating = not self.waves and view in self.calibration
                labels = self.labels(LIN=line, PAR=partition, ECO=echo, IMA=calibrating)
                seq.add_block(fse.gx, *waves, fse.adc, *labels)
            else:
                seq.add_block(fse.gx, *waves)
            seq.add_block(
                fse.gx_bridge_post,
                pp.scale_grad(fse.gy_rew, ky),
                pp.scale_grad(fse.gz_rew, kz),
            )
        fse.rf_ref.amplitude = self.nominal
        for _ in range(self.n_navigators):
            for block in self.navigator.blocks:
                seq.add_block(*block)
        seq.add_block(pp.make_delay(closing))

    def finalize(self) -> None:
        """Write the prescription, the echo timing and the trains as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": float(self.fse.echo_times[self.te_echo]),
            "TR": self.repetition_time,
            "EchoSpacing": self.fse.esp,
            "EchoTrainLength": self.etl,
            "Excitation": self.excitation,
            "ViewOrdering": self.ordering,
            "FlipModulation": self.flip_modulation,
            "RefocusingFlipAngles": [
                float(f) for f in self.flips[0][: self.lengths[0]]
            ],
            "NumShots": len(self.trains),
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": self.fse.center_sample,
            "SliceThickness": self.fov[2],
        }
        if self.individual:
            definitions["TRPeriphery"] = self.tr_periphery
            definitions["EchoTrainLengthPeriphery"] = self.etl_periphery
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Fse3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="fse_3d.seq"))
