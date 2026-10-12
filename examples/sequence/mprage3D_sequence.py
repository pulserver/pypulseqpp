"""3D MPRAGE: inversion-prepared, RF-spoiled Cartesian gradient echo, one train of views per inversion."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The line orders ``ordering`` selects from.
ORDERINGS = ("radial", "shuffling")

NAME = "mprage_3d"
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
#: Seed of the shuffled line order and of its Poisson-disc views.
SHUFFLE_SEED = 0
#: Pacing of one three-plane navigator (s), and the most one recovery
#: carries: each takes a little longitudinal magnetisation from the volume
#: the recovery is restoring.
NAVIGATOR_TR = 100e-3
NAVIGATOR_COUNT = 5


def radius(view: tuple[int, int], n_y: int, n_z: int) -> float:
    """Squared distance of a view ``(y, z)`` from the k-space centre, normalised to the grid."""
    return ((view[0] - n_y // 2) / n_y) ** 2 + ((view[1] - n_z // 2) / n_z) ** 2


def trains(
    views: list[tuple[int, int]], length: int, n_y: int, n_z: int, ordering: str
) -> list[list[tuple[int, int]]]:
    """Deal views ``(y, z)`` into trains of ``length``.

    ``radial`` cuts the plane into angular wedges, each read centre-out
    (:func:`~pypulseqpp.make_radial_order`), so every train opens near the
    k-space centre; ``shuffling`` deals the views at random.
    """
    if not views:
        return []
    if ordering == "shuffling":
        rng = np.random.default_rng(SHUFFLE_SEED)
        shuffled = [views[i] for i in rng.permutation(len(views))]
        return [shuffled[i : i + length] for i in range(0, len(views), length)]
    coords = (np.asarray(views, dtype=float) - [n_y // 2, n_z // 2]) / [n_y, n_z]
    return [[views[i] for i in train] for train in pp.make_radial_order(coords, length)]


def mprage3d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 256e-3,
    fov_y: float = 256e-3,
    fov_z: float = 176e-3,
    n_x: int = 256,
    n_y: int = 256,
    n_z: int = 176,
    flip_angle_deg: float = 9.0,
    te: float | None = None,
    esp: float | None = None,
    ti: float | None = 900e-3,
    tr: float | None = 5000e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    rz: int = 1,
    caipi_shift: int = 0,
    partial_fourier_x: float = 1.0,
    partial_fourier_y: float = 1.0,
    partial_fourier_z: float = 1.0,
    n_dummy: int = 1,
    excitation: str = "slab",
    readout_oversampling: float = 2.0,
    n_acs_y: int = 24,
    n_acs_z: int = 16,
    elliptical_acs: bool = False,
    ordering: str = "radial",
    wave: str = "both",
    wave_cycles: int = 8,
    wave_amplitude: float = 0.0,
    navigator: bool = False,
) -> pp.Sequence:
    """3D MPRAGE: one inversion per shot, then a train of spoiled low-flip views.

    Each shot is the inversion, a wait that puts the first view's excitation
    centre at TI, one :class:`LineReadout3D` repetition per view of its train,
    and a recovery that makes every inversion-to-inversion interval the TR.
    Every train holds ``ceil(n_y / ry)`` ky-kz views: only views inside the
    inscribed ky-kz ellipse are sampled, and the unsampled views nearest the
    k-space centre are added until the views fill a whole number of trains, so
    every shot acquires the same number of views. ``ordering`` deals the trains
    as angular wedges of the ky-kz plane, each played centre-out, so that the
    centre view's excitation occurs at TI and its ADC echo at TI + TE, or as a
    random partition for a time-resolved reconstruction. Each acquisition
    carries its view as ``LIN`` and ``PAR``, and, shuffled, its place in the
    train, the time a time-resolved reconstruction resolves, as ``ECO``;
    calibration views are marked ``IMA``, and under wave-CAIPI the calibration
    region is first acquired wave-free in trains of its own, marked ``REF``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x : float, default=0.256
        Field of view along the readout (m).
    fov_y : float, default=0.256
        Field of view along the phase encode (m).
    fov_z : float, default=0.176
        Field of view along the partition encode (m). The slab excited is
        ``fov_z`` thick.
    n_x : int, default=256
        Matrix size along the readout.
    n_y : int, default=256
        Matrix size along the phase encode.
    n_z : int, default=176
        Matrix size along the partition encode.
    flip_angle_deg : float, default=9.0
        Readout excitation flip angle (degrees).
    te : float | None, default=None
        Echo time of every line (s). ``None`` is as short as the readout
        admits.
    esp : float | None, default=None
        Spacing of successive line excitations (s). ``None`` is as short
        as the readout admits.
    ti : float | None, default=0.9
        Inversion time (s), from the inversion pulse's centre to the first
        line's excitation. ``None`` is as short as the inversion module
        admits.
    tr : float | None, default=5.0
        Inversion-to-inversion interval (s). ``None`` leaves one raster of
        recovery after the train.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry, rz : int, default=1
        Undersampling along the phase and the partition encode.
    caipi_shift : int, default=0
        CAIPIRINHA shift: partitions by which the lattice is displaced per
        acquired line, in ``[0, rz)``.
        Unused by ``shuffling``.
    partial_fourier_x : float, default=1.0
        Fraction of the echo acquired, in ``[0.75, 1]``. Truncates the samples
        before the echo.
    partial_fourier_y, partial_fourier_z : float, default=1.0
        Fraction of the phase- and partition-encode extent acquired, in
        ``[0.75, 1]``.
    n_dummy : int, default=1
        Whole shots played without acquiring before the first.
    excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
        A slab-selective SLR pulse, a hard pulse, or a slab- and
        water-selective spectral-spatial pulse.
    readout_oversampling : float, default=2.0
        Readout oversampling factor, at least one.
    n_acs_y : int, default=24
        Extent of the fully sampled calibration region along the phase encode,
        acquired ahead of the rest when undersampled.
    n_acs_z : int, default=16
        Extent of the fully sampled calibration region along the partition
        encode, acquired ahead of the rest when undersampled.
    elliptical_acs : bool, default=False
        Make the calibration region the ellipse inscribed in the
        ``n_acs_y x n_acs_z`` rectangle rather than the rectangle.
    ordering : {'radial', 'shuffling'}, default='radial'
        Train order, and the support it is drawn from. ``'radial'``:
        CAIPIRINHA lattice support, each train an angular wedge of the ky-kz
        plane acquired centre-out (:func:`~pypulseqpp.make_radial_order`).
        ``'shuffling'``: variable-density Poisson-disc support
        (``sampling='poisson'`` of
        :func:`~pypulseqpp.make_cartesian_plane_sampling`), views dealt into
        trains in random order.
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
        Play three-plane spiral navigators in the recovery after each
        shot, as many as it holds up to ``NAVIGATOR_COUNT``.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``excitation`` or ``ordering`` is unknown, a partial Fourier
        fraction is outside ``[0.75, 1]``, an undersampling factor is below
        one, ``caipi_shift`` is outside ``[0, rz)``, ``wave`` names no wave
        mode, or the TE, the spacing, the TI or the TR is shorter than the
        shot takes.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.mprage3D_sequence(
    ...     n_x=32, n_y=16, n_z=8, ti=100e-3, tr=300e-3
    ... )
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if excitation not in sequences.EXCITATIONS:
        raise ValueError(
            f"excitation must be one of {sequences.EXCITATIONS}, got {excitation!r}"
        )
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
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

    fov = (fov_x, fov_y, fov_z)
    matrix = (n_x, n_y, n_z)
    inv = sequences.InversionPreparation(
        system, voxel_size_m=min(fov_x / n_x, fov_z / n_z)
    )
    exc = sequences.make_excitation(
        system,
        excitation,
        flip_angle_deg,
        fov_z,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        hard_duration_s=HARD_PULSE_DURATION,
        fat_shift_ppm=FAT_SHIFT_PPM,
    )
    gz = getattr(exc, "gz", None)
    ro = sequences.LineReadout3D(
        system,
        exc.rf,
        gz,
        fov=fov,
        matrix=matrix,
        te=te,
        tr=esp,
        partial_echo=partial_fourier_x,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=SPOILING_CYCLES,
        wave=wave,
        wave_cycles=wave_cycles,
        wave_amplitude=wave_amplitude,
    )
    waves = [
        gradient
        for gradient in (getattr(ro, "gy_wave", None), getattr(ro, "gz_wave", None))
        if gradient is not None
    ]
    no_waves = [pp.scale_grad(g, 0.0) for g in waves]
    spacing = ro.duration

    # Every shot plays one train of as many views as a line has at this
    # undersampling, and the views fill a whole number of trains.
    n_readouts = -(-n_y // ry)
    calibrating, imaging = pp.make_cartesian_plane_sampling(
        (n_y, n_z),
        (ry, rz),
        (n_acs_y, n_acs_z),
        caipi_shift=caipi_shift,
        partial_fourier=(partial_fourier_y, partial_fourier_z),
        elliptical=True,
        elliptical_acs=elliptical_acs,
        sampling="poisson" if ordering == "shuffling" else "lattice",
        seed=SHUFFLE_SEED,
        train_length=n_readouts,
    )
    calibration = set(calibrating)
    views = sorted({*calibrating, *imaging})
    shots = trains(views, n_readouts, n_y, n_z, ordering)
    # A wave-encoded view calibrates nothing, so with the wave on the
    # calibration region is acquired again wave-free, in shots of its own,
    # the last filled with the views nearest the centre.
    reference = []
    if waves:
        nearest = sorted(views, key=lambda v: (radius(v, n_y, n_z), v))
        region = [v for v in nearest if v in calibration]
        extra = [v for v in nearest if v not in calibration]
        region += extra[: -len(region) % n_readouts]
        reference = trains(region, n_readouts, n_y, n_z, "radial")

    # From the inversion centre to the first excitation centre: the rest
    # of the inversion module, the TI wait and the pulse's place in its
    # block.
    raster = system.block_duration_raster
    inversion_tail = inv.duration - (inv.rf_prep.delay + inv.rf_prep.center)
    ti_floor = inversion_tail + ro.rf.delay + ro.rf.center
    if ti is None:
        ti = ti_floor + raster
    elif ti < ti_floor + raster - 1e-9:
        raise ValueError(
            f"the requested TI of {ti * 1e3:.3f} ms is shorter than the "
            f"{(ti_floor + raster) * 1e3:.3f} ms the inversion takes"
        )
    wait_ti = pp.make_delay(pp.round_to_raster(ti - ti_floor, raster))
    inversion_time = ti_floor + wait_ti.delay

    body = inv.duration + wait_ti.delay + n_readouts * spacing
    if tr is None:
        tr = body + raster
    recovery = tr - body
    if recovery < raster - 1e-9:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
            f"{(body + raster) * 1e3:.3f} ms one shot takes"
        )
    # Navigators ride in the recovery, where they cost no scan time.
    spiral_navigator, n_navigators, navigating = None, 0, 0.0
    if navigator:
        spiral_navigator = sequences.SpiralNavigator(system, navigator_tr=NAVIGATOR_TR)
        n_navigators = spiral_navigator.fit(
            recovery - raster, "auto", limit=NAVIGATOR_COUNT
        )
        navigating = n_navigators * spiral_navigator.duration
    wait_recovery = pp.make_delay(pp.round_to_raster(recovery - navigating, raster))
    repetition_time = body + navigating + wait_recovery.delay

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(
        views: list[tuple[int, int] | None], phases, kind: str = "image"
    ) -> None:
        """Add one inversion-prepared shot over ``views``.

        ``phases`` are the RF-spoiling phases (rad) of its repetitions. A
        ``dummy`` shot acquires nothing, a ``reference`` shot plays without
        the wave, and a repetition past the end of ``views``, or at ``None``,
        acquires nothing either.
        """
        wave_events = no_waves if kind == "reference" else waves
        wait_te = getattr(ro, "wait_te", None)
        wait_tr = getattr(ro, "wait_tr", None)
        views = list(views) + [None] * (n_readouts - len(views))
        if kind == "dummy":
            flags = {"ONCE": 1}
        else:
            flags = {"ONCE": 0} if n_dummy else {}
            if waves:
                flags["REF"] = int(kind == "reference")

        seq.add_block(inv.rf_prep, *labels(**flags))
        seq.add_block(inv.gz_spoil)
        seq.add_block(wait_ti)
        for echo, (view, phase) in enumerate(zip(views, phases, strict=True)):
            # The slab sits at isocentre, so the excitation has no frequency
            # offset and its phase is the spoiling phase alone.
            ro.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            line, partition = (n_y // 2, n_z // 2) if view is None else view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            seq.add_block(ro.rf, *([] if gz is None else [gz]))
            if wait_te is not None:
                seq.add_block(wait_te)
            seq.add_block(
                ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
            )
            if kind != "dummy" and view is not None:
                is_calibration = not waves and view in calibration
                counters = {"LIN": line, "PAR": partition, "IMA": is_calibration}
                if ordering == "shuffling":
                    counters["ECO"] = echo
                label_events = labels(**counters)
                seq.add_block(ro.gx, *wave_events, ro.adc, *label_events)
            else:
                seq.add_block(ro.gx, *wave_events)
            seq.add_block(
                ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
            )
            if wait_tr is not None:
                seq.add_block(wait_tr)
        for _ in range(n_navigators):
            for block in spiral_navigator.blocks:
                seq.add_block(*block)
        seq.add_block(wait_recovery)

    # The dummy shots, the wave-free reference shots, then every shot.
    blank = [None] * n_readouts
    played = [(blank, "dummy")] * n_dummy
    played += [(shot, "reference") for shot in reference]
    played += [(shot, "image") for shot in shots]
    schedule = pp.make_rf_spoiling_schedule(
        len(played) * n_readouts, increment=np.deg2rad(RF_SPOILING_INCREMENT_DEG)
    ).reshape(len(played), n_readouts)
    for (views, kind), phases in zip(played, schedule, strict=True):
        kernel(views, phases, kind)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": list(fov),
        "Matrix": [n_x, n_y, n_z],
        "Name": NAME,
        "TE": ro.echo_time,
        "TR": repetition_time,
        "TI": inversion_time,
        "EchoSpacing": spacing,
        "EchoTrainLength": n_readouts,
        "Excitation": excitation,
        "ViewOrdering": ordering,
        "kSpaceCenterLine": n_y // 2,
        "kSpaceCenterPartition": n_z // 2,
        "kSpaceCenterSample": ro.center_sample,
        "SliceThickness": fov[2],
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = mprage3d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="mprage_3d.seq"))
