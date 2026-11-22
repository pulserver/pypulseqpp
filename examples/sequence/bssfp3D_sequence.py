"""Balanced SSFP, 3D Cartesian, with phase cycling."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences
from pypulseqpp.sequences.excitation._base import rf_reference

EXCITATIONS = ("nonselective", "slab")
GATINGS = ("none", "prospective")
TRIGGER_CHANNELS = ("physio1", "physio2")
AXES = ("x", "y", "z")
#: Excitations over which a segment's flip angle ramps linearly up to its own:
#: a catalyst that damps the oscillating transient of a train opened from rest.
CATALYST_RAMP = 10

#: Adiabaticity of the T2 preparation where the peak B1 is too low for its
#: default: lowered as far as 0.75 of it, it stays above an inversion's.
T2_PREP_LOW_B1_ADIABATICITY = 6.0

NAME = "bssfp_3d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the slab-selective pulse.
PULSE_DURATION = 1.0e-3
TIME_BW_PRODUCT = 2.0
#: Duration of the nonselective hard pulse (s).
HARD_PULSE_DURATION = 0.5e-3


def t2_preparation(
    system: pp.Opts, echo_time_s: float, voxel_size_m: float
) -> sequences.T2Preparation:
    """Return the T2 preparation at its default adiabaticity, or at ``T2_PREP_LOW_B1_ADIABATICITY`` where the peak B1 is too low for it.

    Raises
    ------
    ValueError
        If the pulses do not fit the peak B1 at the lower adiabaticity either.
    """
    try:
        return sequences.T2Preparation(system, echo_time_s, voxel_size_m=voxel_size_m)
    except ValueError as error:
        if "adiabatic" not in str(error):
            raise
    return sequences.T2Preparation(
        system,
        echo_time_s,
        adiabaticity=T2_PREP_LOW_B1_ADIABATICITY,
        voxel_size_m=voxel_size_m,
    )


def segment_views(
    views: list[tuple[int, int]], n_y: int, n_z: int, views_per_segment: int
) -> list[list[tuple[int, int]]]:
    """Deal ``(line, partition)`` views to segments of at most ``views_per_segment``, each a wedge read centre-out.

    The views are sorted by their angle in the ky-kz plane, each axis scaled
    to its own extent, and cut into as few contiguous runs as hold
    ``views_per_segment`` each; a run is read in order of distance from the
    k-space centre. Every segment so starts at the centre and steps between
    neighbouring views.

    Parameters
    ----------
    views : list of tuple of int
        ``(line, partition)`` of every view.
    n_y, n_z : int
        Matrix size along the phase and the partition encode.
    views_per_segment : int
        Largest number of views in a segment.

    Returns
    -------
    list of list of tuple of int
        The views of each segment, in play order.

    Examples
    --------
    >>> views = [(y, z) for y in range(4) for z in range(4)]
    >>> segments = segment_views(views, 4, 4, 4)
    >>> [len(segment) for segment in segments]
    [4, 4, 4, 4]
    >>> sorted(v for segment in segments for v in segment) == views
    True

    Each quadrant of a 4 x 4 plane is a segment, read from the view next to
    the centre:

    >>> [segment[0] for segment in segments]
    [(1, 1), (2, 1), (2, 2), (1, 2)]
    """
    ky = np.array([(line - n_y // 2) / (n_y / 2) for line, _ in views])
    kz = np.array([(partition - n_z // 2) / (n_z / 2) for _, partition in views])
    radius = np.hypot(ky, kz)
    order = np.lexsort((radius, np.arctan2(kz, ky)))
    n_segments = math.ceil(len(views) / views_per_segment)
    return [
        [views[i] for i in run[np.argsort(radius[run], kind="stable")]]
        for run in np.array_split(order, n_segments)
    ]


def bssfp3d(
    system: pp.Opts | None = None,
    *,
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
    excitation: str = "nonselective",
    readout_oversampling: float = 2.0,
    n_acs_y: int = 24,
    n_acs_z: int = 16,
    elliptical_acs: bool = False,
    n_phase_cycles: int = 1,
    n_dummy: int | None = None,
    gating: str = "none",
    trigger_channel: str = "physio2",
    trigger_delay: float = 0.0,
    trigger_period: float = 1.0,
    views_per_segment: int = 32,
    n_ramp: int = CATALYST_RAMP,
    navigator: bool = False,
    t2_prep_te: float = 0.0,
    fat_saturation: bool = False,
    inversion_time: float = 0.0,
    inversion_thickness: float = 0.0,
    inversion_axis: str = "x",
    inversion_offset: float = 0.0,
) -> list[pp.Sequence]:
    """Balanced SSFP, 3D Cartesian: one train per phase cycle, each opened by a half flip.

    Every repetition returns all three gradient moments to zero, with TE at
    TR/2 (:class:`BssfpReadout3D`). Only views inside the inscribed ky-kz
    ellipse are sampled, lines in order and each line's partitions back and
    forth. Phase cycle ``k`` of ``n_phase_cycles`` steps the RF and receiver
    phase by ``pi + 2 pi k / n_phase_cycles`` per repetition, which shifts the
    off-resonance bands; two cycles is CISS. Each cycle is a file of its own, a
    repeating unit from its first excitation, and a ``catalyst`` file ahead of
    it plays the half flip half a repetition before the first of ``n_dummy``
    unencoded repetitions that acquire nothing, each at the phase it would have
    had in the cycle, while the oscillating transient settles. The files are chained
    in that order, so the scanner plays them as one table. Acquisitions carry
    their line, partition and cycle as ``LIN``, ``PAR`` and ``SET``, and
    calibration views are marked ``IMA``.

    Under ``prospective`` gating the views are acquired in segments of
    ``views_per_segment``, one per trigger on ``trigger_channel``: an ECG
    trigger for whole-heart coronary MRA, a respiratory one for inflow MRA.
    Each segment waits for its trigger and ``trigger_delay``, then plays in
    order a slab-selective or non-selective adiabatic inversion and
    ``inversion_time`` of recovery, an adiabatic T2 preparation, a fat
    saturation, and opens from rest with ``n_ramp`` excitations whose flip
    angle ramps linearly to ``flip_angle_deg`` before its views. The views
    are dealt to the segments by angle in the ky-kz plane, and each segment
    reads its wedge from the k-space centre outwards (:func:`segment_views`),
    so every segment samples the centre right after its preparations and
    steps between neighbouring views. With ``navigator`` the ramp
    excitations read lines of the central partition, ``NAV`` set: an image
    navigator, the projection of the slab along the partition axis, from
    which a reconstruction tracks each segment's motion. Segments carry
    ``SEG``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x : float, default=0.22
        Field of view along the readout (m).
    fov_y : float, default=0.22
        Field of view along the phase encode (m).
    fov_z : float, default=0.128
        Field of view along the partition encode (m). The slab excited is
        ``fov_z`` thick.
    n_x : int, default=256
        Matrix size along the readout.
    n_y : int, default=256
        Matrix size along the phase encode.
    n_z : int, default=128
        Matrix size along the partition encode.
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
        CAIPIRINHA shift: partitions by which the lattice is displaced per
        acquired line, in ``[0, rz)``.
    partial_fourier_y, partial_fourier_z : float, default=1.0
        Fraction of the phase- and partition-encode extent acquired, in
        ``[0.75, 1]``.
    excitation : {'nonselective', 'slab'}, default='nonselective'
        A hard pulse, or a slab-selective SLR pulse whose rephasers the
        balanced readout builds.
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
    n_phase_cycles : int, default=1
        Trains acquired, each with its own RF phase increment.
    n_dummy : int | None, default=None
        Repetitions each catalyst plays without acquiring after the half
        flip. ``None`` is as many as bring the balanced steady state within
        1 % (:func:`~pypulseqpp.sequences.steady_state_dummies` at
        :data:`~pypulseqpp.sequences.STEADY_STATE_T2_S`). Unused under
        ``prospective`` gating, whose segments open with their ramp.
    gating : {'none', 'prospective'}, default='none'
        Free-running acquisition, or segments each waiting for a trigger.
    trigger_channel : {'physio1', 'physio2'}, default='physio2'
        Physiological signal a segment waits for.
    trigger_delay : float, default=0.0
        Time from a trigger to the segment's first preparation (s); zero, or
        at least 1 ms.
    trigger_period : float, default=1.0
        Nominal interval between triggers (s): the R-R interval, or the
        respiratory period. A segment must fit in it.
    views_per_segment : int, default=32
        Views a segment acquires.
    n_ramp : int, default=CATALYST_RAMP
        Excitations of a segment's catalyst ramp, at least one.
    navigator : bool, default=False
        Read an image navigator during the catalyst ramp.
    t2_prep_te : float, default=0.0
        Echo time of an adiabatic T2 preparation ahead of each segment (s);
        zero for none.
    fat_saturation : bool, default=False
        Saturate fat ahead of each segment.
    inversion_time : float, default=0.0
        Time from an adiabatic inversion to the segment's first imaging
        excitation (s); zero for none.
    inversion_thickness : float, default=0.0
        Thickness of the inverted slab (m); zero inverts the whole transmit
        volume.
    inversion_axis : {'x', 'y', 'z'}, default='x'
        Logical axis the inverted slab is selected along.
    inversion_offset : float, default=0.0
        Offset of the inverted slab from the field-of-view centre along
        ``inversion_axis`` (m).

    Returns
    -------
    list of pypulseqpp.Sequence
        The designed sequences in play order: each phase cycle's catalyst, then
        its train. The last is the main sequence.

    Raises
    ------
    ValueError
        If ``excitation`` is unknown, a partial Fourier fraction is outside
        ``[0.75, 1]``, a count is below one, ``caipi_shift`` is outside
        ``[0, rz)``, or the TR is shorter than the repetition, or too short
        beside the pulse for the half flip; if ``gating``,
        ``trigger_channel`` or ``inversion_axis`` is unknown, a prospective
        scan has more than one phase cycle, a trigger delay below 1 ms, an
        inversion time shorter than what follows the inversion, or a segment
        longer than ``trigger_period``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> chain = sequences.bssfp3D_sequence(n_x=64, n_y=16, n_z=4)
    >>> [seq.get_definition("Name") for seq in chain]
    ['bssfp_3d_catalyst', 'bssfp_3d']
    >>> all(seq.check_timing()[0] for seq in chain)
    True

    A whole-heart coronary MRA: ECG-triggered segments with a T2
    preparation, fat saturation and an image navigator:

    >>> [coronary] = sequences.bssfp3D_sequence(
    ...     n_x=64, n_y=16, n_z=4, gating="prospective", trigger_delay=0.5,
    ...     t2_prep_te=0.04, fat_saturation=True, navigator=True,
    ...     views_per_segment=8,
    ... )
    >>> coronary.get_definition("NumSegments")
    [6.0]
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if excitation not in EXCITATIONS:
        raise ValueError(f"excitation must be one of {EXCITATIONS}, got {excitation!r}")
    for name, fraction in (
        ("partial_fourier_y", partial_fourier_y),
        ("partial_fourier_z", partial_fourier_z),
    ):
        if not 0.75 <= fraction <= 1.0:
            raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
    for name, count in (("ry", ry), ("rz", rz), ("n_phase_cycles", n_phase_cycles)):
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    if n_dummy is not None and n_dummy < 0:
        raise ValueError(f"n_dummy must be at least 0, got {n_dummy}")
    if not 0 <= caipi_shift < rz:
        raise ValueError(f"caipi_shift must lie in [0, {rz}), got {caipi_shift}")
    if gating not in GATINGS:
        raise ValueError(f"gating must be one of {GATINGS}, got {gating!r}")
    if trigger_channel not in TRIGGER_CHANNELS:
        raise ValueError(
            f"trigger_channel must be one of {TRIGGER_CHANNELS}, got {trigger_channel!r}"
        )
    if inversion_axis not in AXES:
        raise ValueError(
            f"inversion_axis must be one of {AXES}, got {inversion_axis!r}"
        )
    for name, count in (("views_per_segment", views_per_segment), ("n_ramp", n_ramp)):
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    if gating == "prospective" and n_phase_cycles != 1:
        raise ValueError(
            f"a prospective scan acquires one phase cycle, not {n_phase_cycles}"
        )
    if trigger_delay != 0.0 and trigger_delay < 1e-3:
        raise ValueError(
            f"trigger_delay must be zero or at least 1 ms, got {trigger_delay * 1e3:g} ms"
        )

    fov = (fov_x, fov_y, fov_z)
    matrix = (n_x, n_y, n_z)
    if excitation == "slab":
        exc = sequences.SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            fov_z,
            duration_s=PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
            rephase=False,
        )
        rf, gz = exc.rf, exc.gz
    else:
        # An even number of block rasters puts the pulse centre on the raster.
        raster = system.block_duration_raster
        hard = 2 * raster * math.ceil(HARD_PULSE_DURATION / (2 * raster) - 1e-9)
        rf, gz = (
            sequences.NonSelectiveExcitation(
                system, flip_angle_deg, duration_s=hard
            ).rf,
            None,
        )
    ro = sequences.BssfpReadout3D(
        system,
        rf,
        gz,
        fov=fov,
        matrix=matrix,
        tr=tr,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
    )
    ro_gz = getattr(ro, "gz", None)
    nominal = ro.rf.amplitude
    if n_dummy is None:
        n_dummy = sequences.steady_state_dummies(
            ro.tr, flip_angle_deg, t2_s=sequences.STEADY_STATE_T2_S
        )
    # The slice rephaser and the partition encode share one ramp and one
    # window, so their sum is the partition lobe scaled.
    z_pre = getattr(ro, "gz_pre", None)
    z_rew = getattr(ro, "gz_rew", None)

    calibrating, imaging = pp.make_cartesian_plane_sampling(
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
    views = sorted({*calibrating, *imaging})
    calibration = set(calibrating)
    increments = [np.pi + 2 * np.pi * k / n_phase_cycles for k in range(n_phase_cycles)]

    def define(seq: pp.Sequence, **definitions) -> None:
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        for key, value in {
            "FOV": list(fov),
            "Matrix": [n_x, n_y, n_z],
            "TE": ro.te,
            "TR": ro.tr,
            "Excitation": excitation,
            "NumPhaseCycles": len(increments),
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "kSpaceCenterSample": ro.center_sample,
            "SliceThickness": fov[2],
            **definitions,
        }.items():
            seq.set_definition(key=key, value=value)

    def z_encode(lobe, rephaser, kz: float):
        """Return ``lobe`` scaled to ``kz``, plus the ``rephaser`` sharing its window."""
        amplitude = kz * lobe.amplitude
        if rephaser is not None:
            amplitude += rephaser.amplitude
        return pp.scale_grad(lobe, amplitude / lobe.amplitude)

    def catalyst(k: int) -> pp.Sequence:
        """Design cycle ``k``'s half flip and dummy repetitions, ahead of its first excitation."""
        seq = pp.Sequence(system)
        increment = increments[k]
        # The first excitation of the cycle is at the increment itself, so
        # repetition n before it is at -n increments.
        ro.rf.amplitude = 0.5 * nominal
        ro.rf.phase_offset = -n_dummy * increment % (2 * np.pi)
        seq.add_block(ro.rf, *([] if ro_gz is None else [ro_gz]))
        ro.rf.amplitude = nominal
        seq.add_block(ro.wait_prep, *([] if ro.gz_prep is None else [ro.gz_prep]))
        for n in range(n_dummy - 1, -1, -1):
            ro.rf.phase_offset = -n * increment % (2 * np.pi)
            seq.add_block(ro.rf, *([] if ro_gz is None else [ro_gz]))
            seq.add_block(
                ro.gx,
                pp.scale_grad(ro.gy_pre, 0.0),
                z_encode(ro.gz_partition, z_pre, 0.0),
            )
            seq.add_block(
                ro.gx_rew,
                pp.scale_grad(ro.gy_rew, 0.0),
                z_encode(ro.gz_partition_rew, z_rew, 0.0),
            )
        define(seq, Name=f"{NAME}_catalyst", PhaseCycle=k)
        return seq

    def cycle(k: int) -> pp.Sequence:
        """Design every view of phase cycle ``k``: one balanced repetition each."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        increment = increments[k]
        for n, view in enumerate(views):
            phase = (n + 1) * increment % (2 * np.pi)
            line, partition = view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            ro.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            label_events = labels(
                LIN=line, PAR=partition, SET=k, IMA=view in calibration
            )
            seq.add_block(ro.rf, *([] if ro_gz is None else [ro_gz]))
            seq.add_block(
                ro.gx,
                ro.adc,
                pp.scale_grad(ro.gy_pre, ky),
                z_encode(ro.gz_partition, z_pre, kz),
                *label_events,
            )
            seq.add_block(
                ro.gx_rew,
                pp.scale_grad(ro.gy_rew, ky),
                z_encode(ro.gz_partition_rew, z_rew, kz),
            )
        define(
            seq,
            Name=NAME,
            PhaseCycle=k,
            PhaseIncrement=float(np.rad2deg(increment)),
        )
        seq.set_definition(
            key="NumGainCalibrationReadouts",
            value=sequences.gain_calibration_readouts(seq),
        )
        return seq

    # Each cycle's file follows its catalyst's, and the last cycle is the main
    # sequence.
    def segmented() -> pp.Sequence:
        """Design every segment of a prospective scan: trigger, preparations, ramp and views."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        segments = segment_views(views, n_y, n_z, views_per_segment)
        voxel = min(f / n for f, n in zip(fov, matrix, strict=True))
        trigger = pp.make_trigger(
            trigger_channel, duration=trigger_delay, system=system
        )
        inversion = (
            sequences.InversionPreparation(
                system,
                voxel_size_m=voxel,
                thickness_m=inversion_thickness,
                selection_axis=inversion_axis,
                offset_m=inversion_offset,
            )
            if inversion_time > 0
            else None
        )
        after = [
            module
            for module in (
                t2_preparation(system, t2_prep_te, voxel) if t2_prep_te > 0 else None,
                sequences.FatSaturation(system, voxel_size_m=voxel)
                if fat_saturation
                else None,
            )
            if module is not None
        ]
        excite = rf_reference(ro.rf)
        recovery = None
        if inversion is not None:
            raster = system.block_duration_raster
            wait = (
                inversion_time
                - (inversion.duration - inversion.center)
                - sum(module.duration for module in after)
                - n_ramp * ro.tr
                - excite
            )
            wait = raster * round(wait / raster)
            if wait < 1e-3:
                raise ValueError(
                    f"inversion_time must leave at least 1 ms of recovery after "
                    f"the inversion, the preparations and the ramp, got "
                    f"{inversion_time * 1e3:.0f} ms"
                )
            # A block is held to the raster above its duration, and a
            # multiple of the raster in floating point can sit just past it.
            recovery = pp.make_delay(wait - 1e-9)
        # Neighbouring central lines: the navigator images the scan's field of
        # view at low resolution, with nothing folded into it.
        nav_lines = [
            min(max(n_y // 2 + j - n_ramp // 2, 0), n_y - 1) for j in range(n_ramp)
        ]
        nav_on = pp.make_label(type="SET", label="NAV", value=1)
        nav_off = pp.make_label(type="SET", label="NAV", value=0)

        def add_module(module, pending: list) -> list:
            for block in module.blocks:
                seq.add_block(*block, *pending)
                pending = []
            return pending

        def excitation(n: int, amplitude: float, pending: list) -> None:
            phase = np.pi * ((n + 1) % 2)
            ro.rf.amplitude = amplitude
            ro.rf.phase_offset = phase
            ro.adc.phase_offset = phase
            seq.add_block(ro.rf, *([] if ro_gz is None else [ro_gz]), *pending)
            ro.rf.amplitude = nominal

        def encode(view, acquire: list) -> tuple[float, float]:
            line, partition = view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            seq.add_block(
                ro.gx,
                *acquire,
                pp.scale_grad(ro.gy_pre, ky),
                z_encode(ro.gz_partition, z_pre, kz),
            )
            seq.add_block(
                ro.gx_rew,
                pp.scale_grad(ro.gy_rew, ky),
                z_encode(ro.gz_partition_rew, z_rew, kz),
            )
            return ky, kz

        segment_duration = 0.0
        for s, segment in enumerate(segments):
            start = seq.duration()[0]
            pending = [trigger]
            if trigger_delay > 0:
                seq.add_block(trigger)
                pending = []
            if inversion is not None:
                pending = add_module(inversion, pending)
                seq.add_block(recovery)
            for module in after:
                pending = add_module(module, pending)
            for j in range(n_ramp):
                excitation(j, nominal * (j + 1) / n_ramp, pending)
                pending = []
                if navigator:
                    view = (nav_lines[j], n_z // 2)
                    events = [
                        ro.adc,
                        *labels(LIN=view[0], PAR=view[1], SEG=s),
                        nav_on,
                    ]
                else:
                    view, events = (n_y // 2, n_z // 2), []
                encode(view, events)
            for v, view in enumerate(segment):
                excitation(n_ramp + v, nominal, [])
                label_events = labels(
                    LIN=view[0], PAR=view[1], SEG=s, IMA=view in calibration
                )
                closing = [nav_off] if v == 0 and navigator else []
                encode(view, [ro.adc, *label_events, *closing])
            segment_duration = max(segment_duration, seq.duration()[0] - start)
        if segment_duration > trigger_period + 1e-9:
            raise ValueError(
                f"a segment takes {segment_duration * 1e3:.0f} ms, longer than the "
                f"{trigger_period * 1e3:.0f} ms trigger period"
            )
        define(
            seq,
            Name=NAME,
            PhaseCycle=0,
            PhaseIncrement=180.0,
            Gating=gating,
            TriggerChannel=trigger_channel,
            TriggerDelay=trigger_delay,
            TriggerPeriod=trigger_period,
            ViewsPerSegment=views_per_segment,
            NumSegments=len(segments),
            SegmentDuration=segment_duration,
            CatalystRamp=n_ramp,
            Navigator=int(navigator),
            InversionTime=inversion_time,
            T2PrepTE=t2_prep_te,
            FatSaturation=int(fat_saturation),
        )
        seq.set_definition(
            key="NumGainCalibrationReadouts",
            value=sequences.gain_calibration_readouts(seq),
        )
        return seq

    if gating == "prospective":
        return [segmented()]

    chain = []
    for k in range(len(increments)):
        chain += [catalyst(k), cycle(k)]
    return chain


main = bssfp3d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="bssfp_3d.seq"))
