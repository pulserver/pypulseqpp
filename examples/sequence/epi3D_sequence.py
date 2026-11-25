"""3D gradient-echo EPI with skipped-CAIPI sampling."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


def caipi_shift(ry: int, rz: int) -> int:
    """Return the CAIPI shift whose lattice keeps its aliases furthest apart.

    The sampled lattice is spanned by ``(0, rz)`` and ``(ry, shift)``, in
    lines and partitions. Its aliases form the dual lattice, which in two
    dimensions is the same lattice turned and scaled, so the shift that
    lengthens the sampling vectors also spreads the aliases most. Shifts are
    compared on the lengths of their two shortest primitive vectors, so a tie
    on the shortest is settled by the next, and then by the smaller shift. The rule
    reproduces the patterns Stirnberg and Stöcker (Magn Reson Med 2021,
    doi:10.1002/mrm.28486) found best, such as 2x2z1, 3x2z1, 2x4z2 and 1x6z2.

    Parameters
    ----------
    ry : int
        Acceleration along the phase-encode axis.
    rz : int
        Acceleration along the partition axis, which is also the shifts to
        choose among.

    Returns
    -------
    int
        The shift, in partitions per ``ry`` lines.
    """

    def shortest(shift: int) -> list[int]:
        # One of each pair of opposite vectors, b > 0 or b == 0 < a, and no
        # multiple of a shorter one.
        return sorted(
            (b * ry) ** 2 + (a * rz + b * shift) ** 2
            for a in range(-rz, rz + 1)
            for b in range(rz + 1)
            if (b or a > 0) and math.gcd(a, b) == 1
        )[:2]

    return max(range(rz), key=lambda shift: (shortest(shift), -shift))


def train_lines(
    n: int, ry: int, n_shots: int, partial_fourier: float
) -> tuple[int, int]:
    """Return the first line of shot 0 and the lines per shot.

    Shot ``s`` reads lines ``start + (s + i * n_shots) * ry``. The lattice
    keeps every line ``i`` with ``(i - n // 2) % ry == 0``, so the centre line
    is always read. The shots end on the lattice's last line and cover the
    lines from ``n - round(partial_fourier * n)`` up; when the shots cannot
    share them evenly, the extra lines extend below that, or are dropped where
    there is no room.

    Parameters
    ----------
    n : int
        Phase-encode lines across the full field of view.
    ry : int
        Acceleration along the phase-encode axis.
    n_shots : int
        Shots the train is segmented into.
    partial_fourier : float
        Fraction of k-space acquired, from the far side of the centre.

    Returns
    -------
    start : int
        The first line of shot 0.
    etl : int
        Lines each shot reads.

    Raises
    ------
    ValueError
        If the lines cannot be shared among ``n_shots`` shots.
    """
    first = n - round(partial_fourier * n)
    lattice = [i for i in range(n) if (i - n // 2) % ry == 0]
    count = sum(1 for i in lattice if i >= first)
    etl = -(-count // n_shots)
    if lattice[-1] - (etl * n_shots - 1) * ry < 0:
        etl = count // n_shots
    if etl < 1:
        raise ValueError(
            f"{count} lines cannot be shared among {n_shots} shots; lower n_shots"
        )
    return lattice[-1] - (etl * n_shots - 1) * ry, etl


def shell_bases(n: int, rz: int, partial_fourier: float) -> list[int]:
    """Return the first partition of every shell, the centre one first.

    Shells of ``rz`` partitions end on the last partition and cover those from
    ``n - round(partial_fourier * n)`` up; a shell that would start below
    partition zero is dropped. They are played centre-out.

    Parameters
    ----------
    n : int
        Partitions across the full field of view.
    rz : int
        Partitions one shell spans, which is the partition acceleration.
    partial_fourier : float
        Fraction of k-space acquired, from the far side of the centre.

    Returns
    -------
    list of int
        The first partition of each shell, centre-out.

    Raises
    ------
    ValueError
        If the partitions cannot hold one shell.
    """
    kept = round(partial_fourier * n)
    count = -(-kept // rz)
    if n - count * rz < 0:
        count = kept // rz
    if count < 1:
        raise ValueError(f"{kept} partitions cannot hold one shell of {rz}")
    bases = [n - (m + 1) * rz for m in range(count)]
    return sorted(bases, key=lambda b: (abs(b + (rz - 1) / 2 - n // 2), b))


NAME = "epi_3d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the slab-selective pulse. A time-bandwidth product of 12
#: keeps the slab profile's transition within the encoded margin of
#: ``SLAB_OVERSAMPLING``.
PULSE_DURATION = 4e-3
TIME_BW_PRODUCT = 12.0
#: Thickness of the excited slab, as a fraction of ``fov_z``.
SLAB_COVERAGE = 0.9
#: Duration of the nonselective hard pulse (s).
HARD_PULSE_DURATION = 0.5e-3
#: Fat methylene shift from water (ppm), converted against ``system.B0``
#: when the spectral-spatial pulse is built.
FAT_SHIFT_PPM = -3.4
#: Centre lines read without blips at the start of every shot when
#: ``navigator_lines`` asks for a navigator.
NAVIGATOR_LINES = 3
#: Counter holding each train line's position in its train, from zero, which
#: matches an imaging line to the reference line of the same echo.
ECHO_COUNTER = "USER0"
#: Dephasing left on the readout axis at the end of each shot, in cycles
#: across one voxel.
SPOILING_CYCLES = 4.0
#: Quadratic RF-spoiling increment of every excitation (degrees). The
#: readout dephasing alone leaves the slab in a gradient-echo steady state
#: whose coherent pathways brighten CSF over a spoiled gradient echo.
RF_SPOILING_INCREMENT_DEG = 117.0
#: Digital output marking every volume under ``volume_output``, and how
#: long it lasts (s); it never outlasts the excitation block it rides.
OUTPUT_CHANNEL = "ext1"
OUTPUT_DURATION = 1e-3
#: Flip angle of the gradient-echo calibration (degrees), whatever the
#: imaging flip. Its views follow one another at its shortest TR, from
#: relaxed magnetization, so a large imaging flip would leave the centre of
#: k-space, read halfway through, at a small steady-state signal, and each
#: view's RF alone can exceed the SAR limit. Ten degrees, near the Ernst angle
#: of that TR, plays a fraction of the imaging pulse's RF power. The
#: calibration measures coil sensitivities, which do not depend on the
#: contrast.
CALIBRATION_FLIP_DEG = 10.0
#: Calibrations the scan opens with.
CALIBRATIONS = ("fleet", "gre")
#: Non-acquiring shots that open the FLEET calibration, so its first
#: acquiring shot does not read relaxed magnetization.
FLEET_DUMMY_SHOTS = 1

#: ``SET`` labels of the full-k phase reference volume: forward, reversed.
FULL_K_REFERENCE_SETS = (4, 5)

#: ``SET`` labels of the field pair: forward, reversed.
FIELD_PAIR_SETS = (2, 3)


def fleet_lines(n: int, extent: int, shots: int) -> range:
    """Return the lines of a FLEET calibration: ``extent`` about the centre, rounded up to whole shots.

    Each shot reads at least two lines, so the shot holding the centre line
    echoes it within its train.

    Parameters
    ----------
    n : int
        Phase-encode lines across the full field of view.
    extent : int
        Lines asked of the calibration.
    shots : int
        Interleaved shots that share them.

    Returns
    -------
    range
        Consecutive lines, as many as ``shots`` share evenly, at least two
        each, and ``n`` holds.
    """
    count = min(max(-(-extent // shots), 2) * shots, n - n % shots)
    low = min(max(n // 2 - count // 2, 0), n - count)
    return range(low, low + count)


def fleet_train(
    system: pp.Opts,
    rf,
    gz,
    epi,
    lines: range,
    shots: int,
    **arguments,
):
    """Return a FLEET calibration train over ``lines``: ``epi``'s read lobe and blip window, at its shortest echo time.

    Shot ``s`` reads ``lines[s::shots]``, a step of ``shots`` lines, and is
    delayed by ``esp / shots``, so the echo time grows across the lines as it
    does across the lines of the imaging and the calibration is distorted as
    the imaging is. The echo time is the shortest the train admits: the
    phase an off-resonance accrues by the imaging echo time at 7T wraps
    within one calibration voxel near air-tissue interfaces, and coil maps
    fitted to that calibration take on its structure. ``arguments`` are those
    of the imaging train: matrix, field of view, navigator, sampling and
    spoiling.

    Raises
    ------
    ValueError
        If its echo spacing is not ``epi``'s.
    """
    n_y = arguments["matrix"][1]
    common = dict(
        arguments,
        order=np.column_stack(
            (np.arange(len(lines) // shots) * shots, np.zeros(len(lines) // shots))
        ),
        te_line=(n_y // 2 - lines[0]) / shots,
        echo_shifts=shots,
        blip_span=epi.blip_span,
    )
    train = sequences.EpiReadout3D(system, rf, gz, **dict(common, te=None))
    if abs(train.esp - epi.esp) > 1e-9:
        raise ValueError(
            f"the FLEET calibration's echo spacing {train.esp * 1e3:.3f} ms is not "
            f"the imaging's {epi.esp * 1e3:.3f} ms"
        )
    return train


def epi3d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 220e-3,
    fov_y: float = 220e-3,
    fov_z: float = 96e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_z: int = 32,
    flip_angle_deg: float = 20.0,
    te: float | None = None,
    tr: float | None = None,
    n_frames: int = 1,
    readout_bandwidth_hz: float = 500e3,
    ry: int = 1,
    rz: int = 1,
    partial_fourier_y: float = 1.0,
    partial_fourier_z: float = 1.0,
    n_shots: int = 1,
    n_dummy: int | None = None,
    excitation: str = "slab",
    readout_oversampling: float = 1.0,
    z_oversampling: float = 0.125,
    n_acs_y: int = 32,
    n_acs_z: int = 32,
    calibration: str = "fleet",
    volume_output: bool = False,
    phase_reference: bool = False,
    navigator_lines: int | None = None,
    field_pair: bool = True,
    field_pair_tr: float | None = None,
    full_k_reference: bool = True,
) -> list[pp.Sequence]:
    """3D gradient-echo EPI: one train per ``(shot, shell)``, skipped-CAIPI sampled.

    The views sampled form a CAIPIRINHA lattice holding the centre of
    k-space: line ``y`` is read when ``(y - n_y // 2) % ry == 0``, at the
    partitions ``z`` with ``(z - n_z // 2 - d * j) % rz == 0``, ``j`` being the
    line's lattice index and ``d`` the shift :func:`caipi_shift` picks. A
    shell is ``rz`` consecutive partitions, and each of its lines holds one
    lattice partition. Shot ``s`` of a shell reads every ``n_shots``-th
    lattice line from the ``s``-th, blipping to that line's partition
    (skipped-CAIPI, Stirnberg and Stöcker, Magn Reson Med 2021,
    doi:10.1002/mrm.28486); one shot per shell is blipped-CAIPI. A volume is
    every shot of every shell, shots outer and shells centre-out inner.

    With ``phase_reference``, the last dummy shot (the FLEET calibration's
    dummy shot) is a reference train for odd/even phase correction: the
    shot's own train with every phase- and partition-encode prewinder, blip
    and rewinder at zero, its lines acquired (``PHASECORR``). Every train line
    carries its position in the train in ``ECHO_COUNTER``, which matches an
    imaging line to the reference line of the same echo. A scan without dummy
    shots plays one for the reference. Each shot can also read
    ``navigator_lines`` centre lines without blips (``NAV``) before its
    train, after one unsampled lobe that lets the read axis settle. Every line
    carries ``REV`` for its read polarity, and shots are delayed by successive fractions of the echo
    spacing so the echo time grows smoothly across the lines; ``te`` is the
    echo time of the centre line. Acquisitions carry ``LIN``, ``PAR``,
    ``SEG`` (the shot) and ``REP``.

    The ``calibration`` of the central ``n_acs_y x n_acs_z`` rectangle
    (``REF``) at ``CALIBRATION_FLIP_DEG`` is linked ahead of the imaging; the
    coil sensitivities of the volume are estimated from it once. The FLEET
    calibration (Polimeni et al., Magn Reson Med 2016, doi:10.1002/mrm.25628)
    reads the lines of each partition in ``n_shots * ry`` interleaved EPI
    shots, which step as far per echo spacing as an imaging shot, with its
    read lobe, navigator and blip, so the calibration is distorted as the
    imaging is, at the shortest echo time they admit. A partition's shots follow one another, forward
    (``SET = 0``) and then reversed (``SET = 1``), after
    ``FLEET_DUMMY_SHOTS`` unsampled ones that open the calibration, and the
    two polarities give the field that corrects the distortion. The
    gradient-echo calibration reads one view per excitation, undistorted;
    after it, a ``reference`` volume with the phase encode reversed
    (``SET = 1``) is linked for distortion correction.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x : float, default=0.22
        Field of view along the readout (m).
    fov_y : float, default=0.22
        Field of view along the phase encode (m).
    fov_z : float, default=0.096
        Field of view along the partition encode (m). The slab excited is
        ``SLAB_COVERAGE`` times ``fov_z`` thick, and ``z_oversampling``
        encodes partitions beyond ``fov_z``; ``Matrix`` and ``FOV`` state the
        prescription without them.
    n_x : int, default=128
        Matrix size along the readout.
    n_y : int, default=128
        Matrix size along the phase encode.
    n_z : int, default=32
        Matrix size along the partition encode.
    flip_angle_deg : float, default=20.0
        Excitation flip angle (degrees) of the imaging and the reference; the
        calibration plays ``CALIBRATION_FLIP_DEG``.
    te : float | None, default=None
        Echo time of the centre line (s). ``None`` is as short as the
        train, and its navigator lines when there are any, admit.
    tr : float | None, default=None
        Volume repetition time (s): every shot of every shell. ``None`` is
        as short as possible.
    n_frames : int, default=1
        Volumes in the time series, each carrying its ``REP`` counter.
    readout_bandwidth_hz : float, default=500000.0
        Requested receiver bandwidth (Hz).
    ry, rz : int, default=1
        Undersampling along the phase and the partition encode; ``rz`` is
        also the shell height.
    partial_fourier_y, partial_fourier_z : float, default=1.0
        Fraction of the phase- and partition-encode extent read, in
        ``[0.5, 1]``. Truncates the lines and the shells before the centre.
    n_shots : int, default=1
        Interleaved shots each shell's lines are split into.
    n_dummy : int | None, default=None
        Non-acquiring volumes before a time series; with one frame,
        non-acquiring shots.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
        A slab-selective SLR pulse, a hard pulse, or a slab- and
        water-selective spectral-spatial pulse.
    z_oversampling : float, default=0.125
        Partitions encoded beyond ``n_z``, as a fraction of it, half on each
        side and rounded up to whole partitions. The slab profile's edges, and
        tissue past them, are encoded in this margin instead of wrapping onto
        the opposite edge, whose aliases under ``rz`` fall on the centre
        partitions; the reconstruction crops it.
    readout_oversampling : float, default=1.0
        Minimum EPI readout oversampling factor, at least one. Additional
        samples preserve the requested dwell under the gradient limits.
    n_acs_y : int, default=32
        Extent of the calibration rectangle along the phase encode; FLEET
        rounds it up to a whole number of lines per shot.
    n_acs_z : int, default=32
        Extent of the calibration rectangle along the partition encode.
    calibration : {'fleet', 'gre'}, default='fleet'
        The calibration: FLEET, segmented EPI at
        the imaging train's echo spacing, or a gradient echo per view.
    volume_output : bool, default=False
        Play a digital output on ``OUTPUT_CHANNEL`` at the first
        excitation of every volume, dummy volumes included.
    phase_reference : bool, default=False
        Acquire a blip-free reference train in a dummy shot, for odd/even
        phase correction.
    navigator_lines : int | None, default=None
        Centre lines read without blips at the start of every shot, for
        odd/even phase correction. ``None`` is ``NAVIGATOR_LINES`` without
        ``phase_reference`` and none with it.
    field_pair : bool, default=True
        Play two volumes of the scan before it, at its echo time, flip angle
        and echo spacing, every line of a partial-Fourier scan read: one with
        the phase encode forward, labelled ``SET = 2``, and one reversed,
        labelled ``SET = 3``. The two are
        displaced in opposite directions by the susceptibility field along
        the phase encode, and lose and pile up signal where the scan does, so
        the reconstruction estimates the displacement field of the scan from
        the intensity the scan has.
    field_pair_tr : float or None, default=None
        Volume repetition time (s) of the field pair; ``None`` plays it at
        the scan's ``tr``.
    full_k_reference : bool, default=True
        Play one fully sampled volume at the echo time of a partial-Fourier
        scan, labelled ``SET = 4``, from whose phase the reconstruction
        completes the unsampled side of k-space. Without a prescribed
        ``te``, the scan takes the shortest echo time this volume admits.

    Returns
    -------
    list of pypulseqpp.Sequence
        The designed sequences in play order: the ``calibration``, the
        ``reference`` after a gradient-echo calibration, the forward and the
        reversed volume of the ``field_pair``, the ``full_k_reference``, then
        the scan, which is the main sequence. ``sequences.write`` writes them
        as files linked through ``NextSequence``.

    Raises
    ------
    ValueError
        If ``excitation`` or ``calibration`` is unknown, a partial Fourier
        fraction is outside
        ``[0.5, 1]``, a count is below one, the shots cannot share the lines
        or the partitions hold no shell, the TE or TR is shorter than
        the shots take, or the TE is shorter than the ``full_k_reference``
        admits.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> *prescans, seq = sequences.epi3D_sequence(n_x=32, n_y=16, n_z=4, n_dummy=0)
    >>> [prescan.definitions["Name"] for prescan in prescans]
    ['epi_3d_calibration', 'epi_3d_field_forward', 'epi_3d_field_reversed']
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 4.0], 'epi_3d')
    """
    if z_oversampling < 0:
        raise ValueError(f"z_oversampling must be at least 0, got {z_oversampling}")
    margin = math.ceil(n_z * z_oversampling / 2)
    protocol = {
        "fov_x": fov_x,
        "fov_y": fov_y,
        "fov_z": fov_z * (n_z + 2 * margin) / n_z,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z + 2 * margin,
        "prescribed_z": (n_z, fov_z),
        "flip_angle_deg": flip_angle_deg,
        "te": te,
        "tr": tr,
        "n_frames": n_frames,
        "readout_bandwidth_hz": readout_bandwidth_hz,
        "ry": ry,
        "rz": rz,
        "partial_fourier_y": partial_fourier_y,
        "partial_fourier_z": partial_fourier_z,
        "n_shots": n_shots,
        "n_dummy": n_dummy,
        "excitation": excitation,
        "readout_oversampling": readout_oversampling,
        "n_acs_y": n_acs_y,
        "n_acs_z": n_acs_z,
        "calibration": calibration,
        "volume_output": volume_output,
        "phase_reference": phase_reference,
        "navigator_lines": navigator_lines,
    }
    if not (full_k_reference and min(partial_fourier_y, partial_fourier_z) < 1.0):
        *prescans, seq = _designed(system, **protocol, encode_sets=(0, 1))
        pair = (
            _field_pair(system, protocol, protocol["te"], field_pair_tr)
            if field_pair
            else []
        )
        return [*prescans, *pair, seq]
    try:
        *_, reference = _designed(
            system,
            **{
                **protocol,
                "partial_fourier_y": 1.0,
                "partial_fourier_z": 1.0,
                "n_frames": 1,
                "volume_output": False,
            },
            encode_sets=FULL_K_REFERENCE_SETS,
        )
    except ValueError as error:
        raise ValueError(
            f"the full-k phase reference cannot be designed: {error}"
        ) from error
    reference.set_definition(key="Name", value=f"{NAME}_full_k_reference")
    *prescans, seq = _designed(
        system,
        **{**protocol, "te": reference.definitions["TE"][0]},
        encode_sets=(0, 1),
    )
    full_k = {**protocol, "partial_fourier_y": 1.0, "partial_fourier_z": 1.0}
    pair = (
        _field_pair(system, full_k, reference.definitions["TE"][0], field_pair_tr)
        if field_pair
        else []
    )
    return [*prescans, *pair, reference, seq]


def _add_phased(seq, events, phase):
    """Add a block of ``events`` to ``seq``, its RF pulses and ADCs advanced by ``phase`` (rad)."""
    phased = [
        event for event in events if getattr(event, "type", None) in ("rf", "adc")
    ]
    offsets = [event.phase_offset for event in phased]
    for event in phased:
        event.phase_offset += phase
    try:
        seq.add_block(*events)
    finally:
        for event, offset in zip(phased, offsets, strict=True):
            event.phase_offset = offset


def _field_pair(
    system: pp.Opts | None, protocol: dict, te: float | None, tr: float | None
) -> list[pp.Sequence]:
    """Design the field pair of :func:`epi3d`: one volume of the scan at echo time ``te``, forward then reversed, at ``tr``."""
    *_, reversed_volume, forward_volume = _designed(
        system,
        **{
            **protocol,
            "te": te,
            "tr": protocol["tr"] if tr is None else tr,
            "n_frames": 1,
            "calibration": "gre",
            "volume_output": False,
        },
        encode_sets=FIELD_PAIR_SETS,
    )
    forward_volume.set_definition(key="Name", value=f"{NAME}_field_forward")
    reversed_volume.set_definition(key="Name", value=f"{NAME}_field_reversed")
    return [forward_volume, reversed_volume]


def _designed(
    system: pp.Opts | None,
    *,
    fov_x,
    fov_y,
    fov_z,
    n_x,
    n_y,
    n_z,
    prescribed_z,
    flip_angle_deg,
    te,
    tr,
    n_frames,
    readout_bandwidth_hz,
    ry,
    rz,
    partial_fourier_y,
    partial_fourier_z,
    n_shots,
    n_dummy,
    excitation,
    readout_oversampling,
    n_acs_y,
    n_acs_z,
    calibration,
    volume_output,
    phase_reference,
    navigator_lines,
    encode_sets: tuple[int, int],
) -> list[pp.Sequence]:
    """Design :func:`epi3d`, its forward phase encodes labelled ``SET = encode_sets[0]`` and its reversed ones ``SET = encode_sets[1]``."""
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if navigator_lines is None:
        navigator_lines = 0 if phase_reference else NAVIGATOR_LINES
    if excitation not in sequences.EXCITATIONS:
        raise ValueError(
            f"excitation must be one of {sequences.EXCITATIONS}, got {excitation!r}"
        )
    if calibration not in CALIBRATIONS:
        raise ValueError(
            f"calibration must be one of {CALIBRATIONS}, got {calibration!r}"
        )
    for name, fraction in (
        ("partial_fourier_y", partial_fourier_y),
        ("partial_fourier_z", partial_fourier_z),
    ):
        if not 0.5 <= fraction <= 1.0:
            raise ValueError(f"{name} must lie in [0.5, 1], got {fraction}")
    for name, count in (
        ("ry", ry),
        ("rz", rz),
        ("n_shots", n_shots),
        ("n_frames", n_frames),
    ):
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")

    fov = (fov_x, fov_y, fov_z)
    matrix = (n_x, n_y, n_z)
    stated_fov = (fov_x, fov_y, prescribed_z[1])
    stated_matrix = (n_x, n_y, prescribed_z[0])
    raster = system.block_duration_raster
    exc = sequences.make_excitation(
        system,
        excitation,
        flip_angle_deg,
        SLAB_COVERAGE * prescribed_z[1],
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        hard_duration_s=HARD_PULSE_DURATION,
        fat_shift_ppm=FAT_SHIFT_PPM,
    )
    gz = getattr(exc, "gz", None)

    # Shot s reads lattice lines s, s + n_shots, ...; the centre line is the
    # c-th, echoed at the (c / n_shots)-th line of the unshifted train.
    start, etl = train_lines(n_y, ry, n_shots, partial_fourier_y)
    origins = [start + s * ry for s in range(n_shots)]
    shells = shell_bases(n_z, rz, partial_fourier_z)
    shift = caipi_shift(ry, rz)
    steps = np.arange(etl) * n_shots * ry
    # Every shell starts at the same residue modulo rz, so a shot's offsets
    # within its shell depend on the shot alone.
    trains = {}
    shot_trains = []
    for s in range(n_shots):
        lines = start + (s + np.arange(etl) * n_shots) * ry
        lattice = n_z // 2 + shift * ((lines - n_y // 2) // ry)
        offsets = (lattice - shells[0]) % rz
        key = tuple(offsets)
        if key not in trains:
            trains[key] = sequences.EpiReadout3D(
                system,
                exc.rf,
                gz,
                order=np.column_stack((steps, offsets)),
                te=te,
                te_line=(n_y // 2 - start) // ry / n_shots,
                navigator_lines=navigator_lines,
                echo_shifts=n_shots,
                fov=fov,
                matrix=matrix,
                oversampling=readout_oversampling,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=SPOILING_CYCLES,
                labels=("LIN", "PAR"),
            )
        shot_trains.append(trains[key])
    epi = shot_trains[0]
    echo_time = epi.echo_time

    # A shot is the train and a closing delay of at least one raster.
    volume = [(s, b) for s in range(n_shots) for b in shells]
    shortest_shot = epi.duration + raster
    pad = raster
    if tr is not None:
        per_shot = tr / len(volume)
        if per_shot < shortest_shot - 1e-9:
            raise ValueError(
                f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
                f"{len(volume) * shortest_shot * 1e3:.3f} ms the "
                f"{len(volume)} shots of a volume take"
            )
        pad += pp.round_to_raster(per_shot - shortest_shot, raster)
    shot_duration = shortest_shot - raster + pad
    repetition_time = len(volume) * shot_duration
    if n_dummy is None:
        # Every shot excites the slab; a volume holds len(volume) shots.
        excitations = sequences.steady_state_dummies(shot_duration, flip_angle_deg)
        n_dummy = -(-excitations // len(volume)) if n_frames > 1 else excitations
    n_dummy_shots = n_dummy * (len(volume) if n_frames > 1 else 1)
    if phase_reference:
        # The reference plays in the last dummy shot, so there is one.
        n_dummy_shots = max(n_dummy_shots, 1)
    dummies = [volume[i % len(volume)] for i in range(n_dummy_shots)]
    output_pulse = pp.make_digital_output_pulse(
        OUTPUT_CHANNEL,
        duration=min(
            OUTPUT_DURATION,
            pp.calc_duration(exc.rf, *([] if gz is None else [gz])),
        ),
        system=system,
    )

    # The calibration: a gradient echo per view, or FLEET's interleaved shots
    # per partition.
    acs_shots = n_shots * ry
    low_z = n_z // 2 - n_acs_z // 2
    partitions = list(range(max(low_z, 0), min(low_z + n_acs_z, n_z)))
    lines_y = (
        fleet_lines(n_y, n_acs_y, acs_shots)
        if calibration == "fleet"
        else range(
            max(n_y // 2 - n_acs_y // 2, 0), min(n_y // 2 - n_acs_y // 2 + n_acs_y, n_y)
        )
    )
    views = [(y, z) for y in lines_y for z in partitions]
    gre = acs = None
    if views:
        # The imaging pulse at the calibration's flip angle and at its
        # duration, so both play one waveform, scaled.
        held = getattr(exc, "pulse_duration", None)
        weak = sequences.make_excitation(
            system,
            excitation,
            CALIBRATION_FLIP_DEG,
            SLAB_COVERAGE * prescribed_z[1],
            duration_s=held or PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
            hard_duration_s=held or HARD_PULSE_DURATION,
            fat_shift_ppm=FAT_SHIFT_PPM,
        )
        weak_gz = getattr(weak, "gz", None)
        if calibration == "gre":
            gre = sequences.LineReadout3D(
                system,
                weak.rf,
                weak_gz,
                fov=fov,
                matrix=matrix,
                oversampling=readout_oversampling,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=SPOILING_CYCLES,
                labels=("LIN", "PAR"),
            )
        else:
            acs = fleet_train(
                system,
                weak.rf,
                weak_gz,
                epi,
                lines_y,
                acs_shots,
                navigator_lines=navigator_lines,
                fov=fov,
                matrix=matrix,
                oversampling=readout_oversampling,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=SPOILING_CYCLES,
                labels=("LIN", "PAR"),
            )

    def spoiling(count: int) -> list[float]:
        """Return the RF-spoiling phase of each of ``count`` excitations (rad)."""
        return list(
            pp.make_rf_spoiling_schedule(
                count, increment=np.deg2rad(RF_SPOILING_INCREMENT_DEG)
            )
        )

    def define(seq: pp.Sequence, **definitions) -> None:
        for key, value in {
            "FOV": list(stated_fov),
            "Matrix": list(stated_matrix),
            **definitions,
        }.items():
            seq.set_definition(key=key, value=value)
        seq.set_definition(
            key="NumGainCalibrationReadouts",
            value=sequences.gain_calibration_readouts(seq),
        )

    def calibration_kernel(
        seq: pp.Sequence,
        labels: sequences.Labels,
        view: tuple[int, int],
        phase: float,
    ) -> None:
        """Add one gradient echo at ``(line, partition)``, excited and read at RF phase ``phase``."""
        line, partition = view
        gre.rf.phase_offset = gre.adc.phase_offset = phase
        gre.adc_labels[0].value, gre.adc_labels[1].value = line, partition
        ky = (line - n_y // 2) / (n_y / 2)
        kz = (partition - n_z // 2) / (n_z / 2)
        seq.add_block(gre.rf, *([] if weak_gz is None else [weak_gz]), *labels(REF=1))
        wait_te = getattr(gre, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            gre.gx_pre, pp.scale_grad(gre.gy_pre, ky), pp.scale_grad(gre.gz_pre, kz)
        )
        seq.add_block(gre.gx, gre.adc, *gre.adc_labels)
        seq.add_block(
            gre.gx_spoil, pp.scale_grad(gre.gy_rew, ky), pp.scale_grad(gre.gz_rew, kz)
        )

    def calibrate() -> pp.Sequence:
        """Design the gradient-echo calibration over the central rectangle."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        for view, phase in zip(views, spoiling(len(views)), strict=True):
            calibration_kernel(seq, labels, view, phase)
        define(
            seq,
            Name=f"{NAME}_calibration",
            kSpaceCenterLine=n_y // 2,
            kSpaceCenterPartition=n_z // 2,
        )
        return seq

    def fleet() -> pp.Sequence:
        """Design the FLEET calibration: every shot of a partition, then the next."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        starts = list(lines_y[:acs_shots])
        phases = iter(spoiling(FLEET_DUMMY_SHOTS + 2 * len(partitions) * len(starts)))
        for dummy in range(FLEET_DUMMY_SHOTS):
            reference = phase_reference and dummy == FLEET_DUMMY_SHOTS - 1
            shot_blocks(
                seq,
                labels,
                acs,
                starts[0],
                n_z // 2,
                0,
                labels(SEG=0, SET=0, PHASECORR=1) if reference else [],
                raster,
                reference=reference,
                phase=next(phases),
            )
        for partition in partitions:
            for polarity in (0, 1):
                for shot, line in enumerate(starts):
                    shot_blocks(
                        seq,
                        labels,
                        acs,
                        line,
                        partition,
                        shot,
                        labels(
                            REF=1,
                            SEG=shot,
                            SET=polarity,
                            **({"PHASECORR": 0} if phase_reference else {}),
                        ),
                        raster,
                        acquire=True,
                        reversed_encode=bool(polarity),
                        phase=next(phases),
                    )
        define(
            seq,
            Name=f"{NAME}_calibration",
            TE=acs.echo_time,
            EchoSpacing=acs.esp,
            kSpaceCenterLine=n_y // 2,
            kSpaceCenterPartition=n_z // 2,
        )
        return seq

    def kernel(
        seq: pp.Sequence,
        labels: sequences.Labels,
        shot: int,
        base: int,
        frame: int | None,
        output: bool = False,
        reversed_encode: bool = False,
        reference: bool = False,
        phase: float = 0.0,
    ) -> None:
        """Add one shot of the shell starting at partition ``base``, at RF phase ``phase``; ``frame=None`` is a dummy, a ``reference`` one the blip-free reference train."""
        train = shot_trains[shot]
        acquire = frame is not None
        if not acquire:
            flags = {"ONCE": 1}
            if reference:
                flags |= {"SEG": shot, "PHASECORR": 1}
                if encode_sets[reversed_encode]:
                    flags["SET"] = encode_sets[reversed_encode]
        else:
            flags = {"REP": frame, "SEG": shot, "ONCE": 0}
            if encode_sets[reversed_encode]:
                flags["SET"] = encode_sets[reversed_encode]
            if phase_reference:
                flags["PHASECORR"] = 0
            if not n_dummy_shots:
                del flags["ONCE"]
        shot_blocks(
            seq,
            labels,
            train,
            origins[shot],
            base + int(train.order[0, 1]),
            shot,
            [*labels(**flags), *([output_pulse] if output else [])],
            pad,
            acquire=acquire,
            reversed_encode=reversed_encode and not reference,
            reference=reference,
            phase=phase,
        )

    def shot_blocks(
        seq: pp.Sequence,
        labels: sequences.Labels,
        train,
        line: int,
        partition: int,
        shot: int,
        first_events: list,
        pad: float,
        acquire: bool = False,
        reversed_encode: bool = False,
        reference: bool = False,
        phase: float = 0.0,
    ) -> None:
        """Add one shot of ``train`` from ``(line, partition)``, ``first_events`` on its first block.

        The excitation and every readout of the shot play at RF phase
        ``phase``.
        ``shot`` sets the echo shift. ``reversed_encode`` negates every line
        encode, so each line L of the forward shot is read at its mirror,
        ``LIN = n_y - L``; the mirror of line 0 lies outside the matrix and is
        played unsampled. ``reference`` scales every phase- and
        partition-encode event to zero and acquires the train lines, its
        navigator unsampled: the blip-free reference train.
        """
        sign = -1.0 if reversed_encode else 1.0
        train.shot_labels[0].value = n_y - line if reversed_encode else line
        train.shot_labels[1].value = partition

        # The prewinders and the rewinders closing the shot are scaled alike.
        ky = sign * (line - n_y // 2) / (n_y / 2)
        kz = (partition - n_z // 2) / (n_z / 2)
        swap = {
            id(train.gy_pre): pp.scale_grad(train.gy_pre, ky),
            id(train.gy_rew): pp.scale_grad(train.gy_rew, ky),
            id(train.gz_pre): pp.scale_grad(train.gz_pre, kz),
            id(train.gz_rew): pp.scale_grad(train.gz_rew, kz),
        }
        if reversed_encode:
            swap.update(
                (id(blip), pp.scale_grad(blip, -1.0))
                for blip in train.gy_blips
                if blip is not None
            )
            swap.update(
                (
                    id(step[0]),
                    pp.make_label(type="INC", label="LIN", value=-step[0].value),
                )
                for step in train.line_labels
                if step
            )
        if reference:
            # The same events at zero amplitude, so the reference plays the
            # imaging shot's blocks.
            encodes = [train.gy_pre, train.gy_rew, train.gz_pre, train.gz_rew]
            swap.update(
                (id(event), pp.scale_grad(event, 0.0))
                for event in (*encodes, *train.gy_blips, *train.gz_blips)
                if event is not None
            )
        lines = line + train.order[:, 0]
        step = shot * train.echo_shift_step
        wait_shift = getattr(train, "wait_shift", None)
        if wait_shift is not None:
            swap[id(wait_shift)] = pp.make_delay(wait_shift.delay + step)
        wait_tr = getattr(train, "wait_tr", None)
        closing = pad + (0.0 if wait_tr is None else wait_tr.delay) - step

        navigator = {id(event) for event in getattr(train, "gx_navigator", ())}
        readouts = {id(event) for event in train.gx}
        polarity, read = 0, 0
        for index, block in enumerate(train.blocks):
            head = block[0]
            if head is wait_tr:
                continue
            reading = id(head) in readouts and id(head) not in navigator
            echo = read
            sampled = (acquire or (reference and reading)) and not (
                reversed_encode and reading and lines[read] == 0
            )
            read += reading
            events = [
                swap.get(id(event), event)
                for event in block
                if sampled or event.type != "adc"
            ]
            if index == 0:
                events += first_events
            elif id(head) in navigator or id(head) in readouts:
                if acquire or sampled:
                    flags = {"NAV": int(id(head) in navigator), "REV": polarity}
                    if reading and phase_reference:
                        flags[ECHO_COUNTER] = echo
                    events += labels(**flags)
                polarity ^= 1
            _add_phased(seq, events, phase)
        seq.add_block(pp.make_delay(closing))

    def play(frames, reversed_encode: bool = False) -> pp.Sequence:
        """Design the dummies and then ``frames``, shot by shot."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        shots = [(None, view) for view in dummies]
        shots += [(frame, view) for frame in frames for view in volume]
        phases = spoiling(len(shots))
        for index, ((frame, (shot, base)), phase) in enumerate(
            zip(shots, phases, strict=True)
        ):
            output = volume_output and not reversed_encode and (shot, base) == volume[0]
            # The last dummy shot plays the reference.
            reference = phase_reference and index == len(dummies) - 1
            kernel(
                seq,
                labels,
                shot,
                base,
                frame,
                output,
                reversed_encode,
                reference=reference,
                phase=phase,
            )
        return seq

    def reference() -> pp.Sequence:
        """Design one volume with the phase encode reversed, after its dummies."""
        seq = play(frames=[0], reversed_encode=True)
        define(
            seq,
            Name=f"{NAME}_reference",
            TE=echo_time,
            TR=repetition_time,
            EchoSpacing=epi.esp,
            kSpaceCenterLine=n_y // 2,
            kSpaceCenterPartition=n_z // 2,
        )
        return seq

    prescans = [calibrate()] if gre is not None else []
    if acs is not None:
        prescans.append(fleet())
    else:
        prescans.append(reference())
    seq = play(frames=range(n_frames))

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV (compat=False: the lines are sampled on the ramps).
    definitions = {
        "FOV": list(stated_fov),
        "Matrix": list(stated_matrix),
        "Name": NAME,
        "TE": echo_time,
        "TR": repetition_time,
        "EchoSpacing": epi.esp,
        "EPIFactor": epi.etl,
        "Excitation": excitation,
        "CaipiShift": shift,
        "kSpaceCenterLine": n_y // 2,
        "kSpaceCenterPartition": n_z // 2,
        "SliceThickness": stated_fov[2],
    }
    if navigator_lines:
        # The reconstruction scales each excitation's navigator phase by
        # TrainEchoTime / NavigatorEchoTime, the phase a field offset gains by
        # the echo time; the forward navigator lines are the even ones.
        definitions["NavigatorEchoTime"] = float(np.mean(epi.navigator_times[::2]))
        definitions["TrainEchoTime"] = echo_time
        definitions["ReconParameters"] = "NavigatorEchoTime TrainEchoTime"
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=sequences.gain_calibration_readouts(seq),
    )
    return [*prescans, seq]


main = epi3d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="epi_3d.seq"))
