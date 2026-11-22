"""Multi-slice 2D gradient-echo or spin-echo EPI, optionally segmented, multiband or diffusion-weighted."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: Slew-rate ceiling (T/m/s) of the fat saturation, whose spoilers play on three
#: axes at once: designed at the sequence's slew they would set its nerve
#: stimulation, and the slew search would slow every gradient for them.
PREPARATION_MAX_SLEW = 40.0


def _received_at(adc, phase: float, system: pp.Opts):
    """Return ``adc`` with its receiver phase advanced by ``phase`` rad."""
    return pp.make_adc(
        num_samples=adc.num_samples,
        system=system,
        dwell=adc.dwell,
        delay=adc.delay,
        freq_offset=adc.freq_offset,
        phase_offset=adc.phase_offset + phase,
    )


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


def packets_of(n: int, per_packet: int) -> list[list[int]]:
    """Deal ``n`` slices round-robin into packets, even slices first in each.

    Parameters
    ----------
    n : int
        Slices to deal.
    per_packet : int
        Most slices one packet may hold.

    Returns
    -------
    list of list of int
        The slices of each packet, in excitation order.
    """
    n_packets = -(-n // per_packet)
    packets = [range(start, n, n_packets) for start in range(n_packets)]
    return [[*packet[::2], *packet[1::2]] for packet in packets]


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
    gz_reph,
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
        order=np.arange(len(lines) // shots) * shots,
        te_line=(n_y // 2 - lines[0]) / shots,
        echo_shifts=shots,
        blip_span=epi.blip_span,
    )
    train = sequences.EpiReadout2D(system, rf, gz, gz_reph, **dict(common, te=None))
    if abs(train.esp - epi.esp) > 1e-9:
        raise ValueError(
            f"the FLEET calibration's echo spacing {train.esp * 1e3:.3f} ms is not "
            f"the imaging's {epi.esp * 1e3:.3f} ms"
        )
    return train


def spin_echo_timing(system, exc, refocus, make_train, te, b_value):
    """Return the timing of a spin-echo EPI shot and its diffusion lobes.

    The shot plays the excitation and its rephaser, the first diffusion lobe,
    a wait, the refocusing pulse between its crushers, the second lobe and
    the train. Each half of the echo time holds one lobe; the lobes are
    lengthened by ``DIFFUSION_LOBE_STEP`` at ``system.max_grad`` until they
    reach ``b_value``, and the echo time is the shortest the longer half
    admits unless ``te`` is given.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    exc : SpatialSelectiveExcitation
        The 90 degree excitation.
    refocus : SpatialSelectiveRefocusing
        The refocusing pulse and its crushers.
    make_train : callable
        Returns the train opened by the refocusing pulse at a given echo time
        from that pulse (s), ``None`` for the shortest.
    te : float | None
        Echo time (s); ``None`` is the shortest.
    b_value : float | None
        b-value (s/mm^2) of the lobes; ``None`` plays no lobes.

    Returns
    -------
    types.SimpleNamespace
        ``train``, ``echo_time`` (s), ``wait_half`` (the excitation half's
        wait, or None), ``lobes`` (the lobe at ``system.max_grad`` on x, y
        and z, or None), ``lobe_duration`` (s), ``b_value`` (s/mm^2, that of
        the lobes at full amplitude) and ``excitation_half`` (s, from the
        excitation's start to the refocusing block).

    Raises
    ------
    ValueError
        If ``te`` is shorter than either half admits, or the lobes cannot
        reach ``b_value`` within it.
    """
    import types

    raster = system.block_duration_raster
    exc_tail = float(exc.seq.duration()[0]) - float(exc.rf.delay + exc.rf.center)
    refocus_block = pp.calc_duration(refocus.rf_ref, refocus.gz)
    refocus_centre = float(refocus.rf_ref.delay + refocus.rf_ref.center)
    probe = make_train(None)

    def lobe(duration, channel="x"):
        return pp.make_trapezoid(
            channel=channel,
            amplitude=system.max_grad,
            duration=duration,
            system=system,
        )

    def halves(duration, half=None):
        excitation = exc_tail + duration + refocus_centre
        readout = float(probe.echo_time) + duration
        shortest = max(excitation, readout)
        if half is None:
            half = shortest
        if half < shortest - 1e-9:
            return None
        return half, pp.round_to_raster(half - excitation, raster)

    def search(half):
        """Return the shortest lobe that reaches ``b_value`` at ``half``, with its b-value."""
        duration, reached = 0.0, 0.0
        if b_value is None:
            return (duration, reached) if halves(duration, half) else None
        rise = system.max_grad / system.max_slew
        duration = pp.round_to_raster(2 * rise + DIFFUSION_LOBE_STEP, raster)
        while True:
            timing = halves(duration, half)
            if timing is None:
                return None
            reached = sequences.diffusion_pair_b_value(
                lobe(duration), timing[1] + refocus_block, system.grad_raster_time
            )
            if reached >= b_value:
                return duration, reached
            duration = pp.round_to_raster(duration + DIFFUSION_LOBE_STEP, raster)

    shortest = search(None)
    shortest_half = halves(shortest[0])[0]
    if te is not None and te / 2 < shortest_half - 1e-9:
        raise ValueError(
            f"TE {te * 1e3:.2f} ms is shorter than the spin echo admits; "
            f"the minimum is {2 * shortest_half * 1e3:.2f} ms"
        )
    duration, reached = shortest if te is None else search(te / 2)
    half, wait = halves(duration, None if te is None else te / 2)
    train = make_train(half - duration)
    echo_time = (
        exc_tail + duration + wait + refocus_centre + float(train.echo_time) + duration
    )
    return types.SimpleNamespace(
        train=train,
        echo_time=echo_time,
        wait_half=pp.make_delay(wait) if wait > 0 else None,
        lobes=None
        if b_value is None
        else tuple(lobe(duration, channel) for channel in "xyz"),
        lobe_duration=duration,
        b_value=reached,
        excitation_half=float(exc.seq.duration()[0]) + duration + wait,
    )


NAME = "epi_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the selective pulses. The selection amplitude, which slice
#: offsets are converted against, is the pulse's measured bandwidth over the
#: thickness.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Dephasing of each crusher around the refocusing pulse of a spin echo, in
#: cycles across the slice.
CRUSHER_CYCLES = 4.0
#: Step (s) by which the diffusion lobes are lengthened until they reach the
#: b-value.
DIFFUSION_LOBE_STEP = 0.2e-3
#: Centre lines read without blips at the start of every shot when
#: ``navigator_lines`` asks for a navigator.
NAVIGATOR_LINES = 3
#: Counter holding each train line's position in its train, from zero, which
#: matches an imaging line to the reference line of the same echo.
ECHO_COUNTER = "USER0"
#: Dephasing left on the readout axis at the end of each shot, in cycles
#: across one voxel.
SPOILING_CYCLES = 4.0
#: Digital output marking every volume under ``volume_output``, and how
#: long it lasts (s); it never outlasts the excitation block it rides.
OUTPUT_CHANNEL = "ext1"
OUTPUT_DURATION = 1e-3
#: Flip angle of the gradient-echo calibration (degrees), whatever the
#: imaging flip. Its lines follow one another at its shortest TR, from relaxed
#: magnetization: at an imaging flip such as 70 degrees the signal falls
#: within a few lines to a hundredth of the first line's, and each line's RF
#: alone can exceed the SAR limit. Near the Ernst angle of that TR, ten degrees
#: keeps most of the first line's signal across the calibration region and
#: plays a fiftieth of a 70 degree pulse's RF power. The calibration measures
#: coil sensitivities, which do not depend on the contrast.
CALIBRATION_FLIP_DEG = 10.0
#: Calibrations the scan opens with.
CALIBRATIONS = ("fleet", "gre")
#: Non-acquiring shots that open the FLEET calibration of each slice, so its
#: first acquiring shot does not read relaxed magnetization.
FLEET_DUMMY_SHOTS = 1
#: Largest fat-nulling flip (degrees) of the calibration's fat saturation; a
#: calibration shot too short for a null at this flip waits until it allows one.
CALIBRATION_FAT_NULL_FLIP_DEG = 120.0

#: ``SET`` of the forward and of the reversed volume of the spin-echo field
#: pair, apart from those of the scan and its reversed reference.
FIELD_PAIR_SETS = (2, 3)
FULL_K_REFERENCE_SETS = (4, 5)


def epi2d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 220e-3,
    fov_y: float = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    flip_angle_deg: float = 70.0,
    te: float | None = None,
    tr: float | None = None,
    n_frames: int = 1,
    readout_bandwidth_hz: float = 500e3,
    ry: int = 1,
    partial_fourier_y: float = 1.0,
    n_shots: int = 1,
    multiband: int = 1,
    fat_saturation: bool = False,
    n_dummy: int | None = None,
    readout_oversampling: float = 1.0,
    n_acs_y: int = 24,
    calibration: str = "fleet",
    volume_output: bool = False,
    phase_reference: bool = False,
    navigator_lines: int | None = None,
    spin_echo: bool = False,
    diffusion_directions: int = 0,
    b_value: float = 1.0e9,
    diffusion_b0: int = 1,
    field_pair: bool = True,
    full_k_reference: bool = True,
) -> list[pp.Sequence]:
    """Multi-slice 2D gradient-echo or spin-echo EPI, optionally segmented, multiband or diffusion-weighted.

    With ``spin_echo``, a 90 degree excitation and a slice-selective
    refocusing pulse between crushers form a spin echo at ``te``, which the
    centre line reads. With ``diffusion_directions``, a pair of identical
    trapezoidal diffusion lobes surrounds the refocusing pulse (Stejskal and
    Tanner, J Chem Phys 1965, doi:10.1063/1.1695690), lengthened at
    ``system.max_grad`` until they reach ``b_value``. The scan plays
    ``diffusion_b0`` volumes at b = 0, then one volume per direction of
    :func:`~pypulseqpp.io.uniform_directions`, in the logical frame, and
    records that table in the file. The table repeats ``n_frames`` times. The FLEET or gradient-echo calibration stays a
    gradient echo.

    With ``phase_reference``, the last dummy shot of each slice (of each
    FLEET slice, its dummy shot) is a reference train for odd/even phase
    correction: the shot's own train with every phase-encode prewinder, blip
    and rewinder at zero, its lines acquired (``PHASECORR``). Every train line
    carries its position in the train in ``ECHO_COUNTER``, which matches an
    imaging line to the reference line of the same echo. A scan without dummy
    shots plays one cycle of them for the reference. Each shot can also read
    ``navigator_lines`` centre lines without blips (``NAV``) before its
    train, after one unsampled lobe that lets the read axis settle. Every line
    carries ``REV`` for its read polarity. The trains are ramp-sampled with
    the blips on the read ramps.
    Segmented shots interleave on the phase-encode lattice, each delayed by
    a fraction of the echo spacing so the echo time grows smoothly across
    k-space; ``te`` is the echo time of the centre line.

    Slices (under ``multiband``, slice groups) are excited once per shot in
    packets, even ones first. With one frame, slices one TR cannot hold are
    dealt round-robin into packets; a time series must fit every slice in
    one. A multiband shot excites ``multiband`` slices at once and a
    blipped-CAIPI train encodes the band in ``PAR``.

    The ``calibration`` of the central ``n_acs_y`` lines (``REF``) at
    ``CALIBRATION_FLIP_DEG``, single band, is linked ahead of the imaging;
    the coil sensitivities of the whole volume are estimated from it once.
    The FLEET calibration (Polimeni et al., Magn Reson Med 2016,
    doi:10.1002/mrm.25628) reads those lines in ``n_shots * ry`` interleaved
    EPI shots, which step as far per echo spacing as an imaging shot, with
    its read lobe, navigator and blip, so the calibration is distorted as the
    imaging is, at the shortest echo time they admit. The shots of a slice follow one another
    after ``FLEET_DUMMY_SHOTS`` unsampled ones, forward (``SET = 0``) and then
    reversed (``SET = 1``), so each phase-encode polarity has its own, and
    the two polarities give the field that corrects the distortion. The
    gradient-echo calibration reads one line per excitation, undistorted;
    after it, a ``reference`` volume with the phase encode reversed
    (``SET = 1``) is linked for distortion correction.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x, fov_y : float, default=0.22
        Field of view along the readout and the phase encode (m).
    n_x : int, default=128
        Readout matrix size.
    n_y : int, default=128
        Phase-encode matrix size.
    n_slices : int, default=1
        Number of slices.
    slice_thickness : float, default=0.005
        Slice thickness (m).
    slice_spacing : float, default=0.0
        Gap between adjacent slices (m); zero is contiguous.
    flip_angle_deg : float, default=70.0
        Excitation flip angle (degrees) of the imaging and the reference; the
        calibration plays ``CALIBRATION_FLIP_DEG``.
    te : float | None, default=None
        Echo time of the centre line (s). ``None`` is as short as the
        train, and its navigator lines when there are any, admit.
    tr : float | None, default=None
        Volume repetition time (s): every shot of every slice. ``None`` is
        as short as possible, and puts every slice in one packet.
    n_frames : int, default=1
        Volumes in the time series, each carrying its ``REP`` counter.
    readout_bandwidth_hz : float, default=500000.0
        Requested receiver bandwidth (Hz).
    ry : int, default=1
        Phase-encode undersampling: one line in every ``ry`` is read, the
        centre line among them.
    partial_fourier_y : float, default=1.0
        Fraction of the phase-encode extent read, in ``[0.5, 1]``.
        Truncates the lines before the centre, which shortens the minimum
        TE.
    n_shots : int, default=1
        Interleaved shots each volume's phase encode is split into.
    multiband : int, default=1
        Slices excited at once. It must divide ``n_slices``.
    fat_saturation : bool, default=False
        Saturate fat before every shot, at the flip
        :func:`~pypulseqpp.sequences.fat_null_flip_deg` gives, which leaves it
        at zero when the shot's excitation comes.
    n_dummy : int | None, default=None
        Non-acquiring volumes before a time series; with one frame,
        non-acquiring shots per slice before each packet.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    readout_oversampling : float, default=1.0
        Minimum EPI readout oversampling factor, at least one. Additional
        samples preserve the requested dwell under the gradient limits.
    n_acs_y : int, default=24
        Phase-encode lines of the calibration; FLEET rounds them up to a
        whole number of lines per shot.
    calibration : {'fleet', 'gre'}, default='fleet'
        The calibration: FLEET,
        segmented EPI at the imaging train's echo spacing, or a gradient echo
        per line.
    volume_output : bool, default=False
        Play a digital output on ``OUTPUT_CHANNEL`` at the first
        excitation of every volume, dummy volumes included.
    phase_reference : bool, default=False
        Acquire a blip-free reference train in a dummy shot of each slice, for
        odd/even phase correction.
    navigator_lines : int | None, default=None
        Centre lines read without blips at the start of every shot, for
        odd/even phase correction. ``None`` is ``NAVIGATOR_LINES`` without
        ``phase_reference`` and none with it.
    spin_echo : bool, default=False
        Refocus the echo with a slice-selective 180 degree pulse; the
        excitation is then 90 degrees, whatever ``flip_angle_deg``.
    diffusion_directions : int, default=0
        Count of diffusion-encoding directions; 0 plays no diffusion
        weighting. It needs ``spin_echo``.
    b_value : float, default=1000000000.0
        b-value (s/m^2) of each diffusion-encoding direction.
    diffusion_b0 : int, default=1
        Count of b = 0 volumes played before the directions.
    field_pair : bool, default=True
        Play a spin-echo pair before the scan, at the imaging train and the
        shortest spin echo: one volume with the phase encode forward, labelled
        ``SET = 2``, and one reversed, labelled ``SET = 3``. The two are displaced
        in opposite directions by the susceptibility field and differ in
        nothing else, as a gradient-echo pair does not at a long echo time,
        and the reconstruction estimates the displacement field of the scan
        from them. The pair is played single-band.
    full_k_reference : bool, default=True
        Play one fully sampled volume at the echo time of a partial-Fourier
        gradient-echo scan, labelled ``SET = 4``, from whose phase the
        reconstruction completes the unsampled side of k-space. Without a
        prescribed ``te``, the scan takes the shortest echo time this volume
        admits. A spin-echo scan, whose phase is smooth, plays none.

    Returns
    -------
    list of pypulseqpp.Sequence
        The designed sequences in play order: the ``calibration``, the
        ``reference`` after a gradient-echo calibration, the forward and the
        reversed volume of the ``field_pair``, the ``full_k_reference``, then
        the scan, which is the main sequence. ``sequences.write`` writes them as files linked
        through ``NextSequence``.

    Raises
    ------
    ValueError
        If ``partial_fourier_y`` is outside ``[0.5, 1]``, ``calibration`` is
        unknown, a count is below one, ``multiband`` does not divide
        ``n_slices``, the shots cannot
        share the lines, the TE is shorter than the train admits, or the TR
        cannot hold one slice, or every slice when ``n_frames`` is above
        one, or the TE is shorter than the ``full_k_reference`` admits.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> *prescans, seq = sequences.epi2D_sequence(n_x=32, n_y=16, n_dummy=0)
    >>> [prescan.definitions["Name"] for prescan in prescans]
    ['epi_2d_calibration', 'epi_2d_field_forward', 'epi_2d_field_reversed']
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["EPIFactor"], seq.definitions["Name"]
    ([16.0], 'epi_2d')
    """
    protocol = {
        "fov_x": fov_x,
        "fov_y": fov_y,
        "n_x": n_x,
        "n_y": n_y,
        "n_slices": n_slices,
        "slice_thickness": slice_thickness,
        "slice_spacing": slice_spacing,
        "flip_angle_deg": flip_angle_deg,
        "te": te,
        "tr": tr,
        "n_frames": n_frames,
        "readout_bandwidth_hz": readout_bandwidth_hz,
        "ry": ry,
        "partial_fourier_y": partial_fourier_y,
        "n_shots": n_shots,
        "multiband": multiband,
        "fat_saturation": fat_saturation,
        "n_dummy": n_dummy,
        "readout_oversampling": readout_oversampling,
        "n_acs_y": n_acs_y,
        "calibration": calibration,
        "volume_output": volume_output,
        "phase_reference": phase_reference,
        "navigator_lines": navigator_lines,
        "spin_echo": spin_echo,
        "diffusion_directions": diffusion_directions,
        "b_value": b_value,
        "diffusion_b0": diffusion_b0,
        "field_pair": field_pair,
    }
    if not (full_k_reference and partial_fourier_y < 1.0 and not spin_echo):
        return _designed(system, **protocol, encode_sets=(0, 1))
    try:
        *_, reference = _designed(
            system,
            **{
                **protocol,
                "partial_fourier_y": 1.0,
                "n_frames": 1,
                "volume_output": False,
                "field_pair": False,
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
    return [*prescans, reference, seq]


def _designed(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 220e-3,
    fov_y: float = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    flip_angle_deg: float = 70.0,
    te: float | None = None,
    tr: float | None = None,
    n_frames: int = 1,
    readout_bandwidth_hz: float = 500e3,
    ry: int = 1,
    partial_fourier_y: float = 1.0,
    n_shots: int = 1,
    multiband: int = 1,
    fat_saturation: bool = False,
    n_dummy: int | None = None,
    readout_oversampling: float = 1.0,
    n_acs_y: int = 24,
    calibration: str = "fleet",
    volume_output: bool = False,
    phase_reference: bool = False,
    navigator_lines: int | None = None,
    spin_echo: bool = False,
    diffusion_directions: int = 0,
    b_value: float = 1.0e9,
    diffusion_b0: int = 1,
    field_pair: bool = False,
    encode_sets: tuple[int, int],
) -> list[pp.Sequence]:
    """Design :func:`epi2d`, its forward phase encodes labelled ``SET = encode_sets[0]`` and its reversed ones ``SET = encode_sets[1]``."""
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if navigator_lines is None:
        navigator_lines = 0 if phase_reference else NAVIGATOR_LINES
    if not 0.5 <= partial_fourier_y <= 1.0:
        raise ValueError(
            f"partial_fourier_y must lie in [0.5, 1], got {partial_fourier_y}"
        )
    if calibration not in CALIBRATIONS:
        raise ValueError(
            f"calibration must be one of {CALIBRATIONS}, got {calibration!r}"
        )
    for name, count in (
        ("ry", ry),
        ("n_shots", n_shots),
        ("multiband", multiband),
        ("n_frames", n_frames),
    ):
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    if n_slices % multiband:
        raise ValueError(
            f"the slice count {n_slices} is not a multiple of the multiband "
            f"factor {multiband}"
        )

    if diffusion_directions < 0 or diffusion_b0 < 0:
        raise ValueError(
            "diffusion_directions and diffusion_b0 count volumes and cannot be negative"
        )
    if diffusion_directions and not spin_echo:
        raise ValueError("diffusion weighting needs spin_echo")
    if spin_echo and multiband > 1:
        raise ValueError("a spin echo is designed single-band; set multiband to 1")
    b_table = g_table = None
    if diffusion_directions:
        g_table = np.concatenate(
            [
                np.zeros((diffusion_b0, 3)),
                pp.io.uniform_directions(diffusion_directions),
            ]
        )
        b_table = np.where(np.any(g_table != 0, axis=1), b_value * 1e-6, 0.0)
    volumes = n_frames * (1 if b_table is None else len(b_table))

    fov = (fov_x, fov_y)
    raster = system.block_duration_raster
    slice_step = slice_thickness + slice_spacing
    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * slice_step
    slab_thickness = n_slices * slice_step - slice_spacing

    single = sequences.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )
    excited_thickness = single.slice_thickness
    slice_gap = slice_step - excited_thickness

    # Shot s reads lines start + (s + i * n_shots) * ry, and the centre line
    # is the c-th of the lattice: with the echo shifts, it is echoed at the
    # (c / n_shots)-th line of the unshifted train.
    start, etl = train_lines(n_y, ry, n_shots, partial_fourier_y)
    origins = [start + s * ry for s in range(n_shots)]
    train_arguments = {
        "te": te,
        "te_line": (n_y // 2 - start) // ry / n_shots,
        "navigator_lines": navigator_lines,
        "echo_shifts": n_shots,
        "oversampling": readout_oversampling,
        "readout_bandwidth_hz": readout_bandwidth_hz,
        "spoiling_cycles": SPOILING_CYCLES,
    }
    steps = np.arange(etl) * n_shots * ry
    n_groups = n_slices // multiband
    if multiband > 1:
        exc = sequences.SmsExcitation(
            system,
            flip_angle_deg,
            thickness_m=slice_thickness,
            slice_gap_m=n_groups * slice_step,
            n_bands=multiband,
            duration_s=PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
        )
        # The rephaser folds onto the selection lobe, since the band phase
        # takes the z channel of the phase-encode prewinder.
        gz = pp.concatenate_gradients(exc.gz, exc.gz_reph, system=system)
        # The band axis is a partition axis of multiband partitions, with
        # its lattice anchored on the centre band like the lines are.
        shift = caipi_shift(ry, multiband)
        trains = {}
        shot_trains = []
        for s in range(n_shots):
            lines = start + (s + np.arange(etl) * n_shots) * ry
            bands = (multiband // 2 + shift * ((lines - n_y // 2) // ry)) % multiband
            key = tuple(bands)
            if key not in trains:
                trains[key] = sequences.EpiReadout3D(
                    system,
                    exc.rf,
                    gz,
                    **train_arguments,
                    order=np.column_stack((steps, bands)),
                    fov=(fov_x, fov_y, multiband * n_groups * slice_step),
                    matrix=(n_x, n_y, multiband),
                    labels=("LIN", "PAR"),
                )
            shot_trains.append(trains[key])
        selection_amplitude = float(exc.gz.amplitude)
        # The pulse places its bands at band_positions_m from the slice its
        # frequency offset selects, which is not their mean for an even band
        # count: the offset aims that mean at the group's centre.
        lead = float(np.mean(exc.band_positions_m))
        centers = [
            (g - (n_slices - 1) / 2 + (multiband - 1) / 2 * n_groups) * slice_step
            - lead
            for g in range(n_groups)
        ]
        group_centres = [c + lead for c in centers]
        partition_fov = multiband * n_groups * slice_step
    elif spin_echo:
        exc = sequences.SpatialSelectiveExcitation(
            system,
            90.0,
            slice_thickness,
            duration_s=PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
        )
        refocus = sequences.SpatialSelectiveRefocusing(
            system,
            slice_thickness,
            duration_s=PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
            spoiling_cycles=CRUSHER_CYCLES,
        )
        echo = spin_echo_timing(
            system,
            exc,
            refocus,
            lambda half: sequences.EpiReadout2D(
                system,
                refocus.rf_ref,
                refocus.gz,
                **{**train_arguments, "te": half},
                order=steps,
                fov=fov,
                matrix=(n_x, n_y),
                labels=("LIN",),
            ),
            te,
            None if b_table is None else b_value * 1e-6,
        )
        shot_trains = [echo.train] * n_shots
        selection_amplitude = refocus.selection_amplitude
        centers = list(positions)
        group_centres = [0.0] * len(centers)
        partition_fov = 1.0
    else:
        readout = sequences.EpiReadout2D(
            system,
            single.rf,
            single.gz,
            single.gz_reph,
            **train_arguments,
            order=steps,
            fov=fov,
            matrix=(n_x, n_y),
            labels=("LIN",),
        )
        shot_trains = [readout] * n_shots
        selection_amplitude = single.selection_amplitude
        centers = list(positions)
        group_centres = [0.0] * len(centers)
        partition_fov = 1.0
    epi = shot_trains[0]
    echo_time = echo.echo_time if spin_echo else epi.echo_time
    # The excitation half of a spin echo and its second diffusion lobe play
    # around the train, which opens with the refocusing pulse.
    echo_extra = echo.excitation_half + echo.lobe_duration if spin_echo else 0.0
    excited = exc.rf if spin_echo else epi.rf

    fatsat = (
        sequences.FatSaturation(
            pp.cap_system(system, max_slew=PREPARATION_MAX_SLEW),
            voxel_size_m=min(fov_x / n_x, fov_y / n_y, slice_thickness),
        )
        if fat_saturation
        else None
    )

    # A shot is the fat saturation, the train, and a closing delay of at
    # least one raster; the last shot of a packet waits out the cycle. A
    # cycle is one shot of every slice of a packet, and a volume is
    # n_shots cycles.
    prep = [fatsat] if fatsat is not None else []
    prep_duration = sum(module.duration for module in prep)
    shot_duration = prep_duration + echo_extra + epi.duration + raster
    cycle = None if tr is None else tr / n_shots
    per_packet = (
        n_groups if cycle is None else max(1, int(cycle / shot_duration + 1e-9))
    )
    packets = packets_of(n_groups, per_packet)
    if volumes > 1 and len(packets) > 1:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms cannot hold the "
            f"{n_groups} excitations of a volume, {shot_duration * 1e3:.3f} ms each "
            f"per shot"
        )
    if cycle is None:
        cycle = max(map(len, packets)) * shot_duration
    if fatsat is not None:
        # The flip that leaves fat at zero when the train's excitation comes,
        # in the steady state of one saturation every shot. The pulse's
        # duration is its bandwidth's, so the timing stands.
        to_excitation = (
            fatsat.duration
            - fatsat.center
            + float(excited.delay)
            + float(excited.center)
        )
        fatsat = sequences.FatSaturation(
            pp.cap_system(system, max_slew=PREPARATION_MAX_SLEW),
            flip_angle_deg=sequences.fat_null_flip_deg(to_excitation, shot_duration),
            voxel_size_m=min(fov_x / n_x, fov_y / n_y, slice_thickness),
        )
        prep = [fatsat]
    pads = {
        size: pp.round_to_raster(cycle - size * shot_duration, raster) + raster
        for size in {len(packet) for packet in packets}
    }
    if min(pads.values()) < raster:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
            f"{n_shots * shot_duration * 1e3:.3f} ms the shots of one slice take"
        )
    packet_time = {n: n * shot_duration - raster + pad for n, pad in pads.items()}
    repetition_time = n_shots * max(packet_time.values())
    if n_dummy is None:
        # A slice is excited once a cycle; a volume holds n_shots cycles.
        excitations = sequences.steady_state_dummies(
            cycle, 90.0 if spin_echo else flip_angle_deg
        )
        n_dummy = -(-excitations // n_shots) if volumes > 1 else excitations
    dummy_cycles = n_dummy * (n_shots if volumes > 1 else 1)
    if phase_reference:
        # The reference plays in the last dummy cycle, so there is one.
        dummy_cycles = max(dummy_cycles, 1)
    output_pulse = pp.make_digital_output_pulse(
        OUTPUT_CHANNEL,
        duration=min(
            OUTPUT_DURATION,
            pp.calc_duration(excited, exc.gz if spin_echo else epi.gz),
        ),
        system=system,
    )

    # The calibration lines: centred, and a whole number per shot for FLEET.
    acs_shots = n_shots * ry if calibration == "fleet" else 1
    acs_lines = fleet_lines(n_y, n_acs_y, acs_shots)
    weak = None
    if acs_lines:
        # The single-band pulse's samples scaled to the calibration's flip
        # angle, a small-tip pulse's profile being its waveform's: one
        # waveform and one timing for both, with no second design.
        weak = pp.make_arbitrary_rf(
            signal=np.asarray(single.rf.signal)
            * (CALIBRATION_FLIP_DEG / flip_angle_deg),
            flip_angle=0.0,
            no_signal_scaling=True,
            delay=single.rf.delay,
            dwell=float(single.rf.t[1] - single.rf.t[0]),
            system=system,
            use=single.rf.use,
            center=single.rf.center,
        )
    gre = acs = None
    if weak is not None and calibration == "gre":
        gre = sequences.LineReadout2D(
            system,
            weak,
            single.gz,
            single.gz_reph,
            fov=fov,
            matrix=(n_x, n_y),
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=SPOILING_CYCLES,
            labels=("LIN",),
        )
    elif weak is not None:
        acs = fleet_train(
            system,
            weak,
            single.gz,
            single.gz_reph,
            epi,
            acs_lines,
            acs_shots,
            **{
                key: value
                for key, value in train_arguments.items()
                if key not in ("te", "te_line", "echo_shifts")
            },
            fov=fov,
            matrix=(n_x, n_y),
            labels=("LIN",),
        )

    def calibration_prep(probe) -> tuple[list, float]:
        """Return the fat saturation of a calibration shot and the wait (s) that closes the shot.

        The calibration saturates fat as the imaging does: coil maps fitted to
        a calibration whose scalp fat is displaced along the phase encode do
        not hold for water-only imaging data. ``probe`` adds one calibration
        shot to a sequence. The saturation nulls fat at the calibration's
        excitation; a shot too short for a null at
        ``CALIBRATION_FAT_NULL_FLIP_DEG`` waits until it allows one.
        """
        if fatsat is None:
            return [], 0.0
        shot = pp.Sequence(system)
        probe(shot)
        delay = fatsat.duration - fatsat.center + float(weak.delay) + float(weak.center)
        period = fatsat.duration + float(shot.duration()[0])
        cosine = np.cos(np.deg2rad(CALIBRATION_FAT_NULL_FLIP_DEG))
        shortest = delay - sequences.FAT_T1_S * np.log(
            1.0 - (1.0 - np.exp(delay / sequences.FAT_T1_S)) / cosine
        )
        wait = pp.round_to_raster(max(shortest - period, 0.0), raster)
        flip = sequences.fat_null_flip_deg(delay, period + wait)
        prep = [
            sequences.FatSaturation(
                pp.cap_system(system, max_slew=PREPARATION_MAX_SLEW),
                flip_angle_deg=flip,
                voxel_size_m=min(fov_x / n_x, fov_y / n_y, slice_thickness),
            )
        ]
        return prep, wait

    def define(seq: pp.Sequence, **definitions) -> None:
        # Every file of the chain places its slices, so a reconstruction reads
        # a prescan's geometry from the prescan, and each acquiring file states
        # the k-space centre line: one that did not was centred on the lines it
        # read, which partial Fourier leaves off centre.
        for key, value in {
            "FOV": [*fov, slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            "SlicePositions": positions.tolist(),
            "SliceThickness": excited_thickness,
            "SliceGap": slice_gap,
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
        s: int,
        line: int,
        prep: list = (),
        wait: float = 0.0,
    ) -> None:
        """Add one single-band gradient echo of slice ``s`` at ``line``, between ``prep`` and a ``wait`` (s)."""
        gre.rf.freq_offset = single.selection_amplitude * positions[s]
        gre.rf.phase_offset = -2 * np.pi * gre.rf.freq_offset * gre.rf.center
        gre.adc_labels.value = line
        ky = (line - n_y // 2) / (n_y / 2)
        for module in prep:
            for block in module.blocks:
                seq.add_block(*block)
        seq.add_block(gre.rf, gre.gz, *labels(REF=1, SLC=s))
        wait_te = getattr(gre, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, gre.gz_reph)
            seq.add_block(gre.gx_pre, pp.scale_grad(gre.gy_pre, ky))
        else:
            seq.add_block(gre.gx_pre, pp.scale_grad(gre.gy_pre, ky), gre.gz_reph)
        seq.add_block(gre.gx, gre.adc, gre.adc_labels)
        seq.add_block(gre.gx_spoil, pp.scale_grad(gre.gy_rew, ky))
        if wait > 0:
            seq.add_block(pp.make_delay(wait))

    def calibrate() -> pp.Sequence:
        """Design the gradient-echo calibration: every line of every slice."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        order = [s for packet in packets_of(n_slices, n_slices) for s in packet]
        prep, wait = calibration_prep(
            lambda probe: calibration_kernel(probe, sequences.Labels(), 0, acs_lines[0])
        )
        for s in order:
            for line in acs_lines:
                calibration_kernel(seq, labels, s, line, prep, wait)
        define(
            seq,
            Name=f"{NAME}_calibration",
            kSpaceCenterLine=n_y // 2,
        )
        return seq

    def fleet() -> pp.Sequence:
        """Design the FLEET calibration: every shot of a slice, then the next slice."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        order = [s for packet in packets_of(n_slices, n_slices) for s in packet]
        starts = [acs_lines[0] + s for s in range(acs_shots)]
        prep, wait = calibration_prep(
            lambda probe: shot_blocks(
                probe, sequences.Labels(), acs, starts[0], 0, 0.0, [], raster
            )
        )
        for s in order:
            offset = single.selection_amplitude * positions[s]
            for dummy in range(FLEET_DUMMY_SHOTS):
                reference = phase_reference and dummy == FLEET_DUMMY_SHOTS - 1
                shot_blocks(
                    seq,
                    labels,
                    acs,
                    starts[0],
                    0,
                    offset,
                    labels(SLC=s, SEG=0, SET=0, PHASECORR=1) if reference else [],
                    raster + wait,
                    reference=reference,
                    prep=prep,
                )
            for polarity in (0, 1):
                for shot, origin in enumerate(starts):
                    flags = labels(
                        REF=1,
                        SLC=s,
                        SEG=shot,
                        SET=polarity,
                        **({"PHASECORR": 0} if phase_reference else {}),
                    )
                    shot_blocks(
                        seq,
                        labels,
                        acs,
                        origin,
                        shot,
                        offset,
                        flags,
                        raster + wait,
                        acquire=True,
                        reversed_encode=bool(polarity),
                        prep=prep,
                    )
        define(
            seq,
            Name=f"{NAME}_calibration",
            TE=acs.echo_time,
            EchoSpacing=acs.esp,
            kSpaceCenterLine=n_y // 2,
        )
        return seq

    def kernel(
        seq: pp.Sequence,
        labels: sequences.Labels,
        g: int,
        shot: int,
        frame: int | None,
        pad: float,
        output: bool = False,
        reversed_encode: bool = False,
        reference: bool = False,
    ) -> None:
        """Add one shot of slice (multiband group) ``g``; ``frame=None`` is a dummy.

        ``pad`` is the closing delay the shot would have without echo shift,
        in s. A ``reference`` dummy acquires the blip-free reference train.
        """
        acquire = frame is not None
        if not acquire:
            flags = {"ONCE": 1}
            if reference:
                flags |= {"SLC": g, "SEG": shot, "PHASECORR": 1}
                if encode_sets[reversed_encode]:
                    flags["SET"] = encode_sets[reversed_encode]
        else:
            flags = {"SLC": g, "REP": frame, "SEG": shot, "ONCE": 0}
            if multiband > 1:
                flags["SMS"] = 1
            if encode_sets[reversed_encode]:
                flags["SET"] = encode_sets[reversed_encode]
            if phase_reference:
                flags["PHASECORR"] = 0
            if not dummy_cycles:
                del flags["ONCE"]
        half, lobe = (), None
        if spin_echo:
            exc.rf.freq_offset = exc.selection_amplitude * centers[g]
            exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
            half = [(exc.rf, exc.gz), (exc.gz_reph,)]
            if echo.lobes is not None:
                lobe = lobes(None if frame is None else frame % len(b_table))
                half.append(lobe)
            if echo.wait_half is not None:
                half.append((echo.wait_half,))
        shot_blocks(
            seq,
            labels,
            shot_trains[shot],
            origins[shot],
            shot,
            selection_amplitude * centers[g],
            [*labels(**flags), *([output_pulse] if output else [])],
            pad,
            group_centre=group_centres[g],
            acquire=acquire,
            reversed_encode=reversed_encode and not reference,
            prep=prep,
            reference=reference,
            half=half,
            lobe=lobe,
        )

    made_lobes: dict = {}

    def lobes(row: int | None) -> tuple:
        """Return the diffusion lobe of table row ``row`` on the three channels; ``None`` and b = 0 are a delay."""
        if row not in made_lobes:
            if row is None or b_table[row] <= 0:
                made_lobes[row] = (pp.make_delay(echo.lobe_duration),)
            else:
                scale = math.sqrt(b_table[row] / echo.b_value)
                made_lobes[row] = tuple(
                    pp.scale_grad(event, scale * float(component))
                    for event, component in zip(echo.lobes, g_table[row], strict=True)
                    if component != 0.0
                ) or (pp.make_delay(echo.lobe_duration),)
        return made_lobes[row]

    def shot_blocks(
        seq: pp.Sequence,
        labels: sequences.Labels,
        train,
        origin: int,
        shot: int,
        offset: float,
        first_events: list,
        pad: float,
        group_centre: float = 0.0,
        acquire: bool = False,
        reversed_encode: bool = False,
        prep: list = (),
        reference: bool = False,
        half: list = (),
        lobe: tuple | None = None,
    ) -> None:
        """Add one shot of ``train`` from line ``origin``, its pulse ``offset`` Hz off resonance.

        A spin echo's ``half`` (its excitation, rephaser, first diffusion lobe
        and wait) plays before the train, whose pulse refocuses, and ``lobe``,
        the second diffusion lobe, after that pulse.

        A multiband line is received at the phase its partition encode gives
        a spin at ``group_centre`` (m), so its bands carry the phases of a
        partition volume centred on their group.

        ``first_events`` ride the first block, and ``shot`` sets the echo
        shift. ``reversed_encode`` negates every phase-encode event, so each
        line L of the forward shot is read at its mirror, ``LIN = n_y - L``;
        the mirror of line 0 lies outside the matrix and is played unsampled.
        ``reference`` scales every phase-encode event to zero and acquires the
        train lines, its navigator unsampled: the blip-free reference train.
        """
        train.rf.freq_offset = offset
        train.rf.phase_offset = -2 * np.pi * train.rf.freq_offset * train.rf.center
        if half:
            train.rf.phase_offset += np.pi / 2
        sign = -1.0 if reversed_encode else 1.0
        train.shot_labels[0].value = n_y - origin if reversed_encode else origin
        # The prewinder and the rewinder closing the shot are scaled alike.
        ky = sign * (origin - n_y // 2) / (n_y / 2)
        swap = {
            id(train.gy_pre): pp.scale_grad(train.gy_pre, ky),
            id(train.gy_rew): pp.scale_grad(train.gy_rew, ky),
        }
        if len(train.shot_labels) > 1:
            band = int(train.order[0, 1])
            train.shot_labels[1].value = band
            kz = (band - multiband // 2) / (multiband / 2)
            swap[id(train.gz_pre)] = pp.scale_grad(train.gz_pre, kz)
            swap[id(train.gz_rew)] = pp.scale_grad(train.gz_rew, kz)
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
            encodes = [train.gy_pre, train.gy_rew, *train.gy_blips]
            if len(train.shot_labels) > 1:
                encodes += [train.gz_pre, train.gz_rew, *train.gz_blips]
            swap.update(
                (id(event), pp.scale_grad(event, 0.0))
                for event in encodes
                if event is not None
            )
        lines = origin + np.asarray(train.order).reshape(train.etl, -1)[:, 0]
        step = shot * train.echo_shift_step
        wait_shift = getattr(train, "wait_shift", None)
        if wait_shift is not None:
            swap[id(wait_shift)] = pp.make_delay(wait_shift.delay + step)
        wait_tr = getattr(train, "wait_tr", None)
        closing = pad + (0.0 if wait_tr is None else wait_tr.delay) - step

        label_events = list(first_events)
        for module in prep:
            for block in module.blocks:
                seq.add_block(*block, *label_events)
                label_events = []
        for block in half:
            seq.add_block(*block, *label_events)
            label_events = []
        navigator = {id(event) for event in getattr(train, "gx_navigator", ())}
        readouts = {id(event) for event in train.gx}
        polarity, line = 0, 0
        for index, block in enumerate(train.blocks):
            head = block[0]
            if head is wait_tr:
                continue
            reading = id(head) in readouts and id(head) not in navigator
            echo = line
            sampled = (acquire or (reference and reading)) and not (
                reversed_encode and reading and lines[line] == 0
            )
            line += reading
            events = [
                swap.get(id(event), event)
                for event in block
                if sampled or event.type != "adc"
            ]
            if reading and sampled and not reference and len(train.shot_labels) > 1:
                kz = (int(train.order[echo, 1]) - multiband // 2) / partition_fov
                events = [
                    _received_at(event, 2 * np.pi * kz * group_centre, system)
                    if event.type == "adc"
                    else event
                    for event in events
                ]
            if index == 0:
                events += label_events
            elif id(head) in navigator or id(head) in readouts:
                if acquire or sampled:
                    flags = {"NAV": int(id(head) in navigator), "REV": polarity}
                    if reading and phase_reference:
                        flags[ECHO_COUNTER] = echo
                    events += labels(**flags)
                polarity ^= 1
            seq.add_block(*events)
            if index == 0 and lobe is not None:
                seq.add_block(*lobe)
        seq.add_block(pp.make_delay(closing))

    def play(frames, reversed_encode: bool = False):
        """Design the dummies and then ``frames``, packet by packet.

        Returns the sequence, and the first block and the number of blocks of
        its first acquiring shot.
        """
        seq, labels = pp.Sequence(system), sequences.Labels()
        first_image_shot = None
        for packet in packets:
            cycles = [(None, c % n_shots) for c in range(dummy_cycles)]
            cycles += [(frame, shot) for frame in frames for shot in range(n_shots)]
            pad = pads[len(packet)]
            for c, (frame, shot) in enumerate(cycles):
                # The last dummy cycle plays the reference of every slice.
                reference = phase_reference and c == dummy_cycles - 1
                for i, g in enumerate(packet):
                    last = i == len(packet) - 1
                    before = seq.num_blocks
                    kernel(
                        seq,
                        labels,
                        g,
                        shot,
                        frame,
                        pad if last else raster,
                        output=volume_output
                        and not reversed_encode
                        and i == 0
                        and shot == 0,
                        reversed_encode=reversed_encode,
                        reference=reference,
                    )
                    if frame is not None and first_image_shot is None:
                        first_image_shot = (before + 1, seq.num_blocks - before)
        return seq, first_image_shot

    def reference() -> pp.Sequence:
        """Design one volume with the phase encode reversed, after its dummies."""
        seq, _ = play(frames=[0], reversed_encode=True)
        define(
            seq,
            Name=f"{NAME}_reference",
            TE=echo_time,
            TR=repetition_time,
            EchoSpacing=epi.esp,
            kSpaceCenterLine=n_y // 2,
            **({"MultibandFactor": multiband} if multiband > 1 else {}),
        )
        return seq

    prescans = [calibrate()] if gre is not None else []
    if acs is not None:
        prescans.append(fleet())
    else:
        prescans.append(reference())
    seq, (plot_tr_start, plot_tr_size) = play(frames=range(volumes))

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV (compat=False: the lines are sampled on the ramps).
    definitions = {
        "FOV": [*fov, slab_thickness],
        "Matrix": [n_x, n_y, n_slices],
        "Name": NAME,
        "TE": echo_time,
        "TR": repetition_time,
        "EchoSpacing": epi.esp,
        "EPIFactor": epi.etl,
        "PlotTRsize": plot_tr_size,
        "PlotTRstart": plot_tr_start,
        "kSpaceCenterLine": n_y // 2,
        "SlicePositions": positions.tolist(),
        "SliceThickness": excited_thickness,
        "SliceGap": slice_gap,
    }
    if multiband > 1:
        definitions["MultibandFactor"] = multiband
    if len(packets) == 1:
        # Excitation time of each slice from the start of its volume.
        (packet,) = packets
        when = {g: i * shot_duration for i, g in enumerate(packet)}
        definitions["SliceTiming"] = [when[s % n_groups] for s in range(n_slices)]
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    if b_table is not None:
        pp.io.record_diffusion_table(seq, b_table, g_table)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=sequences.gain_calibration_readouts(seq),
    )
    if field_pair:
        *_, reversed_volume, forward_volume = _designed(
            system,
            fov_x=fov_x,
            fov_y=fov_y,
            n_x=n_x,
            n_y=n_y,
            n_slices=n_slices,
            slice_thickness=slice_thickness,
            slice_spacing=slice_spacing,
            flip_angle_deg=flip_angle_deg,
            readout_bandwidth_hz=readout_bandwidth_hz,
            ry=ry,
            partial_fourier_y=partial_fourier_y,
            n_shots=n_shots,
            multiband=1,
            fat_saturation=fat_saturation,
            n_dummy=n_dummy,
            readout_oversampling=readout_oversampling,
            n_acs_y=n_acs_y,
            phase_reference=phase_reference,
            navigator_lines=navigator_lines,
            b_value=b_value,
            diffusion_b0=diffusion_b0,
            spin_echo=True,
            calibration="gre",
            te=None,
            tr=None,
            n_frames=1,
            diffusion_directions=0,
            volume_output=False,
            field_pair=False,
            encode_sets=FIELD_PAIR_SETS,
        )
        forward_volume.set_definition(key="Name", value=f"{NAME}_field_forward")
        reversed_volume.set_definition(key="Name", value=f"{NAME}_field_reversed")
        prescans += [forward_volume, reversed_volume]
    return [*prescans, seq]


main = epi2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="epi_2d.seq"))
