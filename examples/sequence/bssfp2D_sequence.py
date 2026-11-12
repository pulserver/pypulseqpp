"""Balanced SSFP 2D Cartesian, multi-slice, with optional cardiac cine gating."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The cardiac gating modes ``gating`` selects from.
GATINGS = ("none", "retrospective", "prospective")

NAME = "bssfp_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: Default SLR design of the selective pulse: short, at a time-bandwidth
#: product of 2, as product balanced SSFP excites. At the same flip angle it
#: deposits about half the energy of a pulse at a time-bandwidth product of 4,
#: and a slice played on its own has no neighbour for a softer profile to
#: excite. The selection amplitude, which slice offsets are converted against,
#: is the pulse's measured bandwidth over the thickness.
PULSE_DURATION = 0.48e-3
TIME_BW_PRODUCT = 2.0
#: Excitations over which a slice's flip angle ramps linearly up to its own,
#: the first of its dummies: a catalyst that damps the oscillating transient
#: off resonance better than a half flip half a repetition ahead, and leaves
#: the repetition free of the half flip's timing.
CATALYST_RAMP = 20
#: Distance from the balanced steady state the default dummies leave.
STEADY_STATE_TOLERANCE = 0.05
#: Physiological signal a prospective heartbeat waits for.
TRIGGER_CHANNEL = "physio1"
#: Dummies a prospective slice opens with when ``n_dummy`` is None: its first
#: heartbeat has to hold them with its cardiac phases, so they are its
#: catalyst ramp alone.
PROSPECTIVE_DUMMIES = CATALYST_RAMP


def prospective_starts(
    segments: list[int],
    n_phases: int,
    n_dummy: int,
    rr: float,
    tr: float,
    trigger_delay: float = 0.0,
) -> list[int]:
    """Return the repetition at which each heartbeat's cardiac phases start under ``prospective`` gating.

    Repetitions are counted from the slice's first excitation, which its one
    trigger starts. The slice then runs free: the first
    heartbeat's phases follow the dummies and ``trigger_delay`` to the nearest
    repetition, and those of heartbeat ``s`` start ``s`` nominal heartbeats
    and ``trigger_delay`` after the trigger, to the nearest repetition, with
    repetitions that do not acquire between.

    Parameters
    ----------
    segments : list of int
        Phase-encode lines of each heartbeat's segment.
    n_phases : int
        Cardiac phases each heartbeat acquires.
    n_dummy : int
        Repetitions played without acquiring at the start of the slice.
    rr : float
        Nominal heartbeat (s).
    tr : float
        Repetition time (s).
    trigger_delay : float, default=0.0
        Time from the trigger to the first excitation's place in the first
        heartbeat, played as repetitions that do not acquire (s).

    Examples
    --------
    >>> prospective_starts([12, 12, 4], n_phases=10, n_dummy=10, rr=1.0, tr=5e-3)
    [10, 200, 400]
    """
    lead = n_dummy + round(trigger_delay / tr)
    starts, taken = [], lead
    for s, views in enumerate(segments):
        start = max(taken, round((s * rr + trigger_delay) / tr)) if s else lead
        starts.append(start)
        taken = start + n_phases * views
    return starts


def bssfp2d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 300e-3,
    fov_y: float = 300e-3,
    n_x: int = 192,
    n_y: int = 192,
    n_slices: int = 1,
    slice_thickness: float = 6e-3,
    slice_spacing: float = 0.0,
    flip_angle_deg: float = 45.0,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    pulse_duration: float = PULSE_DURATION,
    time_bw_product: float = TIME_BW_PRODUCT,
    ry: int = 1,
    partial_fourier_y: float = 1.0,
    n_phases: int = 25,
    n_dummy: int | None = None,
    readout_oversampling: float = 1.0,
    n_acs_y: int = 24,
    gating: str = "none",
    heart_rate_bpm: float = 60.0,
    views_per_segment: int = 12,
    trigger_delay: float = 0.0,
) -> pp.Sequence:
    """Balanced SSFP 2D Cartesian: one complete train per slice, optionally cardiac-gated.

    Every axis returns to k = 0 between pulse centres and TE is TR/2
    (:class:`BssfpReadout2D`). Excitations alternate between phase pi and 0,
    the ADC following, and the first ``min(CATALYST_RAMP, n_dummy)`` of a
    slice ramp the flip angle linearly up to ``flip_angle_deg``, a catalyst
    that damps the oscillating transient. ``ONCE`` marks the slice's first
    excitation (1), the train (0) and the closing rewind (2), so the whole
    slice is one repetition of the scan. Slices play one after another, since
    a steady state does not survive interleaving.

    Without gating, each slice's lines are played ``n_phases`` times in a
    row, the steady state carried from one frame into the next with no
    dummies between them, the frame counted in ``PHS``. With gating, the
    phase-encode lines are played ``views_per_segment`` at a time. Under
    ``retrospective`` gating each segment is cycled as many times as a full
    segment fits in a heartbeat, the same number for every segment, so each
    cycle samples every line; the interpreter's ECG log bins them into cardiac
    phases. Under ``prospective`` gating each slice's first excitation waits
    for one trigger event on ``TRIGGER_CHANNEL``, and the slice then runs free
    at the nominal heart rate: every heartbeat plays its segment once per
    cardiac phase, and repetitions that do not acquire fill the rest of it
    and the trigger delay (:func:`prospective_starts`), so the steady state
    is kept from one heartbeat into the next and a heart rate away from the
    nominal one drifts against the phases. A slice held in one heartbeat is
    a triggered single shot. A gated slice's blocks join on the slice axis
    (``bridge``), so the gated train has no rest in it and its timing against
    the heartbeat is the designed one. Without gating, every repetition
    opens from rest and closes to it, so a scanner that prepares a segment
    between blocks at rest plays one repetition per segment and adds the same
    preparation time to every repetition time. With dummies, each such slice
    is two files, its catalyst and dummies and then its frames, each
    repeating one repetition. Acquisitions carry ``LIN``, ``SLC``, the segment as
    ``SEG`` and the frame, the cardiac phase or the cycle within the
    heartbeat as ``PHS``; calibration lines are marked ``IMA``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x, fov_y : float, default=0.3
        Field of view along the readout and the phase encode (m).
    n_x : int, default=192
        Readout matrix size.
    n_y : int, default=192
        Phase-encode matrix size.
    n_slices : int, default=1
        Number of slices, each acquired as its own complete train.
    slice_thickness : float, default=0.006
        Slice thickness (m).
    slice_spacing : float, default=0.0
        Gap between adjacent slices (m); zero is contiguous.
    flip_angle_deg : float, default=45.0
        Excitation flip angle (degrees).
    tr : float | None, default=None
        Repetition time (s); TE is TR/2. ``None`` is as short as possible.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    pulse_duration : float, default=0.00048
        Duration of the selective pulse (s); a longer pulse deposits less RF
        energy at the same flip angle and lengthens the repetition.
    time_bw_product : float, default=2.0
        Time-bandwidth product of the selective pulse; a higher value gives a
        squarer slice profile at more RF energy.
    ry : int, default=1
        Phase-encode undersampling: one line in every ``ry`` is acquired,
        the centre line among them.
    partial_fourier_y : float, default=1.0
        Fraction of the phase-encode extent acquired, in ``[0.75, 1]``.
    n_phases : int, default=25
        Frames. Without gating, the times each slice's k-space is acquired
        back to back, a free-running dynamic series; under ``prospective``
        gating, the cardiac phases each heartbeat acquires. Unused under
        ``retrospective`` gating, where the heartbeat sets the cycles.
    n_dummy : int | None, default=None
        Repetitions played without acquiring at the start of each slice, the
        first ``CATALYST_RAMP`` of them its catalyst ramp. ``None`` is as many
        as bring the balanced steady state within ``STEADY_STATE_TOLERANCE``
        (:func:`~pypulseqpp.sequences.steady_state_dummies` at
        :data:`~pypulseqpp.sequences.STEADY_STATE_T2_S`), at most one
        heartbeat under ``retrospective`` gating, or
        :data:`PROSPECTIVE_DUMMIES` under ``prospective`` gating.
    readout_oversampling : float, default=1.0
        Readout oversampling factor, at least one.
    n_acs_y : int, default=24
        Fully sampled calibration lines at the centre of k-space, acquired
        ahead of the rest when ``ry > 1``.
    gating : {'none', 'retrospective', 'prospective'}, default='none'
        No gating; each segment cycled over one heartbeat; or each slice
        triggered once and every nominal heartbeat acquired once per cardiac
        phase.
    heart_rate_bpm : float, default=60.0
        Nominal heart rate (beats per minute) the segments are timed to.
    views_per_segment : int, default=12
        Phase-encode lines per segment, one segment per heartbeat. Unused
        without gating.
    trigger_delay : float, default=0.0
        Time from a slice's prospective trigger to its first excitation's
        place in its first heartbeat, played as repetitions that do not
        acquire, to the nearest repetition (s).

    Returns
    -------
    pypulseqpp.Sequence or list of pypulseqpp.Sequence
        The designed sequence; without gating and with dummies, the chain of
        each slice's dummies and frames, in play order, slice by slice.

    Raises
    ------
    ValueError
        If ``gating`` is unknown, ``partial_fourier_y`` is outside
        ``[0.75, 1]``, a count is below one, a prospective heartbeat cannot
        hold its phases, or the TR is shorter than the balanced repetition.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.bssfp2D_sequence(
    ...     n_x=64, n_y=16, readout_bandwidth_hz=50e3, n_dummy=0
    ... )
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if gating not in GATINGS:
        raise ValueError(f"gating must be one of {GATINGS}, got {gating!r}")
    if not 0.75 <= partial_fourier_y <= 1.0:
        raise ValueError(
            f"partial_fourier_y must lie in [0.75, 1], got {partial_fourier_y}"
        )
    for name, count in (
        ("ry", ry),
        ("n_phases", n_phases),
        ("views_per_segment", views_per_segment),
    ):
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")

    fov = (fov_x, fov_y)
    exc = sequences.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=pulse_duration,
        rephase=False,
        time_bw_product=time_bw_product,
    )
    # Solving the balance and TE = TR/2 is the feasibility check: an
    # unreachable TR raises ValueError here.
    ro = sequences.BssfpReadout2D(
        system,
        exc.rf,
        exc.gz,
        fov=fov,
        matrix=(n_x, n_y),
        tr=tr,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
        half_flip_prep=False,
        bridge=gating != "none",
    )
    # A gated slice is bridged: its blocks join on the slice axis, so that a
    # scanner preparing segments between blocks at rest plays nothing between
    # its repetitions and the train keeps its designed timing against the
    # heartbeat. It opens from rest, on its trigger under prospective gating,
    # and its last rewind closes to rest. A free-running slice is not: every
    # repetition rests at its ends, so each plays as a segment of its own and
    # every repetition time carries the same preparation, where a segment
    # holding a frame or more would lengthen only the repetition at its end and
    # set off a transient in the steady state there.
    nominal = ro.rf.amplitude
    rr = 60.0 / heart_rate_bpm
    if n_dummy is None:
        n_dummy = (
            PROSPECTIVE_DUMMIES
            if gating == "prospective"
            else sequences.steady_state_dummies(
                ro.tr,
                flip_angle_deg,
                t2_s=sequences.STEADY_STATE_T2_S,
                tolerance=STEADY_STATE_TOLERANCE,
            )
        )
        if gating == "retrospective":
            # One heartbeat: the residual approach spreads over every
            # segment and cardiac phase, as in product cine.
            n_dummy = min(n_dummy, max(1, round(rr / ro.tr)))
    trigger_event = pp.make_trigger(TRIGGER_CHANNEL, system=system)

    calibrating, imaging = pp.make_cartesian_axis_sampling(
        n_y, ry, n_acs_y, partial_fourier=partial_fourier_y
    )
    # The calibration block leads, so a reconstruction can estimate
    # coil sensitivities while the rest is still arriving.
    lines = [*calibrating, *imaging]
    calibration = set(calibrating)
    # A train is a list of (line, segment, phase) repetitions, line None for
    # one that does not acquire.
    segments = (
        [lines]
        if gating == "none"
        else [
            lines[i : i + views_per_segment]
            for i in range(0, len(lines), views_per_segment)
        ]
    )
    train = [(None, 0, 0)] * n_dummy
    cycles = max(1, round(rr / (views_per_segment * ro.tr)))
    starts = (
        prospective_starts(
            [len(segment) for segment in segments],
            n_phases,
            n_dummy,
            rr,
            ro.tr,
            trigger_delay,
        )
        if gating == "prospective"
        else []
    )
    for s, segment in enumerate(segments):
        if gating == "none":
            # A free-running series: the frames follow one another with the
            # steady state kept, so no dummies between them.
            train += [(line, 0, f) for f in range(n_phases) for line in segment]
        elif gating == "retrospective":
            train += [(line, s, c) for c in range(cycles) for line in segment]
        else:
            acquired = n_phases * len(segment) * ro.tr + trigger_delay
            if s == 0:
                acquired += n_dummy * ro.tr
            if acquired > rr + 1e-9:
                raise ValueError(
                    f"{n_phases} cardiac phases of {len(segment)} lines take "
                    f"{acquired * 1e3:.0f} ms, longer than the "
                    f"{rr * 1e3:.0f} ms heartbeat"
                )
            # The heartbeat is filled to its nominal length, and the first
            # one to the trigger delay, with repetitions that do not acquire:
            # the slice has one trigger, at its first excitation.
            train += [(None, 0, 0)] * (starts[s] - len(train))
            train += [(line, s, f) for f in range(n_phases) for line in segment]
    n_segments = len(segments)
    # A free-running series with dummies is a chain of two files per slice:
    # the catalyst and dummies, then the frames.
    chained = gating == "none" and n_dummy > 0

    slice_step = slice_thickness + slice_spacing
    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * slice_step
    slab_thickness = n_slices * slice_step - slice_spacing
    slice_gap = slice_step - exc.slice_thickness

    def ky_of(line: int | None) -> float:
        return 0.0 if line is None else (line - n_y // 2) / (n_y / 2)

    n_ramp = min(CATALYST_RAMP, n_dummy)

    def rewind(target: pp.Sequence, ky: float, closes: bool, ends: bool) -> None:
        """Rewind the encode ``ky``; ``closes`` brings the slice axis to rest, ``ends`` marks the train's end."""
        target.add_block(
            ro.gx_rew,
            pp.scale_grad(ro.gy_rew, ky),
            ro.gz_close if closes and ro.bridged else ro.gz_rew,
            *(ro.end_labels if ends else []),
        )

    def repetition(
        target: pp.Sequence,
        s: int,
        shot: int,
        line: int | None,
        segment: int,
        phase: int,
        opens: bool,
    ) -> None:
        """Excite slice ``s`` and read ``line``, opening from rest with ``opens``; ``line=None`` plays a dummy.

        The slice's first repetition carries ``ONCE`` and, under prospective
        gating, the trigger, and the catalyst's repetitions ramp the flip angle.
        """
        rf = ro.rf
        rf.freq_offset = exc.selection_amplitude * positions[s]
        alternation = np.pi * ((shot + 1) % 2)
        rf.phase_offset = alternation - 2 * np.pi * rf.freq_offset * rf.center
        ro.adc.phase_offset = alternation
        rf.amplitude = nominal * (shot + 1) / n_ramp if shot < n_ramp else nominal
        first = shot == 0
        opening = [
            *(ro.prep_labels if first else []),
            *([trigger_event] if first and gating == "prospective" else []),
        ]
        target.add_block(rf, ro.gz_open if opens and ro.bridged else ro.gz, *opening)
        rf.amplitude = nominal
        ky = ky_of(line)
        once = ro.train_labels if first else []
        if line is None:
            target.add_block(ro.gx, pp.scale_grad(ro.gy_pre, ky), ro.gz_pre, *once)
            return
        label_events = labels(
            LIN=line, SLC=s, SEG=segment, PHS=phase, IMA=line in calibration
        )
        target.add_block(
            ro.gx,
            ro.adc,
            pp.scale_grad(ro.gy_pre, ky),
            ro.gz_pre,
            *label_events,
            *once,
        )

    # Each slice's whole train is played before the next slice's.
    seq, labels = pp.Sequence(system), sequences.Labels()
    chain = []
    for s in range(n_slices):
        target = pp.Sequence(system) if chained else seq
        first = target.num_blocks
        # The slice's labels restart, as a ONCE through labels() would.
        labels.restart()
        previous = None
        for shot, (line, segment, phase) in enumerate(train):
            # The slice opens from rest, and chained it rests ahead of every
            # frame as well.
            opens = shot == 0 or (
                chained and shot >= n_dummy and (shot - n_dummy) % len(lines) == 0
            )
            if previous is not None:
                rewind(target, previous, closes=opens, ends=False)
            if chained and shot == n_dummy:
                chain.append(target)
                target = pp.Sequence(system)
                labels.restart()
            repetition(target, s, shot, line, segment, phase, opens=opens)
            previous = ky_of(line)
        rewind(target, previous, closes=True, ends=not chained)
        if chained:
            chain.append(target)
        if s == 0:
            plot_tr_start = first + 1
            plot_tr_size = target.num_blocks - first
    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [*fov, slab_thickness],
        "Matrix": [n_x, n_y, n_slices],
        "Name": NAME,
        "TE": ro.te,
        "TR": ro.tr,
        "Gating": gating,
        "PlotTRsize": plot_tr_size,
        "PlotTRstart": plot_tr_start,
        "kSpaceCenterLine": n_y // 2,
        "kSpaceCenterSample": ro.center_sample,
        "SlicePositions": positions.tolist(),
        "SliceThickness": exc.slice_thickness,
        "SliceGap": slice_gap,
    }
    if gating != "none":
        definitions.update(
            HeartRate=heart_rate_bpm,
            ViewsPerSegment=views_per_segment,
            NumSegments=n_segments,
        )
    if gating == "prospective":
        definitions.update(CardiacPhases=n_phases, TriggerDelay=trigger_delay)
    if chained:
        # Each file repeats one repetition, so a scanner that segments a
        # repetition plays every repetition as the same segment; a frame is
        # what the frames file plots.
        frame = 3 * len(lines)
        for index, part in enumerate(chain):
            catalyst = index % 2 == 0
            size = part.num_blocks if catalyst else frame
            stated = definitions | {"PlotTRstart": 1, "PlotTRsize": size}
            if catalyst:
                stated["Name"] = f"{NAME}_catalyst"
            for key, value in stated.items():
                part.set_definition(key=key, value=value)
            if not catalyst:
                part.set_definition(
                    key="NumGainCalibrationReadouts",
                    value=sequences.gain_calibration_readouts(part),
                )
        return chain
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=sequences.gain_calibration_readouts(seq),
    )
    return seq


main = bssfp2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="bssfp_2d.seq"))
