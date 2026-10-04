"""Multi-slice 2D gradient-echo EPI, optionally segmented and multiband."""

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

    Shot ``s`` reads lines ``start + (s + i * n_shots) * ry``. The lattice keeps
    every line ``i`` with ``(i - n // 2) % ry == 0``, so the centre line is
    always read. The shots end on the lattice's last line and cover the lines
    from ``n - round(partial_fourier * n)`` up; when the shots cannot share
    them evenly, the extra lines extend below that, or are dropped where
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


NAME = "epi_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the selective pulses. The selection amplitude, which slice
#: offsets are converted against, is the pulse's measured bandwidth over the
#: thickness.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Centre lines read without blips at the start of every shot.
NAVIGATOR_LINES = 3
#: Dephasing left on the readout axis at the end of each shot, in cycles
#: across one voxel.
SPOILING_CYCLES = 4.0
#: Digital output marking every volume under ``volume_output``, and how
#: long it lasts (s); it never outlasts the excitation block it rides.
OUTPUT_CHANNEL = "ext1"
OUTPUT_DURATION = 1e-3


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
    n_dummy: int = 2,
    readout_oversampling: float = 1.0,
    n_acs_y: int = 24,
    volume_output: bool = False,
) -> list[pp.Sequence]:
    """Multi-slice 2D gradient-echo EPI: single-shot or segmented, optionally multiband.

    Every shot reads ``NAVIGATOR_LINES`` centre lines without blips
    (``NAV``) before its train, and every line carries ``REV`` for its read
    polarity. The trains are ramp-sampled with the blips on the read ramps.
    Segmented shots interleave on the phase-encode lattice, each delayed by
    a fraction of the echo spacing so the echo time grows smoothly across
    k-space; ``te`` is the echo time of the centre line.

    Slices (under ``multiband``, slice groups) are excited once per shot in
    packets, even ones first. With one frame, slices one TR cannot hold are
    dealt round-robin into packets; a time series must fit every slice in
    one. A multiband shot excites ``multiband`` slices at once and a
    blipped-CAIPI train encodes the band in ``PAR``.

    Two prescans are linked ahead of the imaging: the
    ``calibration``, a single-band gradient echo per slice over the central
    ``n_acs_y`` lines (``REF``), when undersampled or multiband; and the
    ``reference``, one volume with the phase encode reversed (``SET = 1``),
    for distortion correction.

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
        Excitation flip angle (degrees).
    te : float | None, default=None
        Echo time of the centre line (s). ``None`` is as short as the
        navigator and the train admit.
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
        Saturate fat before every shot.
    n_dummy : int, default=2
        Non-acquiring volumes before a time series; with one frame,
        non-acquiring shots per slice before each packet.
    readout_oversampling : float, default=1.0
        Readout oversampling factor, at least one.
    n_acs_y : int, default=24
        Phase-encode lines of the gradient-echo calibration.
    volume_output : bool, default=False
        Play a digital output on ``OUTPUT_CHANNEL`` at the first
        excitation of every volume, dummy volumes included.

    Returns
    -------
    list of pypulseqpp.Sequence
        The designed sequences in play order: the ``calibration`` when
        undersampled or multiband, the ``reference``, then the scan, which is
        the main sequence. ``sequences.write`` writes them as files linked
        through ``NextSequence``.

    Raises
    ------
    ValueError
        If ``partial_fourier_y`` is outside ``[0.5, 1]``, a count is below
        one, ``multiband`` does not divide ``n_slices``, the shots cannot
        share the lines, the TE is shorter than the train admits, or the TR
        cannot hold one slice, or every slice when ``n_frames`` is above
        one.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> *prescans, seq = sequences.epi2D_sequence(n_x=32, n_y=16, n_dummy=0)
    >>> [prescan.definitions["Name"] for prescan in prescans]
    ['epi_2d_reference']
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["EPIFactor"], seq.definitions["Name"]
    ([16.0], 'epi_2d')
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if not 0.5 <= partial_fourier_y <= 1.0:
        raise ValueError(
            f"partial_fourier_y must lie in [0.5, 1], got {partial_fourier_y}"
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
        "navigator_lines": NAVIGATOR_LINES,
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
        centers = [
            (g - (n_slices - 1) / 2 + (multiband - 1) / 2 * n_groups) * slice_step
            for g in range(n_groups)
        ]
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
    epi = shot_trains[0]
    echo_time = epi.echo_time

    fatsat = (
        sequences.FatSaturation(
            system, voxel_size_m=min(fov_x / n_x, fov_y / n_y, slice_thickness)
        )
        if fat_saturation
        else None
    )

    # A shot is the fat saturation, the train, and a closing delay of at
    # least one raster; the last shot of a packet waits out the cycle. A
    # cycle is one shot of every slice of a packet, and a volume is
    # n_shots cycles.
    fatsat_duration = fatsat.duration if fatsat is not None else 0.0
    shot_duration = fatsat_duration + epi.duration + raster
    cycle = None if tr is None else tr / n_shots
    per_packet = (
        n_groups if cycle is None else max(1, int(cycle / shot_duration + 1e-9))
    )
    packets = packets_of(n_groups, per_packet)
    if n_frames > 1 and len(packets) > 1:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms cannot hold the "
            f"{n_groups} excitations of a volume, {shot_duration * 1e3:.3f} ms each "
            f"per shot"
        )
    if cycle is None:
        cycle = max(map(len, packets)) * shot_duration
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
    dummy_cycles = n_dummy * (n_shots if n_frames > 1 else 1)
    output_pulse = pp.make_digital_output_pulse(
        OUTPUT_CHANNEL,
        duration=min(OUTPUT_DURATION, pp.calc_duration(epi.rf, epi.gz)),
        system=system,
    )

    # A gradient echo keeps EPI distortion out of the coil maps. An
    # undersampled scan calibrates from it, and a multiband scan always does.
    calibration = []
    if ry > 1 or multiband > 1:
        low = n_y // 2 - n_acs_y // 2
        calibration = list(range(max(low, 0), min(low + n_acs_y, n_y)))
    gre = None
    if calibration:
        gre = sequences.LineReadout2D(
            system,
            single.rf,
            single.gz,
            single.gz_reph,
            fov=fov,
            matrix=(n_x, n_y),
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=SPOILING_CYCLES,
            labels=("LIN",),
        )

    def define(seq: pp.Sequence, **definitions) -> None:
        for key, value in {
            "FOV": [*fov, slab_thickness],
            "Matrix": [n_x, n_y, n_slices],
            **definitions,
        }.items():
            seq.set_definition(key=key, value=value)

    def calibration_kernel(
        seq: pp.Sequence, labels: sequences.Labels, s: int, line: int
    ) -> None:
        """Add one single-band gradient echo of slice ``s`` at ``line``."""
        gre.rf.freq_offset = single.selection_amplitude * positions[s]
        gre.rf.phase_offset = -2 * np.pi * gre.rf.freq_offset * gre.rf.center
        gre.adc_labels.value = line
        ky = (line - n_y // 2) / (n_y / 2)
        seq.add_block(gre.rf, gre.gz, *labels(REF=1, SLC=s))
        wait_te = getattr(gre, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, gre.gz_reph)
            seq.add_block(gre.gx_pre, pp.scale_grad(gre.gy_pre, ky))
        else:
            seq.add_block(gre.gx_pre, pp.scale_grad(gre.gy_pre, ky), gre.gz_reph)
        seq.add_block(gre.gx, gre.adc, gre.adc_labels)
        seq.add_block(gre.gx_spoil, pp.scale_grad(gre.gy_rew, ky))

    def calibrate() -> pp.Sequence:
        """Design the gradient-echo calibration: every line of every slice."""
        seq, labels = pp.Sequence(system), sequences.Labels()
        order = [s for packet in packets_of(n_slices, n_slices) for s in packet]
        for s in order:
            for line in calibration:
                calibration_kernel(seq, labels, s, line)
        define(seq, Name=f"{NAME}_calibration")
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
    ) -> None:
        """Add one shot of slice (multiband group) ``g``; ``frame=None`` is a dummy.

        ``pad`` is the closing delay the shot would have without echo shift,
        in s. ``reversed_encode`` negates every phase-encode event and keeps
        the labels of the forward shot.
        """
        train = shot_trains[shot]
        train.rf.freq_offset = selection_amplitude * centers[g]
        train.rf.phase_offset = -2 * np.pi * train.rf.freq_offset * train.rf.center
        sign = -1.0 if reversed_encode else 1.0
        origin = origins[shot]
        acquire = frame is not None

        if not acquire:
            flags = {"ONCE": 1}
        else:
            flags = {"SLC": g, "REP": frame, "SEG": shot, "ONCE": 0}
            if multiband > 1:
                flags["SMS"] = 1
            if reversed_encode:
                flags["SET"] = 1
            if not n_dummy:
                del flags["ONCE"]
        label_events = labels(**flags)
        train.shot_labels[0].value = origin
        # The prewinder and the rewinder closing the shot are scaled alike.
        ky = sign * (origin - n_y // 2) / (n_y / 2)
        swap = {
            id(train.gy_pre): pp.scale_grad(train.gy_pre, ky),
            id(train.gy_rew): pp.scale_grad(train.gy_rew, ky),
        }
        if multiband > 1:
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
        step = shot * train.echo_shift_step
        wait_shift = getattr(train, "wait_shift", None)
        if wait_shift is not None:
            swap[id(wait_shift)] = pp.make_delay(wait_shift.delay + step)
        wait_tr = getattr(train, "wait_tr", None)
        closing = pad + (0.0 if wait_tr is None else wait_tr.delay) - step

        if fatsat is not None:
            for i, block in enumerate(fatsat.blocks):
                seq.add_block(*block, *(label_events if i == 0 else ()))
            label_events = []
        navigator = {id(event) for event in getattr(train, "gx_navigator", ())}
        readouts = {id(event) for event in train.gx}
        polarity = 0
        for index, block in enumerate(train.blocks):
            head = block[0]
            if head is wait_tr:
                continue
            events = [
                swap.get(id(event), event)
                for event in block
                if acquire or event.type != "adc"
            ]
            if index == 0:
                events += [*label_events, *([output_pulse] if output else [])]
            elif id(head) in navigator or id(head) in readouts:
                if acquire:
                    events += labels(NAV=int(id(head) in navigator), REV=polarity)
                polarity ^= 1
            seq.add_block(*events)
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
            for frame, shot in cycles:
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
        )
        return seq

    prescans = [calibrate()] if gre is not None else []
    prescans.append(reference())
    seq, (plot_tr_start, plot_tr_size) = play(frames=range(n_frames))

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
    return [*prescans, seq]


main = epi2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="epi_2d.seq"))
