"""2D PROPELLER spin echo, multi-slice."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "se_propeller_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design shared by the excitation and the refocusing pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Readout oversampling factor.
READOUT_OVERSAMPLING = 2.0
#: Dephasing each crusher beside the refocusing pulse winds, in cycles
#: across one voxel.
CRUSHER_CYCLES = 4.0
#: Dephasing left on the slice axis at the end of each repetition, in
#: cycles across the slice.
SPOILING_CYCLES = 4.0


def se_propeller2d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    te: float | None = 15e-3,
    tr: float | None = 500e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    n_dummy: int | None = None,
    blade_width: int = 16,
) -> pp.Sequence:
    """Multi-slice 2D PROPELLER spin echo: one blade line per excitation.

    A slice-selective SLR 90, one slice-selective SLR 180 between crushers
    half a TE later, and one frequency-encoded line at the echo. A blade is
    ``blade_width`` lines centred on k = 0, turned by a rotation extension;
    blades are chosen as :mod:`gre_propeller2D_sequence` chooses them, and the
    slices dealt into packets as :mod:`gre2D_sequence` deals them. Every
    acquisition carries its line as ``LIN``, its blade as ``SEG`` and its
    slice as ``SLC``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov : float, default=0.22
        Isotropic in-plane field of view (m).
    n : int, default=128
        In-plane matrix size.
    n_slices : int, default=1
        Number of slices.
    slice_thickness : float, default=0.005
        Slice thickness (m).
    slice_spacing : float, default=0.0
        Gap between adjacent slices (m); zero is contiguous.
    te : float | None, default=0.015
        Echo time (s), excitation centre to echo, with the refocusing
        pulse at its midpoint. ``None`` is as short as possible.
    tr : float | None, default=0.5
        Repetition time between successive excitations of one slice (s).
        ``None`` is as short as possible, and puts every slice in one
        packet.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry : int, default=1
        Angular undersampling: one blade in every ``ry`` of the Nyquist set
        is played.
    n_dummy : int or None, default=None
        Non-acquiring repetitions, at the first blade's angle, before each
        packet.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    blade_width : int, default=16
        Phase-encode lines per blade, at most ``n``.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``ry`` is below one, ``blade_width`` is outside ``[1, n]``, or
        the TE or TR is shorter than the pulses and the readout take.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_propeller2D_sequence(n=32, blade_width=8, te=None, tr=None)
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if ry < 1:
        raise ValueError(f"ry must be at least 1, got {ry}")
    if not 1 <= blade_width <= n:
        raise ValueError(f"blade_width must lie in [1, {n}], got {blade_width}")

    exc = sequences.SpatialSelectiveExcitation(
        system,
        90.0,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )
    # A nonselective 180 would invert every slice of the packet whichever
    # one it refocuses, so the refocusing pulse selects the slice too.
    ref = sequences.SpatialSelectiveRefocusing(
        system,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        spoiling_cycles=CRUSHER_CYCLES,
    )
    ref_phase = float(ref.rf_ref.phase_offset)
    # The in-plane axes turn with the blade, so a spoiler on them would
    # point a different way from one blade to the next: it plays on z.
    gz_spoil, _, _ = pp.make_crusher(
        SPOILING_CYCLES, slice_thickness, "z", system=system
    )

    # TE is solved in two halves about the 180. The readout owns the second,
    # from the 180's centre to the echo; a delay before the 180 sets the
    # first, which is never shorter than the excitation's blocks allow.
    half_floor = exc.duration - exc.center + ref.center

    def readout(half_te: float | None):
        return sequences.LineReadout2D(
            system,
            ref.rf_ref,
            ref.gz,
            fov=(fov, fov),
            matrix=(n, n),
            te=half_te,
            oversampling=READOUT_OVERSAMPLING,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )

    te_min = 2 * max(readout(None).echo_time, half_floor)
    if te is not None and te < te_min - 1e-9:
        raise ValueError(
            f"TE {te * 1e3:.2f} ms is shorter than the {te_min * 1e3:.2f} ms "
            "the excitation and the readout admit"
        )
    # The 180 must sit midway, so both halves are solved on the block raster:
    # the excitation half waits a whole number of rasters, and the readout
    # is asked for the same span.
    raster = system.block_duration_raster
    target = (te_min if te is None else te) / 2 - half_floor
    wait = pp.ceil_to_raster(max(target, 0.0) - 1e-9, raster)
    ro = readout(half_floor + wait)
    wait_half_te = pp.make_delay(wait) if wait > 0 else None
    echo_time = 2 * (half_floor + wait)

    # A blade turned by half a turn is the same blade, which the Nyquist
    # set divides evenly.
    n_nyquist = math.ceil(np.pi * n / (2 * blade_width))
    angles = np.pi * np.arange(0, n_nyquist, ry) / n_nyquist
    rotations = [pp.make_rotation(float(angle)) for angle in angles]

    # Slices one TR cannot hold are dealt round-robin into packets, even
    # slices of a packet first. Every shot closes with a pure delay: one
    # raster, and on the last slice of a packet whatever is left of the TR.
    spoil = pp.ceil_to_raster(pp.calc_duration(gz_spoil), raster)
    shot = exc.duration + wait + ro.duration + spoil + raster
    per_packet = n_slices if tr is None else max(1, int(tr / shot + 1e-9))
    n_packets = -(-n_slices // per_packet)
    dealt = [range(start, n_slices, n_packets) for start in range(n_packets)]
    packets = [[*packet[::2], *packet[1::2]] for packet in dealt]
    cycle = tr if tr is not None else max(map(len, packets)) * shot
    pads = {
        size: pp.round_to_raster(cycle - size * shot, raster) + raster
        for size in {len(packet) for packet in packets}
    }
    if min(pads.values()) < raster:
        raise ValueError(
            f"TR {cycle * 1e3:.1f} ms is shorter than one repetition takes "
            f"({shot * 1e3:.1f} ms)"
        )
    packet_time = {size: size * shot - raster + pad for size, pad in pads.items()}
    repetition_time = max(packet_time.values())
    if n_dummy is None:
        n_dummy = sequences.steady_state_dummies(repetition_time, 90.0)

    views = [
        (blade, line) for blade in range(len(angles)) for line in range(blade_width)
    ]
    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_spacing
    )
    slab_thickness = n_slices * (slice_thickness + slice_spacing) - slice_spacing
    slice_gap = slice_thickness + slice_spacing - exc.slice_thickness

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(s: int, view: tuple[int, int] | None, pad: float) -> None:
        """Add one spin echo of slice ``s`` at ``(blade, line)``; ``None`` is a dummy.

        A dummy plays the first blade's centre line without its ADC. Every
        block that drives an in-plane gradient carries the blade's rotation.
        """
        position = positions[s]
        # Each pulse selects at its own plateau; a crushed refocusing
        # gradient's amplitude is its crusher peak.
        exc.rf.freq_offset = exc.selection_amplitude * position
        exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ref.rf_ref.freq_offset = ref.selection_amplitude * position
        ref.rf_ref.phase_offset = (
            ref_phase - 2 * np.pi * ref.rf_ref.freq_offset * ref.rf_ref.center
        )

        if view is None:
            blade, ky = 0, 0.0
            label_events = labels(SLC=s, ONCE=1)
        else:
            blade, line = view
            ky = (line - blade_width // 2) / (n / 2)
            label_events = labels(SLC=s, SEG=blade, LIN=line, ONCE=0)
        rotation = rotations[blade]

        seq.add_block(exc.rf, exc.gz, *label_events)
        seq.add_block(exc.gz_reph)
        if wait_half_te is not None:
            seq.add_block(wait_half_te)
        seq.add_block(ref.rf_ref, ref.gz)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), rotation)
        seq.add_block(ro.gx, *([] if view is None else [ro.adc]), rotation)
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), rotation)
        seq.add_block(gz_spoil)
        seq.add_block(pp.make_delay(pad))

    played = [None] * n_dummy + views
    for packet in packets:
        for view in played:
            for i, s in enumerate(packet):
                last = i == len(packet) - 1
                kernel(s, view, pads[len(packet)] if last else raster)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov, fov, slab_thickness],
        "Matrix": [n, n, n_slices],
        "Name": NAME,
        "TE": echo_time,
        "TR": repetition_time,
        "Trajectory": "propeller",
        "BladeWidth": blade_width,
        "NumBlades": len(angles),
        "kSpaceCenterLine": blade_width // 2,
        "kSpaceCenterSample": ro.center_sample,
        "SlicePositions": positions.tolist(),
        "SliceThickness": exc.slice_thickness,
        "SliceGap": slice_gap,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = se_propeller2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_propeller_2d.seq"))
