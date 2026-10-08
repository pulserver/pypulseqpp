"""2D spiral spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The spiral densities ``density`` selects from.
DENSITIES = ("constant", "variable", "dual")

NAME = "se_spiral_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design shared by the excitation and the refocusing pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Dephasing each crusher beside the refocusing pulse winds, in cycles
#: across one voxel.
CRUSHER_CYCLES = 4.0
#: Dephasing left on the slice axis at the end of each repetition, in
#: cycles across one voxel.
SPOILING_CYCLES = 4.0
#: Exponent of the normalised radius, for variable density.
VARIABLE_DENSITY_POWER = 2.0
#: Normalised radius of the dual-density transition.
TRANSITION_RADIUS = 0.5


def se_spiral2d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    te: float | None = None,
    tr: float | None = 500e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    n_dummy: int | None = None,
    n_shots: int = 16,
    density: str = "constant",
    periphery_undersampling: float = 2.0,
    transition_speed: float = 12.0,
) -> pp.Sequence:
    """Multi-slice 2D spiral spin echo: one interleaf per excitation.

    A slice-selective SLR 90, one slice-selective SLR 180 between crushers
    half a TE later, and one outward interleaf starting at the echo, turned
    per shot by a rotation extension. The interleaves are designed and chosen
    as :mod:`gre_spiral2D_sequence` designs and chooses them, and the slices
    dealt into packets as :mod:`gre2D_sequence` deals them. Every acquisition
    carries its interleaf as ``LIN`` and its slice as ``SLC``.

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
    te : float | None, default=None
        Echo time (s), excitation centre to the start of the outward path,
        with the refocusing pulse at its midpoint. ``None`` is as short as
        possible.
    tr : float | None, default=0.5
        Repetition time between successive excitations of one slice (s).
        ``None`` is as short as possible, and puts every slice in one
        packet.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry : int, default=1
        Angular undersampling: one interleaf in every ``ry`` of the
        ``n_shots`` is played.
    n_dummy : int or None, default=None
        Non-acquiring repetitions, at the first interleaf's angle, before
        each packet.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    n_shots : int, default=16
        Interleaves that sample the centre of k-space at Nyquist.
    density : {'constant', 'variable', 'dual'}, default='constant'
        Constant pitch, a radial power-law transition to the periphery, or
        a logistic one.
    periphery_undersampling : float, default=2.0
        How much sparser the periphery is sampled than the centre, at
        least one. Unused at constant density.
    transition_speed : float, default=12.0
        Steepness of the dual-density transition.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``density`` is unknown, ``ry`` or ``periphery_undersampling`` is
        below one, or the TE or TR is shorter than the pulses and the
        readout take.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_spiral2D_sequence(n=32, n_shots=4, tr=None)
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if density not in DENSITIES:
        raise ValueError(f"density must be one of {DENSITIES}, got {density!r}")
    if ry < 1:
        raise ValueError(f"ry must be at least 1, got {ry}")
    if periphery_undersampling < 1:
        raise ValueError(
            f"periphery_undersampling must be at least 1, got {periphery_undersampling}"
        )

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

    # The centre is designed for n_shots interleaves and the periphery for
    # proportionally more, which is what spreads an interleaf's turns there.
    shaped = {}
    if density != "constant":
        shaped = {
            "inner_design_interleaves": n_shots,
            "outer_design_interleaves": n_shots * periphery_undersampling,
            "variable_density_power": VARIABLE_DENSITY_POWER,
            "transition_radius": TRANSITION_RADIUS,
            "transition_speed": transition_speed,
        }

    # TE is solved in two halves about the 180. The readout owns the second,
    # from the 180's centre to the echo; a delay before the 180 sets the
    # first, which is never shorter than the excitation's blocks allow.
    half_floor = exc.duration - exc.center + ref.center

    def readout(half_te: float | None):
        return sequences.SpiralReadout2D(
            system,
            ref.rf_ref,
            ref.gz,
            fov=fov,
            matrix=n,
            design_interleaves=n_shots,
            density=density,
            te=half_te,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=SPOILING_CYCLES,
            **shaped,
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

    # An interleaf covers a full turn, which the n_shots divide evenly.
    angles = 2 * np.pi * np.arange(0, n_shots, ry) / n_shots
    rotations = [pp.make_rotation(float(angle)) for angle in angles]

    # Slices one TR cannot hold are dealt round-robin into packets, even
    # slices of a packet first. Every shot closes with a pure delay: one
    # raster, and on the last slice of a packet whatever is left of the TR.
    shot = exc.duration + wait + ro.duration + raster
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

    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_spacing
    )
    slab_thickness = n_slices * (slice_thickness + slice_spacing) - slice_spacing
    slice_gap = slice_thickness + slice_spacing - exc.slice_thickness

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(s: int, arm: int | None, pad: float) -> None:
        """Add one spin echo of slice ``s`` reading ``arm``; ``None`` is a dummy.

        The readout's blocks, from the refocusing pulse on, are played as it
        laid them out, each turned to the interleaf's angle where it drives
        an in-plane gradient.
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

        acquire = arm is not None
        if acquire:
            label_events = labels(SLC=s, LIN=arm, ONCE=0)
        else:
            label_events = labels(SLC=s, ONCE=1)
        rotation = rotations[arm if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *label_events)
        seq.add_block(exc.gz_reph)
        if wait_half_te is not None:
            seq.add_block(wait_half_te)
        for block in ro.blocks:
            events = [event for event in block if acquire or event is not ro.adc]
            if any(getattr(event, "channel", None) in ("x", "y") for event in events):
                events.append(rotation)
            seq.add_block(*events)
        seq.add_block(pp.make_delay(pad))

    arms = [None] * n_dummy + list(range(len(angles)))
    for packet in packets:
        for arm in arms:
            for i, s in enumerate(packet):
                last = i == len(packet) - 1
                kernel(s, arm, pads[len(packet)] if last else raster)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov, fov, slab_thickness],
        "Matrix": [n, n, n_slices],
        "Name": NAME,
        "TE": echo_time,
        "TR": repetition_time,
        "Trajectory": "spiral",
        "NumArms": len(angles),
        "kSpaceCenterSample": ro.center_sample,
        "SlicePositions": positions.tolist(),
        "SliceThickness": exc.slice_thickness,
        "SliceGap": slice_gap,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = se_spiral2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_spiral_2d.seq"))
