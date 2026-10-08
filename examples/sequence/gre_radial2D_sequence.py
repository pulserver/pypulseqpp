"""RF-spoiled 2D radial gradient echo, multi-slice."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "gre_radial_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the selective pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Quadratic RF spoiling phase increment (degrees), counted per slice.
RF_SPOILING_INCREMENT_DEG = 117.0
#: Dephasing left on the slice axis at the end of each repetition, in
#: cycles across one voxel.
SPOILING_CYCLES = 4.0


def gre_radial2d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    flip_angle_deg: float = 12.0,
    te: float | None = None,
    tr: float | None = 20e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    n_dummy: int | None = None,
    readout_oversampling: float = 2.0,
) -> pp.Sequence:
    """RF-spoiled, multi-slice 2D radial gradient echo: one full spoke per repetition.

    One spoke waveform is turned per shot by a rotation extension. The Nyquist
    set is ``ceil(pi / 2 * n)`` spokes spread evenly over half a turn, and
    every ``ry``-th of them is played, in order. Slices are dealt into packets,
    and ordered within one, as :mod:`gre2D_sequence` deals them. Every
    acquisition carries its spoke as ``LIN`` and its slice as ``SLC``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov : float, default=0.22
        Isotropic in-plane field of view (m).
    n : int, default=128
        In-plane matrix size; a spoke reads it edge to edge.
    n_slices : int, default=1
        Number of slices.
    slice_thickness : float, default=0.005
        Slice thickness (m).
    slice_spacing : float, default=0.0
        Gap between adjacent slices (m); zero is contiguous.
    flip_angle_deg : float, default=12.0
        Excitation flip angle (degrees).
    te : float | None, default=None
        Echo time to the spoke's centre crossing (s). ``None`` is as short
        as possible.
    tr : float | None, default=0.02
        Repetition time between successive excitations of one slice (s).
        ``None`` is as short as possible, and puts every slice in one
        packet.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry : int, default=1
        Angular undersampling: one spoke in every ``ry`` of the Nyquist set
        is played.
    n_dummy : int or None, default=None
        Non-acquiring repetitions, at the first spoke's angle, before each
        packet.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    readout_oversampling : float, default=2.0
        Readout oversampling factor, at least one.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``ry`` is below one, or the TR cannot hold one slice.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_radial2D_sequence(n=32, tr=None)
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

    exc = sequences.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )
    ro = sequences.RadialReadout2D(
        system,
        exc.rf,
        exc.gz,
        exc.gz_reph,
        fov=fov,
        matrix=n,
        te=te,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=SPOILING_CYCLES,
    )
    # A full spoke covers half a turn, which the Nyquist set divides evenly.
    n_nyquist = math.ceil(np.pi / 2 * n)
    angles = np.pi * np.arange(0, n_nyquist, ry) / n_nyquist
    rotations = [pp.make_rotation(float(angle)) for angle in angles]

    # Slices one TR cannot hold are dealt round-robin into packets, even
    # slices of a packet first. Every shot closes with a pure delay: one
    # raster, and on the last slice of a packet whatever is left of the TR.
    raster = system.block_duration_raster
    shot = ro.duration + raster
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
            f"the requested TR of {cycle * 1e3:.3f} ms is shorter than the "
            f"{shot * 1e3:.3f} ms one slice takes"
        )
    packet_time = {size: size * shot - raster + pad for size, pad in pads.items()}
    repetition_time = max(packet_time.values())
    if n_dummy is None:
        n_dummy = sequences.steady_state_dummies(repetition_time, flip_angle_deg)

    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_spacing
    )
    slab_thickness = n_slices * (slice_thickness + slice_spacing) - slice_spacing
    slice_gap = slice_thickness + slice_spacing - exc.slice_thickness

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(s: int, spoke: int | None, phase: float, pad: float) -> None:
        """Add one excitation of slice ``s`` reading ``spoke``; ``None`` is a dummy.

        The readout's blocks after the pulse are played as it laid them out,
        each turned to the spoke's angle where it drives an in-plane gradient.
        """
        exc.rf.freq_offset = exc.selection_amplitude * positions[s]
        exc.rf.phase_offset = phase - 2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ro.adc.phase_offset = phase

        acquire = spoke is not None
        if acquire:
            label_events = labels(SLC=s, LIN=spoke, ONCE=0)
        else:
            label_events = labels(SLC=s, ONCE=1)
        rotation = rotations[spoke if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *label_events)
        for block in ro.blocks[1:]:
            events = [event for event in block if acquire or event is not ro.adc]
            if any(getattr(event, "channel", None) in ("x", "y") for event in events):
                events.append(rotation)
            seq.add_block(*events)
        seq.add_block(pp.make_delay(pad))

    spokes = [None] * n_dummy + list(range(len(angles)))
    phases = pp.make_rf_spoiling_schedule(
        len(spokes), increment=np.deg2rad(RF_SPOILING_INCREMENT_DEG)
    )
    for packet in packets:
        for spoke, phase in zip(spokes, phases, strict=True):
            for i, s in enumerate(packet):
                last = i == len(packet) - 1
                pad = pads[len(packet)] if last else raster
                kernel(s, spoke, phase, pad)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov, fov, slab_thickness],
        "Matrix": [n, n, n_slices],
        "Name": NAME,
        "TE": ro.echo_time,
        "TR": repetition_time,
        "Trajectory": "radial",
        "NumSpokes": len(angles),
        "kSpaceCenterSample": ro.center_sample,
        "SlicePositions": positions.tolist(),
        "SliceThickness": exc.slice_thickness,
        "SliceGap": slice_gap,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = gre_radial2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_radial_2d.seq"))
