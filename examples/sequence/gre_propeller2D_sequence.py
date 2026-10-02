"""RF-spoiled 2D PROPELLER gradient echo, multi-slice."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "gre_propeller_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the selective pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Readout oversampling factor.
READOUT_OVERSAMPLING = 2.0
#: Quadratic RF spoiling phase increment (degrees), counted per slice.
RF_SPOILING_INCREMENT_DEG = 117.0
#: Dephasing left on the slice axis at the end of each repetition, in
#: cycles across the slice.
SPOILING_CYCLES = 4.0


def gre_propeller2d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    n_dummy: int = 16,
    blade_width: int = 16,
) -> pp.Sequence:
    """RF-spoiled, multi-slice 2D PROPELLER gradient echo: one blade line per repetition.

    A blade is ``blade_width`` Cartesian lines centred on k = 0, and every
    blade is the same lines turned by a rotation extension. The Nyquist set is
    ``ceil(pi * n / (2 * blade_width))`` blades spread evenly over half a turn;
    every ``ry``-th of them is played, in order, each line by line. Slices are
    dealt into packets, and ordered within one, as :mod:`gre2D_sequence` deals
    them. Every acquisition carries its line as ``LIN``, its blade as ``SEG``
    and its slice as ``SLC``.

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
    flip_angle_deg : float, default=12.0
        Excitation flip angle (degrees).
    te : float | None, default=0.008
        Echo time (s). ``None`` is as short as the readout admits.
    tr : float | None, default=0.25
        Repetition time between successive excitations of one slice (s).
        ``None`` is as short as possible, and puts every slice in one
        packet.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry : int, default=1
        Angular undersampling: one blade in every ``ry`` of the Nyquist set
        is played.
    n_dummy : int, default=16
        Non-acquiring repetitions, at the first blade's angle, before each
        packet.
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
        the TR cannot hold one slice.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre_propeller2D_sequence(n=32, blade_width=8, tr=None)
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
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )
    ro = sequences.LineReadout2D(
        system,
        exc.rf,
        exc.gz,
        exc.gz_reph,
        fov=(fov, fov),
        matrix=(n, n),
        te=te,
        oversampling=READOUT_OVERSAMPLING,
        readout_bandwidth_hz=readout_bandwidth_hz,
    )
    # The in-plane axes turn with the blade, so a spoiler on them would
    # point a different way from one blade to the next: it plays on z.
    gz_spoil, _, _ = pp.make_crusher(
        SPOILING_CYCLES, slice_thickness, "z", system=system
    )
    # A blade turned by half a turn is the same blade, which the Nyquist
    # set divides evenly.
    n_nyquist = math.ceil(np.pi * n / (2 * blade_width))
    angles = np.pi * np.arange(0, n_nyquist, ry) / n_nyquist
    rotations = [pp.make_rotation(float(angle)) for angle in angles]

    # Slices one TR cannot hold are dealt round-robin into packets, even
    # slices of a packet first. Every shot closes with a pure delay: one
    # raster, and on the last slice of a packet whatever is left of the TR.
    raster = system.block_duration_raster
    spoil = pp.ceil_to_raster(pp.calc_duration(gz_spoil), raster)
    shot = ro.duration + spoil + raster
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

    views = [
        (blade, line) for blade in range(len(angles)) for line in range(blade_width)
    ]
    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_spacing
    )
    slab_thickness = n_slices * (slice_thickness + slice_spacing) - slice_spacing
    slice_gap = slice_thickness + slice_spacing - exc.slice_thickness

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(s: int, view: tuple[int, int] | None, phase: float, pad: float) -> None:
        """Add one excitation of slice ``s`` at ``(blade, line)``; ``None`` is a dummy.

        A dummy plays the first blade's centre line without its ADC. Every
        block that drives an in-plane gradient carries the blade's rotation.
        """
        rf, gz = exc.rf, exc.gz
        rf.freq_offset = exc.selection_amplitude * positions[s]
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center
        ro.adc.phase_offset = phase

        if view is None:
            blade, ky = 0, 0.0
            label_events = labels(SLC=s, ONCE=1)
        else:
            blade, line = view
            ky = (line - blade_width // 2) / (n / 2)
            label_events = labels(SLC=s, SEG=blade, LIN=line, ONCE=0)
        rotation = rotations[blade]
        gy_pre = pp.scale_grad(ro.gy_pre, ky)

        seq.add_block(rf, gz, *label_events)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, ro.gz_reph)
            seq.add_block(ro.gx_pre, gy_pre, rotation)
        else:
            seq.add_block(ro.gx_pre, gy_pre, ro.gz_reph, rotation)
        seq.add_block(ro.gx, *([] if view is None else [ro.adc]), rotation)
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), rotation)
        seq.add_block(gz_spoil)
        seq.add_block(pp.make_delay(pad))

    played = [None] * n_dummy + views
    phases = pp.make_rf_spoiling_schedule(
        len(played), increment=np.deg2rad(RF_SPOILING_INCREMENT_DEG)
    )
    for packet in packets:
        for view, phase in zip(played, phases, strict=True):
            for i, s in enumerate(packet):
                last = i == len(packet) - 1
                pad = pads[len(packet)] if last else raster
                kernel(s, view, phase, pad)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov, fov, slab_thickness],
        "Matrix": [n, n, n_slices],
        "Name": NAME,
        "TE": ro.echo_time,
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


main = gre_propeller2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_propeller_2d.seq"))
