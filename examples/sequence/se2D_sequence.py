"""2D Cartesian spin echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "se_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design shared by the excitation and the refocusing pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Dephasing each crusher beside the refocusing pulse winds, in cycles
#: across one voxel.
CRUSHER_CYCLES = 4.0
#: Dephasing left on the readout axis at the end of each repetition, in
#: cycles across one voxel.
SPOILING_CYCLES = 4.0
#: Sinc design of the spatial saturation bands.
BAND_DURATION = 3e-3
BAND_TIME_BW_PRODUCT = 4.0


def se2d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 220e-3,
    fov_y: float = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    te: float | None = 15e-3,
    tr: float | None = 500e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    partial_fourier_x: float = 1.0,
    partial_fourier_y: float = 1.0,
    n_dummy: int | None = None,
    readout_oversampling: float = 2.0,
    n_acs_y: int = 24,
    sat1_normal_x: float = 0.0,
    sat1_normal_y: float = 0.0,
    sat1_normal_z: float = 0.0,
    sat1_position: float = 0.0,
    sat1_thickness: float = 0.0,
    sat2_normal_x: float = 0.0,
    sat2_normal_y: float = 0.0,
    sat2_normal_z: float = 0.0,
    sat2_position: float = 0.0,
    sat2_thickness: float = 0.0,
) -> pp.Sequence:
    """Multi-slice 2D Cartesian spin echo: one line per excitation.

    A slice-selective SLR 90, one slice-selective SLR 180 between crushers
    half a TE later, and one frequency-encoded line at the echo. Both pulses
    are offset to the slice, each at its own selection gradient. Slices are
    dealt into packets, and ordered within one, as :mod:`gre2D_sequence` deals
    them; under undersampling the calibration lines are acquired first.

    Up to two spatial saturation bands (``sat1_*``, ``sat2_*``) are played
    before every excitation. Each lies in the physical frame, centred
    ``position`` from the isocentre along its normal, and the field-of-view
    offset and rotation leave it there.

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
        Phase-encode undersampling: one line in every ``ry`` is acquired,
        the centre line among them.
    partial_fourier_x : float, default=1.0
        Fraction of the echo acquired, in ``[0.75, 1]``.
    partial_fourier_y : float, default=1.0
        Fraction of the phase-encode extent acquired, in ``[0.75, 1]``.
    n_dummy : int or None, default=None
        Non-acquiring repetitions before the first line of each packet.
        ``None`` plays as many as bring the steady state within 1 % (:func:`~pypulseqpp.sequences.steady_state_dummies`).
    readout_oversampling : float, default=2.0
        Readout oversampling factor, at least one.
    n_acs_y : int, default=24
        Fully sampled calibration lines at the centre of k-space, acquired
        ahead of the rest when ``ry > 1``.
    sat1_normal_x, sat1_normal_y, sat1_normal_z : float, default=0.0
        Normal of the first spatial saturation band along the physical
        gradient axes. Normalised; not the zero vector for a band with a
        thickness.
    sat1_position : float, default=0.0
        Centre of the first band along its normal, from the isocentre (m).
    sat1_thickness : float, default=0.0
        Thickness of the first band (m). Zero plays no band.
    sat2_normal_x, sat2_normal_y, sat2_normal_z : float, default=0.0
        Normal of the second band, as for the first.
    sat2_position : float, default=0.0
        Centre of the second band along its normal, from the isocentre (m).
    sat2_thickness : float, default=0.0
        Thickness of the second band (m). Zero plays no band.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If a partial Fourier fraction is outside ``[0.75, 1]``, ``ry`` is
        below one, a saturation band has a thickness and a zero normal or a
        negative thickness, or the TE or TR is shorter than the pulses and
        the readout take.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se2D_sequence(n_x=32, n_y=16, te=15e-3, tr=None)
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    for name, fraction in (
        ("partial_fourier_x", partial_fourier_x),
        ("partial_fourier_y", partial_fourier_y),
    ):
        if not 0.75 <= fraction <= 1.0:
            raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
    if ry < 1:
        raise ValueError(f"ry must be at least 1, got {ry}")

    fov = (fov_x, fov_y)
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

    # TE is solved in two halves about the 180. The readout owns the second,
    # from the 180's centre to the echo; a delay before the 180 sets the
    # first, which is never shorter than the excitation's blocks allow.
    half_floor = exc.duration - exc.center + ref.center

    def readout(half_te: float | None):
        return sequences.LineReadout2D(
            system,
            ref.rf_ref,
            ref.gz,
            fov=fov,
            matrix=(n_x, n_y),
            te=half_te,
            partial_echo=partial_fourier_x,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=SPOILING_CYCLES,
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

    bands = sequences.spatial_saturations(
        system,
        (
            (
                (sat1_normal_x, sat1_normal_y, sat1_normal_z),
                sat1_position,
                sat1_thickness,
            ),
            (
                (sat2_normal_x, sat2_normal_y, sat2_normal_z),
                sat2_position,
                sat2_thickness,
            ),
        ),
        duration_s=BAND_DURATION,
        time_bw_product=BAND_TIME_BW_PRODUCT,
        voxel_size_m=min(fov_x / n_x, fov_y / n_y, slice_thickness),
    )
    band_duration = sum(band.duration for band in bands)

    # Slices one TR cannot hold are dealt round-robin into packets, even
    # slices of a packet first; every shot closes with a pure delay, as
    # gre2D_sequence closes them.
    shot = band_duration + exc.duration + wait + ro.duration + raster
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
    # The last slice of a packet plays its pad in place of the closing raster.
    packet_time = {n: n * shot - raster + pad for n, pad in pads.items()}
    repetition_time = max(packet_time.values())
    if n_dummy is None:
        n_dummy = sequences.steady_state_dummies(repetition_time, 90.0)

    calibrating, imaging = pp.make_cartesian_axis_sampling(
        n_y, ry, n_acs_y, partial_fourier=partial_fourier_y
    )
    # The calibration block leads, so a reconstruction can estimate
    # coil sensitivities while the rest is still arriving.
    lines = [*calibrating, *imaging]
    calibration = set(calibrating)
    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_spacing
    )
    slab_thickness = n_slices * (slice_thickness + slice_spacing) - slice_spacing
    slice_gap = slice_thickness + slice_spacing - exc.slice_thickness

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(s: int, line: int | None, pad: float) -> None:
        """Add one excitation of slice ``s`` at one line; ``None`` is a dummy."""
        position = positions[s]
        # Each pulse selects at its own plateau; a crushed refocusing
        # gradient's amplitude is its crusher peak.
        exc.rf.freq_offset = exc.selection_amplitude * position
        exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ref.rf_ref.freq_offset = ref.selection_amplitude * position
        ref.rf_ref.phase_offset = (
            ref_phase - 2 * np.pi * ref.rf_ref.freq_offset * ref.rf_ref.center
        )

        if line is None:
            ky, label_events = 0.0, labels(SLC=s, ONCE=1)
        else:
            ky = (line - n_y // 2) / (n_y / 2)
            is_calibration = line in calibration
            label_events = labels(
                SLC=s, LIN=line, IMA=is_calibration, SEG=not is_calibration, ONCE=0
            )

        for band in bands:
            for block in band.blocks:
                seq.add_block(*block, *label_events)
                label_events = []
        seq.add_block(exc.rf, exc.gz, *label_events)
        seq.add_block(exc.gz_reph)
        if wait_half_te is not None:
            seq.add_block(wait_half_te)
        seq.add_block(ref.rf_ref, ref.gz)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky))
        seq.add_block(ro.gx, *([] if line is None else [ro.adc]))
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky))
        seq.add_block(pp.make_delay(pad))

    played = [None] * n_dummy + lines
    for packet in packets:
        for line in played:
            for i, s in enumerate(packet):
                last = i == len(packet) - 1
                kernel(s, line, pads[len(packet)] if last else raster)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [*fov, slab_thickness],
        "Matrix": [n_x, n_y, n_slices],
        "Name": NAME,
        "TE": echo_time,
        "TR": repetition_time,
        "kSpaceCenterLine": n_y // 2,
        "kSpaceCenterSample": ro.center_sample,
        "SlicePositions": positions.tolist(),
        "SliceThickness": exc.slice_thickness,
        "SliceGap": slice_gap,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = se2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="se_2d.seq"))
