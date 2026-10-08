"""RF-spoiled 2D Cartesian gradient echo, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "gre_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the selective pulse. The selection amplitude, which slice
#: offsets are converted against, is the pulse's measured bandwidth over the
#: thickness.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0
#: Quadratic RF spoiling phase increment (degrees), counted per slice.
RF_SPOILING_INCREMENT_DEG = 117.0
#: Dephasing left on the readout axis at the end of each repetition, in
#: cycles across one voxel.
SPOILING_CYCLES = 4.0

#: Sinc design of the spatial saturation bands.
BAND_DURATION = 3e-3
BAND_TIME_BW_PRODUCT = 4.0


def gre2d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 220e-3,
    fov_y: float = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_spacing: float = 0.0,
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    partial_fourier_x: float = 1.0,
    partial_fourier_y: float = 1.0,
    n_dummy: int = 16,
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
    """RF-spoiled, multi-slice 2D Cartesian gradient echo.

    One line per repetition. Slices one TR cannot hold are dealt round-robin
    into packets, played one after another; within a packet the even slices
    are excited before the odd ones. Every packet starts with its own dummy
    repetitions, and the last slice of a packet waits out the rest of the TR.
    Under undersampling the calibration lines are acquired first, marked
    ``IMA``.

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
    flip_angle_deg : float, default=12.0
        Excitation flip angle (degrees).
    te : float | None, default=0.008
        Echo time (s). ``None`` is as short as the readout admits.
    tr : float | None, default=0.25
        Repetition time between successive excitations of one slice (s).
        ``None`` is as short as possible, and puts every slice in one
        packet.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz). What was achieved is the
        readout module's ``bandwidth_hz``.
    ry : int, default=1
        Phase-encode undersampling: one line in every ``ry`` is acquired,
        the centre line among them.
    partial_fourier_x : float, default=1.0
        Fraction of the echo acquired, in ``[0.75, 1]``. Truncates the
        samples before the echo, which shortens the minimum TE.
    partial_fourier_y : float, default=1.0
        Fraction of the phase-encode extent acquired, in ``[0.75, 1]``.
        Truncates the lines before the centre.
    n_dummy : int, default=16
        Non-acquiring repetitions before the first line of each packet.
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
        negative thickness, or the TR cannot hold one slice.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_x=32, n_y=16, tr=None)
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 1.0], 'gre_2d')
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
        fov=fov,
        matrix=(n_x, n_y),
        te=te,
        partial_echo=partial_fourier_x,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=SPOILING_CYCLES,
    )

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

    # Slices one TR cannot hold are dealt round-robin into packets, so the
    # slices of a packet sit a packet count apart and neighbours are never
    # excited back to back.
    raster = system.block_duration_raster
    shot = band_duration + ro.duration + raster
    per_packet = n_slices if tr is None else max(1, int(tr / shot + 1e-9))
    n_packets = -(-n_slices // per_packet)
    dealt = [range(start, n_slices, n_packets) for start in range(n_packets)]
    packets = [[*packet[::2], *packet[1::2]] for packet in dealt]

    # Every shot closes with a pure delay: one raster, and on the last slice
    # of a packet whatever is left of the TR. Packets of different sizes
    # then differ in a duration, not a definition, so the scan is one
    # repeating shot whatever the slices divide into.
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
    # The last shot of a packet closes with its pad rather than the raster.
    packet_time = {n: n * shot - raster + pad for n, pad in pads.items()}
    repetition_time = max(packet_time.values())

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

    def kernel(s: int, line: int | None, phase: float, pad: float) -> None:
        """Add one excitation of slice ``s`` at one line; ``None`` is a dummy."""
        rf, gz = exc.rf, exc.gz
        rf.freq_offset = exc.selection_amplitude * positions[s]
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center
        ro.adc.phase_offset = phase

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
        seq.add_block(rf, gz, *label_events)
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te, ro.gz_reph)
            seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky))
        else:
            seq.add_block(ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), ro.gz_reph)
        seq.add_block(ro.gx, *([] if line is None else [ro.adc]))
        seq.add_block(ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky))
        seq.add_block(pp.make_delay(pad))

    # Every slice sees the same line order, so the RF spoiling phase of a
    # slice's ``k``-th excitation is the schedule's ``k``-th entry.
    played = [None] * n_dummy + lines
    phases = pp.make_rf_spoiling_schedule(
        len(played), increment=np.deg2rad(RF_SPOILING_INCREMENT_DEG)
    )
    for packet in packets:
        for line, phase in zip(played, phases, strict=True):
            for i, s in enumerate(packet):
                last = i == len(packet) - 1
                pad = pads[len(packet)] if last else raster
                kernel(s, line, phase, pad)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV; which way the logical axes point is the
    # interpreter's business.
    definitions = {
        "FOV": [*fov, slab_thickness],
        "Matrix": [n_x, n_y, n_slices],
        "Name": NAME,
        "TE": ro.echo_time,
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


main = gre2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="gre_2d.seq"))
