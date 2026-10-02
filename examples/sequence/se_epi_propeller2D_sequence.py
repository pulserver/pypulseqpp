"""2D PROPELLER spin echo with EPI blades, multi-slice."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "se_epi_propeller_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design shared by the excitation and the refocusing pulse.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0


def se_epi_propeller2d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n_x: int = 128,
    blade_width: int = 16,
    n_blades: int | None = None,
    angle_scheme: str = "uniform",
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    te: float | None = 80e-3,
    tr: float | None = 2000e-3,
    readout_bandwidth_hz: float = 250e3,
    crusher_cycles: float = 4.0,
    n_dummy: int = 0,
    n_gain_calibration_readouts: int | None = None,
) -> pp.Sequence:
    """Multi-slice 2D PROPELLER spin echo: one EPI blade per excitation.

    A slice-selective excitation and one refocusing pulse with crushers put
    the spin echo on the blade's central line, so ``te`` runs from the
    excitation centre to that line with the refocusing pulse at its midpoint.
    The blade straddles the centre of k-space and is turned per shot by a
    rotation extension. Slices one TR cannot hold are dealt round-robin into
    passes, and each slice of a pass is played at ``tr / len(pass)``. Every
    acquisition carries its line within the blade as ``LIN``, its slice as
    ``SLC`` and its blade as ``SEG``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov : float, default=0.22
        Isotropic in-plane field of view (m).
    n_x : int, default=128
        In-plane matrix size.
    blade_width : int, default=16
        Phase-encode lines per blade.
    n_blades : int or None, default=None
        Blades in the set. ``None`` is the smallest count that samples the
        rim of k-space at Nyquist.
    angle_scheme : {'uniform', 'golden'}, default='uniform'
        Spacing of the blade angles, spread over half a turn.
    n_slices : int, default=1
        Number of slices.
    slice_thickness : float, default=0.005
        Slice thickness (m).
    slice_gap : float, default=0.0
        Gap between adjacent slices (m).
    slice_order : {'sequential','reverse','interleaved','center_out','outside_in','random'}, default='interleaved'
        Order the slices of one pass are excited in, as
        :func:`~pypulseqpp.make_traversal_order` visits them.
    te : float or None, default=0.08
        Effective echo time, excitation centre to the blade's central
        line (s). ``None`` is as short as possible.
    tr : float or None, default=2.0
        Repetition time between successive excitations of one slice (s).
        ``None`` is as short as possible, and puts every slice in one
        pass.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    crusher_cycles : float, default=4.0
        Cycles of dephasing each crusher beside the refocusing pulse winds.
    n_dummy : int, default=0
        Non-acquiring blades, at the first blade's angle, before the first
        acquired blade of each pass.
    n_gain_calibration_readouts : int or None, default=None
        Written as the ``NumGainCalibrationReadouts`` definition. ``None``
        is one per slice.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If the TE is shorter than the blade half or the excitation half
        admits, or the TR is shorter than one blade takes.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.se_epi_propeller2D_sequence(
    ...     n_x=32, blade_width=8, n_blades=4, te=None, tr=None
    ... )
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if n_gain_calibration_readouts is None:
        n_gain_calibration_readouts = n_slices
    exc = sequences.SpatialSelectiveExcitation(
        system,
        90.0,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )
    ref = sequences.SpatialSelectiveRefocusing(
        system,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
        spoiling_cycles=crusher_cycles,
    )

    def make_blade(first_line_te: float | None):
        return sequences.PropellerReadout2D(
            system,
            ref.rf_ref,
            ref.gz,
            fov=fov,
            matrix=n_x,
            blade_width=blade_width,
            n_blades=n_blades,
            scheme=angle_scheme,
            te=first_line_te,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )

    # The blade's own te is its first line's, while the spin echo belongs
    # to its central line: the readout half is TE/2 less the lines before
    # the centre, and the excitation half is a wait.
    excitation_span = exc.seq.duration()[0]
    exc_center = exc.rf.delay + exc.rf.center
    half_te_floor = (excitation_span - exc_center) + ref.center
    probe = make_blade(None)
    centre_delta = float(probe.echo_times[blade_width // 2] - probe.echo_times[0])
    half_te = (
        max(probe.echo_time + centre_delta, half_te_floor) if te is None else te / 2
    )
    first_line_te = half_te - centre_delta
    if first_line_te < probe.echo_time - 1e-9:
        raise ValueError(
            f"TE {2 * half_te * 1e3:.1f} ms is shorter than the blade half "
            f"admits; the minimum is "
            f"{2 * (probe.echo_time + centre_delta) * 1e3:.1f} ms"
        )
    if half_te_floor > half_te + 1e-9:
        raise ValueError(
            f"TE {2 * half_te * 1e3:.1f} ms is shorter than the excitation "
            f"half admits; the minimum is {2 * half_te_floor * 1e3:.1f} ms"
        )
    blade = make_blade(first_line_te)
    raster = system.block_duration_raster
    half_te_wait = pp.round_to_raster(half_te - half_te_floor, raster)
    wait_half_te = pp.make_delay(half_te_wait) if half_te_wait > 0 else None
    # Each half is rounded to the raster on its own, so the echo time
    # played is their sum.
    echo_time = half_te_floor + half_te_wait + blade.echo_time + centre_delta
    gy_pre = pp.scale_grad(blade.gy_pre, blade.blade_start)
    # The last line sits this fraction of gy_pre from the centre, and the
    # closing block brings the phase-encode axis back from it.
    end = blade.blade_start + blade.order[-1, 0] / (blade_width / 2)
    gy_rew = pp.scale_grad(blade.gy_rew, end)
    # One event per distinct angle.
    made: dict[float, object] = {}
    rotations = [
        made.setdefault(float(a), pp.make_rotation(float(a)))
        for a in blade.blade_angles
    ]

    shot = excitation_span + max(half_te_wait, 0.0) + blade.seq.duration()[0]
    if tr is not None and tr < shot - 1e-9:
        raise ValueError(
            f"TR {tr * 1e3:.1f} ms is shorter than one blade takes "
            f"({shot * 1e3:.1f} ms)"
        )
    # Slices one TR cannot hold are dealt round-robin into passes, so
    # neighbours are never excited back to back.
    per_pass = n_slices if tr is None else max(1, int(tr / shot))
    n_passes = -(-n_slices // per_pass)
    groups = [list(range(start, n_slices, n_passes)) for start in range(n_passes)]
    passes = [
        [group[i] for i in pp.make_traversal_order(len(group), slice_order)]
        for group in groups
    ]
    # Each slice of a pass closes with the wait that makes its shot tr / size.
    waits = {}
    for size in {len(group) for group in passes}:
        pad = 0.0 if tr is None else pp.round_to_raster(tr / size - shot, raster)
        waits[size] = pp.make_delay(pad) if pad > 0 else None
    pass_time = {
        size: size * (shot + (w.delay if w is not None else 0.0))
        for size, w in waits.items()
    }
    repetition_time = max(pass_time.values())

    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )
    slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(s: int, index: int | None, wait=None) -> None:
        """Add one spin echo of slice ``s`` reading blade ``index``; ``None`` is a dummy.

        A dummy plays the first blade's orientation without acquiring.
        """
        position = positions[s]
        exc.rf.freq_offset = exc.selection_amplitude * position
        exc.rf.phase_offset = -2 * np.pi * exc.rf.freq_offset * exc.rf.center
        ref.rf_ref.freq_offset = ref.selection_amplitude * position
        ref.rf_ref.phase_offset = (
            np.pi / 2 - 2 * np.pi * ref.rf_ref.freq_offset * ref.rf_ref.center
        )

        acquire = index is not None
        once = {"ONCE": int(not acquire)} if n_dummy else {}
        if acquire:
            label_events = labels(SLC=s, SEG=index, **once)
        else:
            label_events = labels(SLC=s, **once)
        rotation = rotations[index if acquire else 0]

        seq.add_block(exc.rf, exc.gz, *label_events)
        seq.add_block(exc.gz_reph)
        if wait_half_te is not None:
            seq.add_block(wait_half_te)
        seq.add_block(ref.rf_ref, ref.gz)
        wait_te = getattr(blade, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(blade.gx_pre, gy_pre, rotation)
        for line in range(blade.etl):
            blip = blade.gy_blips[line]
            acquired = (blade.adc, *labels(LIN=line)) if acquire else ()
            seq.add_block(
                blade.gx[line],
                *acquired,
                *([blip] if blip is not None else []),
                rotation,
            )
        seq.add_block(blade.gx_spoil, gy_rew, rotation)
        if wait is not None:
            seq.add_block(wait)

    # Each pass plays its dummy blades, then every blade at each of its slices.
    blades = [None] * n_dummy + list(range(blade.n_blades))
    for group in passes:
        wait = waits[len(group)]
        for index in blades:
            for s in group:
                kernel(s, index, wait)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov, fov, slab_thickness],
        "Matrix": [n_x, n_x, n_slices],
        "Name": NAME,
        "TE": echo_time,
        "TR": repetition_time,
        "Trajectory": "propeller",
        "BladeWidth": blade.blade_width,
        "NumBlades": blade.n_blades,
        "NumGainCalibrationReadouts": n_gain_calibration_readouts,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = se_epi_propeller2d

if __name__ == "__main__":
    raise SystemExit(
        cli.run(main, sys.argv[1:], default_output="se_epi_propeller_2d.seq")
    )
