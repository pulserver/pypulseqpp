"""2D single-shot fast spin echo, multi-slice, with an optional slab inversion."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

NAME = "ssfse_2d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: Shortest duration of the excitation and refocusing pulses (s); each is
#: lengthened until it fits under the system's peak B1.
PULSE_DURATION = 1e-3
#: Dephasing each crusher beside a refocusing pulse winds, in cycles across
#: the slice.
CRUSHER_CYCLES = 2.0
#: Shortest duration of the adiabatic inversion (s).
INVERSION_DURATION = 10e-3
#: The refocusing trains ``flip_modulation`` selects from.
MODULATIONS = ("constant", "traps", "pd_target")


def refocusing_train(
    flip_modulation: str, etl: int, angle_deg: float, esp: float
) -> np.ndarray:
    """Return the refocusing angles of one train (degrees).

    Parameters
    ----------
    flip_modulation : {'constant', 'traps', 'pd_target'}
        Every angle at ``angle_deg``; a train that settles from above onto
        ``angle_deg`` (:func:`~pypulseqpp.make_traps_schedule`); or the
        target-signal train no angle of which exceeds ``angle_deg``
        (:func:`~pypulseqpp.make_pd_target_schedule`).
    etl : int
        Echoes in the train.
    angle_deg : float
        Refocusing angle (degrees), as ``flip_modulation`` reads it.
    esp : float
        Echo spacing (s).

    Returns
    -------
    numpy.ndarray
        ``(etl,)`` refocusing angles (degrees).
    """
    if flip_modulation == "constant":
        return np.full(etl, angle_deg)
    if flip_modulation == "traps":
        return np.degrees(pp.make_traps_schedule(etl, math.radians(angle_deg)))
    return np.degrees(
        pp.make_pd_target_schedule(etl, esp, maximum_flip_angle=math.radians(angle_deg))
    )


def ssfse2d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 240e-3,
    fov_y: float = 180e-3,
    n_x: int = 256,
    n_y: int = 192,
    n_slices: int = 1,
    slice_thickness: float = 4e-3,
    slice_spacing: float = 0.0,
    partial_fourier_y: float = 0.55,
    ry: int = 2,
    n_acs_y: int = 8,
    readout_bandwidth_hz: float = 250e3,
    readout_oversampling: float = 1.0,
    time_bw_product: float = 2.0,
    flip_modulation: str = "pd_target",
    refocusing_angle_deg: float = 150.0,
    ti: float | None = None,
    inversion_slab: float = 2.0,
    shot_interval: float | None = None,
) -> pp.Sequence:
    """Multi-slice 2D single-shot fast spin echo: one CPMG train reads one slice.

    Each slice is excited once and read by one train of refocusing pulses,
    one phase-encoded line per echo, in increasing ky from the first line
    the partial-Fourier fraction keeps, so the centre of k-space is read
    early and the effective echo time is short. The calibration lines at the
    centre are read in the same train. Slices are played even ones first,
    then odd ones, so that one slice's train does not reach into the next
    slice played.

    The refocusing angles follow ``flip_modulation``, each pulse scaled from
    one design at the train's largest angle. With ``ti``, a slice-selective
    adiabatic inversion over ``inversion_slab`` slice thicknesses precedes
    every excitation by ``ti``.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x : float, default=0.24
        Field of view along the readout (m).
    fov_y : float, default=0.18
        Field of view along the phase encode (m).
    n_x : int, default=256
        Readout matrix size.
    n_y : int, default=192
        Phase-encode matrix size.
    n_slices : int, default=1
        Number of slices.
    slice_thickness : float, default=0.004
        Slice thickness (m).
    slice_spacing : float, default=0.0
        Gap between adjacent slices (m); zero is contiguous.
    partial_fourier_y : float, default=0.55
        Fraction of the phase-encode extent acquired, in ``[0.5, 1]``.
    ry : int, default=2
        Phase-encode undersampling outside the calibration lines.
    n_acs_y : int, default=8
        Fully sampled calibration lines at the centre of k-space.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    readout_oversampling : float, default=1.0
        Readout oversampling factor, at least one.
    time_bw_product : float, default=2.0
        Time-bandwidth product of the excitation and refocusing pulses.
    flip_modulation : {'constant', 'traps', 'pd_target'}, default='pd_target'
        Refocusing train (:func:`refocusing_train`).
    refocusing_angle_deg : float, default=150.0
        Refocusing angle (degrees): every angle of a constant train, the
        angle a TRAPS train settles on, the largest of a target-signal
        train.
    ti : float | None, default=None
        Inversion time (s), from the centre of the inversion to the centre
        of the excitation. ``None`` plays no inversion.
    inversion_slab : float, default=2.0
        Thickness of the inverted slab, in slice thicknesses.
    shot_interval : float | None, default=None
        Time from one excitation to the next (s). ``None`` is as short as
        the shot allows.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``partial_fourier_y`` is outside ``[0.5, 1]``, ``ry`` is below one,
        ``flip_modulation`` is unknown, ``inversion_slab`` is not positive,
        or ``ti`` or ``shot_interval`` is shorter than the pulses allow.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.ssfse2D_sequence(n_x=64, n_y=48, n_slices=2)
    >>> seq.check_timing()[0]
    True
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
    if ry < 1:
        raise ValueError(f"ry must be at least 1, got {ry}")
    if flip_modulation not in MODULATIONS:
        raise ValueError(
            f"flip_modulation must be one of {MODULATIONS}, got {flip_modulation!r}"
        )
    if inversion_slab <= 0:
        raise ValueError("inversion_slab must be positive")

    calibrating, imaging = pp.make_cartesian_axis_sampling(
        n_y, ry, n_acs_y, partial_fourier=partial_fourier_y
    )
    lines = sorted({*calibrating, *imaging})
    calibration = set(calibrating)
    etl = len(lines)

    exc = sequences.SpatialSelectiveExcitation(
        system,
        90.0,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=time_bw_product,
    )

    def design(angle_deg: float):
        return sequences.SpatialSelectiveRefocusing(
            system,
            slice_thickness,
            angle_deg,
            duration_s=PULSE_DURATION,
            voxel_size_m=slice_thickness,
            spoiling_cycles=CRUSHER_CYCLES,
            time_bw_product=time_bw_product,
        )

    def train_of(ref):
        return sequences.FseReadout2D(
            system,
            exc.rf,
            exc.gz,
            exc.gz_reph,
            rf_ref=ref.rf_ref,
            gz_ref=ref.gz,
            fov=(fov_x, fov_y),
            matrix=(n_x, n_y),
            etl=etl,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
            voxel_size_m=slice_thickness,
        )

    # The train's angles depend on its echo spacing, and the spacing on the
    # pulse designed at the largest angle, so the pulse is designed at the
    # prescribed angle, the train read from it, and the pulse designed again
    # at the train's largest angle where that is larger.
    ref = design(refocusing_angle_deg)
    fse = train_of(ref)
    flips = refocusing_train(flip_modulation, etl, refocusing_angle_deg, fse.esp)
    largest = float(flips.max())
    if largest > refocusing_angle_deg + 1e-9:
        ref = design(largest)
        fse = train_of(ref)
        flips = refocusing_train(flip_modulation, etl, refocusing_angle_deg, fse.esp)
    nominal = float(ref.rf_ref.amplitude)
    designed_deg = max(largest, refocusing_angle_deg)
    ref_phase = float(ref.rf_ref.phase_offset)

    raster = system.block_duration_raster
    inversion = wait_ti = None
    if ti is not None:
        inversion = sequences.InversionPreparation(
            system,
            INVERSION_DURATION,
            slab_thickness_m=inversion_slab * slice_thickness,
            voxel_size_m=slice_thickness,
        )
        floor = (
            inversion.duration
            - inversion.center
            + float(exc.rf.delay)
            + float(exc.rf.center)
        )
        if ti < floor - 1e-9:
            raise ValueError(
                f"TI {ti * 1e3:.1f} ms is shorter than the {floor * 1e3:.1f} ms the "
                "inversion and the excitation take"
            )
        wait = pp.round_to_raster(ti - floor, raster)
        wait_ti = pp.make_delay(wait) if wait > 0 else None

    shot = (
        (0.0 if inversion is None else inversion.duration)
        + (0.0 if wait_ti is None else wait_ti.delay)
        + fse.duration
    )
    interval = shot + raster if shot_interval is None else shot_interval
    pad = pp.round_to_raster(interval - shot, raster)
    if pad < raster - 1e-12:
        raise ValueError(
            f"a shot interval of {interval * 1e3:.1f} ms is shorter than one shot "
            f"takes ({shot * 1e3:.1f} ms)"
        )

    positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_spacing
    )
    order = [*range(0, n_slices, 2), *range(1, n_slices, 2)]
    slab_thickness = n_slices * (slice_thickness + slice_spacing) - slice_spacing
    seq, labels = pp.Sequence(system), sequences.Labels()

    def place(rf, amplitude: float, position: float, phase: float = 0.0) -> None:
        rf.freq_offset = amplitude * position
        rf.phase_offset = phase - 2 * np.pi * rf.freq_offset * rf.center

    for s in order:
        position = positions[s]
        if inversion is not None:
            place(inversion.rf_prep, float(inversion.gz_prep.amplitude), position)
            for block in inversion.blocks:
                seq.add_block(*block)
            if wait_ti is not None:
                seq.add_block(wait_ti)
        place(exc.rf, exc.selection_amplitude, position)
        place(ref.rf_ref, ref.selection_amplitude, position, ref_phase)
        seq.add_block(exc.rf, exc.gz)
        seq.add_block(fse.gx_pre, exc.gz_reph)
        for line, flip in zip(lines, flips, strict=True):
            ky = (line - n_y // 2) / (n_y / 2)
            ref.rf_ref.amplitude = nominal * float(flip) / designed_deg
            seq.add_block(ref.rf_ref, ref.gz)
            seq.add_block(fse.gx_bridge_pre, pp.scale_grad(fse.gy_pre, ky))
            seq.add_block(
                fse.gx,
                fse.adc,
                *labels(SLC=s, LIN=line, IMA=line in calibration),
            )
            seq.add_block(fse.gx_bridge_post, pp.scale_grad(fse.gy_rew, ky))
        ref.rf_ref.amplitude = nominal
        seq.add_block(pp.make_delay(pad))

    centre_echo = lines.index(n_y // 2)
    definitions = {
        "FOV": [fov_x, fov_y, slab_thickness],
        "Matrix": [n_x, n_y, n_slices],
        "Name": NAME,
        "TE": float(fse.echo_times[centre_echo]),
        "TR": n_slices * interval,
        "EchoSpacing": fse.esp,
        "EchoTrainLength": etl,
        "FlipModulation": flip_modulation,
        "RefocusingFlipAngles": [float(f) for f in flips],
        "kSpaceCenterLine": n_y // 2,
        "kSpaceCenterSample": fse.center_sample,
        "SlicePositions": positions.tolist(),
        "SliceOrder": order,
        "SliceThickness": exc.slice_thickness,
        "SliceGap": slice_thickness + slice_spacing - exc.slice_thickness,
    }
    if ti is not None:
        definitions["TI"] = ti
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=sequences.gain_calibration_readouts(seq),
    )
    return seq


main = ssfse2d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="ssfse_2d.seq"))
