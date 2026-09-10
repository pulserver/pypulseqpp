"""RF-spoiled 2D Cartesian gradient echo, multi-slice.

``main`` builds the sequence and returns it; running this module as a script
writes a ``.seq`` from the same controls.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: SLR design of the selective pulse, held here rather than left at the
#: module's default so a script can retune the excitation without touching the
#: loop. The selection amplitude follows as
#: ``time_bw_product / (duration * thickness)``, which is also what a slice
#: offset is converted against.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0

#: Ceilings on the gradient and slew limits, in mT/m and T/m/s. The sequence
#: is held below the smaller of these and what the scanner reports, so
#: lowering them here reruns the whole script under gentler gradients -- for
#: nerve-stimulation headroom, acoustic comfort, eddy currents -- without
#: touching anything else. They sit above typical hardware, so they cap
#: nothing until you lower them.
MAX_GRAD = 80.0
MAX_SLEW = 200.0


def main(
    # Restore with Sequence.plot: plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = "gre_2d.seq",
    *,
    system: pp.Opts | None = None,
    fov: float | tuple[float, float] = 220e-3,
    # Restore with pp.TransformFOV, which applies the offset to the finished
    # sequence: fov_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_gap: float = 0.0,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    # Restore with pp.tile, which repeats the block table: the averages are
    # written out rather than left to the interpreter's repeat count, so every
    # acquisition carries the AVG it belongs to and the dummies -- marked ONCE
    # -- appear in the first average only.
    # n_averages: int = 1,
    n_dummy: int = 16,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Build an RF-spoiled 2D Cartesian gradient-echo sequence.

    One frequency-encoded line per repetition, from a slice-selective SLR
    excitation. Phase encoding may be undersampled with a fully sampled
    autocalibration block and truncated by partial Fourier; the readout may be
    a partial echo.

    The autocalibration block leads the traversal and closes a segment of its
    own, so a reconstruction can calibrate while the rest of the scan is still
    arriving -- which puts the centre of k-space in the transient, hence
    ``n_dummy``. More slices than one TR can hold are split into passes.

    Three parameters the sequence this was ported from also took are
    commented out rather than dropped: ``plot``, ``fov_offset`` and
    ``n_averages``, each waiting on one operation this package has not grown
    yet -- ``Sequence.plot``, ``TransformFOV`` and ``tile``.

    Parameters
    ----------
    test_report : bool, optional
        Print a report on the finished sequence.
    write_seq : bool, optional
        Write the sequence to a .seq file.
    seq_filename : str, optional
        Where to write it.
    system : pypulseqpp.Opts, optional
        System limits.
    fov : float or tuple of float, optional
        In-plane field of view, in metres; one value for both axes, or
        ``(fov_x, fov_y)``.
    n_x : int, optional
        Readout samples.
    n_y : int, optional
        Phase-encode steps.
    n_slices : int, optional
        Number of slices.
    slice_thickness : float, optional
        Slice thickness, in metres.
    slice_gap : float, optional
        Gap between adjacent slices, in metres.
    slice_order : str, optional
        Order the slices of one pass are excited in, as
        :func:`pypulseqpp.calc_traversal_order` accepts.
    flip_angle_deg : float, optional
        Excitation flip angle, in degrees.
    te : float or None, optional
        Echo time, in seconds. ``None`` is as short as the readout admits.
    tr : float or None, optional
        Repetition time, in seconds, between successive excitations of one
        slice. ``None`` is as short as possible, and puts every slice in one
        pass.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth, in Hz. What was achieved is on the
        readout module's ``bandwidth_hz``, and is generally lower.
    partial_echo : float, optional
        Fraction of the echo acquired, in (0.5, 1]. Truncates the samples
        before the echo, which shortens the minimum TE.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in (0.5, 1]. Truncates
        the lines before the centre, which shortens the scan.
    acceleration : int, optional
        Uniform phase-encode undersampling factor.
    n_acs : int, optional
        Fully sampled autocalibration lines at the centre of k-space,
        acquired ahead of the rest of the scan.
    n_dummy : int, optional
        Repetitions played without acquiring, before the first line of each
        pass, to bring the magnetisation to steady state. The calibration
        block leads the traversal, so these are what keeps the centre of
        k-space -- which sets the contrast -- out of the transient.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment, in degrees.
    spoiling_cycles : float, optional
        Cycles of dephasing left on the readout axis at the end of each
        repetition, counted across one voxel.

    Returns
    -------
    pypulseqpp.Sequence
        The sequence.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_x=32, n_y=16, n_acs=0, n_dummy=0, tr=None)

    Sixteen lines, one repetition each, and the same blocks every time:

    >>> len(seq.block_events) % 16
    0
    >>> seq.check_timing()[0]
    True

    The prescription is written into the file, so what reads it back does not
    have to be told the geometry again:

    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 1.0], 'gre_2d')
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    # Designing the repetitions is also what checks that TE, TR and the rest
    # can be had: a TE shorter than one repetition admits makes the readout
    # module raise, so there is no second timing path to drift out of step.
    kernel = GREKernel(
        system,
        fov=fov,
        n_x=n_x,
        n_y=n_y,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        slice_order=slice_order,
        flip_angle_deg=flip_angle_deg,
        te=te,
        tr=tr,
        readout_bandwidth_hz=readout_bandwidth_hz,
        partial_echo=partial_echo,
        partial_fourier=partial_fourier,
        acceleration=acceleration,
        n_acs=n_acs,
        n_dummy=n_dummy,
        spoiling_cycles=spoiling_cycles,
    )
    excitation = kernel.excitation
    fov_x, fov_y = kernel.fov
    sampled_lines = kernel.sampled_lines

    # The rest of the encoding plan: where each slice sits, and the RF-spoiling
    # phase every repetition is played with. The dummies share the schedule, so
    # the spoiling phase the first acquired line sees is the one it would have
    # seen mid-scan.
    acs_start = max(0, n_y // 2 - n_acs // 2)
    acs_stop = min(n_y, acs_start + n_acs)
    last_calibration_line = kernel.n_calibration - 1
    slice_positions = (np.arange(n_slices) - (n_slices - 1) / 2) * (
        slice_thickness + slice_gap
    )
    rf_phases = pp.make_rf_spoiling_schedule(
        (len(sampled_lines) + n_dummy) * n_slices,
        increment=np.deg2rad(rf_spoiling_increment_deg),
    )

    seq = pp.Sequence(system)
    spoiling_phase = iter(rf_phases)

    def repetition(readout, slices, ky: float, acquire: bool, mark=None) -> None:
        """Play one TR of every slice of a pass, acquiring or not.

        ``mark`` rides the first excitation, for a label whose value has just
        changed. Label state is sticky, so it stays set until it is set again.
        """
        # Present only when a TE longer than the minimum was asked for; the
        # repetition time is carried by the delay that closes each shot.
        wait_te = getattr(readout, "wait_te", None)
        pad = kernel.pads[len(slices)]
        for i, i_slice in enumerate(slices):
            rf_phase = next(spoiling_phase)
            readout.rf.freq_offset = excitation.gz.amplitude * slice_positions[i_slice]
            readout.rf.phase_offset = (
                rf_phase - 2 * np.pi * readout.rf.freq_offset * readout.rf.center
            )
            readout.adc.phase_offset = rf_phase

            # Which slice this excitation is, by position: `slice_positions` is
            # ascending, so the loop index is the geometric index a
            # reconstruction stacks by, whatever order the passes visit.
            slc_label.value = i_slice
            seq.add_block(readout.rf, readout.gz, *([mark] if mark is not None else []))
            mark = None
            if wait_te is not None:
                seq.add_block(wait_te, readout.gz_reph)
                seq.add_block(readout.gx_pre, pp.scale_grad(readout.gy_pre, ky))
            else:
                seq.add_block(
                    readout.gx_pre,
                    pp.scale_grad(readout.gy_pre, ky),
                    readout.gz_reph,
                )
            if acquire:
                seq.add_block(readout.gx, readout.adc, *readout.adc_labels)
            else:
                seq.add_block(readout.gx)
            seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, ky))
            # One raster on every slice but the last of the pass, which
            # carries the rest of the repetition time.
            last = i == len(slices) - 1
            seq.add_block(pp.make_delay(pad if last else system.block_duration_raster))

    for slices in kernel.passes:
        readout = kernel.readout
        lin_label, slc_label, ima_label, seg_label = readout.adc_labels

        # Steady state first: the same repetition without its ADC, so the
        # magnetisation the first acquired line sees is the one every later
        # line sees. The centre of k-space is acquired first and sets the
        # contrast, so this is what the ordering costs.
        for i_dummy in range(n_dummy):
            repetition(
                readout,
                slices,
                0.0,
                acquire=False,
                mark=pp.make_label("ONCE", "SET", 1) if i_dummy == 0 else None,
            )

        clear_once = pp.make_label("ONCE", "SET", 0) if n_dummy else None
        for i_phase, line in enumerate(sampled_lines):
            ky = (line - n_y / 2) / (n_y / 2)
            # Every calibration line is imaging data too: the block is a fully
            # sampled centre of the same k-space rather than a separate
            # acquisition. Label state persists, so the flag is written every
            # repetition.
            lin_label.value = line
            ima_label.value = int(acs_start <= line < acs_stop)
            # The calibration block is segment zero and the rest segment one,
            # which is what lets a reconstruction calibrate the moment the
            # block is complete rather than at the end of the scan.
            seg_label.value = int(i_phase > last_calibration_line)

            repetition(readout, slices, ky, acquire=True, mark=clear_once)
            clear_once = None

    # Where the prescribed volume sits. Which way the logical axes point in
    # the magnet is the interpreter's business, so only the offset is applied
    # here. Restore with pp.TransformFOV:
    #
    #     pp.TransformFOV(
    #         translation=tuple(offset * 1e3 for offset in fov_offset),
    #         system=system,
    #     ).apply_to_sequence(seq, in_place=True)

    slab_thickness = n_slices * (slice_thickness + slice_gap) - slice_gap
    seq.set_definition(key="FOV", value=[fov_x, fov_y, slab_thickness])
    seq.set_definition(key="Matrix", value=[n_x, n_y, n_slices])
    seq.set_definition(key="Name", value="gre_2d")
    seq.set_definition(key="TE", value=kernel.echo_time)
    seq.set_definition(key="TR", value=kernel.repetition_time)

    # Where the centre of k-space is, and the slice geometry the excitation
    # actually produces: the gap is the prescribed spacing less one measured
    # thickness.
    seq.set_definition(key="kSpaceCenterLine", value=n_y // 2)
    seq.set_definition(
        key="kSpaceCenterSample",
        value=kernel.readout.center_sample,
    )
    seq.set_definition(key="SlicePositions", value=slice_positions.tolist())
    seq.set_definition(key="SliceThickness", value=kernel.excitation.slice_thickness)
    seq.set_definition(
        key="SliceGap",
        value=slice_thickness + slice_gap - kernel.excitation.slice_thickness,
    )

    # Last, because it multiplies the block table. Restore with pp.tile:
    #
    #     seq = pp.tile(seq, n_averages, in_place=True)

    if test_report:
        print(seq.test_report())

    # if plot:
    #     seq.plot()

    if write_seq:
        cli.write_sequence(seq, seq_filename)

    return seq


# ======================================================================
# Subroutines of main()
# ======================================================================


def GREKernel(
    system: pp.Opts,
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 128,
    n_y: int = 128,
    n_slices: int = 1,
    slice_thickness: float = 5e-3,
    slice_order: str = "interleaved",
    flip_angle_deg: float = 12.0,
    te: float | None = 8e-3,
    tr: float | None = 250e-3,
    readout_bandwidth_hz: float = 250e3,
    partial_echo: float = 1.0,
    partial_fourier: float = 1.0,
    acceleration: int = 1,
    n_acs: int = 24,
    # n_averages: int = 1,  # restore with pp.tile; see main
    n_dummy: int = 16,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design the repetitions, and the plan that repeats them.

    Whatever the scan loop needs: the excitation, one readout per distinct
    pass size, which slices each pass holds, the phase-encode lines the
    sampling asks for, and the time they add up to.

    Building the modules *is* the feasibility check. A TE shorter than one
    repetition can achieve makes
    :class:`~pypulseqpp.sequences.LineReadout2D` raise, as does any
    out-of-range matrix, fov or fraction.

    A TR too short for every slice does *not* raise. The slices are dealt into
    as many passes as it takes, spread across the slab so the slices of one
    pass are not neighbours, and each pass gets a repetition of its own length
    ``tr / (slices in the pass)``. A pass therefore lasts exactly one TR
    whatever its size, which is what makes the requested TR exact for every
    slice rather than exact for most of them.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    fov, n_x, n_y, n_slices, slice_thickness, slice_order, flip_angle_deg, te, \
tr, readout_bandwidth_hz, partial_echo, partial_fourier, acceleration, n_acs, \
n_dummy, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readout``, ``pads`` (the closing delay per pass
        size), ``passes`` (slice indices per pass, in excitation order),
        ``fov``, ``sampled_lines``,
        ``n_calibration`` (how many of them lead the traversal),
        ``echo_time``, ``repetition_time``, ``bandwidth_hz`` and ``duration``.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    excitation = sequences.SpatialSelectiveExcitation(
        system,
        flip_angle_deg,
        slice_thickness,
        duration_s=PULSE_DURATION,
        time_bw_product=TIME_BW_PRODUCT,
    )

    def readout(module_tr: float | None):
        return sequences.LineReadout2D(
            system,
            excitation.rf,
            excitation.gz,
            excitation.gz_reph,
            fov=(fov_x, fov_y),
            matrix=(n_x, n_y),
            te=te,
            tr=module_tr,
            partial_echo=partial_echo,
            readout_bandwidth_hz=readout_bandwidth_hz,
            spoiling_cycles=spoiling_cycles,
            labels=("LIN", "SLC", "IMA", "SEG"),
        )

    # The shortest repetition the prescription admits says how many slices one
    # TR can hold, and so how many passes the slices have to be dealt into.
    shortest = readout(None)
    per_pass = n_slices if tr is None else max(1, int(tr / shortest.duration))
    n_passes = -(-n_slices // per_pass)

    # Dealt round-robin: the sizes come out differing by at most one, and the
    # slices of a pass come out n_passes apart, which is the same thing that
    # keeps a multi-slice acquisition from exciting neighbours back to back.
    passes = [
        [int(i) for i in range(start, n_slices, n_passes)] for start in range(n_passes)
    ]
    passes = [
        [group[int(i)] for i in pp.calc_traversal_order(len(group), slice_order)]
        for group in passes
        if group
    ]

    # One readout, at its shortest, whatever a pass holds. What differs
    # between a pass of 18 slices and one of 17 is then a *duration* and not a
    # definition: every shot closes with a pure delay, one raster on each
    # slice and, on the last of a pass, whatever is left of the repetition
    # time. A pure delay is one definition however long it waits, so the block
    # stream reads as one shot repeating whatever the slices divide into --
    # which is what lets the repeating unit be found at the first block rather
    # than after the odd pass.
    raster = system.block_duration_raster
    shot_span = shortest.duration + raster
    cycle = tr if tr is not None else max(len(g) for g in passes) * shot_span
    pads = {
        size: pp.round_to_raster(cycle - size * shot_span, raster) + raster
        for size in {len(group) for group in passes}
    }
    if min(pads.values()) < raster:
        raise ValueError(
            f"the requested TR of {cycle * 1e3:.3f} ms is shorter than the "
            f"{max(size * shot_span for size in pads) * 1e3:.3f} ms the slices of a "
            "pass take"
        )

    # The autocalibration block is acquired first, so a reconstruction can
    # estimate coil sensitivities from it while the rest of the scan is still
    # running. It leads the traversal, so its length is where it ends.
    sampled_lines = pp.calc_sampled_lines(
        n_y,
        acceleration,
        n_acs,
        order="calibration_first",
        partial_fourier=partial_fourier,
    )
    n_calibration = len(
        pp.calc_calibration_lines(n_y, n_acs, partial_fourier=partial_fourier)
    )

    # One repetition per acquired line per slice, plus the dummies that bring
    # each pass to steady state; the readout has already padded itself to the
    # per-slice TR, so a pass is simply their sum.
    # Every pass lasts one repetition time by construction, so the scan is
    # one per line per pass, dummies included.
    # With averages: n_dummy * pass_time + n_averages * len(sampled_lines) *
    # pass_time -- they repeat the body and not the dummies.
    pass_time = sum(len(g) * shot_span + pads[len(g)] for g in passes)
    duration = (n_dummy + len(sampled_lines)) * pass_time

    return SimpleNamespace(
        excitation=excitation,
        readout=shortest,
        pads=pads,
        passes=passes,
        fov=(fov_x, fov_y),
        sampled_lines=sampled_lines,
        n_calibration=n_calibration,
        echo_time=shortest.echo_time,
        repetition_time=max(len(g) * shot_span + pads[len(g)] for g in passes),
        bandwidth_hz=shortest.bandwidth_hz,
        duration=duration,
    )


if __name__ == "__main__":
    raise SystemExit(
        cli.run(
            main,
            sys.argv[1:],
            description="Write a 2D Cartesian gradient-echo .seq.",
            default_output="gre_2d.seq",
        )
    )
