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

#: Duration (s) and time-bandwidth product of the SLR excitation. The
#: selection amplitude, ``time_bw_product / (duration * thickness)``, also
#: converts a slice offset into a frequency offset.
PULSE_DURATION = 3e-3
TIME_BW_PRODUCT = 4.0

#: Ceilings on gradient amplitude (mT/m) and slew rate (T/m/s). The sequence
#: uses the smaller of these and the limits of the system it is given.
MAX_GRAD = 80.0
MAX_SLEW = 200.0


def main(
    plot: bool = False,
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
    # Restore with block-table tiling, which is deferred. The averages are
    # then written out rather than left to the interpreter's repeat count, so
    # every acquisition carries its AVG and the dummies, marked ONCE, appear
    # in the first average only.
    # n_averages: int = 1,
    n_dummy: int = 16,
    rf_spoiling_increment_deg: float = 117.0,
    spoiling_cycles: float = 4.0,
) -> pp.Sequence:
    """Build an RF-spoiled, multi-slice 2D Cartesian gradient echo.

    The autocalibration lines are acquired first, then the remaining lines.
    Slices are divided into passes when they do not all fit within one TR.

    Parameters
    ----------
    plot : bool, optional
        Draw the finished sequence in SeqEyes.
    test_report : bool, optional
        Print a report on the finished sequence.
    write_seq : bool, optional
        Write the sequence to a .seq file.
    seq_filename : str, optional
        Path of the .seq file.
    system : pypulseqpp.Opts, optional
        System limits.
    fov : float or tuple of float, optional
        In-plane field of view (m), one value or ``(fov_x, fov_y)``.
    n_x : int, optional
        Readout samples.
    n_y : int, optional
        Phase-encode steps.
    n_slices : int, optional
        Number of slices.
    slice_thickness : float, optional
        Slice thickness (m).
    slice_gap : float, optional
        Gap between adjacent slices (m).
    slice_order : str, optional
        Order in which the slices of one pass are excited.
        Any order :func:`pypulseqpp.calc_traversal_order` accepts.
    flip_angle_deg : float, optional
        Excitation flip angle, in degrees.
    te : float or None, optional
        Echo time (s); None is as short as the readout admits.
    tr : float or None, optional
        Repetition time (s) between excitations of the same slice.
        ``None`` is as short as possible and puts every slice in one pass.
    readout_bandwidth_hz : float, optional
        Requested receiver bandwidth (Hz).
        The achieved one is the readout module's ``bandwidth_hz``.
    partial_echo : float, optional
        Fraction of the echo acquired, in (0.5, 1].
        Drops samples before the echo, which shortens the minimum TE.
    partial_fourier : float, optional
        Fraction of the phase-encode extent acquired, in (0.5, 1].
        Drops lines before the centre of k-space.
    acceleration : int, optional
        Uniform phase-encode undersampling factor.
    n_acs : int, optional
        Fully sampled autocalibration lines at the centre of k-space.
    n_dummy : int, optional
        Repetitions played without acquisition before each pass.
        They bring the RF-spoiled signal to steady state.
    rf_spoiling_increment_deg : float, optional
        Quadratic RF spoiling phase increment (degrees).
    spoiling_cycles : float, optional
        Readout-axis dephasing at the end of each TR, in cycles per voxel.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.gre2D_sequence(n_x=32, n_y=16, n_acs=0, n_dummy=0, tr=None)

    Sixteen lines, one repetition each, and the same blocks every time:

    >>> len(seq.block_events) % 16
    0
    >>> seq.check_timing()[0]
    True

    The prescription is stored in the definitions:

    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 1.0], 'gre_2d')
    """
    system = pp.Opts() if system is None else system
    system = pp.cap_system(system, max_grad=MAX_GRAD, max_slew=MAX_SLEW)

    # Designing the repetitions is also the feasibility check: an unreachable
    # TE or TR raises there, so there is no second timing path.
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

    # Averages belong here, last, because repeating the block table
    # multiplies it; see the n_averages note on the signature.

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot()

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
    # n_averages: int = 1,  # see main
    n_dummy: int = 16,
    spoiling_cycles: float = 4.0,
) -> SimpleNamespace:
    """Design GRE event templates and the sampling plan.

    Slices that do not fit in one TR are dealt round-robin into passes. Each
    pass lasts one TR: its slices play back to back at the shortest shot,
    and the closing delay of the last one (``pads``) takes up the rest.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    fov, n_x, n_y, n_slices, slice_thickness, slice_order, flip_angle_deg, te, tr, readout_bandwidth_hz, partial_echo, partial_fourier, acceleration, n_acs, n_dummy, spoiling_cycles
        As for :func:`main`.

    Returns
    -------
    types.SimpleNamespace
        ``excitation``, ``readout``, ``pads`` (the last shot's closing delay
        (s), per pass size), ``passes`` (slice indices per pass, in excitation
        order), ``fov``, ``sampled_lines``, ``n_calibration`` (how many of
        them lead the traversal), ``echo_time`` and ``repetition_time`` (s),
        ``bandwidth_hz`` and ``duration`` (s, the whole scan).

    Raises
    ------
    ValueError
        If the TE is unreachable or the TR is shorter than a pass.
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

    # One readout at its shortest, whatever a pass holds. Every shot closes
    # with a pure delay: one raster on each slice and, on the last of a pass,
    # the rest of the TR. Passes of different sizes then differ only in a
    # delay's duration, not in block definitions, so the block stream repeats
    # one shot from the first block.
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

    # Every line and every dummy repetition plays each pass once. Averages
    # would repeat the lines but not the dummies.
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
