"""Non-Cartesian readouts: one designed interleave, played as a whole repetition."""

from __future__ import annotations

__all__ = [
    "NonCartesianReadout",
    "RadialProjectionReadout",
    "RadialReadout2D",
    "RadialStackReadout",
    "RosetteProjectionReadout",
    "RosetteReadout2D",
    "RosetteStackReadout",
    "SpiralProjectionReadout",
    "SpiralReadout2D",
    "SpiralStackReadout",
]

from typing import Any

import numpy as np

import pypulseqpp as pp

from .._module import SequenceModule
from ._common import AXES, left_align_rephaser, present, solve_delay, solve_rephasing
from ._trajectories import NonCartesianGradient, Rosette, Spiral

_READOUT_GRAD_MARGIN = 0.8


class _ArmedReadout(SequenceModule):
    """Block layouts for compact or explicitly rotated interleaves.

    A compact layout reuses one base arm for any arm index; the acquisition
    loop supplies its rotation. An explicit layout stores each arm separately.
    """

    def arm(self, index: int) -> list[tuple]:
        """Blocks of one arm, as the module laid them out.

        Parameters
        ----------
        index : int
            Which arm, counting from zero.

        Returns
        -------
        list of tuple
            One tuple of events per block.
        """
        return self._arms[index if len(self._arms) > 1 else 0]

    def _lay_out_arms(self, n_arms: int, tail: Any = None) -> None:
        """Split what has been laid out so far into one block layout per arm.

        The TR wait cannot be laid out with its arm -- the minimum TR is only
        known once every arm is -- so it is passed here and closes each one.
        """
        blocks = self.blocks
        if tail is not None:
            blocks = blocks[: len(blocks) - n_arms]
        span = len(blocks) // n_arms
        self._arms = [
            [
                *blocks[i * span : (i + 1) * span],
                *(((tail,),) if tail is not None else ()),
            ]
            for i in range(n_arms)
        ]
        self.n_arms = n_arms


# ======================================================================
# Radial
# ======================================================================


class _RadialReadout(_ArmedReadout):
    """Radial repetition with spoke, prephaser and rewinder in one waveform.

    The ADC samples the plateau. A slice rephaser may share an axis left
    invariant by the shot rotation. With explicit=True, angles selects
    materialised spokes; otherwise the loop supplies rotation extensions.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given, left-aligned in whichever block follows
        the pulse.
    gx, gy : GradEvent
        The spoke, prephaser through rewinder. ``gy`` is the same waveform at
        zero amplitude, so a block always has a y slot for a rotation to fill.
        Lists of one entry per angle when ``explicit``.
    gz_pre, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler, when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window, delayed onto the readout plateau.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.
    echo_spacing : float
        Time between consecutive echoes (s), zero for a single-echo readout.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    rf : RfEvent
        The pulse that opens the repetition.
    gz : GradEvent, optional
        A selection gradient played in the same block as ``rf``.
    gz_reph : GradEvent, optional
        The rephaser that unwinds ``gz``. Carried left-aligned in the first
        block after the pulse -- the TE wait when there is one, otherwise
        alongside the prephaser at the head of the spoke, where it costs no
        echo time at all. Only an axis the loop's rotation leaves alone can
        carry one, so an in-plane acquisition takes a rephaser on z and a
        projection takes none.
    fov : float
        Isotropic in-plane field of view (m).
    matrix : int
        In-plane matrix size. Sets ``kmax = matrix / (2 * fov)``.
    fov_z, matrix_z : float, int
        Partition field of view (m) and count. Stacks only.
    te, tr : float, optional
        Echo time (s) from the RF isodelay to the centre crossing, and
        repetition time (s). ``None`` is as short as possible.
    oversampling : float, optional
        Readout oversampling: more samples along the same spoke.
    readout_bandwidth_hz : float, optional
        Requested ADC sampling rate (Hz). ``bandwidth_hz`` reports the
        achieved raster-compatible rate.
    spoiling_cycles : float, optional
        Dephasing left at the end of the TR, in cycles across
        ``voxel_size_m``. Zero leaves the spoke rewound.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m); the resolution by default.
    spoiling_axis : {'z', 'x', 'y'}, optional
        Axis the spoiler is played on.
    n_echoes : int, optional
        Path traversals per repetition, separated by ``echo_spacing``.
        Each retraces the same k-space coordinates and carries its own ``ECO``
        label. Non-Cartesian arms use rewinders/prewinders between echoes.
    explicit : bool, optional
        Write out one spoke per entry of ``angles`` instead of one base spoke.
    angles : array-like, optional
        In-plane rotations (rad). Required when ``explicit``, refused
        otherwise -- a sampling pattern is not the readout's to hold.
    labels : sequence of str, optional
        Counters emitted on the acquisition block.
    trigger : event, optional
        A trigger or digital output armed on the block that opens the readout.

    Raises
    ------
    ValueError
        If a count is out of range, ``angles`` and ``explicit`` disagree, or
        the requested TE or TR is shorter than the module can achieve.
    """

    #: Set by the stack variants to the axis they encode partitions on.
    _phase_axis: str | None = None

    #: Axes the loop's rotation mixes, and so cannot carry a slice rephaser.
    _rotated_axes: tuple[str, ...] = ("x", "y")

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: float,
        matrix: int,
        fov_z: float | None = None,
        matrix_z: int | None = None,
        te: float | None = None,
        tr: float | None = None,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_axis: str = "z",
        n_echoes: int = 1,
        explicit: bool = False,
        angles: Any = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        n_echoes = _checked_layout(
            n_echoes, spoiling_cycles, spoiling_axis, explicit, angles, labels
        )
        if fov <= 0 or int(matrix) < 2:
            raise ValueError("fov must be positive and matrix must be >= 2")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        # A spoke spans 2 * kmax, sampled n_samples times.
        delta_kx = 1.0 / fov
        n_samples = max(2, round(oversampling * int(matrix)))
        readout_area = int(matrix) * delta_kx
        dwell, readout_duration = pp.calc_adc_timing(
            n_samples,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=readout_area
            / (_READOUT_GRAD_MARGIN * system.max_grad),
        )

        gx = pp.make_trapezoid(
            channel="x",
            flat_area=readout_area,
            flat_time=readout_duration,
            system=system,
        )
        # The spoke runs out through the centre and back, so one lobe of half
        # the readout area serves as both prephaser and rewinder.
        gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * gx.area, system=system)
        gx_rew = pp.make_trapezoid(channel="x", area=-0.5 * gx.area, system=system)
        rise_time = gx.rise_time
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=rise_time, system=system
        )

        # Shift each piece to where it is played, then sum them into one
        # continuous waveform.
        gx_pre_duration = pp.calc_duration(gx_pre)
        gx.delay += gx_pre_duration
        adc.delay += gx_pre_duration
        gx_rew.delay += pp.calc_duration(gx)
        gx = pp.add_gradients(grads=[gx_pre, gx, gx_rew], system=system)
        gy = pp.scale_grad(gx, 0.0)
        gy.channel = "y"

        echo_offset = gx_pre_duration + rise_time + 0.5 * readout_duration
        if explicit:
            angles = np.atleast_1d(np.asarray(angles, dtype=float))
            base = gx
            gx = [pp.scale_grad(base, float(np.cos(angle))) for angle in angles]
            gy = []
            for angle in angles:
                lobe = pp.scale_grad(base, float(np.sin(angle)))
                lobe.channel = "y"
                gy.append(lobe)

        gz_pre, gz_rew = _partition_encode(
            self._phase_axis, fov_z, matrix_z, type(self).__name__, system
        )
        gz_reph = _accept_rephaser(gz_reph, self, present(self._phase_axis))
        reph_span = pp.calc_duration(gz_reph) if gz_reph is not None else 0.0

        gz_spoil = None
        if spoiling_cycles:
            if voxel_size_m is None:
                voxel_size_m = fov / int(matrix)
            if voxel_size_m <= 0:
                raise ValueError("voxel_size_m must be positive")
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, spoiling_axis, system=system
            )

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]
        # The readout plays the train, so the readout writes the counter that
        # indexes it: a scan loop hands over a whole arm and never sees the
        # echo boundaries inside it.
        echo_labels = [
            pp.make_label(type="SET", label="ECO", value=i_echo)
            for i_echo in range(n_echoes)
        ]
        n_arms = len(gx) if isinstance(gx, list) else 1

        self.seq = pp.Sequence(system)

        rf_center = float(rf.delay) + float(rf.center)
        rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz) if gz is not None else pp.calc_duration(rf),
            system.block_duration_raster,
        )
        pre_span = (
            pp.ceil_to_raster(pp.calc_duration(gz_pre), system.block_duration_raster)
            if gz_pre is not None
            else 0.0
        )
        # The rephaser has to follow the selection lobe with nothing in
        # between, so it rides the TE wait when that is long enough to hold it.
        # Otherwise it sits alongside the prephaser at the head of the spoke,
        # which costs no echo time at all -- until it outlasts the head room
        # there, when it has to be given a block of its own before the spoke.
        head_room = float(adc.delay)
        reph_alone = reph_span > head_room
        te_min = (
            rf_block
            - rf_center
            + pre_span
            + (reph_span if reph_alone else 0.0)
            + echo_offset
        )
        te_delay = solve_delay(te, te_min, "TE", system)

        reph_on_wait = reph_alone or te_delay >= reph_span > 0
        wait_span = te_delay + (reph_span if reph_alone else 0.0)
        wait_te = pp.make_delay(wait_span) if wait_span else None

        for i_arm in range(n_arms):
            if gz is not None:
                self.seq.add_block(rf, gz)
            else:
                self.seq.add_block(rf)
            if wait_te is not None:
                self.seq.add_block(wait_te, *(present(gz_reph) if reph_on_wait else ()))
            if gz_pre is not None:
                self.seq.add_block(gz_pre, *_armed(trigger))
            for i_echo in range(n_echoes):
                self.seq.add_block(
                    _at(gx, i_arm),
                    _at(gy, i_arm),
                    adc,
                    *adc_labels,
                    *([echo_labels[i_echo]] if n_echoes > 1 else []),
                    *(present(gz_reph) if not reph_on_wait and i_echo == 0 else ()),
                    *(_armed(trigger) if gz_pre is None else ()),
                )
            if gz_rew is not None or gz_spoil is not None:
                self.seq.add_block(
                    *(event for event in (gz_rew, gz_spoil) if event is not None)
                )

        tr_min = self.seq.duration()[0] / n_arms
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        wait_tr = None
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            for _ in range(n_arms):
                self.seq.add_block(wait_tr)
        self._lay_out_arms(n_arms, wait_tr)

        self.echo_time = te_min + te_delay
        self.echo_spacing = pp.calc_duration(_at(gx, 0)) if n_echoes > 1 else 0.0
        self.center = self.echo_time + rf_center
        self.duration = tr_min + tr_delay
        self.bandwidth_hz = 1.0 / dwell
        self.n_samples = n_samples
        self.readout_duration = readout_duration
        # A full spoke runs edge to edge, so it crosses the centre halfway.
        self.center_sample = n_samples // 2


class RadialReadout2D(_RadialReadout):
    """A full radial spoke through the centre of a plane.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.RadialReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=128,
    ... )
    >>> len(readout.blocks), readout.gx.channel
    (2, 'x')

    >>> any(event is readout.gz_reph for event in readout.blocks[1])
    True
    """


class RadialStackReadout(_RadialReadout):
    """Radial spokes in-plane, Cartesian partitions along z: stack of stars."""

    _phase_axis = "z"


class RadialProjectionReadout(_RadialReadout):
    """Radial spokes for a spherical projection acquisition.

    The loop supplies 3D rotations. Partition encoding is not supported.
    """

    _rotated_axes = AXES


# ======================================================================
# Solved trajectories: spiral and rosette
# ======================================================================


class NonCartesianReadout(_ArmedReadout):
    """A solved interleave with moment bridges and a repetition-time budget.

    Nonzero k-space endpoints require prewinder or rewinder blocks. A
    subclass supplies a NonCartesianGradient; the acquisition loop controls
    per-shot orientation.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given, left-aligned in whichever block follows
        the pulse.
    gx, gy : GradEvent
        The interleave. Lists of one entry per angle when ``explicit``.
    gx_pre, gy_pre : GradEvent
        Bridges reaching the start of the path, when it is not k = 0.
    gx_rew, gy_rew : GradEvent
        Bridges returning to k = 0, when the path does not end there.
    gz_pre, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler, when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``; a bare event when there is one.
    wait_te, wait_tr : DelayEvent
        Present only when a TE or TR longer than the minimum was asked for.
    wait_pre, wait_rew : DelayEvent
        Pads setting the span of the prewinder and rewinder blocks, which can
        outlast their gradients. Absent when that block is not played.
    trajectory : NonCartesianGradient
        The designed interleave, for its ``trajectory`` array and timings.
    echo_spacing : float
        Time between consecutive echoes (s), zero for a single-echo readout.

    Parameters
    ----------
    trajectory : NonCartesianGradient
        Solved gradient interleave with ADC sampling and moment bridges.

    Other Parameters
    ----------------
    system, rf, gz, gz_reph, fov_z, matrix_z, te, tr
        As in :class:`~pypulseqpp.sequences.readout.noncartesian._RadialReadout`.
    spoiling_cycles, voxel_size_m, spoiling_axis, n_echoes, explicit, angles, labels, trigger
        As in :class:`~pypulseqpp.sequences.readout.noncartesian._RadialReadout`.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.SpiralReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=64, design_interleaves=8,
    ... )
    >>> isinstance(readout, design.NonCartesianReadout)
    True
    """

    _phase_axis: str | None = None

    _rotated_axes: tuple[str, ...] = ("x", "y")

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        trajectory: NonCartesianGradient,
        fov_z: float | None = None,
        matrix_z: int | None = None,
        te: float | None = None,
        tr: float | None = None,
        spoiling_cycles: float = 0.0,
        voxel_size_m: float | None = None,
        spoiling_axis: str = "z",
        n_echoes: int = 1,
        explicit: bool = False,
        angles: Any = None,
        labels: tuple[str, ...] | None = None,
        trigger: Any = None,
    ) -> None:
        n_echoes = _checked_layout(
            n_echoes, spoiling_cycles, spoiling_axis, explicit, angles, labels
        )
        arms = (
            [trajectory.rotated(float(angle)) for angle in np.atleast_1d(angles)]
            if explicit
            else [trajectory]
        )

        gz_pre, gz_rew = _partition_encode(
            self._phase_axis, fov_z, matrix_z, type(self).__name__, system
        )
        gz_reph = _accept_rephaser(gz_reph, self, present(self._phase_axis))

        gz_spoil = None
        if spoiling_cycles:
            if voxel_size_m is None:
                voxel_size_m = _resolution(trajectory)
            if voxel_size_m <= 0:
                raise ValueError("voxel_size_m must be positive")
            gz_spoil, _, _ = pp.make_crusher(
                spoiling_cycles, voxel_size_m, spoiling_axis, system=system
            )

        # Every explicit arm plays the base bridges' timing corners (rotation
        # turns the amplitude pair, never the timing), so these spans agree
        # across arms by construction; the max() is the shared value.
        raster = system.block_duration_raster
        natural_pre_span = pp.ceil_to_raster(
            max(
                (
                    pp.calc_duration(*arm.prewinders, *present(gz_pre))
                    for arm in arms
                    if arm.prewinders or gz_pre is not None
                ),
                default=0.0,
            ),
            raster,
        )
        rew_span = pp.ceil_to_raster(
            max(
                (
                    pp.calc_duration(
                        *arm.rewinders, *present(gz_rew), *present(gz_spoil)
                    )
                    for arm in arms
                    if arm.rewinders or gz_rew is not None or gz_spoil is not None
                ),
                default=0.0,
            ),
            raster,
        )

        adc = trajectory.adc
        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]
        # The readout plays the train, so the readout writes the counter that
        # indexes it: a scan loop hands over a whole arm and never sees the
        # echo boundaries inside it.
        echo_labels = [
            pp.make_label(type="SET", label="ECO", value=i_echo)
            for i_echo in range(n_echoes)
        ]

        self.seq = pp.Sequence(system)

        rf_center = float(rf.delay) + float(rf.center)
        rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz) if gz is not None else pp.calc_duration(rf), raster
        )
        te_base = (
            rf_block - rf_center + _echo_offset_of(trajectory, system.grad_raster_time)
        )
        wait_span, pre_span, echo_time = solve_rephasing(
            te,
            te_base,
            natural_pre_span,
            pp.calc_duration(gz_reph) if gz_reph is not None else 0.0,
            system,
        )
        wait_te = pp.make_delay(wait_span) if wait_span else None

        gx_pre, gy_pre = _bracket(arms, "prewinders", "right", pre_span, system)
        gx, gy = _bracket(arms, "gradients", None, 0.0, system)
        gx_rew, gy_rew = _bracket(arms, "rewinders", "left", rew_span, system)
        wait_pre = pp.make_delay(pre_span) if pre_span else None
        wait_rew = pp.make_delay(rew_span) if rew_span else None

        # Replaying an arm re-enters k where the last one left it, so an echo
        # train has to come back to the head of the arm between echoes: the
        # traversal's own rewinder unwinds it and its prewinder enters again.
        # This bracket carries the in-plane halves only -- the partition
        # encode, the slice rewinder and the end-of-TR spoiler bracket the
        # repetition, not the echo, and replaying them would spoil the train.
        echo_rew_span = _natural_span(arms, "rewinders", raster)
        echo_pre_span = _natural_span(arms, "prewinders", raster)
        gx_echo_rew, gy_echo_rew = _bracket(
            arms, "rewinders", "left", echo_rew_span, system
        )
        gx_echo_pre, gy_echo_pre = _bracket(
            arms, "prewinders", "right", echo_pre_span, system
        )

        for i_arm in range(len(arms)):
            if gz is not None:
                self.seq.add_block(rf, gz)
            else:
                self.seq.add_block(rf)
            if wait_te is not None:
                self.seq.add_block(wait_te, *present(gz_reph))
            if wait_pre is not None:
                self.seq.add_block(
                    *present(_at(gx_pre, i_arm)),
                    *present(_at(gy_pre, i_arm)),
                    *present(gz_pre),
                    *(() if wait_te else present(gz_reph)),
                    *_armed(trigger),
                    wait_pre,
                )
            for i_echo in range(n_echoes):
                if i_echo:
                    if echo_rew_span:
                        self.seq.add_block(
                            *present(_at(gx_echo_rew, i_arm)),
                            *present(_at(gy_echo_rew, i_arm)),
                            pp.make_delay(echo_rew_span),
                        )
                    if echo_pre_span:
                        self.seq.add_block(
                            *present(_at(gx_echo_pre, i_arm)),
                            *present(_at(gy_echo_pre, i_arm)),
                            pp.make_delay(echo_pre_span),
                        )
                self.seq.add_block(
                    _at(gx, i_arm),
                    _at(gy, i_arm),
                    adc,
                    *adc_labels,
                    *([echo_labels[i_echo]] if n_echoes > 1 else []),
                )
            if wait_rew is not None:
                self.seq.add_block(
                    *present(_at(gx_rew, i_arm)),
                    *present(_at(gy_rew, i_arm)),
                    *present(gz_rew),
                    *present(gz_spoil),
                    wait_rew,
                )

        tr_min = self.seq.duration()[0] / len(arms)
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        wait_tr = None
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            for _ in arms:
                self.seq.add_block(wait_tr)
        self._lay_out_arms(len(arms), wait_tr)

        self.trajectory = trajectory
        self.echo_time = echo_time
        self.echo_spacing = (
            pp.calc_duration(_at(gx, 0)) + echo_rew_span + echo_pre_span
            if n_echoes > 1
            else 0.0
        )
        self.center = echo_time + rf_center
        self.duration = tr_min + tr_delay
        self.bandwidth_hz = 1.0 / float(adc.dwell)
        self.n_samples = int(adc.num_samples)
        # Where in the window k passes through zero: the first sample of a
        # centre-out arm, halfway along one that runs in and back out.
        crossing = _echo_offset_of(trajectory, system.grad_raster_time) - float(
            adc.delay
        )
        self.center_sample = int(
            min(max(round(crossing / float(adc.dwell)), 0), self.n_samples - 1)
        )


class _SpiralReadout(NonCartesianReadout):
    """Spiral design parameters shared by 2D, stack and projection readouts.

    Parameters
    ----------
    fov : float
        Isotropic field of view (m).
    matrix : int
        In-plane matrix size.
    design_interleaves : int, optional
        Nominal pitch, not the number of arms acquired.
    direction : {'outward', 'inward', 'in_out'}, optional
        Centre-to-edge, edge-to-centre, or edge-to-centre-to-edge traversal.
    density : {'constant', 'variable', 'dual'}, optional
        Constant pitch, a radial power-law transition, or a logistic transition.
    inner_design_interleaves, outer_design_interleaves : float, optional
        Local pitch at the centre and edge. Inner defaults to design_interleaves;
        outer defaults to twice inner for variable density and is required for dual.
    variable_density_power : float, optional
        Positive exponent of the normalised radius for variable density.
    transition_radius, transition_speed : float, optional
        Normalised transition radius (between 0 and 1) and positive logistic
        steepness for dual density.
    oversampling : float, optional
        ADC oversampling factor, at least one.
    readout_bandwidth_hz : float, optional
        Requested ADC sampling rate (Hz), not bandwidth per pixel.
    n_points : int, optional
        Geometric path samples supplied to the solver, not ADC samples.
    derate : bool, optional
        Apply the package's system derates before solving.

    See Also
    --------
    NonCartesianReadout : Shared RF, timing, spoiling and orientation parameters.

    """

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: float,
        matrix: int,
        design_interleaves: int = 16,
        direction: str = "outward",
        density: str = "constant",
        inner_design_interleaves: float | None = None,
        outer_design_interleaves: float | None = None,
        variable_density_power: float = 2.0,
        transition_radius: float = 0.5,
        transition_speed: float = 12.0,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        n_points: int = 1024,
        derate: bool = True,
        **kwargs: Any,
    ) -> None:
        trajectory = Spiral(
            system,
            fov,
            matrix,
            design_interleaves,
            direction=direction,
            density=density,
            inner_design_interleaves=inner_design_interleaves,
            outer_design_interleaves=outer_design_interleaves,
            variable_density_power=variable_density_power,
            transition_radius=transition_radius,
            transition_speed=transition_speed,
            num_points=n_points,
            bandwidth_hz_px=readout_bandwidth_hz,
            oversamp=oversampling,
            derate=derate,
        )
        super().init_module(system, rf, gz, gz_reph, trajectory=trajectory, **kwargs)


class SpiralReadout2D(_SpiralReadout):
    """One spiral arm in a plane.

    ``direction`` runs the arm centre-to-edge (``'outward'``), edge-to-centre
    (``'inward'``) or edge-to-centre-to-edge (``'in_out'``); ``density``
    chooses a constant, variable or dual pitch. ``design_interleaves`` sets
    that pitch and is **not** how many arms the loop plays.

    ``readout_bandwidth_hz`` is the requested sample spacing along the arm; the
    solver stretches the waveform to hold it when the time-optimal traversal
    would sample faster. Read ``bandwidth_hz`` for what was achieved.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.SpiralReadout2D(
    ...     system, excitation.rf, excitation.gz,
    ...     fov=0.22, matrix=128, design_interleaves=16, direction="in_out",
    ... )
    >>> readout.trajectory.direction
    'in_out'
    """


class SpiralStackReadout(_SpiralReadout):
    """Spiral arms in-plane, Cartesian partitions along z."""

    _phase_axis = "z"


class SpiralProjectionReadout(_SpiralReadout):
    """Spiral arms turned over a sphere."""

    _rotated_axes = AXES


class _RosetteReadout(NonCartesianReadout):
    """Rosette design parameters shared by 2D, stack and projection readouts.

    Parameters
    ----------
    fov : float
        Isotropic field of view (m).
    matrix : int
        In-plane matrix size.
    petals : int, optional
        Centre-to-centre radial lobes in one interleave, not shots.
    angular_frequency_ratio : float, optional
        Positive angular-to-radial frequency ratio.
    echo_spacing_s : float, optional
        Requested centre-crossing interval (s). The waveform is stretched when
        necessary; requests shorter than the time-optimal spacing are rejected.
    oversampling : float, optional
        ADC oversampling factor, at least one.
    readout_bandwidth_hz : float, optional
        Requested ADC sampling rate (Hz), not bandwidth per pixel.
    derate : bool, optional
        Apply the package's system derates before solving.

    See Also
    --------
    NonCartesianReadout : Shared RF, timing, spoiling and orientation parameters.

    """

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: float,
        matrix: int,
        petals: int = 5,
        angular_frequency_ratio: float = 3.0 / 5.0,
        echo_spacing_s: float | None = None,
        oversampling: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        derate: bool = True,
        **kwargs: Any,
    ) -> None:
        trajectory = Rosette(
            system,
            fov,
            matrix,
            petals=petals,
            angular_frequency_ratio=angular_frequency_ratio,
            echo_spacing_s=echo_spacing_s,
            bandwidth_hz_px=readout_bandwidth_hz,
            oversamp=oversampling,
            derate=derate,
        )
        super().init_module(system, rf, gz, gz_reph, trajectory=trajectory, **kwargs)


class RosetteReadout2D(_RosetteReadout):
    """One multi-petal rosette interleave in a plane."""


class RosetteStackReadout(_RosetteReadout):
    """Rosette petals in-plane, Cartesian partitions along z."""

    _phase_axis = "z"


class RosetteProjectionReadout(_RosetteReadout):
    """Rosette petals turned over a sphere."""

    _rotated_axes = AXES


# ======================================================================
# Shared arithmetic
# ======================================================================


def _checked_layout(
    n_echoes, spoiling_cycles, spoiling_axis, explicit, angles, labels=None
) -> int:
    n_echoes = int(n_echoes)
    if n_echoes < 1:
        raise ValueError("n_echoes must be >= 1")
    if n_echoes > 1 and "ECO" in (labels or ()):
        raise ValueError(
            "ECO counts the echoes of the train the readout itself plays, so it "
            "is the readout that writes it; drop it from labels"
        )
    if spoiling_cycles < 0:
        raise ValueError("spoiling_cycles must be >= 0")
    if spoiling_axis not in AXES:
        raise ValueError(f"spoiling_axis must be one of {AXES}, got {spoiling_axis!r}")
    if explicit and angles is None:
        raise ValueError("explicit=True needs the angles to lay out")
    if angles is not None and not explicit:
        raise ValueError(
            "angles are a sampling pattern, which a readout does not hold; rotate the "
            "interleave in the scan loop, or pass explicit=True to write every arm out"
        )
    return n_echoes


def _accept_rephaser(gz_reph, module, occupied):
    """Return a left-aligned rephaser on an axis invariant under shot rotation.

    Only z is accepted for in-plane or stack readouts; projections accept
    none. Return None when no rephaser is supplied.
    """
    if gz_reph is None:
        return None
    if gz_reph.channel in module._rotated_axes:
        free = tuple(axis for axis in AXES if axis not in module._rotated_axes)
        hint = (
            f"select the slice on {free[0]}"
            if len(free) == 1
            else "excite with is_slab=True so the selection gradient carries its own rephaser"
        )
        turned = (
            ", ".join(module._rotated_axes[:-1]) + f" and {module._rotated_axes[-1]}"
        )
        raise ValueError(
            f"{type(module).__name__} is oriented by rotating {turned}, so a slice rephaser "
            f"on {gz_reph.channel} would be turned with the interleave; {hint}"
        )
    return left_align_rephaser(gz_reph, occupied, type(module).__name__)


def _partition_encode(axis, fov_z, matrix_z, owner, system):
    """Return a stack's partition encode and rewinder.

    Reject partition arguments for non-stack readouts.
    """
    if axis is None:
        if fov_z is not None or matrix_z is not None:
            raise ValueError(
                f"{owner} encodes no partition axis, so fov_z and matrix_z mean nothing to "
                "it; a stack variant is the class that takes them"
            )
        return None, None
    if not fov_z or not matrix_z:
        raise ValueError("a stack needs fov_z and matrix_z")
    gz_pre = pp.make_phase_encoding(axis, float(fov_z) / int(matrix_z), system=system)
    return gz_pre, pp.scale_grad(gz_pre, -1.0)


def _armed(trigger):
    return () if trigger is None else (trigger,)


def _at(events, index):
    return events[index] if isinstance(events, list) else events


def _share_time_grid(system, events):
    """Put each bracket's x/y events on the union of their vertex times.

    Pad shorter events with zero at the aligned edge. Shared timing preserves
    the same piecewise-linear waveform under explicit or extension rotation.
    """
    present = [event for event in events if event is not None]
    if len(present) < 2:
        return events

    times = [
        float(event.delay) + np.asarray(event.tt, dtype=float) for event in present
    ]
    grid = np.unique(np.concatenate(times))
    base = float(grid[0])

    shared = []
    for event in events:
        if event is None:
            shared.append(None)
            continue
        own = float(event.delay) + np.asarray(event.tt, dtype=float)
        amplitudes = np.interp(
            grid, own, np.asarray(event.waveform, dtype=float), left=0.0, right=0.0
        )
        rebuilt = pp.make_extended_trapezoid(
            channel=event.channel,
            amplitudes=amplitudes,
            times=grid - base,
            system=system,
        )
        rebuilt.delay = base
        shared.append(rebuilt)
    return shared


def _natural_span(arms, attribute, raster):
    """Longest an arm's ``attribute`` bracket runs, on the block raster."""
    return pp.ceil_to_raster(
        max(
            (
                pp.calc_duration(*getattr(arm, attribute))
                for arm in arms
                if getattr(arm, attribute)
            ),
            default=0.0,
        ),
        raster,
    )


def _bracket(arms, attribute, alignment, span, system):
    """X and y halves of one bracket, per arm, padded to ``span``.

    Returns a bare event when every arm shares one -- the single-interleave
    case -- and a list of one per arm otherwise, which is what makes the module
    publish ``gx`` as a list exactly when the arms really differ.
    """
    per_arm: dict[str, list] = {"x": [], "y": []}
    for arm in arms:
        events = list(getattr(arm, attribute))
        if alignment and events:
            events = list(pp.align(**{alignment: [*events, pp.make_delay(span)]}))[:-1]
        halves = [
            next((e for e in events if e.channel == channel), None)
            for channel in ("x", "y")
        ]
        if alignment:
            halves = _share_time_grid(system, halves)
        for channel, half in zip(("x", "y"), halves, strict=True):
            per_arm[channel].append(half)

    result = []
    for channel in ("x", "y"):
        entries = per_arm[channel]
        if all(entry is None for entry in entries):
            result.append(None)
        elif len(entries) == 1:
            result.append(entries[0])
        else:
            result.append(entries)
    return result


def _resolution(trajectory: NonCartesianGradient) -> float:
    """Nominal resolution of an interleave: half a period at its largest |k|."""
    path = np.asarray(trajectory.trajectory, dtype=float)
    kmax = float(np.max(np.linalg.norm(path[:, :2], axis=1)))
    return 1.0 / (2.0 * kmax)


def _echo_offset_of(trajectory: NonCartesianGradient, raster: float) -> float:
    """Return the readout's nearest k=0 crossing time in seconds.

    Integrate using each event's stored times, which may be nonuniform
    vertices rather than raster samples.
    """
    events = trajectory.gradients
    span = max(float(e.delay) + float(np.asarray(e.tt)[-1]) for e in events)
    grid = np.unique(
        np.concatenate(
            [float(e.delay) + np.asarray(e.tt, dtype=float) for e in events]
            + [np.arange(0.0, span + raster, raster)]
        )
    )
    grid = grid[grid <= span + 1e-12]

    path = np.zeros((grid.size, len(events)))
    for index, event in enumerate(events):
        times = float(event.delay) + np.asarray(event.tt, dtype=float)
        amplitudes = np.interp(grid, times, np.asarray(event.waveform, dtype=float))
        path[1:, index] = np.cumsum(
            0.5 * (amplitudes[1:] + amplitudes[:-1]) * np.diff(grid)
        )

    for pre in trajectory.prewinders:
        path[:, trajectory.axes.index(pre.channel)] += float(
            np.trapezoid(pre.waveform, pre.tt)
        )
    return float(grid[int(np.argmin(np.linalg.norm(path, axis=1)))])
