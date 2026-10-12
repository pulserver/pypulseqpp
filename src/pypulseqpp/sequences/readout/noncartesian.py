"""Non-Cartesian readouts: one designed interleaf, played as a whole repetition."""

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
    """Block layouts of the arms of a non-Cartesian readout.

    A compact layout holds one base arm, which the caller rotates per shot; an
    explicit layout holds each rotated arm.
    """

    def arm(self, index: int) -> list[tuple]:
        """Return the blocks of one arm, as built.

        Parameters
        ----------
        index : int
            Arm number, counting from zero. A compact layout has one arm, which
            every index returns.

        Returns
        -------
        list of tuple
            One tuple of events per block.
        """
        return self._arms[index if len(self._arms) > 1 else 0]

    def _lay_out_arms(self, n_arms: int, tail: Any = None) -> None:
        """Split the blocks into one layout per arm, each closed by ``tail``, the TR wait."""
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
    invariant by the shot rotation. With ``explicit=True`` the spokes are
    written out, one per entry of ``angles``; otherwise the caller rotates one
    base spoke per shot.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        Pulse that opens the repetition.
    gz : GradEvent, default=None
        Selection gradient played with ``rf``.
    gz_reph : GradEvent, default=None
        Rephaser of ``gz``, left-aligned in the first block after the pulse
        (the TE wait if there is one, else the prephaser block). Only on an
        axis the shot rotation leaves unchanged: z for an in-plane acquisition,
        none for a projection.
    fov : float
        Isotropic in-plane field of view (m).
    matrix : int
        In-plane matrix size; sets ``kmax = matrix / (2 * fov)``.
    fov_z, matrix_z : float, int, default=None
        Partition field of view (m) and count. Stacks only.
    te : float, default=None
        Echo time (s), from the RF centre to the centre crossing of the spoke.
        ``None`` is as short as possible.
    tr : float, default=None
        Repetition time (s). ``None`` is as short as possible.
    oversampling : float, default=1.0
        Readout oversampling: ADC samples per matrix point along the spoke.
    readout_bandwidth_hz : float, default=250000.0
        Requested ADC sampling rate (Hz); ``bandwidth_hz`` is the rate
        achieved on the ADC raster.
    spoiling_cycles : float, default=0.0
        Cycles of dephasing across ``voxel_size_m`` left at the end of the TR.
        Zero leaves the spoke rewound.
    voxel_size_m : float, default=None
        Length the spoiling is counted over (m); ``None`` is ``fov / matrix``.
    spoiling_axis : {'z', 'x', 'y'}, default='z'
        Gradient channel of the spoiler.
    n_echoes : int, default=1
        Traversals of the spoke per repetition, each retracing the same
        k-space coordinates with its own ``ECO`` label.
    explicit : bool, default=False
        Write out one spoke per entry of ``angles``.
    angles : array_like, default=None
        In-plane spoke rotations (rad). Required with ``explicit`` and rejected
        without it.
    labels : sequence of str, default=None
        Labels set to 0 on the acquisition block, to be changed per shot by the
        caller.
    trigger : TriggerEvent or DigitalOutputEvent, default=None
        Event armed on the block that opens the readout.

    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given.
    gx, gy : GradEvent
        The spoke from prephaser through rewinder. ``gy`` is the same waveform
        at zero amplitude, so a rotation always has a y event to fill. Lists
        with one entry per angle when ``explicit``.
    gz_pre, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler; present when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window, delayed onto the readout plateau.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    wait_te, wait_tr : DelayEvent
        Present only when ``te`` or ``tr`` exceeds the minimum.
    echo_time : float
        Echo time from the RF centre to the centre crossing (s).
    echo_spacing : float
        Time between consecutive echoes (s); zero for a single echo.
    bandwidth_hz : float
        Achieved ADC sampling rate ``1 / dwell`` (Hz).
    n_samples : int
        ADC samples per traversal.
    center_sample : int
        Index of the ADC sample at the k-space centre.
    readout_duration : float
        Duration of the readout plateau (s).
    n_arms : int
        Number of arm layouts: the angles when ``explicit``, else 1.

    Raises
    ------
    ValueError
        If a count is out of range, ``angles`` and ``explicit`` disagree, or
        ``te`` or ``tr`` is shorter than the module can achieve.
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

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       excitation = sequences.SpatialSelectiveExcitation(system, 15.0, 5e-3)
       readout = sequences.RadialReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           fov=0.22, matrix=64,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """


class RadialStackReadout(_RadialReadout):
    """Radial spokes in-plane with Cartesian partitions along z, a stack of stars.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> readout = design.RadialStackReadout(
    ...     system, slab.rf, slab.gz,
    ...     fov=0.22, matrix=64, fov_z=0.12, matrix_z=16,
    ... )
    >>> readout.gz_pre.channel
    'z'

    One spoke of one partition, and the k-space line it reads:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       slab = sequences.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
       readout = sequences.RadialStackReadout(
           system, slab.rf, slab.gz,
           fov=0.22, matrix=64, fov_z=0.12, matrix_z=16,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _phase_axis = "z"


class RadialProjectionReadout(_RadialReadout):
    """Radial spokes for a spherical projection acquisition.

    The loop supplies 3D rotations. Partition encoding is not supported.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> hard = design.NonSelectiveExcitation(system, 10.0, 0.5e-3)
    >>> readout = design.RadialProjectionReadout(
    ...     system, hard.rf, fov=0.24, matrix=64
    ... )
    >>> int(readout.adc.num_samples) > 0
    True

    One half-spoke of a spherical projection acquisition, before the loop
    turns it:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       hard = sequences.NonSelectiveExcitation(system, 10.0, 0.5e-3)
       readout = sequences.RadialProjectionReadout(
           system, hard.rf, fov=0.24, matrix=64
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _rotated_axes = AXES


# ======================================================================
# Solved trajectories: spiral and rosette
# ======================================================================


class NonCartesianReadout(_ArmedReadout):
    """A solved interleaf with its prewinder, rewinder and repetition-time budget.

    Nonzero k-space endpoints of the interleaf require prewinder or rewinder
    blocks. ``trajectory`` is any two-channel :class:`NonCartesianGradient`;
    the spiral and rosette readouts design theirs. The caller sets the
    orientation of each shot.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        Pulse that opens the repetition.
    gz : GradEvent, default=None
        Selection gradient played with ``rf``.
    gz_reph : GradEvent, default=None
        Rephaser of ``gz``, left-aligned in the first block after the pulse.
        Only on an axis the shot rotation leaves unchanged.
    trajectory : NonCartesianGradient
        Interleaf with its ADC, prewinders and rewinders.
    fov_z : float, default=None
        Partition field of view (m). Stacks only.
    matrix_z : int, default=None
        Partition count. Stacks only.
    te : float, default=None
        Echo time (s), from the RF centre to the point of the path nearest
        k = 0. ``None`` is as short as possible.
    tr : float, default=None
        Repetition time (s). ``None`` is as short as possible.
    spoiling_cycles : float, default=0.0
        Cycles of dephasing across ``voxel_size_m`` left at the end of the TR.
        Zero leaves the path rewound.
    voxel_size_m : float, default=None
        Length the spoiling is counted over (m); ``None`` is the resolution of
        the trajectory.
    spoiling_axis : {'z', 'x', 'y'}, default='z'
        Gradient channel of the spoiler.
    n_echoes : int, default=1
        Traversals of the path per repetition, joined by rewinders and
        prewinders, each with its own ``ECO`` label.
    explicit : bool, default=False
        Write out one interleaf per entry of ``angles``.
    angles : array_like, default=None
        In-plane interleaf rotations (rad). Required with ``explicit`` and
        rejected without it.
    labels : sequence of str, default=None
        Labels set to 0 on the acquisition block, to be changed per shot by the
        caller.
    trigger : TriggerEvent or DigitalOutputEvent, default=None
        Event armed on the block that opens the readout.


    Attributes
    ----------
    rf : RfEvent
        The pulse the module was given.
    gz : GradEvent
        Its selection gradient, if one was given.
    gz_reph : GradEvent
        Its rephaser, if one was given.
    gx, gy : GradEvent
        The interleaf. Lists with one entry per angle when ``explicit``.
    gx_pre, gy_pre : GradEvent
        Prewinding gradients to the start of the path; present when it does not
        start at k = 0.
    gx_rew, gy_rew : GradEvent
        Rewinding gradients to k = 0; present when the path does not end there.
    gz_pre, gz_rew : TrapEvent
        Partition encode and its rewinder. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler; present when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    wait_te, wait_tr : DelayEvent
        Present only when ``te`` or ``tr`` exceeds the minimum.
    wait_pre, wait_rew : DelayEvent
        Delays that set the duration of the prewinder and rewinder blocks, which
        can outlast their gradients; absent when the block is not played.
    trajectory : NonCartesianGradient
        The interleaf, for its k-space path and timings.
    echo_time : float
        Echo time from the RF centre to the point of the path nearest k = 0 (s).
    echo_spacing : float
        Time between consecutive echoes (s); zero for a single echo.
    bandwidth_hz : float
        ADC sampling rate ``1 / dwell`` (Hz).
    n_samples : int
        ADC samples per traversal.
    center_sample : int
        Index of the ADC sample nearest k = 0, clipped to the window.
    n_arms : int
        Number of arm layouts: the angles when ``explicit``, else 1.

    Raises
    ------
    ValueError
        If a count is out of range, ``angles`` and ``explicit`` disagree, or
        ``te`` or ``tr`` is shorter than the module can achieve.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> kmax = 64 / (2 * 0.22)
    >>> spoke = np.column_stack([np.linspace(-kmax, kmax, 64), np.zeros(64)])
    >>> readout = design.NonCartesianReadout(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     trajectory=design.Arbitrary(system, spoke, matrix=64),
    ... )
    >>> int(readout.adc.num_samples)
    64

    A single spoke supplied as an :class:`Arbitrary` trajectory, played in
    the block layout the readout lays out around it:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       excitation = sequences.SpatialSelectiveExcitation(system, 15.0, 5e-3)
       kmax = 64 / (2 * 0.22)
       spoke = np.column_stack([np.linspace(-kmax, kmax, 64), np.zeros(64)])
       readout = sequences.NonCartesianReadout(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           trajectory=sequences.Arbitrary(system, spoke, matrix=64),
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
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

        # Every explicit arm plays the base prewinder/rewinder timing corners (rotation
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
    design_interleaves : int, default=16
        Interleaf count the spiral pitch is designed for, not the number of
        arms acquired.
    direction : {'outward', 'inward', 'in_out'}, default='outward'
        Traversal from centre to edge, from edge to centre, or from edge
        through the centre to edge.
    density : {'constant', 'variable', 'dual'}, default='constant'
        Pitch constant, rising as a power of the radius, or changing between
        two plateaus by a logistic transition.
    inner_design_interleaves, outer_design_interleaves : float, default=None
        Local interleaf count at the centre and at the edge. The inner count
        defaults to ``design_interleaves``; the outer count to twice the inner
        for variable density, and is required for dual density.
    variable_density_power : float, default=2.0
        Exponent of the normalised radius, for variable density.
    transition_radius : float, default=0.5
        Normalised radius, in (0, 1), at which a dual-density spiral changes
        pitch.
    transition_speed : float, default=12.0
        Logistic rate per unit normalised radius of that change.
    oversampling : float, default=1.0
        ADC oversampling, at least 1.
    readout_bandwidth_hz : float, default=250000.0
        Requested ADC sampling rate (Hz); not the bandwidth per pixel.
    n_points : int, default=1024
        Path samples given to the solver, not ADC samples.
    derate : bool, default=True
        Apply the package's system derates before solving.
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

    ``design_interleaves`` sets the pitch of the arm and is not the number of
    arms the caller plays. Where the time-optimal traversal would space
    adjacent samples further apart than ``1 / (oversampling * fov)`` at the
    requested ``readout_bandwidth_hz``, the waveform is stretched; read
    ``bandwidth_hz`` for the sampling rate achieved.

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

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       excitation = sequences.SpatialSelectiveExcitation(system, 15.0, 5e-3)
       readout = sequences.SpiralReadout2D(
           system, excitation.rf, excitation.gz,
           fov=0.22, matrix=64, design_interleaves=8, n_points=256,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """


class SpiralStackReadout(_SpiralReadout):
    """Spiral arms in-plane, Cartesian partitions along z.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> readout = design.SpiralStackReadout(
    ...     system, slab.rf, slab.gz,
    ...     fov=0.22, matrix=64, design_interleaves=8, n_points=256,
    ...     fov_z=0.12, matrix_z=16,
    ... )
    >>> readout.gz_pre.channel
    'z'

    One interleaf of one partition, and the arm it traverses:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       slab = sequences.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
       readout = sequences.SpiralStackReadout(
           system, slab.rf, slab.gz,
           fov=0.22, matrix=64, design_interleaves=8, n_points=256,
           fov_z=0.12, matrix_z=16,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _phase_axis = "z"


class SpiralProjectionReadout(_SpiralReadout):
    """Spiral arms turned over a sphere.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> hard = design.NonSelectiveExcitation(system, 10.0, 0.5e-3)
    >>> readout = design.SpiralProjectionReadout(
    ...     system, hard.rf, fov=0.22, matrix=64,
    ...     design_interleaves=8, n_points=256,
    ... )
    >>> [gradient.channel for gradient in (readout.gx, readout.gy)]
    ['x', 'y']

    One arm of a spherical spiral acquisition, before the loop turns it:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       hard = sequences.NonSelectiveExcitation(system, 10.0, 0.5e-3)
       readout = sequences.SpiralProjectionReadout(
           system, hard.rf, fov=0.22, matrix=64,
           design_interleaves=8, n_points=256,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _rotated_axes = AXES


class _RosetteReadout(NonCartesianReadout):
    """Rosette design parameters shared by 2D, stack and projection readouts.

    Parameters
    ----------
    fov : float
        Isotropic field of view (m).
    matrix : int
        In-plane matrix size.
    petals : int, default=5
        Centre-to-centre lobes in one interleaf, not shots.
    angular_frequency_ratio : float, default=3.0 / 5.0
        Angular over radial frequency, positive.
    echo_spacing_s : float, default=None
        Mean centre-to-centre petal duration (s); the waveform is stretched to
        it. ``None`` is the minimum the system limits allow, and a shorter
        request is rejected.
    oversampling : float, default=1.0
        ADC oversampling, at least 1.
    readout_bandwidth_hz : float, default=250000.0
        Requested ADC sampling rate (Hz); not the bandwidth per pixel.
    derate : bool, default=True
        Apply the package's system derates before solving.
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
    """One multi-petal rosette interleaf in a plane.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    >>> readout = design.RosetteReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=64,
    ... )
    >>> [gradient.channel for gradient in (readout.gx, readout.gy)]
    ['x', 'y']

    One interleaf, whose petals pass through the centre of k-space once per
    petal:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = sequences.SpatialSelectiveExcitation(system, 15.0, 5e-3)
       readout = sequences.RosetteReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           fov=0.22, matrix=64,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """


class RosetteStackReadout(_RosetteReadout):
    """Rosette petals in-plane, Cartesian partitions along z.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> readout = design.RosetteStackReadout(
    ...     system, slab.rf, slab.gz,
    ...     fov=0.22, matrix=64, fov_z=0.12, matrix_z=16,
    ... )
    >>> readout.gz_pre.channel
    'z'

    One interleaf of one partition, and the petals it traverses:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       slab = sequences.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
       readout = sequences.RosetteStackReadout(
           system, slab.rf, slab.gz, fov=0.22, matrix=64, fov_z=0.12, matrix_z=16
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _phase_axis = "z"


class RosetteProjectionReadout(_RosetteReadout):
    """Rosette petals turned over a sphere.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m",
    ...                  max_slew=150, slew_unit="T/m/s")
    >>> hard = design.NonSelectiveExcitation(system, 10.0, 0.5e-3)
    >>> readout = design.RosetteProjectionReadout(
    ...     system, hard.rf, fov=0.22, matrix=64
    ... )
    >>> int(readout.adc.num_samples) > 0
    True

    One interleaf of a spherical rosette acquisition, before the loop turns
    it:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       hard = sequences.NonSelectiveExcitation(system, 10.0, 0.5e-3)
       readout = sequences.RosetteProjectionReadout(
           system, hard.rf, fov=0.22, matrix=64
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

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
            "interleaf in the scan loop, or pass explicit=True to write every arm out"
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
            f"on {gz_reph.channel} would be turned with the interleaf; {hint}"
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

    Returns a bare event when every arm shares one -- the single-interleaf
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
    """Nominal spatial resolution of an interleaf, ``1 / (2 kmax)`` at its largest |k|."""
    path = np.asarray(trajectory.trajectory, dtype=float)
    kmax = float(np.max(np.linalg.norm(path[:, :2], axis=1)))
    return 1.0 / (2.0 * kmax)


def _echo_offset_of(trajectory: NonCartesianGradient, raster: float) -> float:
    """Return the time in seconds at which the readout passes closest to k = 0.

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
