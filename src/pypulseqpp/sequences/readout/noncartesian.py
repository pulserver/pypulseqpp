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
from ._common import AXES, left_align_rephaser, present, solve_delay
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

    The prephaser and rewinder each cancel half the plateau area, so the spoke
    leaves no in-plane moment at any angle and the ADC samples the plateau. A
    stack plays its partition encode and rewinder as one z waveform in the
    same block, and the spoiler, when asked for, in a block of its own on a
    fixed axis. A slice rephaser may share an axis left invariant by the shot
    rotation. With ``explicit=True`` the spokes are
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
    gz_enc : GradEvent
        Partition encode, ending where the readout starts, and its rewinder,
        starting where it ends, as one waveform at ``+kmax``; scale it by
        ``kz / kmax`` per partition. Its net area is zero. Stacks only.
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
        gx, pre_span, rew_span = _balanced_spoke(readout_area, readout_duration, system)

        lobe = _partition_lobe(
            self._phase_axis, fov_z, matrix_z, type(self).__name__, system
        )
        lobe_span = _partition_span(lobe, system)
        if gz_reph is not None:
            gz_reph = _accept_rephaser(gz_reph, self, ())
        reph_span = pp.calc_duration(gz_reph) if gz_reph is not None else 0.0

        gz_spoil = _spoiler(
            spoiling_cycles,
            voxel_size_m,
            fov / int(matrix),
            spoiling_axis,
            system,
            lobe,
        )

        raster = system.block_duration_raster
        rf_center = float(rf.delay) + float(rf.center)
        rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz) if gz is not None else pp.calc_duration(rf),
            raster,
        )
        lead = max(pre_span, lobe_span)
        te_base = rf_block - rf_center + 0.5 * readout_duration
        placement, lead, wait_span, reph_block = _place_rephaser(
            te,
            te_base,
            lead,
            reph_span,
            has_partitions=lobe is not None,
            system=system,
        )
        block_span = pp.ceil_to_raster(
            lead + readout_duration + max(rew_span, lobe_span), raster
        )

        gx.delay = lead - pre_span
        gy = pp.scale_grad(gx, 0.0)
        gy.channel = "y"
        if explicit:
            angles = np.atleast_1d(np.asarray(angles, dtype=float))
            base = gx
            gx = [pp.scale_grad(base, float(np.cos(angle))) for angle in angles]
            gy = []
            for angle in angles:
                turned = pp.scale_grad(base, float(np.sin(angle)))
                turned.channel = "y"
                gy.append(turned)
        gz_enc = (
            _partition_wave(lobe, lead, lead + readout_duration, system)
            if lobe is not None
            else None
        )
        adc = pp.make_adc(num_samples=n_samples, dwell=dwell, delay=lead, system=system)

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
        wait_te = pp.make_delay(wait_span) if wait_span else None

        self.seq = pp.Sequence(system)
        for i_arm in range(n_arms):
            self.seq.add_block(rf, *present(gz))
            if wait_te is not None:
                self.seq.add_block(
                    wait_te, *(present(gz_reph) if placement == "wait" else ())
                )
            if placement == "block":
                self.seq.add_block(gz_reph, pp.make_delay(reph_block))
            for i_echo in range(n_echoes):
                self.seq.add_block(
                    _at(gx, i_arm),
                    _at(gy, i_arm),
                    *present(gz_enc),
                    *(
                        present(gz_reph)
                        if placement == "readout" and i_echo == 0
                        else ()
                    ),
                    adc,
                    *adc_labels,
                    *([echo_labels[i_echo]] if n_echoes > 1 else []),
                    *(_armed(trigger) if i_echo == 0 else ()),
                )
            if gz_spoil is not None:
                self.seq.add_block(gz_spoil)

        tr_min = self.seq.duration()[0] / n_arms
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        wait_tr = None
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            for _ in range(n_arms):
                self.seq.add_block(wait_tr)
        self._lay_out_arms(n_arms, wait_tr)

        self.echo_time = te_base + wait_span + reph_block + lead
        self.echo_spacing = block_span if n_echoes > 1 else 0.0
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
    >>> readout.gz_enc.channel
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

    The prewinder, interleaf and rewinder play as one waveform per in-plane
    axis, so every interleaf ends at k = 0 whatever its rotation. A stack plays
    its partition encode and rewinder as one z waveform in the same block, and
    the spoiler, when asked for, in a block of its own on a fixed axis. ``trajectory`` is any two-channel :class:`NonCartesianGradient`;
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
        The prewinder, interleaf and rewinder as one waveform per axis, sampled
        at the gradient raster centres. Lists with one entry per angle when
        ``explicit``.
    gz_enc : GradEvent
        Partition encode, ending where the readout starts, and its rewinder,
        starting where it ends, as one waveform at ``+kmax``; scale it by
        ``kz / kmax`` per partition. Its net area is zero. Stacks only.
    gz_spoil : GradEvent
        End-of-TR spoiler; present when ``spoiling_cycles`` is nonzero.
    adc : AdcEvent
        The acquisition window.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    wait_te, wait_tr : DelayEvent
        Present only when ``te`` or ``tr`` exceeds the minimum.
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

        lobe = _partition_lobe(
            self._phase_axis, fov_z, matrix_z, type(self).__name__, system
        )
        lobe_span = _partition_span(lobe, system)
        if gz_reph is not None:
            gz_reph = _accept_rephaser(gz_reph, self, ())
        reph_span = pp.calc_duration(gz_reph) if gz_reph is not None else 0.0
        gz_spoil = _spoiler(
            spoiling_cycles,
            voxel_size_m,
            _resolution(trajectory),
            spoiling_axis,
            system,
            lobe,
        )

        raster = system.block_duration_raster
        grad_raster = system.grad_raster_time
        # Rotation turns the amplitudes of the brackets, never their corners,
        # so every explicit arm has the base arm's spans.
        pre_span = _natural_span(arms, "prewinders", grad_raster)
        rew_span = _natural_span(arms, "rewinders", grad_raster)
        read_span = max(pp.calc_duration(*arm.gradients) for arm in arms)
        echo_offset = _echo_offset_of(trajectory, grad_raster)

        rf_center = float(rf.delay) + float(rf.center)
        rf_block = pp.ceil_to_raster(
            pp.calc_duration(rf, gz) if gz is not None else pp.calc_duration(rf),
            raster,
        )
        te_base = rf_block - rf_center + echo_offset
        placement, lead, wait_span, reph_block = _place_rephaser(
            te,
            te_base,
            max(pre_span, lobe_span),
            reph_span,
            has_partitions=lobe is not None,
            system=system,
        )
        block_span = pp.ceil_to_raster(
            lead + read_span + max(rew_span, lobe_span), raster
        )

        gx, gy = _merged_arms(arms, lead, pre_span, read_span, rew_span, system)
        gz_enc = (
            _partition_wave(lobe, lead, lead + read_span, system)
            if lobe is not None
            else None
        )
        source = trajectory.adc
        adc = pp.make_adc(
            num_samples=int(source.num_samples),
            dwell=float(source.dwell),
            delay=float(source.delay) + lead,
            system=system,
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
        wait_te = pp.make_delay(wait_span) if wait_span else None

        self.seq = pp.Sequence(system)
        for i_arm in range(len(arms)):
            self.seq.add_block(rf, *present(gz))
            if wait_te is not None:
                self.seq.add_block(
                    wait_te, *(present(gz_reph) if placement == "wait" else ())
                )
            if placement == "block":
                self.seq.add_block(gz_reph, pp.make_delay(reph_block))
            for i_echo in range(n_echoes):
                self.seq.add_block(
                    *present(_at(gx, i_arm)),
                    *present(_at(gy, i_arm)),
                    *present(gz_enc),
                    *(
                        present(gz_reph)
                        if placement == "readout" and i_echo == 0
                        else ()
                    ),
                    adc,
                    *adc_labels,
                    *([echo_labels[i_echo]] if n_echoes > 1 else []),
                    *(_armed(trigger) if i_echo == 0 else ()),
                )
            if gz_spoil is not None:
                self.seq.add_block(gz_spoil)

        tr_min = self.seq.duration()[0] / len(arms)
        tr_delay = solve_delay(tr, tr_min, "TR", system)
        wait_tr = None
        if tr_delay:
            wait_tr = pp.make_delay(tr_delay)
            for _ in arms:
                self.seq.add_block(wait_tr)
        self._lay_out_arms(len(arms), wait_tr)

        self.trajectory = trajectory
        self.echo_time = te_base + wait_span + reph_block + lead
        self.echo_spacing = block_span if n_echoes > 1 else 0.0
        self.center = self.echo_time + rf_center
        self.duration = tr_min + tr_delay
        self.bandwidth_hz = 1.0 / float(adc.dwell)
        self.n_samples = int(adc.num_samples)
        # Where in the window k passes through zero: the first sample of a
        # centre-out arm, halfway along one that runs in and back out.
        crossing = echo_offset - float(source.delay)
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
    inner_design_interleaves : float, default=None
        Local interleaf count at the centre; defaults to ``design_interleaves``.
    outer_design_interleaves : float, default=None
        Local interleaf count at the edge; defaults to twice the inner count
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
    >>> readout.gz_enc.channel
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
    >>> readout.gz_enc.channel
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


def _balanced_spoke(readout_area, readout_duration, system):
    """Return a spoke whose prephaser and rewinder share the plateau's ramps.

    The prephaser ramps from zero into the plateau and the rewinder from the
    plateau back to zero, each with half the plateau's area and the opposite
    sign, so the spoke returns to the k-space point it started from.

    Parameters
    ----------
    readout_area : float
        Plateau area (1/m), ``2 * kmax``.
    readout_duration : float
        Plateau duration (s), on the gradient raster.
    system : pypulseqpp.Opts
        System limits.

    Returns
    -------
    gx : GradEvent
        The spoke on x, its plateau from ``pre_span`` to
        ``pre_span + readout_duration``.
    pre_span, rew_span : float
        Durations of the prephaser and the rewinder (s).
    """
    amplitude = readout_area / readout_duration
    _, pre_times, pre_amplitudes = pp.make_extended_trapezoid_area(
        area=-0.5 * readout_area,
        channel="x",
        grad_start=0.0,
        grad_end=amplitude,
        system=system,
    )
    _, rew_times, rew_amplitudes = pp.make_extended_trapezoid_area(
        area=-0.5 * readout_area,
        channel="x",
        grad_start=amplitude,
        grad_end=0.0,
        system=system,
    )
    pre_span = float(pre_times[-1])
    plateau_end = pre_span + readout_duration
    gx = pp.make_extended_trapezoid(
        channel="x",
        times=np.concatenate([pre_times, [plateau_end], plateau_end + rew_times[1:]]),
        amplitudes=np.concatenate([pre_amplitudes, [amplitude], rew_amplitudes[1:]]),
        system=system,
    )
    return gx, pre_span, float(rew_times[-1])


def _merged_arms(arms, lead, pre_span, read_span, rew_span, system):
    """Join each arm's prewinders, readout and rewinders into one waveform per axis.

    The prewinders end where the readout starts, at ``lead``, and the
    rewinders start where it ends; the trajectory designs them to meet the
    readout at its own amplitude, so each axis plays one continuous waveform,
    sampled at the centres of the gradient raster like the readout itself.

    Returns
    -------
    gx, gy : GradEvent or list of GradEvent
        One event per axis, or a list with one per arm when the arms differ.
    """
    pieces = [
        (_bracket(arms, "prewinders", "right", pre_span, system), lead - pre_span),
        (_bracket(arms, "gradients", None, 0.0, system), lead),
        (_bracket(arms, "rewinders", "left", rew_span, system), lead + read_span),
    ]
    raster = system.grad_raster_time
    n_points = round((lead + read_span + rew_span) / raster)
    centres = (np.arange(n_points) + 0.5) * raster
    per_axis = [[], []]
    for i_arm in range(len(arms)):
        samples = np.zeros((2, n_points))
        bracket_mask = np.zeros(n_points, dtype=bool)
        target = np.zeros(2)
        present_axes = [False, False]
        for axis in range(2):
            parts = [_at(halves[axis], i_arm) for halves, _ in pieces]
            for (_, offset), part, role in zip(
                pieces, parts, ("pre", "read", "rew"), strict=True
            ):
                if part is None:
                    continue
                present_axes[axis] = True
                times = offset + float(part.delay) + np.asarray(part.tt, dtype=float)
                inside = (centres > times[0] - 1e-12) & (centres < times[-1] + 1e-12)
                samples[axis, inside] += np.interp(
                    centres[inside], times, np.asarray(part.waveform, dtype=float)
                )
                target[axis] += _part_area(part, raster)
                if role != "read":
                    bracket_mask |= inside
        samples = _close_moment(samples, bracket_mask, target, raster)
        for axis, channel in enumerate(("x", "y")):
            per_axis[axis].append(
                pp.make_arbitrary_grad(
                    channel,
                    samples[axis],
                    first=0.0,
                    last=0.0,
                    delay=0.0,
                    system=system,
                )
                if present_axes[axis]
                else None
            )
    return [axis[0] if len(axis) == 1 else axis for axis in per_axis]


def _close_moment(samples, bracket_mask, target, raster):
    """Restore the designed net moment of an in-plane waveform pair.

    Sampling the corner where a readout meets its bracket at the raster
    centres trims a sliver of area that the bracket was designed to cancel.
    Each axis gets back its missing area as a multiple of one profile shared
    by both axes, the magnitude of the in-plane bracket, so an arm turned in
    the plane is corrected by the same turn.

    Parameters
    ----------
    samples : ndarray, shape (2, n)
        x and y samples at the raster centres.
    bracket_mask : ndarray of bool, shape (n,)
        Samples that belong to a prewinder or rewinder.
    target : ndarray, shape (2,)
        Net area each axis was designed to play (1/m).
    raster : float
        Gradient raster time (s).

    Returns
    -------
    ndarray, shape (2, n)
        The corrected samples.
    """
    profile = np.where(bracket_mask, np.hypot(samples[0], samples[1]), 0.0)
    weight = _played_area(profile, raster)
    if weight == 0.0:
        return samples
    residual = np.array(
        [t - _played_area(row, raster) for t, row in zip(target, samples, strict=True)]
    )
    return samples + residual[:, None] / weight * profile[None, :]


def _played_area(samples, raster, first=0.0, last=0.0):
    """Area of raster-centre samples joined linearly, from ``first`` to ``last``."""
    if samples.size == 0:
        return 0.0
    inner = float(np.sum(samples)) - 0.5 * float(samples[0] + samples[-1])
    return raster * (inner + 0.25 * (first + samples[0] + samples[-1] + last))


def _part_area(part, raster):
    """Played area of one bracket or readout event (1/m)."""
    times = np.asarray(part.tt, dtype=float)
    values = np.asarray(part.waveform, dtype=float)
    if part.type == "trap" or times[0] < 0.25 * raster:
        return (
            float(part.area)
            if part.type == "trap"
            else float(np.trapezoid(values, times))
        )
    return _played_area(
        values, raster, float(part.first or 0.0), float(part.last or 0.0)
    )


def _place_rephaser(te, te_base, lead, reph_span, *, has_partitions, system):
    """Choose where the slice rephaser plays, and solve the TE wait.

    The rephaser rides the TE wait when the wait is long enough to hold it.
    Otherwise a 2D readout plays it at the start of the acquisition block,
    ahead of the plateau, and a stack, whose z axis carries the partition
    encode there, gives it a block of its own.

    Parameters
    ----------
    te : float or None
        Requested echo time (s); ``None`` is as short as possible.
    te_base : float
        Echo time with no wait, no rephaser block and no lead-in (s).
    lead : float
        Time from the start of the acquisition block to the readout (s).
    reph_span : float
        Rephaser duration (s), zero when there is none.
    has_partitions : bool
        Whether the acquisition block carries a partition encode on z.
    system : pypulseqpp.Opts
        System limits.

    Returns
    -------
    placement : {'wait', 'readout', 'block'} or None
        Where the rephaser plays; None when there is none.
    lead : float
        Lead-in of the acquisition block, lengthened to hold the rephaser.
    wait_span : float
        TE wait (s).
    reph_block : float
        Duration of the rephaser's own block (s), zero unless it has one.

    Raises
    ------
    ValueError
        If ``te`` is shorter than the layout can achieve.
    """
    if not reph_span:
        return None, lead, solve_delay(te, te_base + lead, "TE", system), 0.0
    if te is not None:
        wait_span = solve_delay(te, te_base + lead, "TE", system)
        if wait_span >= reph_span - 1e-12:
            return "wait", lead, wait_span, 0.0
    if has_partitions:
        reph_block = pp.ceil_to_raster(reph_span, system.block_duration_raster)
        wait_span = solve_delay(te, te_base + reph_block + lead, "TE", system)
        return "block", lead, wait_span, reph_block
    lead = max(lead, reph_span)
    return "readout", lead, solve_delay(te, te_base + lead, "TE", system), 0.0


def _spoiler(cycles, voxel_size_m, default_voxel_m, axis, system, lobe=None):
    """Return the end-of-TR spoiler, or None when ``cycles`` is zero.

    On the axis of a partition encode ``lobe``, its encode at ``+kmax``, the
    spoiler has the polarity of the rewinder at ``+kmax``. At the first
    partition, ``-kmax``, the rewinder then runs into a spoiler of the
    opposite sign: the largest gradient switching of the scan falls on the
    partition a check of the edge of k-space plays.
    """
    if not cycles:
        return None
    if voxel_size_m is None:
        voxel_size_m = default_voxel_m
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    spoiler = pp.make_crusher(cycles, voxel_size_m, axis, system=system)[0]
    if lobe is not None and lobe.channel == axis and float(lobe.amplitude) > 0:
        return pp.scale_grad(spoiler, -1.0)
    return spoiler


_PARTITION_SLEW_MARGIN = 0.9


def _partition_lobe(axis, fov_z, matrix_z, owner, system):
    """Return a stack's partition encode at the outermost partition, ``+kmax``.

    Return None for a readout without partitions, and reject partition
    arguments given to one.
    """
    if axis is None:
        if fov_z is not None or matrix_z is not None:
            raise ValueError(
                f"{owner} encodes no partition axis, so fov_z and matrix_z mean nothing to "
                "it; a stack variant is the class that takes them"
            )
        return None
    if not fov_z or not matrix_z:
        raise ValueError("a stack needs fov_z and matrix_z")
    # The lobe is sampled at raster centres and rescaled to its area in
    # _partition_wave; designing it below the slew limit leaves room for that.
    return pp.make_trapezoid(
        axis,
        area=0.5 * int(matrix_z) / float(fov_z),
        max_slew=_PARTITION_SLEW_MARGIN * system.max_slew,
        system=system,
    )


def _partition_span(lobe, system):
    """Time a partition encode needs before the readout, its guard and zero samples included."""
    if lobe is None:
        return 0.0
    return pp.calc_duration(lobe) + 2 * system.grad_raster_time


def _partition_wave(lobe, start, end, system):
    """Return the partition encode and its rewinder as one waveform.

    The encode ends one gradient raster before ``start`` and the rewinder,
    its negative, begins one raster after ``end``, both counted from the start
    of the block, with the gradient off in between. The waveform is sampled at
    the centres of the gradient raster and opens and closes on a zero sample,
    so every partition scales one and the same waveform: scaling it by
    ``kz / kmax`` encodes any partition and leaves its net area zero.
    """
    rise, flat, fall = (
        float(lobe.rise_time),
        float(lobe.flat_time),
        float(lobe.fall_time),
    )
    amplitude = float(lobe.amplitude)
    span = rise + flat + fall
    raster = system.grad_raster_time
    # Samples at raster centres are joined linearly, so a lobe that ended on
    # the first sample would still play across its first half raster; one
    # raster of guard keeps the z gradient off throughout the readout, and
    # with it kz constant over every sample.
    encode_end, rewind_start = start - raster, end + raster
    # A zero sample before the encode and after the rewinder: a player that
    # holds a raster waveform's end samples over its edge half rasters plays
    # it exactly as written.
    first = encode_end - span - raster
    lobe_start = first + raster
    corners = np.array(
        [
            lobe_start,
            lobe_start + rise,
            encode_end - fall,
            encode_end,
            rewind_start,
            rewind_start + rise,
            rewind_start + rise + flat,
            rewind_start + span,
        ]
    )
    values = np.array([0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0]) * amplitude
    n_points = round((corners[-1] + raster - first) / raster)
    centres = first + (np.arange(n_points) + 0.5) * raster
    samples = np.interp(centres, corners, values)
    # Joining the samples linearly trims the lobes' corners; rescale so the
    # encode reaches the lobe's own area by the start of the readout.
    played_times = np.concatenate([[first], centres, [first + n_points * raster]])
    played = np.concatenate([[0.0], samples, [0.0]])
    grid = np.concatenate([played_times[played_times < start], [start]])
    samples *= float(lobe.area) / np.trapezoid(
        np.interp(grid, played_times, played), grid
    )
    return pp.make_arbitrary_grad(
        lobe.channel, samples, first=0.0, last=0.0, delay=float(first), system=system
    )


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
