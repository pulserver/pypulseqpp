"""Zero-echo-time readouts: excite on the readout gradient, acquire from k = 0 out."""

from __future__ import annotations

__all__ = ["ZteReadout"]

import math
from itertools import pairwise
from typing import Any

import numpy as np

import pypulseqpp as pp

from ..._angles import calc_projection_shell
from .._module import SequenceModule
from ._common import AXES, solve_delay

#: Fraction of ``max_grad`` a spoke may reach.
_READOUT_GRAD_MARGIN = 0.95


class ZteReadout(SequenceModule):
    """Continuous-gradient ZTE shell, including its initial and final ramps.

    Each view excites on a held gradient, then acquires while the gradient
    slews toward the next direction. The final view ramps to zero. A generated
    shell is replayed under ``shot_rotations``; the views are not to be
    rotated individually as well.

    Transmit ringdown and receiver dead time leave ``n_missing`` central
    samples unacquired; the module reports them and does not fill them. A
    longer TR lengthens the slew between views instead of adding a delay.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        A non-selective pulse, short enough that its bandwidth spans the spoke.
    fov : float
        Isotropic field of view (m).
    matrix : int
        Isotropic matrix size.
    directions : array_like, default=None
        Unit spoke directions ``(n_views, 3)``, in the order the shell visits
        them. When given, ``n_views``, ``n_shots`` and ``scheme`` are ignored
        and the shell is not replayed. Every turn is allotted the duration of
        the widest step, so the TR is set by the largest step.
    n_views : int, default=None
        Spokes in one shell. ``None`` splits a Nyquist-matched sphere of
        ``ceil(pi * matrix ** 2)`` spokes evenly between the shots, so a
        larger ``n_shots`` shortens the shell.
    n_shots : int, default=None
        Shells the sphere is split into, each the shell turned about z by
        ``2 * pi / n_shots`` relative to the previous one. Playing only some of
        the shots undersamples in angle. ``None`` is ``ceil(pi * (matrix - 1))``.
    scheme : {'spiral', 'quiet', 'meridian'}, default='spiral'
        Shape of the shell, as in :func:`pypulseqpp.calc_projection_shell`.
    oversampling : float, default=2.0
        Radial oversampling: the k-space sample spacing along the spoke is
        ``1 / (oversampling * fov)``.
    readout_bandwidth_hz : float, default=62500.0
        Requested ADC sampling rate (Hz); ``bandwidth_hz`` is the rate achieved
        on the ADC raster. It also sets the gradient amplitude, one sample
        per ``delta_k``.
    tr : float, default=None
        Pulse centre to pulse centre (s). ``None`` is as short as the widest
        turn allows; a longer TR slows the turns.
    dead_time_s : float, default=None
        Receiver dead time after the pulse (s), at least ``system.adc_dead_time``
        (the default); the transmit ringdown is added to it.
    labels : sequence of str, default=None
        Labels set to 0 on the acquisition block, to be changed per shot by the
        caller.


    Attributes
    ----------
    rf : RfEvent
        The pulse, delayed by the transmit dead time. It is non-selective and
        short, because it plays on a gradient and its bandwidth has to cover
        the spoke.
    g_ramp : list of GradEvent
        Ramp from zero to the first view's gradient, played once at the head
        of a shell; one event per axis.
    g_hold : list of list of GradEvent
        Per view, the gradient plateau under the pulse, held through the
        dead-time gap.
    g_read : list of list of GradEvent
        Per view, the plateau under the acquisition followed by the turn onto
        the next view; the last entry slews to zero instead.
    adc : AdcEvent
        The acquisition, delayed past the gap.
    adc_labels : LabelSetEvent or list of LabelSetEvent
        One per name in ``labels``.
    shot_rotations : numpy.ndarray
        Rotations about z ``(n_shots, 3, 3)`` that map the shell onto the spoke
        directions of each shot. Present only when the module generated the
        shell, not when ``directions`` is supplied.
    directions : numpy.ndarray
        Unit spoke directions of one shell ``(n_views, 3)``, in play order.
    step_rad : float
        Widest angle between consecutive views (rad). Every turn is allotted
        the duration of this step.
    tr : float
        Time from pulse centre to pulse centre (s).
    view_duration : float
        Duration of the plateaus held for one view, before the turn (s).
    n_samples, n_missing, n_nominal : int
        ADC samples acquired, samples lost to the gap, and samples of the full
        half-spoke.
    gap : float
        Time from the pulse centre to the first sample (s).
    bandwidth_hz : float
        Achieved ADC sampling rate ``1 / dwell`` (Hz).
    delta_k : float
        Radial k-space sample spacing (1/m).
    kmax : float
        Half-spoke extent (1/m).
    gradient_amplitude : float
        Gradient amplitude of a spoke (Hz/m).

    Raises
    ------
    ValueError
        If a count is out of range, the pulse is too long for the dwell, the
        dead-time gap covers the whole spoke, two consecutive views coincide,
        ``dead_time_s`` is below ``system.adc_dead_time``, or ``tr`` is shorter
        than the turn needs.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> hard = design.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)
    >>> zte = design.ZteReadout(
    ...     system, hard.rf, fov=0.24, matrix=96, n_views=64, n_shots=256
    ... )
    >>> len(zte.g_read), zte.shot_rotations.shape
    (64, (256, 3, 3))

    >>> zte.n_samples + zte.n_missing == zte.n_nominal
    True

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts()
       hard = sequences.NonSelectiveExcitation(system, 4.0, duration_s=10e-6)
       readout = sequences.ZteReadout(
           system, hard.rf, fov=0.24, matrix=32, n_views=16, n_shots=32,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        *,
        fov: float,
        matrix: int,
        directions: Any = None,
        n_views: int | None = None,
        n_shots: int | None = None,
        scheme: str = "spiral",
        oversampling: float = 2.0,
        readout_bandwidth_hz: float = 62.5e3,
        tr: float | None = None,
        dead_time_s: float | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        if fov <= 0 or int(matrix) < 2:
            raise ValueError("fov must be positive and matrix must be >= 2")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")

        shot_rotations = None
        if directions is None:
            # Rings hold the shots and the shell walks across them, so the two
            # spacings are reciprocal: ring to ring goes as 1 / n_views, within
            # a ring as 1 / n_shots. They agree at the equator when
            # n_shots = pi * n_views, which with a Nyquist-matched sphere over
            # the whole set puts n_views at the matrix size.
            if n_shots is None:
                n_shots = max(1, math.ceil(np.pi * (int(matrix) - 1)))
            total = math.ceil(np.pi * int(matrix) ** 2)
            n_views = (
                max(3, -(-total // int(n_shots))) if n_views is None else int(n_views)
            )
            directions, shot_rotations = calc_projection_shell(
                n_views, n_shots, scheme=scheme
            )
        directions = _unit(directions)
        dead_time_s = (
            system.adc_dead_time if dead_time_s is None else float(dead_time_s)
        )
        if dead_time_s < system.adc_dead_time:
            raise ValueError(
                f"dead_time_s must be at least the receiver's own "
                f"{system.adc_dead_time * 1e6:.1f} us, got {dead_time_s * 1e6:.1f} us"
            )

        # A spoke is a half line, so the sample count is half a matrix, and the
        # dwell fixes the amplitude: one delta_k of k per sample.
        delta_k = 1.0 / (oversampling * fov)
        kmax = 0.5 * int(matrix) / fov
        n_nominal = max(1, round(kmax / delta_k))
        dwell, _ = pp.calc_adc_timing(
            n_nominal,
            1.0 / readout_bandwidth_hz,
            grad_raster_time=system.grad_raster_time,
            adc_raster_time=system.adc_raster_time,
            min_readout_duration=kmax / (_READOUT_GRAD_MARGIN * system.max_grad),
        )
        amplitude = delta_k / dwell

        if float(rf.shape_dur) >= 2.0 * dwell - 1e-12:
            raise ValueError(
                f"the pulse is {float(rf.shape_dur) * 1e6:.3f} us long, which is not short "
                f"beside a {dwell * 1e6:.3f} us dwell: a ZTE pulse plays on the readout "
                f"gradient, so it has to be brief enough to excite the whole spoke"
            )

        # Encoding starts at the pulse centre, and the acquisition cannot start
        # until the transmit has rung down and the receiver has woken. The
        # block boundary goes at the end of the ringdown and the receiver's own
        # dead time is the ADC's delay, so neither pays for the other; anything
        # further the caller asked for is held before the boundary.
        rf.delay = max(float(rf.delay), system.rf_dead_time)
        rf_center = float(rf.delay) + float(rf.center)
        adc_delay = system.adc_dead_time
        hold_span = pp.ceil_to_raster(
            rf_center
            + float(rf.shape_dur)
            - float(rf.center)
            + float(rf.ringdown_time)
            + dead_time_s
            - adc_delay,
            system.grad_raster_time,
        )
        # A sample reports the centre of its dwell, so the first one is half a
        # dwell into the window.
        gap = hold_span - rf_center + adc_delay + 0.5 * dwell
        n_samples = math.ceil(n_nominal - gap * amplitude / delta_k)
        if n_samples < 1:
            raise ValueError(
                f"the {gap * 1e6:.1f} us gap after the pulse already carries k past the "
                f"{kmax:.1f} /m edge of the spoke, so there is nothing left to acquire; "
                f"raise readout_bandwidth_hz or shorten the pulse"
            )
        # A receiver digitises in groups, so the count the gap leaves is
        # rounded to a whole number of them -- down rather than up, because
        # a sample past kmax is outside the grid the spoke is reconstructed
        # on, where one short of it is merely one fewer.
        divisor = int(getattr(system, "adc_samples_divisor", 0) or 1)
        n_samples = max(divisor, n_samples // divisor * divisor)
        n_missing = n_nominal - n_samples
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell, delay=adc_delay, system=system
        )
        read_span = pp.ceil_to_raster(
            adc_delay + n_samples * dwell, system.grad_raster_time
        )

        # Where the gradient sits under each view, and the chord it crosses to
        # reach the next one. Every turn gets the same slot -- the repetition
        # is what the magnetisation sees, so it has to be one number -- and the
        # slot is the widest chord's. Closing the shell is the long one: from
        # full amplitude down to zero.
        vertices = amplitude * directions
        chords = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        if not np.all(chords > 0.0):
            raise ValueError("consecutive views must not coincide")
        widest_turn = _slew_span(system, float(chords.max()))
        ramp_span = _slew_span(system, amplitude)

        tr_min = pp.ceil_to_raster(
            hold_span + read_span + widest_turn, system.block_duration_raster
        )
        view_span = tr_min + solve_delay(tr, tr_min, "TR", system)
        # Whatever the repetition leaves over is spent turning more slowly, not
        # sitting still: a gentler slew is a quieter one, and nothing is
        # encoded between the acquisitions.
        turn_span = view_span - hold_span - read_span
        close_span = max(ramp_span, turn_span)

        g_ramp = _gradients(system, [0.0, ramp_span], [np.zeros(3), vertices[0]])
        g_hold = [
            _gradients(system, [0.0, hold_span], [vertex, vertex])
            for vertex in vertices
        ]
        g_read = [
            _gradients(
                system,
                [0.0, read_span, read_span + turn_span],
                [start, start, end],
            )
            for start, end in pairwise(vertices)
        ]
        g_read.append(
            _gradients(
                system,
                [0.0, read_span, read_span + close_span],
                [vertices[-1], vertices[-1], np.zeros(3)],
            )
        )
        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]

        self.seq = pp.Sequence(system)
        self.seq.add_block(*g_ramp)
        for view in range(len(directions)):
            self.seq.add_block(rf, *g_hold[view])
            self.seq.add_block(adc, *g_read[view], *adc_labels)

        self.register(g_hold=g_hold, g_read=g_read, directions=directions)
        if shot_rotations is not None:
            self.register(shot_rotations=shot_rotations, reversal=_reversal(directions))
        self.center = ramp_span + rf_center
        self.tr = view_span
        self.view_duration = hold_span + read_span
        self.step_rad = float(
            np.arccos(
                np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)
            ).max()
        )
        self.n_samples = n_samples
        self.n_missing = n_missing
        self.n_nominal = n_nominal
        self.gap = gap
        self.bandwidth_hz = 1.0 / dwell
        self.delta_k = delta_k
        self.kmax = kmax
        self.gradient_amplitude = amplitude


def _unit(directions: Any) -> np.ndarray:
    """Return ``directions`` as read-only unit rows."""
    values = np.asarray(directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(
            "directions must have shape (views, 3) with at least two views"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("directions must be finite")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("directions must be non-zero")
    unit = np.array(values / norms[:, None])
    unit.setflags(write=False)
    return unit


def _reversal(directions: np.ndarray) -> np.ndarray:
    """Return the rotation taking each view to its mirror in the play order.

    A generated shell is symmetric about the equator, heights and azimuth
    increments alike, so a half turn about a horizontal axis reverses it.
    """
    u, _, vt = np.linalg.svd(directions[::-1].T @ directions)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    if not np.allclose(directions @ rotation.T, directions[::-1], atol=1e-9):
        raise AssertionError("a generated shell must be its own reverse")
    rotation.setflags(write=False)
    return rotation


def _slew_span(system: pp.Opts, delta: float) -> float:
    return max(
        system.grad_raster_time,
        pp.ceil_to_raster(delta / system.max_slew, system.grad_raster_time),
    )


def _gradients(system: pp.Opts, times, amplitudes) -> list:
    """One extended trapezoid per axis, every axis present so a rotation can turn it."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    return [
        pp.make_extended_trapezoid(
            channel=axis,
            times=np.asarray(times, dtype=float),
            amplitudes=amplitudes[:, index],
            system=system,
        )
        for index, axis in enumerate(AXES)
    ]
