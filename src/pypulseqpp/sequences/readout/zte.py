"""Zero-echo-time readouts: excite on the readout gradient, acquire from k = 0 out."""

from __future__ import annotations

__all__ = ["ZteReadout"]

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

import pypulseqpp as pp

from ..._angles import calc_projection_shell, calc_raga_size
from .._module import SequenceModule
from ._common import AXES, solve_delay

#: Fraction of ``max_grad`` a spoke may reach.
_READOUT_GRAD_MARGIN = 0.95


class ZteReadout(SequenceModule):
    """Continuous-gradient ZTE shell, including its initial and final ramps.

    Each view excites on a held gradient, then acquires while the gradient
    slews toward the next direction. The final view ramps to zero. Each shot
    plays ``shell(rotation)`` for one of ``shot_rotations``: the gradients of
    the turned shell, with no rotation event, built from a few turn shapes
    per axis scaled by their amplitudes.

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
        the shots undersamples in angle. ``None`` is ``ceil(pi * (matrix - 1))``,
        or under ``'raga'`` the support size nearest it
        (:func:`pypulseqpp.calc_raga_size`).
    scheme : {'spiral', 'quiet', 'meridian'}, default='spiral'
        Shape of the shell, as in :func:`pypulseqpp.calc_projection_shell`.
    ordering : {'sequential', 'golden', 'raga'}, default='sequential'
        Order of ``shot_rotations``, as in
        :func:`pypulseqpp.calc_projection_shell`. Under ``'golden'`` the first
        shots, or any run of consecutive ones, spread evenly in azimuth.
    tiny_index : int, default=1
        Tiny-golden index of the golden step, as in
        :func:`pypulseqpp.calc_projection_shell`.
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
    turn_step : float, default=0.0078125
        Relative step of the geometric grid a view's gradient lies on, on each
        axis: ``+-(1 + turn_step) ** k`` times the spoke amplitude, or zero
        below ``turn_step`` of it. A turn between two grid values is one shape
        per pair of signs and difference of powers, scaled by its larger end,
        so the turns on an axis share a few shapes and meet without a step.
    turn : {'after', 'spread'}, default='after'
        When a view turns onto the next. ``'after'`` holds the gradient
        through the acquisition, so the spoke is straight, and turns in the
        rest of the repetition. ``'spread'`` slews linearly from the end of
        the pulse to the next view, through the acquisition: the spoke bends
        by a fraction of the step between views, and the slew is the lowest
        the repetition allows, which is quieter.
    ramp_duration_s : float, default=None
        Duration of the ramp from zero to the first view and of the ramp back
        to zero after the last (s). ``None`` is as fast as the slew allows; a
        long one is quieter.


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
        ordering: str = "sequential",
        tiny_index: int = 1,
        oversampling: float = 2.0,
        readout_bandwidth_hz: float = 62.5e3,
        tr: float | None = None,
        dead_time_s: float | None = None,
        labels: tuple[str, ...] | None = None,
        turn_step: float = 0.0078125,
        turn: str = "after",
        ramp_duration_s: float | None = None,
    ) -> None:
        if fov <= 0 or int(matrix) < 2:
            raise ValueError("fov must be positive and matrix must be >= 2")
        if oversampling < 1.0:
            raise ValueError("oversampling must be >= 1")
        if readout_bandwidth_hz <= 0:
            raise ValueError("readout_bandwidth_hz must be positive")
        if not 0.0 < turn_step <= 0.5:
            raise ValueError("turn_step must be in (0, 0.5]")
        if turn not in ("after", "spread"):
            raise ValueError(f"turn must be 'after' or 'spread', got {turn!r}")
        if ramp_duration_s is not None and ramp_duration_s < 0.0:
            raise ValueError("ramp_duration_s must not be negative")

        shot_rotations = None
        if directions is None:
            # Rings hold the shots and the shell walks across them, so the two
            # spacings are reciprocal: ring to ring goes as 1 / n_views, within
            # a ring as 1 / n_shots. They agree at the equator when
            # n_shots = pi * n_views, which with a Nyquist-matched sphere over
            # the whole set puts n_views at the matrix size.
            if n_shots is None:
                n_shots = max(1, math.ceil(np.pi * (int(matrix) - 1)))
                if ordering == "raga":
                    n_shots = calc_raga_size(n_shots, tiny_index=tiny_index)
            total = math.ceil(np.pi * int(matrix) ** 2)
            n_views = (
                max(3, -(-total // int(n_shots))) if n_views is None else int(n_views)
            )
            directions, shot_rotations = calc_projection_shell(
                n_views,
                n_shots,
                scheme=scheme,
                ordering=ordering,
                tiny_index=tiny_index,
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
        # slot is the widest chord's, with room for the turn's end to land on
        # the shape grid. Closing the shell is the long one: from full
        # amplitude down to zero.
        vertices = amplitude * directions
        chords = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        if not np.all(chords > 0.0):
            raise ValueError("consecutive views must not coincide")
        widest_turn = _slew_span(
            system, float(chords.max()) + math.sqrt(3.0) * turn_step * amplitude
        )
        ramp_span = max(
            _slew_span(system, amplitude),
            pp.ceil_to_raster(float(ramp_duration_s or 0.0), system.grad_raster_time),
        )

        # A spread turn lasts the whole block, which the receiver's dead time
        # after the last sample may lengthen past the acquisition.
        acquisition = pp.ceil_to_raster(pp.calc_duration(adc), system.grad_raster_time)
        moving = (
            read_span + widest_turn
            if turn == "after"
            else max(acquisition, widest_turn)
        )
        tr_min = pp.ceil_to_raster(hold_span + moving, system.block_duration_raster)
        view_span = tr_min + solve_delay(tr, tr_min, "TR", system)
        # Whatever the repetition leaves over is spent turning more slowly, not
        # sitting still: a gentler slew is a quieter one, and nothing is
        # encoded between the acquisitions.
        turn_span = view_span - hold_span - read_span
        close_span = max(ramp_span, turn_span)

        adc_labels = [
            pp.make_label(type="SET", label=name, value=0) for name in labels or ()
        ]
        self._system = system
        self._turn_step = float(turn_step)
        self._units: dict = {}
        self._times = {
            "ramp": [0.0, ramp_span],
            "hold": [0.0, hold_span],
            "turn": (
                [0.0, read_span, read_span + turn_span]
                if turn == "after"
                else [0.0, read_span + turn_span]
            ),
            "close": [0.0, read_span, read_span + close_span],
        }
        self._directions = directions
        self.gradient_amplitude = amplitude
        played = self.shell()

        self.seq = pp.Sequence(system)
        self.seq.add_block(*played.ramp)
        for view in range(len(directions)):
            self.seq.add_block(rf, *played.hold[view])
            self.seq.add_block(adc, *played.read[view], *adc_labels)

        self.register(
            g_ramp=played.ramp,
            g_hold=played.hold,
            g_read=played.read,
            directions=directions,
        )
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

    def shell(self, rotation: Any = None) -> SimpleNamespace:
        """Return the gradients of the shell turned by ``rotation``, in the logical frame.

        Each view holds its gradient under the pulse, then turns onto the next
        view, after the acquisition or through it as ``turn`` sets. On every axis the held gradient is rounded to
        the geometric grid of ``turn_step``, so the direction a view plays is
        within about ``turn_step`` of its own, and each turn is one of a few
        shapes scaled by its larger end.

        Parameters
        ----------
        rotation : array_like, default=None
            3x3 rotation applied to :attr:`directions`; ``None`` is the
            identity.

        Returns
        -------
        types.SimpleNamespace
            ``ramp``, the three ramps from zero; ``hold`` and ``read``, per
            view, the three plateaus under the pulse and the three readout
            gradients with their turn, the last ramping to zero; and
            ``directions``, the unit directions the views play, ``(n_views,
            3)``.
        """
        targets = self.gradient_amplitude * (
            self._directions
            if rotation is None
            else self._directions @ np.asarray(rotation, dtype=float).T
        )
        signs, powers = _on_grid(targets / self.gradient_amplitude, self._turn_step)
        levels = 1.0 + self._turn_step

        def held(view, axis):
            return (
                signs[view, axis]
                * self.gradient_amplitude
                * levels ** float(powers[view, axis])
            )

        def level(view, axis):
            return int(signs[view, axis]), int(powers[view, axis])

        def turn(axis, view, kind):
            here = level(view, axis)
            if kind == "ramp":
                ends = ((0, 0), here)
            elif kind == "hold":
                ends = (here, here)
            elif kind == "close":
                ends = (here, (0, 0))
            else:
                ends = (here, level(view + 1, axis))
            peak = max((power for sign, power in ends if sign), default=0)
            unit = tuple(
                float(sign) * levels ** float(power - peak) if sign else 0.0
                for sign, power in ends
            )
            return self._scaled(
                axis, kind, unit, self.gradient_amplitude * levels ** float(peak)
            )

        n_views = len(targets)
        values = np.array(
            [[held(view, axis) for axis in range(3)] for view in range(n_views)]
        )
        return SimpleNamespace(
            ramp=[turn(axis, 0, "ramp") for axis in range(3)],
            hold=[
                [turn(axis, view, "hold") for axis in range(3)]
                for view in range(n_views)
            ],
            read=[
                [
                    turn(
                        axis,
                        view,
                        "close" if view == n_views - 1 else "turn",
                    )
                    for axis in range(3)
                ]
                for view in range(n_views)
            ],
            directions=values / np.linalg.norm(values, axis=1, keepdims=True),
        )

    def _scaled(self, axis: int, kind: str, ends: tuple[float, float], scale: float):
        """Return the unit-peak shape ``kind`` with ``ends`` on ``axis``, scaled to ``scale``."""
        key = (axis, kind, ends)
        unit = self._units.get(key)
        if unit is None:
            times = self._times[kind]
            amplitudes = (ends[0],) * (len(times) - 1) + (ends[1],)
            unit = pp.make_extended_trapezoid(
                channel=AXES[axis],
                times=np.asarray(times, dtype=float),
                amplitudes=np.asarray(amplitudes, dtype=float),
                system=self._system,
            )
            self._units[key] = unit
        return pp.scale_grad(unit, float(scale))


def _on_grid(values: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the signs and integer powers of ``1 + step`` nearest ``values``.

    A value is ``sign * (1 + step) ** power``, or zero where its magnitude is
    below ``step``. Any two values on the grid are then in a ratio that is
    itself a power of ``1 + step``, so a turn between them is one shape per
    pair of signs and difference of powers, scaled by its larger end, and
    consecutive turns meet exactly.
    """
    magnitude = np.abs(values)
    on = magnitude >= step
    powers = np.zeros(values.shape, dtype=int)
    powers[on] = np.rint(np.log(magnitude[on]) / math.log1p(step)).astype(int)
    signs = np.where(on, np.sign(values), 0).astype(int)
    return signs, powers


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
