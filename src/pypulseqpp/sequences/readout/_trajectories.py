"""Canonical non-Cartesian gradient interleaves with ADCs and moment bridges."""

from __future__ import annotations

__all__ = [
    "Arbitrary",
    "NonCartesianGradient",
    "Radial",
    "Rosette",
    "Spiral",
]

import copy
import math

import numpy as np

import pypulseqpp as pp

from ..._trajectories import _as_scalar, _validate_common
from ._common import AXES as _AXES

_SPIRAL_DIRECTIONS = ("outward", "inward", "in_out")


def traj2grad(
    trajectory, system, *, oversampling=8, start_at_zero=True, end_at_zero=True
):
    """Gradient tracing ``trajectory`` under the vector gradient and slew limits.

    The path's samples describe geometry only; :func:`pypulseqpp.traj_to_grad`
    assigns the timing.

    Parameters
    ----------
    trajectory : numpy.ndarray
        K-space path, ``(n, 2)`` or ``(n, 3)``, in 1/m.
    system : pypulseq.Opts
        System limits.
    oversampling : int, optional
        Path-resampling factor the solver works at.
    start_at_zero, end_at_zero : bool, optional
        Ramp up from and back down to zero amplitude. Disable an endpoint
        where a moment bridge meets the readout at non-zero amplitude.

    Returns
    -------
    numpy.ndarray
        ``(n_grad, 3)`` on the gradient raster, in Hz/m; zero in the third
        column for a 2D path.
    """
    trajectory = np.atleast_2d(np.asarray(trajectory, dtype=float))
    gradient, _ = pp.traj_to_grad(
        trajectory.T,
        system=system,
        oversampling=oversampling,
        start_at_zero=start_at_zero,
        end_at_zero=end_at_zero,
    )
    gradient = np.atleast_2d(gradient).T
    if gradient.shape[1] < 3:
        gradient = np.hstack(
            [gradient, np.zeros((gradient.shape[0], 3 - gradient.shape[1]))]
        )
    return gradient


def _stretch_gradient(gradient, target_duration, raster):
    """Time-stretch ``gradient`` to at least ``target_duration``, preserving area."""
    gradient = np.asarray(gradient, dtype=float)
    old_n = gradient.shape[0]
    new_n = max(old_n, math.ceil(target_duration / raster - 1e-12))
    if new_n == old_n:
        return gradient
    k_edges = np.vstack(
        (np.zeros((1, gradient.shape[1])), np.cumsum(gradient, axis=0) * raster)
    )
    old_u = np.linspace(0.0, 1.0, old_n + 1)
    new_u = np.linspace(0.0, 1.0, new_n + 1)
    interp = np.column_stack(
        [np.interp(new_u, old_u, k_edges[:, axis]) for axis in range(gradient.shape[1])]
    )
    return np.diff(interp, axis=0) / raster


def _make_grad_events(system, gradient, axes, *, first=None, last=None):
    events = []
    for axis_index, channel in enumerate(axes):
        waveform = np.ascontiguousarray(gradient[:, axis_index])
        events.append(
            pp.make_arbitrary_grad(
                channel=channel,
                waveform=waveform,
                first=None if first is None else float(first[axis_index]),
                last=None if last is None else float(last[axis_index]),
                system=system,
            )
        )
    return tuple(events)


def _adc_samples(system, n_samples):
    """Round down to a multiple of the ADC sample divisor, with at least one group."""
    divisor = int(getattr(system, "adc_samples_divisor", 0) or 1)
    return max(divisor, int(n_samples) // divisor * divisor)


def _make_adc(system, n_samples, read_duration):
    """Make an ADC that fits inside ``read_duration`` on the ADC raster."""
    n_samples = _adc_samples(system, n_samples)
    dwell = (
        math.floor(read_duration / n_samples / system.adc_raster_time + 1e-10)
        * system.adc_raster_time
    )
    if dwell < system.adc_raster_time:
        raise ValueError("readout is too short for the requested ADC oversampling")
    return pp.make_adc(num_samples=n_samples, dwell=dwell, system=system)


def _sample_gradient_trajectory(gradient, raster, adc):
    """Return k (1/m) at the ADC sample centres, with k = 0 at the gradient start."""
    gradient = np.asarray(gradient, dtype=float)
    k_edges = np.vstack(
        (np.zeros((1, gradient.shape[1])), np.cumsum(gradient, axis=0) * raster)
    )
    edge_times = np.arange(gradient.shape[0] + 1, dtype=float) * raster
    sample_times = float(adc.delay) + (
        np.arange(int(adc.num_samples), dtype=float) + 0.5
    ) * float(adc.dwell)
    return np.column_stack(
        [
            np.interp(sample_times, edge_times, k_edges[:, axis])
            for axis in range(gradient.shape[1])
        ]
    )


def _moment_bridges(system, area, grad_start, grad_end, axes):
    """Build trapezoids of ``area`` (1/m) from ``grad_start`` to ``grad_end`` (Hz/m).

    One event per axis with a non-zero area or endpoint; other axes are
    omitted. With several such axes, each axis's ``max_grad`` and ``max_slew``
    are divided by ``sqrt(n_active)`` so the vector stays within the limits,
    except that ``max_grad`` is never lowered below the endpoint amplitudes,
    which the vector-limited readout already fixes.
    """
    active = [
        index
        for index in range(len(axes))
        if abs(area[index]) >= 1e-12
        or abs(grad_start[index]) >= 1e-12
        or abs(grad_end[index]) >= 1e-12
    ]
    if not active:
        return ()

    derate = math.sqrt(len(active)) if len(active) > 1 else 1.0

    events = []
    for index in active:
        limits = system
        if derate > 1.0:
            # The endpoint is the readout's own, so it sets the floor: a
            # ceiling below it would refuse a boundary condition the vector
            # limit already admits.
            floor = max(abs(float(grad_start[index])), abs(float(grad_end[index])))
            limits = copy.copy(system)
            limits.max_slew = system.max_slew / derate
            limits.max_grad = max(system.max_grad / derate, floor)
        grad, _, _ = pp.make_extended_trapezoid_area(
            area=float(area[index]),
            channel=axes[index],
            grad_start=float(grad_start[index]),
            grad_end=float(grad_end[index]),
            system=limits,
        )
        events.append(grad)
    return tuple(events)


class NonCartesianGradient:
    """One canonical non-Cartesian base interleave, independent of its acquisition schedule.

    Prewinders bridge from zero gradient and k = 0 to the readout's start,
    and rewinders from its end back to zero; they play in blocks of their
    own, so ``duration`` is the longest prewinder plus ``read_duration`` plus
    the longest rewinder (s).

    Attributes
    ----------
    trajectory : numpy.ndarray
        ``(n, 2)`` or ``(n, 3)`` path in 1/m: the design polyline, or for
        :class:`Rosette` the k-space at the ADC samples.
    bandwidth_hz_px : float
        ``1 / adc.dwell`` (Hz), the full receiver bandwidth despite the name.
    design_interleaves : int or None
        Interleave count the spiral pitch was designed for.
    recommended_rotations : int or None
        Full spokes for Nyquist sampling at ``kmax``, ``ceil(pi * matrix / 2)``;
        set by :class:`Radial` only.
    kind : str
        ``"arbitrary"``, ``"full"`` (radial), ``"spiral"`` or ``"rosette"``.
    """

    def __init__(
        self,
        *,
        system,
        gradients,
        adc,
        trajectory,
        design_interleaves=None,
        recommended_rotations=None,
        prewinders=(),
        rewinders=(),
        kind=None,
    ):
        self.system = system
        self.gradients = tuple(gradients)
        self.adc = adc
        self.trajectory = np.asarray(trajectory, dtype=float)
        self.design_interleaves = (
            None if design_interleaves is None else int(design_interleaves)
        )
        self.recommended_rotations = (
            None if recommended_rotations is None else int(recommended_rotations)
        )
        self.prewinders = tuple(prewinders)
        self.rewinders = tuple(rewinders)
        self.kind = kind
        self.n_samples = int(adc.num_samples)
        self.adc_dwell_s = float(adc.dwell)
        self.bandwidth_hz_px = 1.0 / self.adc_dwell_s
        self.read_duration = pp.calc_duration(*self.gradients, adc)
        self.duration = (
            (max((pp.calc_duration(g) for g in self.prewinders), default=0.0))
            + self.read_duration
            + (max((pp.calc_duration(g) for g in self.rewinders), default=0.0))
        )

    @property
    def has_prewinder(self):
        return bool(self.prewinders)

    @property
    def has_rewinder(self):
        return bool(self.rewinders)

    @property
    def gx(self):
        """Gradient on channel x, or None when this interleave does not drive it."""
        return next((g for g in self.gradients if g.channel == "x"), None)

    @property
    def gy(self):
        """Gradient on channel y, or None when this interleave does not drive it."""
        return next((g for g in self.gradients if g.channel == "y"), None)

    @property
    def gz(self):
        """Gradient on channel z, or None when this interleave does not drive it."""
        return next((g for g in self.gradients if g.channel == "z"), None)

    @property
    def axes(self) -> tuple[str, ...]:
        """The gradient channels this interleave drives, in waveform order."""
        return tuple(gradient.channel for gradient in self.gradients)

    def rotated(self, angle: float) -> NonCartesianGradient:
        """Return this interleave rotated in its own plane, with unchanged timing.

        The readout waveforms and trajectory turn from the first channel
        towards the second. Prewinders stay right-aligned and rewinders
        left-aligned, each set resampled on the union of its vertex times.
        The ADC object is shared with the source; of the subclass attributes,
        only ``direction`` and ``density`` are copied.

        Parameters
        ----------
        angle : float
            In-plane rotation (rad).

        Returns
        -------
        NonCartesianGradient
            Same class as ``self``.

        Raises
        ------
        ValueError
            If the interleave does not drive exactly two channels, or a rotated
            bridge's vector amplitude exceeds ``system.max_grad``.
        """
        axes = self.axes
        if len(axes) != 2:
            raise ValueError(
                "only a two-channel planar interleave can be rotated in its own plane"
            )

        waveforms = np.column_stack(
            [np.asarray(g.waveform, dtype=float) for g in self.gradients]
        )
        turn = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        rotated = waveforms @ turn.T
        first, last = rotated[0], rotated[-1]

        turned = object.__new__(type(self))
        NonCartesianGradient.__init__(
            turned,
            system=self.system,
            gradients=_make_grad_events(
                self.system, rotated, axes, first=first, last=last
            ),
            adc=self.adc,
            trajectory=self.trajectory[:, :2] @ turn.T,
            design_interleaves=self.design_interleaves,
            recommended_rotations=self.recommended_rotations,
            prewinders=_rotated_bridge_pair(
                self.prewinders, axes, turn, self.system, anchor="right"
            ),
            rewinders=_rotated_bridge_pair(
                self.rewinders, axes, turn, self.system, anchor="left"
            ),
            kind=self.kind,
        )
        for name in ("direction", "density"):
            if hasattr(self, name):
                setattr(turned, name, getattr(self, name))
        return turned


def _bridge_area(event, axes) -> np.ndarray:
    moment = np.zeros(len(axes))
    moment[axes.index(event.channel)] = float(np.trapezoid(event.waveform, event.tt))
    return moment


def _rotated_bridge_pair(events, axes, turn, system, *, anchor):
    """Rotate one bridge per channel as a single vector waveform.

    Prewinders (``anchor="right"``) end together and rewinders
    (``anchor="left"``) start together; each bridge holds its end values
    outside its own extent. Raises ValueError if the rotated vector amplitude
    exceeds ``system.max_grad``, so every further rotation stays feasible.
    """
    if not events:
        return ()
    span = max(float(event.tt[-1]) for event in events)
    vertex_times: set[float] = {0.0, span}
    shifted = {}
    for event in events:
        offset = 0.0 if anchor == "left" else span - float(event.tt[-1])
        tt = np.asarray(event.tt, dtype=float) + offset
        shifted[event.channel] = (tt, np.asarray(event.waveform, dtype=float))
        vertex_times.update(tt.tolist())
    times = np.array(sorted(vertex_times))

    columns = np.zeros((times.size, 2))
    for index, axis in enumerate(axes):
        if axis in shifted:
            tt, wave = shifted[axis]
            columns[:, index] = np.interp(times, tt, wave, left=wave[0], right=wave[-1])
    rotated = columns @ turn.T

    peak = float(np.linalg.norm(rotated, axis=1).max())
    if peak > float(system.max_grad) * (1.0 + 1e-6):
        raise ValueError(
            f"rotated bridge reaches {peak:.0f} Hz/m vector amplitude, above the "
            f"{float(system.max_grad):.0f} Hz/m per-axis ceiling a rotation may land it on"
        )

    events_out = []
    for index, axis in enumerate(axes):
        if np.allclose(rotated[:, index], 0.0):
            continue
        events_out.append(
            pp.make_extended_trapezoid(
                channel=axis,
                amplitudes=rotated[:, index],
                times=times,
                system=system,
            )
        )
    return tuple(events_out)


class Arbitrary(NonCartesianGradient):
    """Base interleave from a caller-supplied 2D or 3D k-space path.

    ``trajectory`` is ``(n, 2)`` or ``(n, 3)`` in 1/m, played on ``axes``
    (default: the first two or three of x, y, z). A path not starting at
    k = 0 gets prewinders, and one not ending there gets rewinders. The ADC
    takes ``round(matrix * oversamp)`` samples, rounded down to
    ``system.adc_samples_divisor``; the readout is stretched to last at least
    that count over ``bandwidth_hz_px`` (Hz), and the dwell is the longest on
    the ADC raster that fits the samples in it. ``derate`` applies
    :func:`pypulseqpp.apply_system_derates` first.
    """

    def __init__(
        self,
        system,
        trajectory,
        *,
        matrix,
        bandwidth_hz_px=250e3,
        oversamp=1.0,
        axes=None,
        solver_oversampling=8,
        derate=True,
    ):
        path = np.asarray(trajectory, dtype=float)
        if path.ndim != 2 or path.shape[1] not in (2, 3) or path.shape[0] < 4:
            raise ValueError("trajectory must have shape (n, 2) or (n, 3), with n >= 4")
        n = _as_scalar(matrix, "matrix", int)
        if n < 2 or oversamp < 1 or bandwidth_hz_px <= 0:
            raise ValueError(
                "matrix must be >= 2, oversamp >= 1, and bandwidth_hz_px positive"
            )
        axes = tuple(_AXES[: path.shape[1]] if axes is None else axes)
        if (
            len(axes) != path.shape[1]
            or len(set(axes)) != len(axes)
            or any(axis not in _AXES for axis in axes)
        ):
            raise ValueError(
                "axes must contain one distinct gradient channel per trajectory dimension"
            )
        if derate:
            system = pp.apply_system_derates(system)

        k_start = path[0]
        has_pre = not np.allclose(k_start, 0.0, atol=1e-12)
        has_rew = not np.allclose(path[-1], 0.0, atol=1e-12)
        n_adc = max(2, round(n * oversamp))
        gradient = traj2grad(
            path,
            system,
            oversampling=solver_oversampling,
            start_at_zero=not has_pre,
            end_at_zero=not has_rew,
        )[:, : path.shape[1]]
        gradient = _stretch_gradient(
            gradient, n_adc / bandwidth_hz_px, system.grad_raster_time
        )
        first, last = gradient[0], gradient[-1]
        gradients = _make_grad_events(system, gradient, axes, first=first, last=last)
        prewinders = (
            _moment_bridges(system, k_start, np.zeros(path.shape[1]), first, axes)
            if has_pre
            else ()
        )
        actual_end = k_start + np.sum(gradient, axis=0) * system.grad_raster_time
        rewinders = (
            _moment_bridges(system, -actual_end, last, np.zeros(path.shape[1]), axes)
            if has_rew
            else ()
        )
        adc = _make_adc(system, n_adc, gradient.shape[0] * system.grad_raster_time)
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=path,
            prewinders=prewinders,
            rewinders=rewinders,
            kind="arbitrary",
        )


class Radial(NonCartesianGradient):
    """Full radial spoke on ``ro_axis`` with a prewinder and a rewinder.

    The readout is a constant-amplitude plateau from ``-kmax`` to ``+kmax``
    lasting ``round(matrix * oversamp) / bandwidth_hz_px``, or longer where
    ``system.max_grad`` requires, ceiled to the gradient raster. The bridges
    carry area ``-kmax`` from and back to zero gradient.
    """

    def __init__(
        self,
        system,
        fov,
        matrix,
        *,
        bandwidth_hz_px=250e3,
        oversamp=1.0,
        ro_axis="x",
        derate=True,
    ):
        fov_m, n = _validate_common(fov, matrix, oversamp, bandwidth_hz_px)
        if derate:
            system = pp.apply_system_derates(system)
        raster = system.grad_raster_time
        kmax = n / (2.0 * fov_m)
        n_adc = max(2, round(n * oversamp))
        target_duration = n_adc / float(bandwidth_hz_px)
        read_duration = pp.ceil_to_raster(
            max(target_duration, 2.0 * kmax / system.max_grad), raster
        )
        amplitude = 2.0 * kmax / read_duration
        grad = pp.make_extended_trapezoid(
            channel=ro_axis,
            times=np.array([0.0, read_duration]),
            amplitudes=np.array([amplitude, amplitude]),
            system=system,
        )
        pre, _, _ = pp.make_extended_trapezoid_area(
            area=-kmax,
            channel=ro_axis,
            grad_start=0.0,
            grad_end=amplitude,
            system=system,
        )
        rew, _, _ = pp.make_extended_trapezoid_area(
            area=-kmax,
            channel=ro_axis,
            grad_start=amplitude,
            grad_end=0.0,
            system=system,
        )
        adc = _make_adc(system, n_adc, read_duration)
        trajectory = pp.calc_radial_trajectory(fov_m, n, num_points=n_adc)
        super().__init__(
            system=system,
            gradients=(grad,),
            adc=adc,
            trajectory=trajectory,
            recommended_rotations=math.ceil(np.pi * n / 2.0),
            prewinders=(pre,),
            rewinders=(rew,),
            kind="full",
        )


class Spiral(NonCartesianGradient):
    """Constant-, variable- or dual-density spiral base interleave.

    ``design_interleaves`` and the density parameters set the pitch as in
    :func:`pypulseqpp.calc_spiral_trajectory`; the caller may acquire any
    number of rotated copies. ``direction`` is ``"outward"`` (centre to edge,
    with rewinders), ``"inward"`` (edge to centre, with prewinders) or
    ``"in_out"`` (edge through centre to edge, with both). Each half of an
    ``"in_out"`` arm is designed for twice the interleave counts, so one arm
    samples like two outward ones.

    The dwell is ``1 / bandwidth_hz_px`` (Hz) floored to the ADC raster.
    Where the time-optimal arm would put adjacent samples more than
    ``1 / (oversamp * fov)`` apart at that dwell, the arm is slowed rather
    than the bandwidth raised. The ADC fills the arm with whole samples.
    """

    def __init__(
        self,
        system,
        fov,
        matrix,
        design_interleaves,
        *,
        direction="outward",
        density="constant",
        inner_design_interleaves=None,
        outer_design_interleaves=None,
        variable_density_power=2.0,
        transition_radius=0.5,
        transition_speed=12.0,
        num_points=1024,
        bandwidth_hz_px=250e3,
        oversamp=1.0,
        axes=("x", "y"),
        solver_oversampling=8,
        derate=True,
    ):
        fov_m, n = _validate_common(fov, matrix, oversamp, bandwidth_hz_px)
        design_interleaves = int(design_interleaves)
        if design_interleaves < 1:
            raise ValueError("design_interleaves must be >= 1")
        if direction not in _SPIRAL_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_SPIRAL_DIRECTIONS}, got {direction!r}"
            )
        axes = tuple(axes)
        if (
            len(axes) != 2
            or len(set(axes)) != 2
            or any(axis not in _AXES for axis in axes)
        ):
            raise ValueError("axes must contain two distinct gradient channels")
        if derate:
            system = pp.apply_system_derates(system)

        factor = 2 if direction == "in_out" else 1
        path_out = pp.calc_spiral_trajectory(
            fov_m,
            n,
            factor * design_interleaves,
            density=density,
            inner_design_interleaves=(
                None
                if inner_design_interleaves is None
                else factor * inner_design_interleaves
            ),
            outer_design_interleaves=(
                None
                if outer_design_interleaves is None
                else factor * outer_design_interleaves
            ),
            variable_density_power=variable_density_power,
            transition_radius=transition_radius,
            transition_speed=transition_speed,
            num_points=num_points,
        )
        grad_out = traj2grad(
            path_out,
            system,
            oversampling=solver_oversampling,
            start_at_zero=True,
            end_at_zero=False,
        )[:, :2]
        # Adjacent samples may be no further apart than 1 / (oversamp * fov),
        # so at the requested dwell the arm may traverse k no faster than
        # that; where the time-optimal arm is faster, stretch it.
        speed = float(np.max(np.linalg.norm(grad_out, axis=1)))
        stretch = max(1.0, speed * fov_m * float(oversamp) / float(bandwidth_hz_px))
        grad_out = _stretch_gradient(
            grad_out,
            stretch * grad_out.shape[0] * system.grad_raster_time,
            system.grad_raster_time,
        )
        out_area = np.sum(grad_out, axis=0) * system.grad_raster_time

        if direction == "outward":
            path = path_out
            gradient = grad_out
            pre_area, rew_area = None, -out_area
        elif direction == "inward":
            path = path_out[::-1]
            gradient = -grad_out[::-1]
            pre_area, rew_area = out_area, None
        else:
            path = np.concatenate((-path_out[:0:-1], path_out), axis=0)
            gradient = np.concatenate((grad_out[::-1], grad_out), axis=0)
            pre_area, rew_area = -out_area, -out_area

        first = gradient[0]
        last = gradient[-1]
        gradients = _make_grad_events(system, gradient, axes, first=first, last=last)
        prewinders = (
            _moment_bridges(system, pre_area, np.zeros(2), first, axes)
            if pre_area is not None
            else ()
        )
        rewinders = (
            _moment_bridges(system, rew_area, last, np.zeros(2), axes)
            if rew_area is not None
            else ()
        )
        read_duration = gradient.shape[0] * system.grad_raster_time
        # Choose the dwell first, then fit whole samples into the arm: the arm's
        # duration is set by the limits, so a dwell derived from a fixed sample
        # count would not honour the requested bandwidth. The dwell is also
        # capped so adjacent samples stay within 1 / (oversamp * fov).
        max_k_speed = float(np.max(np.linalg.norm(gradient, axis=1)))
        dwell = min(
            1.0 / float(bandwidth_hz_px), 1.0 / (float(oversamp) * fov_m * max_k_speed)
        )
        dwell = max(
            float(system.adc_raster_time),
            math.floor(dwell / system.adc_raster_time + 1e-12) * system.adc_raster_time,
        )
        n_adc = _adc_samples(system, max(2, math.floor(read_duration / dwell + 1e-12)))
        adc = pp.make_adc(num_samples=n_adc, dwell=dwell, system=system)
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=path,
            design_interleaves=design_interleaves,
            prewinders=prewinders,
            rewinders=rewinders,
            kind="spiral",
        )
        self.direction = direction
        self.density = density


class Rosette(NonCartesianGradient):
    """Multi-petal rosette base interleave, starting and ending at k = 0.

    The readout needs no bridges. ``trajectory`` is the k-space at the ADC
    samples and ``design_trajectory`` the polyline the gradient was solved
    from; ``echo_spacing_s`` is the realised mean petal duration and
    ``requested_echo_spacing_s`` the request. The ADC takes the same number
    of samples in every petal, the total rounded down to
    ``system.adc_samples_divisor``.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits. Tighter limits lengthen the petals but not their reach.
    fov : float or array-like
        Isotropic field of view (m). With ``matrix`` it sets
        ``kmax = matrix / (2 * fov)``, and it bounds the k-space step between
        adjacent ADC samples to ``1 / (oversamp * fov)``.
    matrix : int or array-like
        Isotropic matrix size.
    petals : int, optional
        Centre-to-centre lobes within this one interleave, not rotated shots.
    angular_frequency_ratio : float, optional
        Angular over radial frequency: below one the petals are open, one is
        the circular limit, above one they wind more tightly.
    echo_spacing_s : float, optional
        Mean centre-to-centre petal duration (s); ``None`` is the minimum the
        limits allow, and a longer value stretches the waveform uniformly.
        The ramps at each end can make the first and last crossing intervals
        differ slightly from the mean.
    bandwidth_hz_px : float, optional
        Requested ``1 / dwell`` (Hz). The dwell is also bounded by the step
        limit and floored to the ADC raster, so the realised bandwidth can be
        higher.
    oversamp : float, optional
        ADC oversampling; tightens the step limit without changing the
        gradient.
    axes : tuple[str, str], optional
        Channels for the path's two components.
    solver_oversampling : int, optional
        Path-resampling factor of the time-optimal solver.
    derate : bool, optional
        Apply :func:`pypulseqpp.apply_system_derates` first.

    Raises
    ------
    ValueError
        If ``echo_spacing_s`` is below the minimum the limits allow, or a
        parameter is out of range.
    """

    def __init__(
        self,
        system,
        fov,
        matrix,
        *,
        petals=5,
        angular_frequency_ratio=3.0 / 5.0,
        echo_spacing_s=None,
        bandwidth_hz_px=250e3,
        oversamp=1.0,
        axes=("x", "y"),
        solver_oversampling=8,
        derate=True,
    ):
        fov_m, n = _validate_common(fov, matrix, oversamp, bandwidth_hz_px)
        petals = int(petals)
        angular_frequency_ratio = float(angular_frequency_ratio)
        if (
            petals < 1
            or not math.isfinite(angular_frequency_ratio)
            or angular_frequency_ratio <= 0
        ):
            raise ValueError("petals and angular_frequency_ratio must be positive")
        if echo_spacing_s is not None:
            echo_spacing_s = float(echo_spacing_s)
            if not math.isfinite(echo_spacing_s) or echo_spacing_s <= 0:
                raise ValueError("echo_spacing_s must be positive")
        axes = tuple(axes)
        if (
            len(axes) != 2
            or len(set(axes)) != 2
            or any(axis not in _AXES for axis in axes)
        ):
            raise ValueError("axes must contain two distinct gradient channels")
        if derate:
            system = pp.apply_system_derates(system)
        path = pp.calc_rosette_trajectory(
            fov_m,
            n,
            petals=petals,
            angular_frequency_ratio=angular_frequency_ratio,
            num_points=2049,
        )
        gradient = traj2grad(
            path,
            system,
            oversampling=solver_oversampling,
            start_at_zero=True,
            end_at_zero=True,
        )[:, :2]
        raster = float(system.grad_raster_time)
        min_read_duration = gradient.shape[0] * raster
        min_echo_spacing = min_read_duration / petals
        if echo_spacing_s is not None:
            requested_duration = petals * echo_spacing_s
            if requested_duration < min_read_duration - 1e-12:
                raise ValueError(
                    f"echo_spacing_s={echo_spacing_s:.9g} is shorter than the minimum feasible {min_echo_spacing:.9g} s"
                )
            gradient = _stretch_gradient(gradient, requested_duration, raster)

        # The sampled time-optimal solve can retain a small numerical zeroth
        # moment even though the ideal rosette ends at k=0.  Remove that
        # constant residual without changing slew or requiring a rewinder.
        gradient = gradient - np.mean(gradient, axis=0, keepdims=True)

        read_duration = gradient.shape[0] * raster
        max_k_speed = float(np.max(np.linalg.norm(gradient, axis=1)))
        dwell_from_fov = 1.0 / (float(oversamp) * fov_m * max_k_speed)
        dwell_from_bandwidth = 1.0 / float(bandwidth_hz_px)
        max_dwell = min(dwell_from_fov, dwell_from_bandwidth)
        # Choose the dwell first, on the ADC raster, then fit a whole number of
        # samples into every petal, so the ADC covers every petal even when the
        # gradient and ADC rasters differ.
        dwell = (
            math.floor(max_dwell / system.adc_raster_time + 1e-12)
            * system.adc_raster_time
        )
        dwell = max(float(system.adc_raster_time), dwell)
        samples_per_petal = max(1, math.floor(read_duration / petals / dwell + 1e-12))
        n_adc = _adc_samples(system, petals * samples_per_petal)
        adc = pp.make_adc(num_samples=n_adc, dwell=dwell, system=system)
        acquired_trajectory = _sample_gradient_trajectory(gradient, raster, adc)
        gradients = _make_grad_events(
            system, gradient, axes, first=np.zeros(2), last=np.zeros(2)
        )
        super().__init__(
            system=system,
            gradients=gradients,
            adc=adc,
            trajectory=acquired_trajectory,
            kind="rosette",
        )
        self.design_trajectory = path
        self.petals = petals
        self.angular_frequency_ratio = angular_frequency_ratio
        self.samples_per_petal = samples_per_petal
        self.min_echo_spacing_s = min_echo_spacing
        self.echo_spacing_s = read_duration / petals
        self.requested_echo_spacing_s = echo_spacing_s
        self.requested_bandwidth_hz_px = float(bandwidth_hz_px)
        self.resolution_m = fov_m / n
