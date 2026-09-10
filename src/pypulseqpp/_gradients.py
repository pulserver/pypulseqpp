"""Gradient factories stated in imaging terms rather than in area.

Each is a thin wrapper over one of PyPulseq's own factories, taking the
quantity a sequence actually knows -- a resolution, a field of view, cycles of
dephasing across a voxel -- and doing the one division that turns it into an
area.
"""

from __future__ import annotations

__all__ = [
    "concatenate_gradients",
    "make_crusher",
    "make_phase_blip",
    "make_phase_encoding",
    "make_wave_gradients",
]

from typing import Any

import numpy as np

from . import _events
from ._opts import default_system


def concatenate_gradients(*grads: Any, system=None):
    """Lay gradients on one channel end to end and sum them into one event.

    Each gradient after the first is delayed to begin where the previous one
    ends, so the result plays them back to back on the shared channel. This is
    what folds a slice rephaser onto its selection gradient -- exactly what a
    slab excitation does internally -- so a spectral-spatial or multiband pulse,
    whose rephaser is a separate event, can hand a 3D or SMS readout one merged
    ``z`` lobe rather than a second gradient the readout has no block for. The
    inputs are not modified; ``None`` entries are dropped.

    Parameters
    ----------
    *grads : GradEvent or TrapEvent or None
        Gradients on the **same** channel, in play order. ``None`` is skipped.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    GradEvent
        The gradients concatenated into a single event, or the sole gradient
        unchanged when only one is given.

    Raises
    ------
    ValueError
        If no gradient is given, or they are not all on one channel.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> system = pp.Opts()
    >>> select = pp.make_trapezoid("z", area=200.0, duration=2e-3, system=system)
    >>> rephase = pp.make_trapezoid("z", area=-100.0, duration=1e-3, system=system)
    >>> merged = pp.concatenate_gradients(select, rephase, system=system)
    >>> round(pp.calc_duration(merged), 6) == round(
    ...     pp.calc_duration(select) + pp.calc_duration(rephase), 6
    ... )
    True

    The rephaser starts where the selection lobe ends, so the inputs are left
    as they were:

    >>> float(rephase.delay)
    0.0
    """
    from . import add_gradients, calc_duration, scale_grad

    events = [grad for grad in grads if grad is not None]
    if not events:
        raise ValueError("concatenate_gradients needs at least one gradient")
    channels = {grad.channel for grad in events}
    if len(channels) != 1:
        raise ValueError(
            f"all gradients must be on one channel, got {sorted(channels)}"
        )
    if len(events) == 1:
        return events[0]

    system = default_system(system)
    placed = [events[0]]
    offset = calc_duration(events[0])
    for grad in events[1:]:
        # scale_grad(1.0) is a copy; the input keeps its own delay.
        shifted = scale_grad(grad, 1.0, system=system)
        shifted.delay = float(shifted.delay) + offset
        placed.append(shifted)
        offset += calc_duration(grad)
    return add_gradients(placed, system=system)


def make_phase_encoding(
    channel: str, resolution: float, system=None, duration: float | None = None
):
    """Encode one phase-encode axis at full amplitude, for a target resolution.

    The largest phase-encode step an acquisition needs is set by resolution
    alone: index ``i`` of ``n`` encodes area ``(i - n/2) / fov``, so the extreme
    is ``(n/2) / fov``, and since ``resolution = fov / n`` that is
    ``1 / (2 * resolution)``. Field of view and matrix size cancel, so neither
    is asked for.

    This builds **one** gradient. Which views are played, and in what order, is
    a scan loop's business; scale this template per view.

    Parameters
    ----------
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    resolution : float
        Target resolution along ``channel`` (m) -- ``fov / matrix`` for an
        in-plane encode, the partition spacing for a 3D slab encode.
    system : pypulseq.Opts, optional
        System limits.
    duration : float, optional
        Force a duration (s); the default is the shortest feasible.

    Returns
    -------
    TrapEvent
        Trapezoid of area ``1 / (2 * resolution)``, to be
        :func:`~pypulseq.scale_grad`-ed per view.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> gy = pp.make_phase_encoding("y", 0.22 / 64, system=pp.Opts())
    >>> round(gy.area, 3)
    145.455

    The same call encodes a 3D slab's partitions, the resolution along ``z``
    being the partition spacing:

    >>> round(pp.make_phase_encoding("z", 1e-3, system=pp.Opts()).area, 1)
    500.0

    See Also
    --------
    make_phase_blip : the small step played between echoes of a train.
    """
    resolution = float(resolution)
    if not resolution > 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    return _trapezoid(channel, 1.0 / (2.0 * resolution), system, duration)


def make_phase_blip(
    channel: str,
    fov: float,
    steps: float = 1.0,
    system=None,
    duration: float | None = None,
):
    """Blip the phase encode by whole k-space cells, between echoes of a train.

    One cell is ``1 / fov``, so ``steps=1`` moves one line and ``steps=R``
    implements an acceleration of ``R`` with no further arithmetic. Negative
    ``steps`` blip backwards.

    Parameters
    ----------
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    fov : float
        Field of view along ``channel`` (m).
    steps : float, optional
        Cells to traverse; non-zero, may be negative.
    system : pypulseq.Opts, optional
        System limits.
    duration : float, optional
        Force a duration (s); the default is the shortest feasible.

    Returns
    -------
    TrapEvent
        Trapezoid of area ``steps / fov``.

    Raises
    ------
    ValueError
        If ``fov`` is not positive or ``steps`` is zero.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> round(pp.make_phase_blip("y", 0.24, steps=2, system=pp.Opts()).area, 3)
    8.333

    See Also
    --------
    make_phase_encoding : the full-area encode played once per shot.
    """
    fov = float(fov)
    if not fov > 0.0:
        raise ValueError(f"fov must be > 0, got {fov}")
    if steps == 0:
        raise ValueError("steps must be non-zero")
    return _trapezoid(channel, float(steps) / fov, system, duration)


def make_crusher(
    dephasing_cycles: float,
    voxel_size: float,
    channel: str,
    grad_start: float = 0.0,
    grad_end: float = 0.0,
    convert_to_arbitrary: bool = False,
    system=None,
):
    """Crush the transverse signal by a stated number of cycles across a voxel.

    :func:`~pypulseq.make_extended_trapezoid_area` with its ``area`` replaced by
    the two numbers a sequence knows: ``dephasing_cycles / voxel_size`` is the
    area that winds that many cycles of phase across ``voxel_size`` metres along
    ``channel``. Same argument order otherwise, and the same return.

    ``grad_start`` and ``grad_end`` let the crusher ride a gradient that is
    already running -- the slice-select lobe around a refocusing pulse -- rather
    than requiring it to fall to zero first.

    Parameters
    ----------
    dephasing_cycles : float
        Cycles of phase to accrue across ``voxel_size``.
    voxel_size : float
        Length over which the dephasing is counted (m).
    channel : str
        Gradient channel (``"x"``, ``"y"`` or ``"z"``).
    grad_start, grad_end : float, optional
        Amplitudes to begin and end at (Hz/m).
    convert_to_arbitrary : bool, optional
        Return the gradient as an arbitrary waveform.
    system : pypulseq.Opts, optional
        System limits.

    Returns
    -------
    grad : GradEvent
        The crusher.
    times : numpy.ndarray
        Its vertex times (s).
    amplitudes : numpy.ndarray
        Its vertex amplitudes (Hz/m).

    Raises
    ------
    ValueError
        If ``voxel_size`` or ``dephasing_cycles`` is not positive.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> crusher, times, amplitudes = pp.make_crusher(4.0, 5e-3, "z", system=system)
    >>> round(float(amplitudes[0])), round(float(amplitudes[-1]))
    (0, 0)

    Four cycles across a 5 mm voxel is an area of 800 per metre:

    >>> import numpy as np
    >>> round(float(np.trapezoid(amplitudes, times)), 6)
    800.0

    See Also
    --------
    make_hexagon_gradient_area : the same area without holding a plateau.
    """
    dephasing_cycles = float(dephasing_cycles)
    voxel_size = float(voxel_size)
    if not voxel_size > 0.0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if not dephasing_cycles > 0.0:
        raise ValueError(f"dephasing_cycles must be > 0, got {dephasing_cycles}")

    from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area

    grad, times, amplitudes = make_extended_trapezoid_area(
        area=dephasing_cycles / voxel_size,
        channel=channel,
        grad_start=grad_start,
        grad_end=grad_end,
        convert_to_arbitrary=convert_to_arbitrary,
        system=default_system(system),
    )
    return _events.convert(grad), times, amplitudes


def make_wave_gradients(
    flat_time: float,
    cycles: int,
    amplitude: float,
    *,
    sine_channel: str | None = "y",
    cosine_channel: str | None = "z",
    delay: float = 0.0,
    return_amplitude: bool = False,
    system=None,
):
    """Build the corkscrew a wave-encoded readout plays under its flat top.

    A sinusoid on one encoded axis and a cosinusoid on the other, a quarter
    period apart, turn the readout into a corkscrew: every voxel is smeared
    along it, the further from the centre the further, so the aliasing
    parallel imaging has to separate is spread out with it (wave-CAIPI).

    Both events are **self-balanced** -- they enter and leave at zero and
    their net area is exactly zero. So scaling one to zero switches the wave
    off for one readout with no rewinder anywhere having to know whether it
    was played. A cosinusoid does not start at zero on its own, so both axes
    are brought in and out over a quarter period at each end, and whatever
    area that leaves is taken back over the same envelope.

    Parameters
    ----------
    flat_time : float
        The readout's flat top, in s, where the samples are. The corkscrew
        lives entirely inside it.
    cycles : int
        Periods of the sinusoid across the flat top.
    amplitude : float
        Requested peak, in T/m. **A ceiling, not a prescription**: a sinusoid
        of angular frequency ``w`` slews at ``amplitude * w``, so a fast
        corkscrew is bounded by the slew rate rather than by what was asked
        for, and what gets built is the lower of the two.
    sine_channel, cosine_channel : {"x", "y", "z"} or None, optional
        The channels the sine and the cosine are played on. ``None`` leaves
        that one out, which is a single-axis wave.
    delay : float, optional
        Where the flat top starts within the block, in s: the readout lobe's
        rise time.
    return_amplitude : bool, optional
        Also return the peak that survived the slew limit.
    system : Opts, optional
        System limits.

    Returns
    -------
    sine, cosine : SimpleNamespace or None
        The arbitrary gradients, ``None`` for a channel left out.
    amplitude : float
        The peak that was built, in T/m. Only with ``return_amplitude``.

    Raises
    ------
    ValueError
        If neither channel is given or both name the same one, if ``cycles``
        or ``amplitude`` is not positive, or if the flat top is too short to
        hold the cycles asked of it.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> gy, gz = pp.make_wave_gradients(3e-3, 4, 10e-3)
    >>> [abs(round(float(np.sum(g.waveform)), 6)) for g in (gy, gz)]
    [0.0, 0.0]
    """
    channels = [c for c in (sine_channel, cosine_channel) if c is not None]
    if not channels:
        raise ValueError("a wave needs a sine channel, a cosine channel or both")
    if len(set(channels)) != len(channels):
        raise ValueError("the sine and the cosine need channels of their own")
    cycles = int(cycles)
    if cycles < 1:
        raise ValueError("wave cycles must be at least one")
    if amplitude <= 0:
        raise ValueError("wave amplitude must be positive")
    system = default_system(system)

    raster = system.grad_raster_time
    n_flat = round(flat_time / raster)
    # A quarter period is what each end is brought in over, so the corkscrew
    # needs whole periods and enough raster to shape their edges.
    n_edge = n_flat // (4 * cycles)
    if n_edge < 1:
        raise ValueError(
            f"{cycles} wave cycles need at least {4 * cycles} gradient raster periods "
            f"across the readout's flat top, which holds {n_flat}"
        )

    centres = (np.arange(n_flat) + 0.5) * raster
    envelope = np.ones(n_flat)
    edge = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n_edge) + 0.5) / n_edge))
    envelope[:n_edge], envelope[-n_edge:] = edge, edge[::-1]

    rate = 2 * np.pi * cycles / (n_flat * raster)
    shapes = {
        channel: _balanced(wave(rate * centres), envelope)
        for channel, wave in ((sine_channel, np.sin), (cosine_channel, np.cos))
        if channel is not None
    }

    # The waveform is linear in its amplitude, so the slew it costs per unit of
    # amplitude is a property of the shape and the amplitude that fits follows
    # from it. Measuring beats bounding here: a sinusoid and the envelope that
    # brings it in are steepest at different moments, and adding their worst
    # cases would give away amplitude neither of them takes. The samples sit at
    # raster centres, so the steps into and out of zero cross half a raster and
    # cost twice what the interior ones do.
    steepest = max(
        float(
            np.abs(
                np.concatenate([[2.0 * shape[0]], np.diff(shape), [-2.0 * shape[-1]]])
            ).max()
        )
        / raster
        for shape in shapes.values()
    )
    peak = min(
        float(amplitude) * system.gamma, system.max_slew / steepest, system.max_grad
    )

    events = tuple(
        None
        if channel is None
        else _events.make_arbitrary_grad(
            channel=channel,
            waveform=peak * shapes[channel],
            first=0.0,
            last=0.0,
            delay=delay,
            system=system,
        )
        for channel in (sine_channel, cosine_channel)
    )
    if return_amplitude:
        return (*events, peak / system.gamma)
    return events


def _area(waveform: np.ndarray) -> float:
    """Area under an arbitrary gradient entered and left at zero, per raster.

    Its samples sit at raster centres, and what an interpreter draws is the
    straight lines between the *interval boundaries* -- which the samples do
    not carry and `restore_shape_corners` puts back, each boundary twice the
    sample before it less the boundary before that. Integrate that and the
    recurrence collapses: every interval contributes the raster times its own
    sample, and the ends are worth no less than the middle. So the area is the
    plain sum.
    """
    return float(waveform.sum())


def _balanced(shape: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """``shape`` under ``envelope``, offset until its net area is exactly zero.

    The offset rides the envelope too, so correcting the area cannot put the
    waveform's ends anywhere but zero. Everything here is affine in the
    offset, which makes finding it one division rather than a search.
    """
    at_zero = _area(shape * envelope)
    per_unit = _area(envelope)
    return (shape - at_zero / per_unit) * envelope


def _trapezoid(channel: str, area: float, system, duration: float | None):
    """One trapezoid of the given area, with the duration left free unless set."""
    arguments = {"channel": channel, "area": area, "system": default_system(system)}
    if duration is not None:
        arguments["duration"] = duration
    return _events.make_trapezoid(**arguments)
