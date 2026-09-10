"""Conversion between PyPulseq namespaces and compiled event objects.

Compiled RF and arbitrary-gradient events store a normalised shape and a
scalar amplitude. Assigning signal or waveform replaces the shape and
invalidates its registration; changing amplitude preserves the shape.
Event setters do not revalidate hardware limits.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import suppress as _suppress
from types import SimpleNamespace as _SimpleNamespace
from typing import Any

import numpy as _np
import pypulseq as _pp
from pypulseq.utils.tracing import trace as _pp_trace
from pypulseq.utils.tracing import trace_enabled as _pp_trace_enabled

from . import _ext as _cxx

# Only the factories are exported: SLOTTED is built from this list and is
# documented as "upstream's factories, wrapped". ``as_namespace`` and
# ``interoperating`` are the machinery underneath and stay module-private.
__all__ = [
    "make_adc",
    "make_adiabatic_pulse",
    "make_arbitrary_grad",
    "make_arbitrary_rf",
    "make_block_pulse",
    "make_delay",
    "make_digital_output_pulse",
    "make_extended_trapezoid",
    "make_gauss_pulse",
    "make_sinc_pulse",
    "make_soft_delay",
    "make_trapezoid",
    "make_trigger",
]


#: Which converter each PyPulseq event type goes through.
_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "rf": _cxx._rf_from,
    "trap": _cxx._trap_from,
    "grad": _cxx._grad_from,
    "adc": _cxx._adc_from,
    "labelset": _cxx._label_from,
    "labelinc": _cxx._label_from,
    "trigger": _cxx._trigger_from,
    "output": _cxx._trigger_from,
    "rot3D": _cxx._rotation_from,
    "soft_delay": _cxx._soft_delay_from,
    "delay": _cxx._delay_from,
}


def convert(event: Any) -> Any:
    """Convert a PyPulseq event to a compiled event; return other objects unchanged."""
    kind = getattr(event, "type", None)
    if kind is None or isinstance(event, _cxx.Event):
        return event
    convert_one = _CONVERTERS.get(kind)
    return event if convert_one is None else convert_one(event)


def _shape_dur(tt: Any) -> float:
    """Return gradient duration from sample-centre or vertex times.

    Raster samples extend half an interval beyond the last centre. Extended
    trapezoids start at zero and end at their final vertex.
    """
    times = _np.asarray(tt, dtype=float)
    if times.size < 2:
        return 0.0
    if times[0] == 0.0:
        return float(times[-1])
    return float(times[-1] + (times[-1] - times[-2]) / 2)


#: Fields upstream reads off an event that our slots do not carry under that
#: name, per event type: how to compute each from the event we do have.
#:
#: These are the whole difference between the two representations. A trapezoid
#: is flat-topped, so its endpoints are zero and we never stored them; an
#: arbitrary gradient keeps a normalised shape beside an amplitude, so its
#: area and duration are derived rather than held; and a trigger stores the
#: numeric channel code the file format uses rather than the name.
_COMPLETIONS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "trap": {
        "first": lambda _e: 0.0,
        "last": lambda _e: 0.0,
    },
    "grad": {
        "area": lambda e: float(_np.trapezoid(e.waveform, e.tt)),
        "shape_dur": lambda e: _shape_dur(e.tt),
    },
    "trigger": {"channel": lambda e: _trigger_channel(e)},
    "output": {"channel": lambda e: _trigger_channel(e)},
}

#: Pulseq's trigger numbering as ``(control, channel_code) -> name``.
_TRIGGER_NAMES = {
    (1.0, 1.0): "osc0",
    (1.0, 2.0): "osc1",
    (1.0, 3.0): "ext1",
    (2.0, 1.0): "physio1",
    (2.0, 2.0): "physio2",
}


def _trigger_channel(event: Any) -> str:
    key = (
        float(getattr(event, "control", 0)),
        float(getattr(event, "channel_code", 0)),
    )
    return _TRIGGER_NAMES.get(key, "")


#: The readable fields of each event type, discovered once. ``dir()`` on an
#: instance is a sorted list build plus a per-name filter, and this conversion
#: runs once per event on every interop call -- millions of times over a large
#: design.
_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def as_namespace(event: Any) -> Any:
    """Convert a compiled event to a PyPulseq-compatible SimpleNamespace.

    Non-event objects are returned unchanged.
    """
    if not isinstance(event, _cxx.Event):
        return event

    kind = type(event)
    names = _FIELD_NAMES.get(kind)
    if names is None:
        names = tuple(
            name
            for name in dir(event)
            if not name.startswith("_") and name not in ("id", "shape_IDs")
        )
        _FIELD_NAMES[kind] = names

    fields = {}
    for name in names:
        try:
            fields[name] = getattr(event, name)
        except AttributeError:
            # Py_T_OBJECT_EX fields that were never assigned. Their *absence*
            # is load-bearing -- upstream branches on hasattr(event,
            # 'shape_IDs') -- so an unset one must stay unset.
            continue

    for name, derive in _COMPLETIONS.get(fields.get("type", ""), {}).items():
        if name not in fields:
            # A shape too short to derive from leaves the field unset.
            with _suppress(Exception):
                fields[name] = derive(event)

    made = _SimpleNamespace(**fields)
    # Carry the registration across when it exists, so a pre-registered event
    # keeps its ids through a helper that only reshapes it.
    for name in ("id", "shape_IDs"):
        if hasattr(event, name):
            setattr(made, name, getattr(event, name))
    return made


def _lowered(value: Any) -> Any:
    """``value`` with every event in it turned into a namespace, recursively."""
    if isinstance(value, _cxx.Event):
        return as_namespace(value)
    if isinstance(value, list):
        return [_lowered(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_lowered(item) for item in value)
    if isinstance(value, dict):
        return {key: _lowered(item) for key, item in value.items()}
    return value


def _raised(value: Any) -> Any:
    """``value`` with every namespace in it turned into an event, recursively."""
    if isinstance(value, list):
        return [_raised(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_raised(item) for item in value)
    return convert(value)


def interoperating(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable with recursive event conversion.

    Arguments become namespaces for PyPulseq type checks and deepcopy;
    returned events become compiled objects. Other values pass through.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        lowered_args = tuple(_lowered(value) for value in args)
        lowered_kwargs = {key: _lowered(value) for key, value in kwargs.items()}
        return _raised(function(*lowered_args, **lowered_kwargs))

    wrapper.__doc__ = (
        f"{function.__doc__ or ''}\n\n"
        "    Notes\n"
        "    -----\n"
        "    This is PyPulseq's function. Events go in as the namespaces it\n"
        "    expects and come back with their fields in slots. See\n"
        "    the event interoperation layer.\n"
    )
    return wrapper


def _converting(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Convert factory results, including each event in a returned tuple."""

    @functools.wraps(factory)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        made = factory(*args, **kwargs)
        if isinstance(made, tuple):
            return tuple(convert(one) for one in made)
        return convert(made)

    wrapper.__doc__ = (
        f"{factory.__doc__ or ''}\n\n"
        "    Notes\n"
        "    -----\n"
        "    This is PyPulseq's factory; the event it builds is returned with\n"
        "    its fields in slots. See the event interoperation layer.\n"
    )
    return wrapper


def _make_arbitrary_grad(
    channel: str,
    waveform,
    first=None,
    last=None,
    delay: float = 0.0,
    max_grad=None,
    max_slew=None,
    system=None,
    oversampling: bool = False,
) -> _SimpleNamespace:
    """Build the namespace fallback for make_arbitrary_grad."""
    if system is None:
        system = _pp.Opts.default
    if max_grad is None or max_grad == 0:
        max_grad = system.max_grad
    if max_slew is None or max_slew == 0:
        max_slew = system.max_slew
    if channel not in ("x", "y", "z"):
        raise ValueError(
            f"Invalid channel. Must be one of x, y or z. Passed: {channel}"
        )

    if first is None or last is None:
        if oversampling:
            if first is None:
                first = 2 * waveform[0] - waveform[1]
            if last is None:
                last = 2 * waveform[-1] - waveform[-2]
        else:
            if first is None:
                first = 0.5 * (3 * waveform[0] - waveform[1])
            if last is None:
                last = 0.5 * (3 * waveform[-1] - waveform[-2])

    if oversampling:
        edge_scale = system.grad_raster_time * 2
        pre = first - waveform[0]
        post = last - waveform[-1]
    else:
        edge_scale = system.grad_raster_time
        pre = 2 * (first - waveform[0])
        post = 2 * (waveform[-1] - last)

    slew_rate = _np.concatenate([[pre], _np.diff(waveform), [post]]) / edge_scale

    slew_peak = _np.abs(slew_rate).max()
    if slew_peak > max_slew * (1 + _pp.eps):
        raise ValueError(f"Slew rate violation {slew_peak / max_slew * 100}")
    amplitude_peak = _np.abs(waveform).max()
    if amplitude_peak > max_grad + _pp.eps:
        raise ValueError(
            f"Gradient amplitude violation {amplitude_peak / max_grad * 100}"
        )

    grad = _SimpleNamespace()
    grad.type = "grad"
    grad.channel = channel
    grad.waveform = waveform
    grad.delay = delay
    if oversampling:
        if len(waveform) % 2 == 0:
            raise ValueError(
                "When oversampling is active, waveform must have an odd number of samples"
            )
        grad.area = (waveform[::2] * system.grad_raster_time).sum()
        grad.tt = _np.arange(1, len(waveform) + 1) * 0.5 * system.grad_raster_time
        grad.shape_dur = (len(waveform) + 1) * 0.5 * system.grad_raster_time
    else:
        grad.area = (waveform * system.grad_raster_time).sum()
        grad.tt = (_np.arange(len(waveform)) + 0.5) * system.grad_raster_time
        grad.shape_dur = len(waveform) * system.grad_raster_time
    grad.first = first
    grad.last = last

    if _pp_trace_enabled():
        grad.trace = _pp_trace()

    return grad


make_adc = _converting(_pp.make_adc)
make_adiabatic_pulse = _converting(_pp.make_adiabatic_pulse)


def make_arbitrary_grad(
    channel: str,
    waveform,
    first=None,
    last=None,
    delay: float = 0.0,
    max_grad=None,
    max_slew=None,
    system=None,
    oversampling: bool = False,
):
    """Create a gradient from amplitudes sampled at raster centres.

    ``first`` and ``last`` specify boundary amplitudes and are extrapolated
    from the endpoint samples when omitted.

    Parameters
    ----------
    channel : str
        One of ``x``, ``y``, ``z``.
    waveform : numpy.ndarray
        Amplitudes at raster centres, Hz/m.
    first, last : float, optional
        Edge values; extrapolated from the end samples when omitted.
    delay : float
        Seconds before the waveform starts.
    max_grad, max_slew : float, optional
        Limits; ``system``'s when omitted or zero.
    system : Opts, optional
        ``Opts.default`` when omitted.
    oversampling : bool
        The waveform samples a grid twice as fine; its length must be odd.

    Returns
    -------
    GradEvent
        The slotted gradient event.

    Raises
    ------
    ValueError
        On an invalid channel, an amplitude or slew violation, or an even
        oversampled length.
    """
    if (
        not _pp_trace_enabled()
        and isinstance(waveform, _np.ndarray)
        and waveform.ndim == 1
        and waveform.size >= 2
    ):
        if system is None:
            system = _pp.Opts.default
        if max_grad is None or max_grad == 0:
            max_grad = system.max_grad
        if max_slew is None or max_slew == 0:
            max_slew = system.max_slew
        return _cxx._arb_grad_build(
            channel,
            waveform,
            first,
            last,
            delay,
            max_grad,
            max_slew,
            system.grad_raster_time,
            oversampling,
            float(_pp.eps),
        )
    return convert(
        _make_arbitrary_grad(
            channel,
            waveform,
            first=first,
            last=last,
            delay=delay,
            max_grad=max_grad,
            max_slew=max_slew,
            system=system,
            oversampling=oversampling,
        )
    )


make_arbitrary_rf = _converting(_pp.make_arbitrary_rf)
make_block_pulse = _converting(_pp.make_block_pulse)
make_delay = _converting(_pp.make_delay)
make_digital_output_pulse = _converting(_pp.make_digital_output_pulse)
make_extended_trapezoid = _converting(_pp.make_extended_trapezoid)
make_gauss_pulse = _converting(_pp.make_gauss_pulse)
make_sinc_pulse = _converting(_pp.make_sinc_pulse)
make_soft_delay = _converting(_pp.make_soft_delay)
make_trapezoid = _converting(_pp.make_trapezoid)
make_trigger = _converting(_pp.make_trigger)

#: ``_scale_grad(grad, scale)``: a copy of the event with its amplitude
#: multiplied, done in C++ so a phase-encode loop does not walk fields.
#: Raises TypeError on anything that is not a slotted gradient.
scaled_gradient = _cxx._scale_grad
