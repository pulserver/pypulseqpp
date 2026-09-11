"""Gradient amplitude, slew, inter-block continuity and mechanical-resonance checks.

Checks use physical-axis waveforms after applying block rotations.
These checks do not establish scanner or patient safety.
"""

from __future__ import annotations

from types import SimpleNamespace

from .. import _ext as _cxx
from ._resonance import ForbiddenBand, check_mech_resonance, read_forbidden_bands

__all__ = [
    "ForbiddenBand",
    "check_grad_continuity",
    "check_max_grad",
    "check_max_slew",
    "check_mech_resonance",
    "read_forbidden_bands",
]


def _limits(seq, system):
    chosen = system if system is not None else seq.system
    if chosen is None:
        raise ValueError(
            "no limits to judge against: build the Sequence with a system= "
            "argument, or pass one here"
        )
    return chosen


def _of(system, name, fallback=0.0):
    value = getattr(system, name, None)
    return fallback if value is None else float(value)


def check_max_grad(seq, system=None) -> tuple[bool, SimpleNamespace]:
    """Return whether every gradient is within the amplitude limit.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    system : pypulseq.Opts, optional
        System limits; defaults to seq.system.

    Returns
    -------
    is_ok : bool
        True when no axis exceeds the limit.
    report : SimpleNamespace
        ``limit``; the ``per_axis`` and ``vector`` peaks; and ``axes``, the
        peak of each of x, y and z on its own. Each peak carries ``value`` in
        Hz/m, the 1-based ``block`` that plays it, and which ``axis``.

    Notes
    -----
    Block rotations are applied before per-axis peaks are evaluated. Only the
    largest per-axis peak is checked against the limit; a nonpositive limit
    disables this check. The vector peak is the maximum simultaneous Euclidean
    magnitude, not the norm of independently occurring axis peaks.
    """
    limits = _limits(seq, system)
    limit = _of(limits, "max_grad")
    found = _cxx.max_gradient(seq._native)

    report = SimpleNamespace(
        limit=limit,
        per_axis=_peak(found["per_axis"]),
        vector=_peak(found["vector"]),
        axes=[_peak(peak) for peak in found["axes"]],
    )
    return (limit <= 0.0 or report.per_axis.value <= limit), report


def check_max_slew(seq, system=None) -> tuple[bool, SimpleNamespace]:
    """Return whether every gradient is within the slew limit, within a block.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    system : pypulseq.Opts, optional
        The scanner to weigh it against; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when nothing slews too fast.
    report : SimpleNamespace
        ``limit``; the ``per_axis`` and ``vector`` slew peaks in Hz/m/s; and
        ``axes``, the peak of each of x, y and z on its own.

    Notes
    -----
    Checks within-block slew after block rotations, not boundary jumps.
    Only the largest per-axis peak is limited; a nonpositive limit disables
    this check. Vector slew is the maximum simultaneous Euclidean magnitude.
    Use check_grad_continuity for inter-block transitions.
    """
    limits = _limits(seq, system)
    limit = _of(limits, "max_slew")
    raster = _of(limits, "grad_raster_time", 10e-6)
    found = _cxx.max_slew(seq._native, max_slew=limit, grad_raster_time=raster)

    report = SimpleNamespace(
        limit=limit,
        per_axis=_peak(found["per_axis"]),
        vector=_peak(found["vector"]),
        axes=[_peak(peak) for peak in found["axes"]],
    )
    return (limit <= 0.0 or report.per_axis.value <= limit), report


def check_grad_continuity(seq, system=None) -> tuple[bool, SimpleNamespace]:
    """Return whether each gradient carries on from the block before it.

    Parameters
    ----------
    seq : Sequence
        The sequence to check.
    system : pypulseq.Opts, optional
        The scanner whose slew limit a jump is judged against; the sequence's
        own by default.

    Returns
    -------
    is_ok : bool
        True when nothing jumps and the sequence leaves its gradients at zero.
    report : SimpleNamespace
        ``limit``; the ``discontinuities`` found, each naming the ``block``,
        the ``axis``, and what the gradient goes ``before`` and ``after`` the
        jump, with the ``slew`` that step asks for; and ``ends_at_zero``.

    Notes
    -----
    Compare endpoints in physical coordinates after block rotations.
    A missing gradient is zero. Jumps are assessed over one gradient raster,
    and the final gradient must be zero. Block indices are 1-based, axes are
    zero-based integers, amplitudes are in Hz/m and slew is in Hz/m/s.
    """
    limits = _limits(seq, system)
    limit = _of(limits, "max_slew")
    raster = _of(limits, "grad_raster_time", 10e-6)
    found = _cxx.grad_continuity(seq._native, max_slew=limit, grad_raster_time=raster)

    report = SimpleNamespace(
        limit=limit,
        discontinuities=[SimpleNamespace(**jump) for jump in found["discontinuities"]],
        ends_at_zero=found["ends_at_zero"],
    )
    return (not report.discontinuities and report.ends_at_zero), report


def _peak(found: dict) -> SimpleNamespace:
    axis = found["axis"]
    return SimpleNamespace(
        value=found["value"],
        block=found["block"],
        axis="xyz"[axis] if 0 <= axis < 3 else None,
    )
