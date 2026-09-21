"""Gradient amplitude, slew-rate, continuity, mechanical-resonance, PNS and SAR checks.

Checks use physical-axis gradient waveforms, after applying block rotations.
These checks do not establish scanner or patient safety.
"""

from __future__ import annotations

from types import SimpleNamespace

from .. import _ext as _cxx
from ._pns import ChronaxieModel, check_pns, read_safe_model
from ._resonance import (
    ForbiddenBand,
    check_mech_resonance,
    mech_resonance_spectrum,
    read_forbidden_bands,
)
from ._sar import VopModel, check_sar, example_vops, read_vops

__all__ = [
    "ChronaxieModel",
    "ForbiddenBand",
    "VopModel",
    "check_grad_continuity",
    "check_max_grad",
    "check_max_slew",
    "check_mech_resonance",
    "check_pns",
    "check_sar",
    "example_vops",
    "mech_resonance_spectrum",
    "read_forbidden_bands",
    "read_safe_model",
    "read_vops",
]


def _limits(seq, system):
    chosen = system if system is not None else seq.system
    if chosen is None:
        raise ValueError(
            "no system limits available: build the Sequence with a system= "
            "argument, or pass one here"
        )
    return chosen


def _of(system, name, fallback=0.0):
    value = getattr(system, name, None)
    return fallback if value is None else float(value)


def check_max_grad(seq, system=None) -> tuple[bool, SimpleNamespace]:
    """Check the peak gradient amplitude against ``max_grad``.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    system : pypulseqpp.Opts, default=None
        System limits; defaults to seq.system.

    Returns
    -------
    is_ok : bool
        True when the largest per-axis amplitude does not exceed ``max_grad``.
    report : SimpleNamespace
        ``limit``; the ``per_axis`` and ``vector`` peaks; and ``axes``, the
        peak of each of x, y and z on its own. Each peak carries ``value`` in
        Hz/m, the 1-based ``block`` containing it, and its ``axis``.

    Notes
    -----
    Block rotations are applied before per-axis peaks are evaluated. Only the
    largest per-axis peak is compared with the limit; a nonpositive limit
    disables the check. The vector peak is the largest simultaneous Euclidean
    magnitude over the three physical axes; it is reported but not compared
    with a limit, and it is not the norm of independently attained axis peaks.
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
    """Check the within-block slew rate against ``max_slew``.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    system : pypulseqpp.Opts, default=None
        System limits; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when the largest per-axis slew rate does not exceed ``max_slew``.
    report : SimpleNamespace
        ``limit``; the ``per_axis`` and ``vector`` slew-rate peaks in Hz/m/s;
        and ``axes``, the peak of each of x, y and z on its own.

    Notes
    -----
    Slew rates are evaluated within each block after block rotations;
    boundaries between blocks are not covered. Only the largest per-axis peak
    is compared with the limit; a nonpositive limit disables the check. The
    vector slew rate is the largest simultaneous Euclidean magnitude, reported
    but not compared with a limit. Use :func:`check_grad_continuity` for
    transitions between blocks.
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
    """Check gradient amplitude continuity across block boundaries.

    Parameters
    ----------
    seq : Sequence
        The sequence to check.
    system : pypulseqpp.Opts, default=None
        System limits supplying the slew limit a discontinuity is compared
        with; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when no discontinuity is found and every gradient waveform ends
        at zero amplitude.
    report : SimpleNamespace
        ``limit``; the ``discontinuities`` found, each naming the ``block``,
        the ``axis``, the amplitudes ``before`` and ``after`` the boundary and
        the ``slew`` rate the step implies; and ``ends_at_zero``.

    Notes
    -----
    Endpoints are compared in physical coordinates after block rotations.
    An absent gradient contributes zero amplitude. A discontinuity is
    evaluated over one gradient raster period, and the final amplitude on
    every axis must be zero. Block indices are 1-based, axes are zero-based
    integers, amplitudes are in Hz/m and slew rates in Hz/m/s.
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
